"""Parent-child execution records used by workflow-call runtime state."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter, field_validator, model_validator
from pydantic_core import PydanticSerializationError

ChildTerminalStatus = Literal["completed", "failed", "canceled"]
ChildState = Literal["pending", "queued", "running", "completed", "failed", "canceled"]
ChildDependencyStatus = Literal["waiting", "running", "completed", "failed", "canceled"]
ChildFailurePolicy = Literal["fail_parent", "cancel_siblings", "continue"]
ChildCancellationPolicy = Literal["cancel_parent", "cancel_siblings", "continue"]

_JSON_SERIALIZER = TypeAdapter(Any)
_TERMINAL_STATES = frozenset({"completed", "failed", "canceled"})


def _json_safe(value: Any) -> Any:
    try:
        _JSON_SERIALIZER.dump_python(value, mode="json", warnings="error")
    except (PydanticSerializationError, TypeError, ValueError) as exc:
        raise ValueError("must be JSON-serializable") from exc
    return value


def _frame(value: Any) -> tuple[int | str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("frame must be a sequence of integers or strings")
    result: list[int | str] = []
    for part in value:
        if isinstance(part, bool) or not isinstance(part, (int, str)):
            raise ValueError("frame values must be integers or strings")
        if isinstance(part, int) and part < 0:
            raise ValueError("frame values must be non-negative")
        if isinstance(part, str) and not part.strip():
            raise ValueError("frame values must not be blank")
        result.append(part)
    return tuple(result)


def _id(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


class _ChildModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "_ChildModel":
        """Copy durable fields and runtime state without copying process-local locks."""
        memo = {} if memo is None else memo
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        object.__setattr__(copied, "__dict__", deepcopy(self.__dict__, memo))
        object.__setattr__(copied, "__pydantic_extra__", deepcopy(self.__pydantic_extra__, memo))
        object.__setattr__(copied, "__pydantic_fields_set__", deepcopy(self.__pydantic_fields_set__, memo))
        private = {}
        for name, value in (self.__pydantic_private__ or {}).items():
            private[name] = threading.RLock() if name == "_lock" else deepcopy(value, memo)
        object.__setattr__(copied, "__pydantic_private__", private)
        return copied


class ChildExecutionRecord(_ChildModel):
    """Durable identity and payload for one child execution."""

    child_execution_id: str = Field(min_length=1)
    parent_execution_id: str = Field(min_length=1)
    parent_frame: tuple[int | str, ...] = Field(default_factory=tuple)
    parent_reference_id: str | None = Field(default=None, min_length=1)
    child_frame: tuple[int | str, ...] = Field(default_factory=tuple)
    depth: int = Field(default=0, ge=0)
    enqueue_index: int = Field(default=0, ge=0)
    capability_id: str | None = Field(default=None, min_length=1)
    graph: Any | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    state: ChildState = "pending"

    _validate_parent_frame = field_validator("parent_frame", mode="before")(_frame)
    _validate_child_frame = field_validator("child_frame", mode="before")(_frame)
    _validate_graph = field_validator("graph")(_json_safe)
    _validate_inputs = field_validator("inputs")(_json_safe)

    @model_validator(mode="after")
    def _validate_child_scope(self) -> "ChildExecutionRecord":
        if self.depth < 1:
            raise ValueError("child depth must be at least one")
        if (
            len(self.child_frame) <= len(self.parent_frame)
            or self.child_frame[: len(self.parent_frame)] != self.parent_frame
        ):
            raise ValueError("child frame must descend from the authorized parent frame")
        return self

    @field_validator("child_execution_id", "parent_execution_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return _id(value, "execution id")

    @field_validator("parent_reference_id", "capability_id")
    @classmethod
    def _validate_optional_ids(cls, value: str | None) -> str | None:
        return None if value is None else _id(value, "identity")

    @field_validator("inputs")
    @classmethod
    def _validate_input_names(cls, value: dict[str, Any]) -> dict[str, Any]:
        if any(not name.strip() for name in value):
            raise ValueError("child input names must not be blank")
        return value

    @property
    def id(self) -> str:
        return self.child_execution_id


class ChildCompletion(_ChildModel):
    """One terminal child result, including its ownership proof."""

    child_execution_id: str = Field(min_length=1)
    parent_execution_id: str = Field(min_length=1)
    parent_frame: tuple[int | str, ...] = Field(default_factory=tuple)
    parent_reference_id: str | None = Field(default=None, min_length=1)
    child_frame: tuple[int | str, ...] = Field(default_factory=tuple)
    status: ChildTerminalStatus
    outputs: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None

    _validate_parent_frame = field_validator("parent_frame", mode="before")(_frame)
    _validate_child_frame = field_validator("child_frame", mode="before")(_frame)
    _validate_outputs = field_validator("outputs")(_json_safe)

    @field_validator("child_execution_id", "parent_execution_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return _id(value, "execution id")

    @field_validator("parent_reference_id")
    @classmethod
    def _validate_reference_id(cls, value: str | None) -> str | None:
        return None if value is None else _id(value, "parent reference id")

    @model_validator(mode="after")
    def _validate_terminal_message(self) -> "ChildCompletion":
        if self.status == "failed" and self.error_message is None:
            raise ValueError("failed child completion requires an error message")
        if self.error_message is not None:
            _id(self.error_message, "child error message")
        return self


@dataclass(frozen=True)
class ChildDependencyUpdate:
    """Result of applying one child terminal event.

    ``terminal`` means dependency became terminal, not merely that child
    reached a terminal state. Tuple unpacking keeps compatibility with the
    existing ``(is_complete, outputs)`` workflow-call helper shape.
    """

    changed: bool
    terminal: bool
    status: ChildDependencyStatus
    aggregated_outputs: dict[str, list[Any]]
    parent_action: Literal["failed", "canceled"] | None = None
    cancel_child_ids: list[str] | None = None
    error_message: str | None = None

    def __iter__(self):
        yield self.terminal
        yield self.aggregated_outputs

    @property
    def is_complete(self) -> bool:
        return self.terminal

    @property
    def outputs(self) -> dict[str, list[Any]]:
        return self.aggregated_outputs


class ChildExecutionCapability(_ChildModel):
    """Authorized, immutable authority to create children for one parent frame."""

    capability_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    parent_execution_id: str = Field(min_length=1)
    parent_frame: tuple[int | str, ...] = Field(default_factory=tuple)
    parent_reference_id: str | None = Field(default=None, min_length=1)
    authorized: Literal[True] = True
    authorization_context: dict[str, Any] | None = None
    depth: int = Field(default=0, ge=0)
    max_depth: int = Field(default=64, ge=0)
    max_children: int = Field(default=1000, ge=1)
    capacity: int = Field(default=1000, ge=0)
    failure_policy: ChildFailurePolicy = "fail_parent"
    cancellation_policy: ChildCancellationPolicy = "cancel_parent"

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    _validate_parent_frame = field_validator("parent_frame", mode="before")(_frame)
    _validate_authorization_context = field_validator("authorization_context")(_json_safe)

    _issued_children: dict[str, ChildExecutionRecord] = PrivateAttr(default_factory=dict)
    _lock: Any = PrivateAttr(default_factory=threading.RLock)

    @field_validator("parent_execution_id", "capability_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return _id(value, "execution id")

    @field_validator("parent_reference_id")
    @classmethod
    def _validate_reference_id(cls, value: str | None) -> str | None:
        return None if value is None else _id(value, "parent reference id")

    @model_validator(mode="after")
    def _validate_limits(self) -> "ChildExecutionCapability":
        if self.depth > self.max_depth:
            raise ValueError("depth must not exceed max_depth")
        return self

    @property
    def remaining_capacity(self) -> int:
        with self._lock:
            return max(0, min(self.capacity, self.max_children) - len(self._issued_children))

    def _check_available(self, count: int) -> None:
        with self._lock:
            issued = len(self._issued_children)
        if issued + count > self.max_children:
            raise ValueError(f"maximum child count exceeded ({self.max_children})")
        if issued + count > self.capacity:
            raise ValueError(f"child queue capacity exceeded ({self.capacity})")

    def _check_parent(
        self,
        parent_execution_id: str | None,
        parent_frame: Sequence[int | str] | None,
        parent_reference_id: str | None,
    ) -> None:
        if parent_execution_id is not None and parent_execution_id != self.parent_execution_id:
            raise ValueError("parent execution identity does not match capability")
        if parent_frame is not None and _frame(parent_frame) != self.parent_frame:
            raise ValueError("parent frame does not match capability")
        if parent_reference_id is not None and parent_reference_id != self.parent_reference_id:
            raise ValueError("parent reference identity does not match capability")

    def _check_count(self, count: int) -> None:
        if count < 1:
            raise ValueError("child dependency must contain at least one child")
        if count > self.max_children:
            raise ValueError(f"maximum child count exceeded ({self.max_children})")
        if count > self.capacity:
            raise ValueError(f"child queue capacity exceeded ({self.capacity})")
        if self.depth + 1 > self.max_depth:
            raise ValueError(f"maximum child depth exceeded ({self.max_depth})")

    def create_child(
        self,
        child_execution_id: str,
        *,
        graph: Any | None = None,
        inputs: dict[str, Any] | None = None,
        child_frame: Sequence[int | str] | None = None,
        parent_execution_id: str | None = None,
        parent_frame: Sequence[int | str] | None = None,
        parent_reference_id: str | None = None,
    ) -> ChildExecutionRecord:
        self._check_parent(parent_execution_id, parent_frame, parent_reference_id)
        self._check_count(1)
        child_id = _id(child_execution_id, "child execution id")
        resolved_frame = _frame(child_frame) if child_frame is not None else (*self.parent_frame, child_id)
        if (
            len(resolved_frame) <= len(self.parent_frame)
            or resolved_frame[: len(self.parent_frame)] != self.parent_frame
        ):
            raise ValueError("child frame must descend from the authorized parent frame")
        child = ChildExecutionRecord(
            child_execution_id=child_id,
            parent_execution_id=self.parent_execution_id,
            parent_frame=self.parent_frame,
            parent_reference_id=self.parent_reference_id,
            child_frame=resolved_frame,
            depth=self.depth + 1,
            capability_id=self.capability_id,
            graph=graph,
            inputs={} if inputs is None else inputs,
        )
        with self._lock:
            existing = self._issued_children.get(child_id)
            if existing is not None:
                if existing == child:
                    return existing.model_copy(deep=True)
                raise ValueError("child execution id was already issued with different data")
            self._check_available(1)
            self._issued_children[child_id] = child
        return child

    spawn = create_child

    def create_dependency(
        self,
        child_execution_ids: Sequence[str] | None = None,
        *,
        children: Sequence[ChildExecutionRecord] | None = None,
        graph: Any | None = None,
        inputs: dict[str, Any] | None = None,
        child_frames: Sequence[Sequence[int | str]] | None = None,
        child_graphs: Sequence[Any | None] | None = None,
        child_inputs: Sequence[dict[str, Any]] | None = None,
        dependency_id: str | None = None,
        parent_execution_id: str | None = None,
        parent_frame: Sequence[int | str] | None = None,
        parent_reference_id: str | None = None,
        failure_policy: ChildFailurePolicy | None = None,
        cancellation_policy: ChildCancellationPolicy | None = None,
    ) -> "ChildDependencyRecord":
        self._check_parent(parent_execution_id, parent_frame, parent_reference_id)
        if children is not None and child_execution_ids is not None:
            raise ValueError("provide child_execution_ids or children, not both")
        if children is not None:
            child_records = [child.model_copy(deep=True) for child in children]
            child_ids = [child.child_execution_id for child in child_records]
            if len(set(child_ids)) != len(child_ids):
                raise ValueError("child execution ids must be unique")
        else:
            if child_execution_ids is None:
                raise ValueError("child dependency requires child execution ids")
            ids = list(child_execution_ids)
            self._check_count(len(ids))
            if len(set(ids)) != len(ids):
                raise ValueError("child execution ids must be unique")
            if child_frames is not None and len(child_frames) != len(ids):
                raise ValueError("child frame count must match child count")
            if child_graphs is not None and len(child_graphs) != len(ids):
                raise ValueError("child graph count must match child count")
            if child_inputs is not None and len(child_inputs) != len(ids):
                raise ValueError("child input count must match child count")
            child_records = []
            newly_issued_ids: set[str] = set()
            try:
                for index, child_id in enumerate(ids):
                    with self._lock:
                        was_issued = child_id in self._issued_children
                    child_records.append(
                        self.create_child(
                            child_id,
                            graph=child_graphs[index] if child_graphs is not None else graph,
                            inputs=child_inputs[index] if child_inputs is not None else inputs,
                            child_frame=child_frames[index] if child_frames is not None else None,
                        ).model_copy(update={"enqueue_index": index})
                    )
                    if not was_issued:
                        newly_issued_ids.add(child_id)
            except Exception:
                with self._lock:
                    for child_id in newly_issued_ids:
                        self._issued_children.pop(child_id, None)
                raise
        self._check_count(len(child_records))
        expected_parent = self.parent_execution_id
        expected_frame = self.parent_frame
        expected_reference = self.parent_reference_id
        for index, child in enumerate(child_records):
            if child.parent_execution_id != expected_parent:
                raise ValueError("child parent execution identity does not match capability")
            if child.parent_frame != expected_frame:
                raise ValueError("child parent frame does not match capability")
            if child.parent_reference_id != expected_reference:
                raise ValueError("child parent reference identity does not match capability")
            if child.capability_id not in (None, self.capability_id):
                raise ValueError("child capability identity does not match capability")
            if child.depth != self.depth + 1:
                raise ValueError("child depth does not match capability")
            if (
                len(child.child_frame) <= len(self.parent_frame)
                or child.child_frame[: len(self.parent_frame)] != self.parent_frame
            ):
                raise ValueError("child frame must descend from the authorized parent frame")
            child_records[index] = child.model_copy(
                update={"enqueue_index": index, "capability_id": self.capability_id}
            )
        with self._lock:
            new_children = {
                child.child_execution_id: child
                for child in child_records
                if child.child_execution_id not in self._issued_children
            }
            for child in child_records:
                existing = self._issued_children.get(child.child_execution_id)
                if existing is not None and existing.model_copy(update={"enqueue_index": child.enqueue_index}) != child:
                    raise ValueError("child execution id was already issued with different data")
            self._check_available(len(new_children))
            for child_id, child in new_children.items():
                self._issued_children[child_id] = child
        return ChildDependencyRecord(
            dependency_id=dependency_id or str(uuid4()),
            parent_execution_id=expected_parent,
            parent_frame=expected_frame,
            parent_reference_id=expected_reference,
            children=child_records,
            failure_policy=failure_policy or self.failure_policy,
            cancellation_policy=cancellation_policy or self.cancellation_policy,
        )

    create_child_dependency = create_dependency
    spawn_children = create_dependency


class ChildDependencyRecord(_ChildModel):
    """All-of parent dependency with durable enqueue and terminal state."""

    dependency_id: str = Field(min_length=1)
    parent_execution_id: str = Field(min_length=1)
    parent_frame: tuple[int | str, ...] = Field(default_factory=tuple)
    parent_reference_id: str | None = Field(default=None, min_length=1)
    children: list[ChildExecutionRecord] = Field(default_factory=list)
    completions: dict[str, ChildCompletion] = Field(default_factory=dict)
    status: ChildDependencyStatus = "waiting"
    failure_policy: ChildFailurePolicy = "fail_parent"
    cancellation_policy: ChildCancellationPolicy = "cancel_parent"
    error_message: str | None = None
    output_keys: tuple[str, ...] | None = None
    terminal_transition_count: int = Field(default=0, ge=0)
    sealed: bool = True

    _lock: Any = PrivateAttr(default_factory=threading.RLock)

    _validate_parent_frame = field_validator("parent_frame", mode="before")(_frame)

    @field_validator("dependency_id", "parent_execution_id")
    @classmethod
    def _validate_ids(cls, value: str) -> str:
        return _id(value, "dependency identity")

    @field_validator("parent_reference_id")
    @classmethod
    def _validate_reference_id(cls, value: str | None) -> str | None:
        return None if value is None else _id(value, "parent reference id")

    @model_validator(mode="before")
    @classmethod
    def _accept_child_id_list(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        child_ids = data.pop("child_execution_ids", None)
        if child_ids is None:
            return data
        if data.get("children") is not None:
            raise ValueError("provide child_execution_ids or children, not both")
        parent_id = data.get("parent_execution_id")
        parent_frame = data.get("parent_frame", ())
        parent_reference = data.get("parent_reference_id")
        data["children"] = [
            {
                "child_execution_id": child_id,
                "parent_execution_id": parent_id,
                "parent_frame": parent_frame,
                "parent_reference_id": parent_reference,
                "child_frame": (*_frame(parent_frame), child_id),
                "enqueue_index": index,
            }
            for index, child_id in enumerate(child_ids)
        ]
        return data

    @model_validator(mode="after")
    def _validate_relationship(self) -> "ChildDependencyRecord":
        if not self.children:
            raise ValueError("child dependency must contain at least one child")
        child_ids = [child.child_execution_id for child in self.children]
        if len(set(child_ids)) != len(child_ids):
            raise ValueError("child execution ids must be unique")
        if any(child.parent_execution_id != self.parent_execution_id for child in self.children):
            raise ValueError("child parent execution identity does not match dependency")
        if any(child.parent_frame != self.parent_frame for child in self.children):
            raise ValueError("child parent frame does not match dependency")
        if any(child.parent_reference_id != self.parent_reference_id for child in self.children):
            raise ValueError("child parent reference identity does not match dependency")
        if any(
            len(child.child_frame) <= len(self.parent_frame)
            or child.child_frame[: len(self.parent_frame)] != self.parent_frame
            for child in self.children
        ):
            raise ValueError("child frame must descend from the parent frame")
        child_map = {child.child_execution_id: child for child in self.children}
        for child_id, completion in self.completions.items():
            if child_id not in child_map or completion.child_execution_id != child_id:
                raise ValueError("completion child identity does not match dependency")
            self._check_completion_identity(child_map[child_id], completion)
        if self.output_keys is not None:
            for completion in self.completions.values():
                if completion.status == "completed" and tuple(sorted(completion.outputs)) != self.output_keys:
                    raise ValueError("child completions returned different output keys")
        if self.terminal_transition_count > 1:
            raise ValueError("terminal transition count must be exactly once")
        if self.status == "completed":
            if len(self.completions) != len(self.children) or any(
                completion.status != "completed" for completion in self.completions.values()
            ):
                raise ValueError("completed dependency must contain completed results for every child")
        if self.status == "failed" and not self.error_message:
            raise ValueError("failed dependency requires an error message")
        if self.status in _TERMINAL_STATES and self.terminal_transition_count != 1:
            raise ValueError("terminal dependency must have exactly one terminal transition")
        if self.status not in _TERMINAL_STATES and self.terminal_transition_count != 0:
            raise ValueError("active dependency cannot have a terminal transition")
        for child_id, completion in self.completions.items():
            if child_map[child_id].state != completion.status:
                raise ValueError("child state does not match its completion")
        return self

    @property
    def child_execution_ids(self) -> list[str]:
        return [child.child_execution_id for child in self.children]

    @property
    def enqueue_order(self) -> list[str]:
        return self.child_execution_ids

    @property
    def completed_child_execution_ids(self) -> list[str]:
        return [child_id for child_id in self.child_execution_ids if child_id in self.completions]

    @property
    def expected_child_count(self) -> int:
        return len(self.children)

    @property
    def child_outputs(self) -> dict[str, dict[str, Any]]:
        return {child_id: self.completions[child_id].outputs for child_id in self.completed_child_execution_ids}

    @property
    def aggregated_outputs(self) -> dict[str, list[Any]]:
        output_keys: list[str] = []
        for child_id in self.child_execution_ids:
            completion = self.completions.get(child_id)
            if completion is None:
                continue
            for key in completion.outputs:
                if key not in output_keys:
                    output_keys.append(key)
        return {
            key: [
                self.completions[child_id].outputs[key]
                for child_id in self.completed_child_execution_ids
                if key in self.completions[child_id].outputs
            ]
            for key in output_keys
        }

    def _child(self, child_execution_id: str) -> ChildExecutionRecord:
        for child in self.children:
            if child.child_execution_id == child_execution_id:
                return child
        raise ValueError(f"child execution '{child_execution_id}' does not belong to dependency")

    def _check_completion_identity(self, child: ChildExecutionRecord, completion: ChildCompletion) -> None:
        if completion.parent_execution_id != self.parent_execution_id:
            raise ValueError("completion parent execution identity does not match dependency")
        if completion.parent_frame != self.parent_frame:
            raise ValueError("completion parent frame does not match dependency")
        if completion.parent_reference_id != self.parent_reference_id:
            raise ValueError("completion parent reference identity does not match dependency")
        if completion.child_frame != child.child_frame:
            raise ValueError("completion child frame does not match dependency")

    def _check_output_shape(self, outputs: dict[str, Any]) -> None:
        if self.output_keys is not None and tuple(sorted(outputs)) != self.output_keys:
            raise ValueError("child completions returned different output keys")

    def _update_for_terminal(self, status: ChildTerminalStatus, message: str | None) -> ChildDependencyUpdate:
        parent_action: Literal["failed", "canceled"] | None = None
        cancel_child_ids: list[str] = []
        if status == "failed":
            if self.failure_policy in ("fail_parent", "cancel_siblings"):
                parent_action = "failed"
            if self.failure_policy == "cancel_siblings":
                cancel_child_ids = [
                    child_id for child_id in self.child_execution_ids if child_id not in self.completions
                ]
        elif status == "canceled":
            if self.cancellation_policy in ("cancel_parent", "cancel_siblings"):
                parent_action = "canceled"
            if self.cancellation_policy == "cancel_siblings":
                cancel_child_ids = [
                    child_id for child_id in self.child_execution_ids if child_id not in self.completions
                ]
        return ChildDependencyUpdate(
            changed=True,
            terminal=True,
            status=self.status,
            aggregated_outputs=self.aggregated_outputs,
            parent_action=parent_action,
            cancel_child_ids=cancel_child_ids,
            error_message=message,
        )

    def _result(
        self, *, changed: bool, parent_action: Literal["failed", "canceled"] | None = None
    ) -> ChildDependencyUpdate:
        return ChildDependencyUpdate(
            changed=changed,
            terminal=self.status in _TERMINAL_STATES,
            status=self.status,
            aggregated_outputs=self.aggregated_outputs,
            parent_action=parent_action,
            cancel_child_ids=[],
            error_message=self.error_message,
        )

    def transition_terminal(self, status: ChildTerminalStatus, message: str | None = None) -> bool:
        with self._lock:
            if self.status in _TERMINAL_STATES:
                if self.status != status or self.error_message != message:
                    raise ValueError("dependency already has a different terminal state")
                return False
            if status == "failed" and not message:
                raise ValueError("failed dependency requires an error message")
            if status == "completed" and len(self.completions) != len(self.children):
                raise ValueError("completed dependency must contain every child result")
            self.status = status
            self.error_message = message
            self.terminal_transition_count += 1
            return True

    def _record_completion(self, completion: ChildCompletion) -> ChildDependencyUpdate:
        child = self._child(completion.child_execution_id)
        self._check_completion_identity(child, completion)
        existing = self.completions.get(completion.child_execution_id)
        if existing is not None:
            if existing == completion:
                return self._result(changed=False)
            raise ValueError("child already has a different terminal completion")
        if self.status in _TERMINAL_STATES:
            return self._result(changed=False)
        if completion.status == "completed":
            self._check_output_shape(completion.outputs)
        self.completions[completion.child_execution_id] = completion
        if self.output_keys is None and completion.status == "completed":
            self.output_keys = tuple(sorted(completion.outputs))
        child.state = completion.status
        self.status = "running"
        if completion.status == "failed" and self.failure_policy != "continue":
            self.transition_terminal("failed", completion.error_message)
            return self._update_for_terminal("failed", completion.error_message)
        if completion.status == "canceled" and self.cancellation_policy != "continue":
            self.transition_terminal("canceled", completion.error_message)
            return self._update_for_terminal("canceled", completion.error_message)
        if len(self.completions) == len(self.children):
            final_status: ChildTerminalStatus = "completed"
            final_message: str | None = None
            if any(item.status == "failed" for item in self.completions.values()):
                final_status = "failed"
                final_message = next(
                    (
                        item.error_message
                        for item in self.completions.values()
                        if item.status == "failed" and item.error_message
                    ),
                    "child failed",
                )
            elif any(item.status == "canceled" for item in self.completions.values()):
                final_status = "canceled"
                final_message = next(
                    (
                        item.error_message
                        for item in self.completions.values()
                        if item.status == "canceled" and item.error_message
                    ),
                    "child canceled",
                )
            self.transition_terminal(final_status, final_message)
            return self._update_for_terminal(final_status, final_message)
        return self._result(changed=True)

    def record_completion(self, completion: ChildCompletion) -> ChildDependencyUpdate:
        """Record one terminal child event atomically and idempotently."""

        with self._lock:
            return self._record_completion(completion)

    def _make_completion(
        self,
        child_execution_id: str,
        status: ChildTerminalStatus,
        outputs: dict[str, Any] | None,
        message: str | None,
        child_frame: Sequence[int | str] | None,
    ) -> ChildCompletion:
        child = self._child(child_execution_id)
        return ChildCompletion(
            child_execution_id=child_execution_id,
            parent_execution_id=self.parent_execution_id,
            parent_frame=self.parent_frame,
            parent_reference_id=self.parent_reference_id,
            child_frame=child.child_frame if child_frame is None else child_frame,
            status=status,
            outputs={} if outputs is None else outputs,
            error_message=message,
        )

    def complete_child(
        self,
        child_execution_id: str,
        outputs: dict[str, Any] | None = None,
        *,
        parent_execution_id: str | None = None,
        parent_frame: Sequence[int | str] | None = None,
        parent_reference_id: str | None = None,
        child_frame: Sequence[int | str] | None = None,
    ) -> ChildDependencyUpdate:
        self._check_identity_arguments(parent_execution_id, parent_frame, parent_reference_id)
        return self.record_completion(
            self._make_completion(child_execution_id, "completed", outputs, None, child_frame)
        )

    def fail_child(
        self,
        child_execution_id: str,
        message: str,
        *,
        parent_execution_id: str | None = None,
        parent_frame: Sequence[int | str] | None = None,
        parent_reference_id: str | None = None,
        child_frame: Sequence[int | str] | None = None,
    ) -> ChildDependencyUpdate:
        self._check_identity_arguments(parent_execution_id, parent_frame, parent_reference_id)
        return self.record_completion(self._make_completion(child_execution_id, "failed", None, message, child_frame))

    def cancel_child(
        self,
        child_execution_id: str,
        message: str = "child canceled",
        *,
        parent_execution_id: str | None = None,
        parent_frame: Sequence[int | str] | None = None,
        parent_reference_id: str | None = None,
        child_frame: Sequence[int | str] | None = None,
    ) -> ChildDependencyUpdate:
        self._check_identity_arguments(parent_execution_id, parent_frame, parent_reference_id)
        return self.record_completion(self._make_completion(child_execution_id, "canceled", None, message, child_frame))

    def _check_identity_arguments(
        self,
        parent_execution_id: str | None,
        parent_frame: Sequence[int | str] | None,
        parent_reference_id: str | None,
    ) -> None:
        if parent_execution_id is not None and parent_execution_id != self.parent_execution_id:
            raise ValueError("parent execution identity does not match dependency")
        if parent_frame is not None and _frame(parent_frame) != self.parent_frame:
            raise ValueError("parent frame does not match dependency")
        if parent_reference_id is not None and parent_reference_id != self.parent_reference_id:
            raise ValueError("parent reference identity does not match dependency")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChildDependencyRecord):
            return NotImplemented
        return self.model_dump(mode="python") == other.model_dump(mode="python")
