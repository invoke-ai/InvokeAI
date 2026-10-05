"""Typed, frame-scoped execution records.

These models are internal runtime records. They deliberately do not reuse or
extend public graph/effect schema models. State-changing methods validate all
inputs before one locked mutation, and repeated identical terminal inputs are
safe to retry.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticSerializationError

_JSON_ADAPTER = TypeAdapter(Any)
_MISSING = object()
FramePart = int | str
T = TypeVar("T")


def _validate_json_value(value: Any) -> Any:
    try:
        _JSON_ADAPTER.dump_python(value, mode="json", warnings="error")
    except (PydanticSerializationError, TypeError, ValueError) as exc:
        raise ValueError("must be JSON-serializable") from exc
    return value


def _validate_name(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class _InternalModel(BaseModel):
    """Strict base for private runtime records."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def __eq__(self, other: object) -> bool:
        """Compare durable fields, excluding process-local synchronization locks."""

        if not isinstance(other, type(self)):
            return NotImplemented
        return self.model_dump(mode="python") == other.model_dump(mode="python")

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "_InternalModel":
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


class ExecutionFrame(_InternalModel):
    """Complete identity of one execution frame.

    Iteration path alone is not identity: state, generated frame ID, and call
    depth remain part of the scope key.
    """

    state_id: str = Field(description="Owning graph execution state identifier.")
    frame_id: str = Field(description="Stable identifier for this frame.")
    iteration_path: tuple[FramePart, ...] = Field(default_factory=tuple, description="Nested iteration coordinates.")
    workflow_call_depth: int = Field(
        default=0,
        ge=0,
        description="Nested workflow-call depth.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    _validate_state_id = field_validator("state_id", "frame_id")(_validate_name)

    @field_validator("iteration_path")
    @classmethod
    def _validate_iteration_path(cls, value: tuple[FramePart, ...]) -> tuple[FramePart, ...]:
        for part in value:
            if isinstance(part, bool) or not isinstance(part, (int, str)):
                raise ValueError("frame path values must be integers or strings")
            if isinstance(part, int) and part < 0:
                raise ValueError("frame path integers must be non-negative")
            if isinstance(part, str) and not part.strip():
                raise ValueError("frame path strings must not be blank")
        return value

    @property
    def execution_id(self) -> str:
        return self.state_id

    @property
    def path(self) -> tuple[FramePart, ...]:
        return self.iteration_path

    @property
    def identity(self) -> tuple[str, str, tuple[FramePart, ...], int]:
        return (self.state_id, self.frame_id, self.iteration_path, self.workflow_call_depth)

    @property
    def key(self) -> tuple[str, str, tuple[FramePart, ...], int]:
        return self.identity

    def __hash__(self) -> int:
        return hash(self.identity)


FrameIdentity = ExecutionFrame
ExecutionFrameIdentity = ExecutionFrame


ActivationStatus = Literal["pending", "resolved"]


class ActivationGate(_InternalModel):
    """One immutable-in-meaning branch decision per owner and frame.

    ``resolve`` is idempotent for the same branch. A different branch after
    resolution is rejected, including when it arrives from a retry.
    """

    gate_id: str = Field(min_length=1, description="Stable gate identifier.")
    owner_id: str = Field(description="Node or runtime owner of this gate.")
    frame: ExecutionFrame
    branches: tuple[str, ...] = Field(
        min_length=1,
        description="Allowed branch identifiers.",
    )
    status: ActivationStatus = "pending"
    selected_branch: str | None = None

    _lock: Any = PrivateAttr(default_factory=threading.RLock)

    _validate_gate_id = field_validator("gate_id", "owner_id")(_validate_name)

    @field_validator("branches")
    @classmethod
    def _validate_branches(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("branches must be unique")
        for branch in value:
            _validate_name(branch)
        return value

    @field_validator("selected_branch")
    @classmethod
    def _validate_selected_branch(cls, value: str | None) -> str | None:
        return None if value is None else _validate_name(value)

    @model_validator(mode="after")
    def _validate_resolution(self) -> "ActivationGate":
        if self.status == "pending" and self.selected_branch is not None:
            raise ValueError("pending gate cannot have a selected branch")
        if self.status == "resolved":
            if self.selected_branch is None:
                raise ValueError("resolved gate requires a selected branch")
            if self.selected_branch not in self.branches:
                raise ValueError("selected branch must be one of branches")
        return self

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def branch(self) -> str | None:
        return self.selected_branch

    def assert_scope(self, owner_id: str, frame: ExecutionFrame) -> None:
        if owner_id != self.owner_id:
            raise ValueError("activation gate owner does not match")
        if frame != self.frame:
            raise ValueError("activation gate frame does not match")

    def resolve(self, branch: str) -> bool:
        """Resolve gate once; return whether state changed."""

        _validate_name(branch)
        with self._lock:
            if branch not in self.branches:
                raise ValueError(f"unknown activation branch: {branch}")
            if self.selected_branch is None:
                object.__setattr__(self, "selected_branch", branch)
                object.__setattr__(self, "status", "resolved")
                return True
            if self.selected_branch == branch:
                return False
            raise ValueError("activation gate is already resolved")

    resolve_once = resolve

    def is_active(
        self,
        branch: str,
        *,
        owner_id: str | None = None,
        frame: ExecutionFrame | None = None,
    ) -> bool:
        with self._lock:
            if owner_id is not None and frame is not None:
                self.assert_scope(owner_id, frame)
            elif owner_id is not None:
                if owner_id != self.owner_id:
                    raise ValueError("activation gate owner does not match")
            elif frame is not None and frame != self.frame:
                raise ValueError("activation gate frame does not match")
            return self.selected_branch == branch


class StreamData(_InternalModel, Generic[T]):
    """One ordered stream data item; ``None`` is a valid value."""

    kind: Literal["data"] = "data"
    sequence: int = Field(ge=0)
    value: T

    @model_validator(mode="after")
    def _validate_value(self) -> "StreamData[T]":
        _validate_json_value(self.value)
        return self


class StreamEnd(_InternalModel):
    """End marker at sequence equal to number of data items."""

    kind: Literal["stream_end"] = "stream_end"
    sequence: int = Field(default=0, ge=0)


StreamEvent: TypeAlias = StreamData[T] | StreamEnd


class StreamBuffer(_InternalModel, Generic[T]):
    """Atomic ordered buffer for one owner/frame stream.

    Data sequences start at zero. End sequence equals data count, allowing an
    empty stream to close at sequence zero. Exact event retries are no-ops;
    conflicting duplicate sequences and gaps are rejected.
    """

    stream_id: str = Field(min_length=1)
    owner_id: str = Field(description="Node or runtime owner of this stream.")
    frame: ExecutionFrame
    events: list[StreamData[T] | StreamEnd] = Field(default_factory=list)
    next_sequence: int = Field(default=0, ge=0)
    closed: bool = False
    end_sequence: int | None = Field(default=None, ge=0)

    _lock: Any = PrivateAttr(default_factory=threading.RLock)

    _validate_stream_id = field_validator("stream_id", "owner_id")(_validate_name)

    @model_validator(mode="after")
    def _validate_buffer_state(self) -> "StreamBuffer[T]":
        expected = 0
        saw_end = False
        for event in self.events:
            if saw_end:
                raise ValueError("stream events cannot follow stream_end")
            if event.sequence != expected:
                raise ValueError("stream events must be ordered without gaps")
            if isinstance(event, StreamData):
                expected += 1
                continue
            saw_end = True
            if self.end_sequence != event.sequence:
                raise ValueError("stream end sequence does not match buffer")
        if self.next_sequence != expected:
            raise ValueError("buffer next sequence does not match events")
        if self.closed != saw_end:
            raise ValueError("buffer closed state does not match stream_end")
        if not saw_end and self.end_sequence is not None:
            raise ValueError("open buffer cannot have an end sequence")
        return self

    @property
    def values(self) -> tuple[T, ...]:
        with self._lock:
            return tuple(cast(StreamData[T], event).value for event in self.events if isinstance(event, StreamData))

    @property
    def data(self) -> tuple[T, ...]:
        return self.values

    @property
    def expected_sequence(self) -> int:
        return self.next_sequence

    def assert_scope(self, owner_id: str, frame: ExecutionFrame) -> None:
        if owner_id != self.owner_id:
            raise ValueError("stream buffer owner does not match")
        if frame != self.frame:
            raise ValueError("stream buffer frame does not match")

    def _data_model(self) -> Any:
        generic_args = getattr(type(self), "__pydantic_generic_metadata__", {}).get("args", ())
        value_type = generic_args[0] if generic_args else Any
        return StreamData[value_type] if value_type is not Any else StreamData[Any]

    def _make_data_event(self, sequence: int, value: Any) -> StreamData[T]:
        return cast(StreamData[T], self._data_model()(sequence=sequence, value=value))

    def _coerce_event(
        self,
        event: StreamData[T] | StreamEnd | str | dict[str, Any],
        *,
        value: Any = _MISSING,
        sequence: int | None = None,
        trusted: bool = False,
    ) -> StreamData[T] | StreamEnd:
        if isinstance(event, StreamData):
            if value is not _MISSING or sequence is not None:
                raise TypeError("event object cannot be combined with value or sequence")
            if trusted:
                return event
            return self._make_data_event(event.sequence, event.value)
        if isinstance(event, StreamEnd):
            if value is not _MISSING or sequence is not None:
                raise TypeError("event object cannot be combined with value or sequence")
            return event
        if isinstance(event, dict):
            kind = event.get("kind", event.get("token_kind", event.get("type")))
            if any(name in event and event[name] != kind for name in ("kind", "token_kind", "type")):
                raise ValueError("conflicting stream event kinds")
            canonical_event = {name: value for name, value in event.items() if name not in {"token_kind", "type"}}
            canonical_event["kind"] = kind
            if kind == "data":
                return cast(StreamData[T], self._data_model().model_validate(canonical_event))
            if kind == "stream_end":
                return StreamEnd.model_validate(canonical_event)
            raise ValueError("stream event kind must be data or stream_end")
        if event not in ("data", "stream_end"):
            raise ValueError("stream event kind must be data or stream_end")
        event_sequence = self.next_sequence if sequence is None else sequence
        if event == "data":
            if value is _MISSING:
                raise ValueError("data event requires a value")
            return self._make_data_event(event_sequence, value)
        if value is not _MISSING:
            raise TypeError("stream_end event cannot have a value")
        return StreamEnd(sequence=event_sequence)

    def accept(
        self,
        event: StreamData[T] | StreamEnd | str | dict[str, Any],
        *,
        value: Any = _MISSING,
        sequence: int | None = None,
        trusted: bool = False,
    ) -> bool:
        """Accept one event atomically; return false for exact retries.

        ``trusted`` is reserved for rehydration of values already decoded from a JSON snapshot. Normal engine
        inputs must use the default validated path.
        """

        with self._lock:
            candidate = self._coerce_event(event, value=value, sequence=sequence, trusted=trusted)
            existing = self.events[candidate.sequence] if candidate.sequence < len(self.events) else None
            if existing is not None:
                if existing == candidate:
                    return False
            if self.closed:
                raise ValueError("stream buffer is closed")
            if existing is not None:
                raise ValueError("duplicate stream sequence has conflicting data")
            if candidate.sequence != self.next_sequence:
                raise ValueError("stream event is out of order")

            self.events.append(candidate)
            if isinstance(candidate, StreamData):
                object.__setattr__(self, "next_sequence", self.next_sequence + 1)
            else:
                object.__setattr__(self, "closed", True)
                object.__setattr__(self, "end_sequence", candidate.sequence)
            return True

    def append_data(self, value: T, *, sequence: int | None = None) -> bool:
        return self.accept("data", value=value, sequence=sequence)

    add_data = append_data

    def close(self, *, sequence: int | None = None) -> bool:
        return self.accept("stream_end", sequence=sequence)

    stream_end = close

    def _restore(self, event_count: int, next_sequence: int, closed: bool, end_sequence: int | None) -> None:
        """Restore the mutable stream fields after a failed execution-state transaction."""

        del self.events[event_count:]
        object.__setattr__(self, "next_sequence", next_sequence)
        object.__setattr__(self, "closed", closed)
        object.__setattr__(self, "end_sequence", end_sequence)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StreamBuffer):
            return NotImplemented
        return self.model_dump(mode="json") == other.model_dump(mode="json")


ContinuationStatus = Literal["pending", "waiting", "running", "completed", "failed", "cancelled"]


class ContinuationRecord(_InternalModel, Generic[T]):
    """Generic owner/frame continuation with a small durable state machine."""

    continuation_id: str = Field(description="Stable continuation identifier.")
    owner_id: str = Field(description="Node or runtime owner of this continuation.")
    frame: ExecutionFrame
    kind: str = Field(min_length=1)
    status: ContinuationStatus = "pending"
    payload: T | None = None
    result: T | None = None
    error: str | None = None

    _lock: Any = PrivateAttr(default_factory=threading.RLock)

    _validate_ids = field_validator("continuation_id", "owner_id", "kind")(_validate_name)

    @field_validator("payload", "result")
    @classmethod
    def _validate_payload_result(cls, value: T | None) -> T | None:
        return _validate_json_value(value)

    @field_validator("error")
    @classmethod
    def _validate_error(cls, value: str | None) -> str | None:
        return None if value is None else _validate_name(value)

    @model_validator(mode="after")
    def _validate_status_data(self) -> "ContinuationRecord[T]":
        if self.status in ("pending", "waiting", "running") and (self.result is not None or self.error is not None):
            raise ValueError("active continuation cannot have result or error")
        if self.status == "completed" and self.error is not None:
            raise ValueError("completed continuation cannot have error")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed continuation requires error")
        return self

    @property
    def terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    def assert_scope(self, owner_id: str, frame: ExecutionFrame) -> None:
        if owner_id != self.owner_id:
            raise ValueError("continuation owner does not match")
        if frame != self.frame:
            raise ValueError("continuation frame does not match")

    def transition(
        self,
        status: ContinuationStatus,
        *,
        result: T | None = None,
        error: str | None = None,
    ) -> bool:
        """Apply valid transition atomically; return false for same-state retry."""

        if error is not None:
            _validate_name(error)
        _validate_json_value(result)
        allowed: dict[ContinuationStatus, frozenset[ContinuationStatus]] = {
            "pending": frozenset({"waiting", "running", "failed", "cancelled"}),
            "waiting": frozenset({"running", "failed", "cancelled"}),
            "running": frozenset({"completed", "failed", "cancelled"}),
            "completed": frozenset(),
            "failed": frozenset(),
            "cancelled": frozenset(),
        }

        with self._lock:
            if status == self.status:
                if status == "completed" and self.result != result:
                    raise ValueError("completed continuation has conflicting result")
                if status == "failed" and self.error != error:
                    raise ValueError("failed continuation has conflicting error")
                if status == "cancelled" and self.error != error:
                    raise ValueError("cancelled continuation has conflicting error")
                return False
            if status not in allowed[self.status]:
                if self.terminal:
                    raise ValueError("continuation is terminal")
                raise ValueError(f"invalid continuation transition: {self.status} to {status}")
            if status == "completed":
                if error is not None:
                    raise ValueError("completed continuation cannot have error")
            elif status == "failed":
                if error is None:
                    raise ValueError("failed continuation requires error")
                result = None
            elif status in ("pending", "waiting", "running"):
                result = None
                error = None
            object.__setattr__(self, "status", status)
            object.__setattr__(self, "result", result)
            object.__setattr__(self, "error", error)
            return True

    def wait(self) -> bool:
        return self.transition("waiting")

    def start(self) -> bool:
        return self.transition("running")

    resume = start

    def complete(self, result: T | None = None) -> bool:
        return self.transition("completed", result=result)

    def fail(self, error: str) -> bool:
        return self.transition("failed", error=error)

    def cancel(self, reason: str | None = None) -> bool:
        return self.transition("cancelled", error=reason)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContinuationRecord):
            return NotImplemented
        return self.model_dump(mode="json") == other.model_dump(mode="json")
