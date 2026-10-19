"""Typed execution effects recorded while an invocation runs.

This module is deliberately independent of graph materialization. It provides
validated references and a per-invocation effect batch for the execution
engine migration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator
from pydantic_core import PydanticSerializationError

if TYPE_CHECKING:
    from invokeai.app.invocations.baseinvocation import BaseInvocationOutput

from invokeai.app.services.shared.execution_engine.child import ChildExecutionCapability

_JSON_SERIALIZER = TypeAdapter(Any)


def _validate_json_serializable(value: Any) -> Any:
    """Reject values that cannot survive GraphExecutionState JSON persistence."""
    try:
        _JSON_SERIALIZER.dump_python(value, mode="json", warnings="error")
    except (PydanticSerializationError, TypeError, ValueError) as exc:
        raise ValueError("must be JSON-serializable") from exc
    return value


def _normalize_aliases(value: Any, aliases: dict[str, tuple[str, ...]]) -> Any:
    if not isinstance(value, dict):
        return value

    data = dict(value)
    for canonical, alternative_names in aliases.items():
        names = (canonical, *alternative_names)
        present = [name for name in names if name in data]
        if not present:
            continue
        if canonical not in data:
            data[canonical] = data[present[0]]
        if any(data[name] != data[canonical] for name in present if name != canonical):
            raise ValueError(f"Conflicting values for {canonical}")
        for name in alternative_names:
            data.pop(name, None)
    return data


class _ExecutionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionToken(_ExecutionModel):
    """Stable identity for one invocation output in one iteration frame."""

    node_id: str = Field(min_length=1, description="The execution node id.")
    field: str = Field(min_length=1, description="The output field name.")
    value: Any | None = Field(default=None, description="The JSON-serializable value carried by this token.")
    frame: tuple[int | str, ...] = Field(default=(), description="The enclosing execution frame path.")
    token_kind: Literal["data", "activation", "stream_end"] = Field(
        default="data", description="The semantic kind of this token."
    )
    sequence: int | None = Field(default=None, ge=0, description="Stable order within a stream or frame.")

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_names(cls, value: Any) -> Any:
        return _normalize_aliases(
            value,
            {
                "node_id": ("invocation_id",),
                "field": ("port", "output", "output_name"),
                "frame": ("iteration_path", "frame_path"),
            },
        )

    @field_validator("node_id", "field")
    @classmethod
    def _reject_blank_names(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("frame")
    @classmethod
    def _reject_invalid_frame_parts(cls, value: tuple[int | str, ...]) -> tuple[int | str, ...]:
        if any(isinstance(part, int) and part < 0 for part in value):
            raise ValueError("frame values must be non-negative")
        if any(isinstance(part, str) and not part.strip() for part in value):
            raise ValueError("frame values must not be blank")
        return value

    @field_validator("value")
    @classmethod
    def _reject_unserializable_value(cls, value: Any | None) -> Any | None:
        return _validate_json_serializable(value)

    @property
    def invocation_id(self) -> str:
        return self.node_id

    @property
    def port(self) -> str:
        return self.field

    @property
    def output_name(self) -> str:
        return self.field

    @property
    def iteration_path(self) -> tuple[int | str, ...]:
        return self.frame

    @property
    def source_node_id(self) -> str:
        return self.node_id

    @property
    def source_field(self) -> str:
        return self.field

    @property
    def frame_path(self) -> tuple[int | str, ...]:
        return self.frame


class ExecutionRef(_ExecutionModel):
    """Frame-aware reference to an execution token."""

    token: ExecutionToken | None = None
    template_node_id: str | None = Field(default=None, min_length=1)
    execution_node_id: str | None = Field(default=None, min_length=1)
    state_id: str | None = Field(default=None, min_length=1)
    frame_path: tuple[int | str, ...] = ()
    frame_id: str | None = Field(default=None, min_length=1)
    workflow_call_depth: int | None = Field(default=None, ge=0)
    scope: Literal["iteration", "final"] = "final"

    @model_validator(mode="before")
    @classmethod
    def _accept_token_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "token" in data and data["token"] is not None:
            token = data["token"]
            if isinstance(token, BaseModel):
                token_data = token.model_dump(mode="python", warnings=False)
            else:
                token_data = token

            def _token_value(names: tuple[str, ...]) -> tuple[bool, Any]:
                if isinstance(token_data, dict):
                    for name in names:
                        if name in token_data:
                            return True, token_data[name]
                return False, None

            def _values_match(left: Any, right: Any) -> bool:
                if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
                    return tuple(left) == tuple(right)
                return left == right

            for aliases, token_names, label in (
                (("node_id", "invocation_id"), ("node_id", "invocation_id"), "node_id"),
                (("field", "port", "output", "output_name"), ("field", "port", "output", "output_name"), "field"),
                (("frame", "iteration_path"), ("frame", "iteration_path", "frame_path"), "frame"),
            ):
                token_present, token_value = _token_value(token_names)
                if not token_present:
                    continue
                for alias in aliases:
                    if alias in data and not _values_match(data[alias], token_value):
                        raise ValueError(f"Token {label} conflicts with legacy alias {alias}")

        if "token" not in data and any(
            name in data for name in ("node_id", "invocation_id", "field", "port", "output", "output_name")
        ):
            token_fields = {
                name: data.pop(name)
                for name in (
                    "node_id",
                    "invocation_id",
                    "field",
                    "port",
                    "output",
                    "output_name",
                    "frame",
                    "iteration_path",
                    "frame_path",
                )
                if name in data
            }
            data["token"] = token_fields
        if "execution_node_id" not in data:
            if "node_id" in data:
                data["execution_node_id"] = data["node_id"]
            elif isinstance(data.get("token"), dict):
                token_data = data["token"]
                data["execution_node_id"] = token_data.get("node_id", token_data.get("invocation_id"))
        if "frame_path" not in data and isinstance(data.get("token"), dict):
            token_data = data["token"]
            data["frame_path"] = token_data.get(
                "frame", token_data.get("iteration_path", token_data.get("frame_path", ()))
            )
        for alias in ("node_id", "invocation_id", "field", "port", "output", "output_name", "frame", "iteration_path"):
            data.pop(alias, None)
        return data

    @model_validator(mode="after")
    def _require_execution_identity(self) -> "ExecutionRef":
        if self.token is None and self.execution_node_id is None:
            raise ValueError("execution reference requires a token or execution_node_id")
        if self.token is not None:
            if "execution_node_id" in self.model_fields_set and self.execution_node_id not in (
                None,
                self.token.node_id,
            ):
                raise ValueError("execution_node_id conflicts with token node_id")
            if "frame_path" in self.model_fields_set and self.frame_path != self.token.frame:
                raise ValueError("frame_path conflicts with token frame")
        return self

    @property
    def node_id(self) -> str:
        if self.execution_node_id is not None:
            return self.execution_node_id
        assert self.token is not None
        return self.token.node_id

    @property
    def field(self) -> str:
        return self.token.field if self.token is not None else ""

    @property
    def frame(self) -> tuple[int | str, ...]:
        return self.frame_path or (self.token.frame if self.token is not None else ())

    @property
    def invocation_id(self) -> str:
        return self.node_id

    @property
    def output_name(self) -> str:
        return self.field

    @property
    def iteration_path(self) -> tuple[int | str, ...]:
        return self.frame


class ExecutionEffect(_ExecutionModel):
    """Base type for a recorded execution effect."""

    kind: str = Field(min_length=1)
    execution_ref: ExecutionRef | None = Field(
        default=None, description="The execution reference that owns this effect, when available."
    )

    @field_validator("kind")
    @classmethod
    def _reject_blank_kind(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class SetValueEffect(ExecutionEffect):
    kind: Literal["set_value"] = "set_value"
    target: ExecutionRef
    value: Any

    _validate_value = field_validator("value")(_validate_json_serializable)


class AddEdgeEffect(ExecutionEffect):
    kind: Literal["add_edge"] = "add_edge"
    source: ExecutionRef
    destination: ExecutionRef


class RemoveEdgeEffect(ExecutionEffect):
    kind: Literal["remove_edge"] = "remove_edge"
    source: ExecutionRef
    destination: ExecutionRef


class EmitEffect(ExecutionEffect):
    kind: Literal["emit"] = "emit"
    token: ExecutionToken
    value: Any | None = None

    _validate_value = field_validator("value")(_validate_json_serializable)


class CloseStreamEffect(ExecutionEffect):
    kind: Literal["close_stream"] = "close_stream"
    token: ExecutionToken


class SpawnExecutionEffect(ExecutionEffect):
    kind: Literal["spawn_execution"] = "spawn_execution"
    parent: ExecutionRef
    graph: Any
    inputs: dict[str, Any] = Field(default_factory=dict)
    child_execution_id: str = Field(min_length=1)
    authorization_context: dict[str, Any] | None = None

    @field_validator("graph")
    @classmethod
    def _reject_missing_graph(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("spawn graph must not be None")
        return _validate_json_serializable(value)

    @field_validator("inputs")
    @classmethod
    def _validate_input_names(cls, value: dict[str, Any]) -> dict[str, Any]:
        if any(not name.strip() for name in value):
            raise ValueError("spawn input names must not be blank")
        return _validate_json_serializable(value)

    _validate_authorization_context = field_validator("authorization_context")(_validate_json_serializable)


class AwaitEffect(ExecutionEffect):
    kind: Literal["await"] = "await"
    dependency: ExecutionRef


class FailEffect(ExecutionEffect):
    kind: Literal["fail"] = "fail"
    message: str = Field(default="")
    error_type: str | None = Field(default=None, min_length=1)
    error_traceback: str | None = None


class ContinuationEffect(ExecutionEffect):
    """A frame-scoped semantic transition for a generic control-flow continuation."""

    kind: Literal["continuation"] = "continuation"
    operation: Literal["start", "complete"]
    continuation_kind: str = Field(min_length=1)
    payload: Any | None = None

    _validate_payload = field_validator("payload")(_validate_json_serializable)


class LegacyExecutionEffect(_ExecutionModel):
    """Typed envelope for the pre-kind effect records retained for snapshot compatibility."""

    model_config = ConfigDict(extra="allow", frozen=True)
    owner_node_id: str | None = None
    source_port: str | None = None


ExecutionEffectModel = Union[
    SetValueEffect,
    AddEdgeEffect,
    RemoveEdgeEffect,
    EmitEffect,
    CloseStreamEffect,
    SpawnExecutionEffect,
    AwaitEffect,
    FailEffect,
    ContinuationEffect,
    LegacyExecutionEffect,
]

_EXECUTION_EFFECT_MODELS: dict[str, type[ExecutionEffect]] = {
    "set_value": SetValueEffect,
    "add_edge": AddEdgeEffect,
    "remove_edge": RemoveEdgeEffect,
    "emit": EmitEffect,
    "close_stream": CloseStreamEffect,
    "spawn_execution": SpawnExecutionEffect,
    "await": AwaitEffect,
    "fail": FailEffect,
    "continuation": ContinuationEffect,
}

_LEGACY_LIFECYCLE_FIELDS = frozenset(
    {
        "execution_ref",
        "execution_reference",
        "owner_ref",
        "parent",
        "dependency",
        "child_execution_id",
        "child_id",
        "graph",
        "inputs",
        "authorization_context",
        "target",
        "source",
        "destination",
        "token",
        "value",
        "message",
        "error_type",
        "error_traceback",
        "operation",
        "continuation_kind",
        "payload",
    }
)


def normalize_persisted_execution_effects(
    value: Any,
    *,
    path: str = "execution_effects",
) -> dict[str, list[ExecutionEffect]]:
    """Validate and normalize the persisted effect ledger into concrete models.

    ``effect_type`` and ``type`` were accepted names for the effect discriminator in older
    snapshots. They are migrated to ``kind`` here so the rest of the runtime only sees the
    typed, canonical representation.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping of execution reference ids to effect lists")

    normalized: dict[str, list[ExecutionEffect]] = {}
    for reference_id, effects in value.items():
        reference_path = f"{path}[{reference_id!r}]"
        if not isinstance(reference_id, str) or not reference_id.strip():
            raise ValueError(f"{reference_path} must use a non-blank string execution reference id")
        if not isinstance(effects, list):
            raise ValueError(f"{reference_path} must be a list of execution effects")

        typed_effects: list[ExecutionEffect] = []
        for index, effect in enumerate(effects):
            effect_path = f"{reference_path}[{index}]"
            if not isinstance(effect, Mapping):
                raise ValueError(f"{effect_path} must be a mapping")

            data = dict(effect)
            present_kind_fields = [name for name in ("kind", "effect_type", "type") if name in data]
            if not present_kind_fields:
                if "owner_node_id" not in data and "source_port" not in data:
                    raise ValueError(f"{effect_path} is missing an execution effect kind ('kind')")
                lifecycle_fields = sorted(_LEGACY_LIFECYCLE_FIELDS.intersection(data))
                if lifecycle_fields:
                    raise ValueError(
                        f"{effect_path} has lifecycle fields {lifecycle_fields} but no execution effect kind ('kind')"
                    )
                try:
                    typed_effects.append(LegacyExecutionEffect.model_validate(data, strict=False))
                except (ValidationError, TypeError, ValueError) as exc:
                    raise ValueError(f"{effect_path} is not a valid legacy execution effect: {exc}") from exc
                continue
            kind = data[present_kind_fields[0]]
            if any(data[name] != kind for name in present_kind_fields[1:]):
                raise ValueError(f"{effect_path} has conflicting execution effect kind aliases")
            if not isinstance(kind, str):
                raise ValueError(f"{effect_path} execution effect kind must be a string")
            model = _EXECUTION_EFFECT_MODELS.get(kind)
            if model is None:
                supported = ", ".join(sorted(_EXECUTION_EFFECT_MODELS))
                raise ValueError(
                    f"{effect_path} has unknown execution effect kind {kind!r}; supported kinds: {supported}"
                )

            data["kind"] = kind
            for alias in ("effect_type", "type"):
                data.pop(alias, None)
            try:
                typed_effects.append(model.model_validate(data, strict=False))
            except (ValidationError, TypeError, ValueError) as exc:
                raise ValueError(f"{effect_path} is not a valid {kind} effect: {exc}") from exc
        normalized[reference_id] = typed_effects
    return normalized


# Descriptive aliases keep call sites free to use execution-specific names.
SetExecutionValueEffect = SetValueEffect
AddExecutionEdgeEffect = AddEdgeEffect
RemoveExecutionEdgeEffect = RemoveEdgeEffect
EmitExecutionEffect = EmitEffect
ContinuationExecutionEffect = ContinuationEffect


class UnsupportedExecutionEffectError(RuntimeError):
    """Raised when the current graph dispatcher cannot apply an effect kind."""


class ExecutionInterface:
    """Restricted recorder facade exposed to invocation code."""

    def __init__(
        self,
        recorder: "ExecutionEffectsRecorder",
        authorize_workflow: Callable[[str], Any] | None = None,
    ) -> None:
        self._recorder = recorder
        self._authorize_workflow = authorize_workflow

    def emit(
        self,
        field: str,
        value: Any,
        *,
        frame: tuple[int | str, ...] | None = None,
        sequence: int | None = None,
        token_kind: Literal["data", "activation"] = "data",
    ) -> None:
        self._recorder.record(
            EmitEffect(
                token=ExecutionToken(
                    node_id=self._recorder.source_node_id,
                    field=field,
                    value=value,
                    frame=frame if frame is not None else self._recorder.frame_path,
                    token_kind=token_kind,
                    sequence=sequence,
                ),
                value=value,
            )
        )

    def close_stream(self, field: str, *, frame: tuple[int | str, ...] | None = None) -> None:
        self._recorder.record(
            CloseStreamEffect(
                token=ExecutionToken(
                    node_id=self._recorder.source_node_id,
                    field=field,
                    frame=frame if frame is not None else self._recorder.frame_path,
                    token_kind="stream_end",
                )
            )
        )

    def spawn(
        self,
        graph: Any,
        inputs: dict[str, Any],
        *,
        child_execution_id: str | None = None,
        authorization_context: dict[str, Any] | None = None,
    ) -> "ChildExecutionHandle":
        capability = self._require_lifecycle_capability("spawn_execution")
        if authorization_context is not None and authorization_context != capability.authorization_context:
            raise PermissionError("child authorization context does not match capability")
        child_id = child_execution_id or str(uuid4())
        child = capability.create_child(child_id, graph=graph, inputs=inputs)
        parent = self._recorder.execution_ref
        self._recorder.record(
            SpawnExecutionEffect(
                execution_ref=parent,
                parent=parent,
                graph=graph,
                inputs=inputs,
                child_execution_id=child.child_execution_id,
                authorization_context=authorization_context
                if authorization_context is not None
                else capability.authorization_context,
            )
        )
        return ChildExecutionHandle(
            child_execution_id=child.child_execution_id,
            parent_execution_id=capability.parent_execution_id,
            authorization_context=authorization_context
            if authorization_context is not None
            else capability.authorization_context,
        )

    def await_dependency(self, dependency: ExecutionRef) -> None:
        self._require_lifecycle_capability("await")
        self._recorder.record(AwaitEffect(execution_ref=self._recorder.execution_ref, dependency=dependency))

    def fail(
        self,
        message: str,
        *,
        error_type: str | None = None,
        error_traceback: str | None = None,
    ) -> None:
        self._require_lifecycle_capability("fail")
        self._recorder.record(
            FailEffect(
                execution_ref=self._recorder.execution_ref,
                message=message,
                error_type=error_type,
                error_traceback=error_traceback,
            )
        )

    def start_continuation(self, continuation_kind: str, *, payload: Any | None = None) -> None:
        self._recorder.record(
            ContinuationEffect(
                execution_ref=self._recorder.execution_ref,
                operation="start",
                continuation_kind=continuation_kind,
                payload=payload,
            )
        )

    def complete_continuation(self, continuation_kind: str, *, payload: Any | None = None) -> None:
        self._recorder.record(
            ContinuationEffect(
                execution_ref=self._recorder.execution_ref,
                operation="complete",
                continuation_kind=continuation_kind,
                payload=payload,
            )
        )

    def _require_lifecycle_capability(self, effect_kind: str) -> ChildExecutionCapability:
        if not self._recorder.allow_lifecycle_effects:
            raise UnsupportedExecutionEffectError(f"Execution effect kind '{effect_kind}' is not supported")
        capability = self._recorder.child_capability
        if capability is None:
            raise PermissionError("child execution capability is unavailable")
        if capability.parent_execution_id != self._recorder.source_node_id:
            raise PermissionError("child execution capability belongs to another execution")
        if capability.parent_frame != tuple(self._recorder.frame_path):
            raise PermissionError("child execution capability belongs to another execution frame")
        return capability

    def authorize_workflow(self, workflow_id: str) -> Any:
        if self._authorize_workflow is None:
            raise PermissionError("workflow authorization is unavailable in this execution context")
        return self._authorize_workflow(workflow_id)


class ChildExecutionHandle(_ExecutionModel):
    """Validated identity and authorization metadata for a spawned child."""

    child_execution_id: str = Field(min_length=1)
    parent_execution_id: str = Field(min_length=1)
    authorization_context: dict[str, Any] | None = None

    _validate_authorization_context = field_validator("authorization_context")(_validate_json_serializable)

    @model_validator(mode="before")
    @classmethod
    def _accept_compatibility_names(cls, value: Any) -> Any:
        return _normalize_aliases(
            value,
            {
                "child_execution_id": ("child_id", "execution_id"),
                "parent_execution_id": ("parent_id",),
                "authorization_context": ("authorization", "auth_context"),
            },
        )

    @field_validator("child_execution_id", "parent_execution_id")
    @classmethod
    def _reject_blank_ids(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @property
    def child_id(self) -> str:
        return self.child_execution_id

    @property
    def execution_id(self) -> str:
        return self.child_execution_id

    @property
    def parent_id(self) -> str:
        return self.parent_execution_id

    @property
    def authorization(self) -> dict[str, Any] | None:
        return self.authorization_context


@dataclass(frozen=True)
class ExecutionEffectBatch:
    effects: tuple[ExecutionEffect, ...]

    def __iter__(self):
        return iter(self.effects)

    def __len__(self) -> int:
        return len(self.effects)

    def __bool__(self) -> bool:
        return bool(self.effects)


class ExecutionEffectsRecorder:
    """Collects effects for one invocation run."""

    _SUPPORTED_EFFECT_KINDS = frozenset({"emit", "close_stream", "continuation"})

    def __init__(
        self,
        source_node_id: str = "context",
        frame_path: tuple[int | str, ...] = (),
        *,
        state_id: str | None = None,
        frame_id: str | None = None,
        workflow_call_depth: int | None = None,
        allow_lifecycle_effects: bool = False,
        child_capability: ChildExecutionCapability | None = None,
    ) -> None:
        self._effects: list[ExecutionEffect] = []
        self.source_node_id = source_node_id
        self.frame_path = frame_path
        self.state_id = state_id
        self.frame_id = frame_id
        self.workflow_call_depth = workflow_call_depth
        self.allow_lifecycle_effects = allow_lifecycle_effects
        self.child_capability = child_capability
        self.execution_ref = ExecutionRef(
            execution_node_id=source_node_id,
            state_id=state_id,
            frame_path=frame_path,
            frame_id=frame_id,
            workflow_call_depth=workflow_call_depth,
        )

    def record(self, effect: ExecutionEffect) -> None:
        if not isinstance(effect, ExecutionEffect):
            raise TypeError(f"Expected ExecutionEffect, got {type(effect).__name__}")
        supported_effect_kinds = self._SUPPORTED_EFFECT_KINDS
        if self.allow_lifecycle_effects:
            supported_effect_kinds = supported_effect_kinds | {"spawn_execution", "await", "fail"}
        if effect.kind not in supported_effect_kinds:
            raise UnsupportedExecutionEffectError(f"Execution effect kind '{effect.kind}' is not supported")
        try:
            effect.model_dump(mode="json", warnings="error")
        except (PydanticSerializationError, TypeError, ValueError) as exc:
            raise ValueError("Execution effect must be JSON-serializable") from exc
        self._effects.append(effect)

    record_effect = record

    def snapshot(self) -> tuple[ExecutionEffect, ...]:
        return tuple(self._effects)

    def drain(self) -> tuple[ExecutionEffect, ...]:
        """Return all recorded effects and reset the recorder."""
        effects = self.snapshot()
        self.clear()
        return effects

    def batch(self) -> ExecutionEffectBatch:
        return ExecutionEffectBatch(effects=self.snapshot())

    def clear(self) -> None:
        self._effects.clear()

    @property
    def effects(self) -> tuple[ExecutionEffect, ...]:
        return self.snapshot()


ExecutionEffectRecorder = ExecutionEffectsRecorder


@dataclass(frozen=True)
class InvocationRunResult:
    """Invocation output plus effects recorded during that run."""

    output: "BaseInvocationOutput"
    effects: tuple[ExecutionEffect, ...]

    @property
    def invocation_output(self) -> "BaseInvocationOutput":
        return self.output

    @property
    def effect_batch(self) -> ExecutionEffectBatch:
        return ExecutionEffectBatch(effects=self.effects)
