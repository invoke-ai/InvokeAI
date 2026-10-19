"""Compatibility helpers for persisted internal execution-state snapshots."""

from collections.abc import Mapping
from typing import Any, Final

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.services.shared.execution_effects import normalize_persisted_execution_effects
from invokeai.app.services.shared.graph import GraphExecutionState

CURRENT_EXECUTION_STATE_VERSION: Final[int] = 2
LEGACY_EXECUTION_STATE_VERSION: Final[int] = 0


def _retain_nullable_execution_token_values(snapshot: dict[str, Any]) -> None:
    """Retain omitted null execution values in this state and any embedded child state."""
    for output in snapshot.get("results", {}).values():
        if isinstance(output, dict) and output.get("type") == "iterate_output":
            output.setdefault("item", None)

    for token in snapshot.get("execution_tokens", {}).values():
        token.setdefault("value", None)

    child_snapshot = snapshot.get("waiting_workflow_call_child_session")
    if isinstance(child_snapshot, dict):
        _retain_nullable_execution_token_values(child_snapshot)


def _object_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _terminal_workflow_call_ids(state: GraphExecutionState) -> set[str]:
    return {
        execution.prepared_call_node_id
        for execution in state.workflow_call_history
        if execution.status in {"completed", "failed"}
    }


def _compact_effects(snapshot: dict[str, Any], state: GraphExecutionState) -> None:
    """Drop terminal saved-workflow lifecycle effects whose history is authoritative.

    Active lifecycle effects and all stream/continuation effects remain authoritative for recovery and validation.
    A terminal saved-workflow call has already recorded its outcome in parent history, so retaining the spawn effect
    would retain the child graph and inputs for no recovery purpose. Unknown effect shapes stay durable for forward
    compatibility.
    """
    references_by_id = {reference.reference_id: reference for reference in state.execution_refs.values()}
    terminal_workflow_calls = _terminal_workflow_call_ids(state)
    active_workflow_call = state.waiting_workflow_call_execution is not None
    compacted: dict[str, list[Any]] = {}
    for reference_id, effects in snapshot["execution_effects"].items():
        reference = references_by_id.get(reference_id)
        node = state.execution_graph.nodes.get(reference.exec_node_id) if reference is not None else None
        drop_workflow_lifecycle = (
            isinstance(node, CallSavedWorkflowInvocation)
            and reference is not None
            and reference.exec_node_id in terminal_workflow_calls
        )
        retained: list[Any] = []
        for effect in effects:
            effect_kind = (
                _object_value(effect, "kind") or _object_value(effect, "effect_type") or _object_value(effect, "type")
            )
            if drop_workflow_lifecycle and effect_kind in {"spawn_execution", "await", "fail"}:
                continue
            if (
                active_workflow_call
                and isinstance(node, CallSavedWorkflowInvocation)
                and effect_kind == "spawn_execution"
            ):
                compacted_effect = effect.model_dump(mode="json") if hasattr(effect, "model_dump") else dict(effect)
                child_execution_id = _object_value(effect, "child_execution_id")
                compacted_effect["graph"] = {"child_execution_id": child_execution_id}
                compacted_effect["inputs"] = {}
                retained.append(compacted_effect)
                continue
            retained.append(effect)
        if retained or not drop_workflow_lifecycle:
            compacted[reference_id] = retained
    snapshot["execution_effects"] = compacted


def _compact_child_dependencies(snapshot: dict[str, Any], state: GraphExecutionState) -> None:
    """Drop terminal dependency records after the parent has left its waiting boundary."""
    if state.waiting_workflow_call_execution is not None:
        return
    snapshot["execution_child_dependencies"] = {
        dependency_id: dependency
        for dependency_id, dependency in snapshot["execution_child_dependencies"].items()
        if _object_value(dependency, "status") not in {"completed", "failed", "canceled"}
    }


def _append_runtime_fields(snapshot: dict[str, Any], state: GraphExecutionState) -> None:
    """Add compact private runtime ledgers to an internal snapshot, including child states."""
    # References are derived from the prepared execution graph and mappings during rehydration. Do not serialize
    # them before dropping them: large loop snapshots otherwise pay the full copy/serialization cost for data that
    # never crosses the persistence boundary.
    snapshot["execution_refs"] = {}
    snapshot["execution_tokens"] = {
        token_id: token.model_dump(mode="json")
        for token_id, token in state.execution_tokens.items()
        if token.token_kind == "activation"
    }
    snapshot["execution_effects"] = {
        reference_id: [
            effect.model_dump(mode="json") if hasattr(effect, "model_dump") else effect for effect in effects
        ]
        for reference_id, effects in state.execution_effects.items()
    }
    snapshot["execution_child_dependencies"] = {
        dependency_id: dependency.model_dump(mode="json", warnings=False)
        for dependency_id, dependency in state.execution_child_dependencies.items()
    }
    for reference_id, effects in state._legacy_execution_effects_for_snapshot().items():
        snapshot["execution_effects"][reference_id] = [
            effect.model_dump(mode="json") if hasattr(effect, "model_dump") else effect for effect in effects
        ]

    _compact_effects(snapshot, state)
    _compact_child_dependencies(snapshot, state)

    if state.waiting_workflow_call_execution is not None:
        # The active child has its own queue row and that row is the recovery authority. Keeping
        # the same runtime graph under the parent doubles snapshot size and permits stale parent
        # copies to diverge from child progress.
        snapshot.pop("waiting_workflow_call_child_session", None)
        return

    child_snapshot = snapshot.get("waiting_workflow_call_child_session")
    child_state = state.waiting_workflow_call_child_session
    if isinstance(child_snapshot, dict) and child_state is not None:
        _append_runtime_fields(child_snapshot, child_state)


class UnsupportedExecutionStateVersionError(ValueError):
    """Raised when an execution-state snapshot cannot be read by this runtime."""


def dump_execution_state(state: GraphExecutionState) -> dict[str, Any]:
    """Dump an internal execution state with an additive version marker.

    The marker stays alongside the existing state fields so rolling deployments
    and diagnostic tools that still deserialize the legacy raw shape continue
    to work.
    """
    snapshot = state.model_dump(mode="json", warnings=False, exclude_none=True)
    _append_runtime_fields(snapshot, state)
    # Persist nullable output tokens explicitly. `ExecutionToken.value` remains required in the
    # public model/schema, but an output port may legitimately carry None and the general
    # exclude_none policy would otherwise make the snapshot impossible to hydrate.
    _retain_nullable_execution_token_values(snapshot)
    snapshot["execution_state_version"] = CURRENT_EXECUTION_STATE_VERSION
    return snapshot


def _normalize_persisted_effects_in_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize effect ledgers before GraphExecutionState rehydrates runtime state."""
    normalized = dict(payload)
    if "execution_effects" in normalized:
        normalized["execution_effects"] = normalize_persisted_execution_effects(normalized["execution_effects"])

    child_snapshot = normalized.get("waiting_workflow_call_child_session")
    if isinstance(child_snapshot, Mapping):
        normalized["waiting_workflow_call_child_session"] = _normalize_persisted_effects_in_snapshot(child_snapshot)
    return normalized


def load_execution_state(snapshot: Mapping[str, Any]) -> GraphExecutionState:
    """Load a flat versioned snapshot, a legacy snapshot, or a supported version/state envelope."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("Execution state snapshot must be a mapping")

    if "version" in snapshot:
        version = snapshot["version"]
        payload = snapshot.get("state")
        if not isinstance(payload, Mapping):
            raise ValueError("Versioned execution state snapshot must contain a mapping in 'state'")
    else:
        version = snapshot.get("execution_state_version", LEGACY_EXECUTION_STATE_VERSION)
        payload = dict(snapshot)
        payload.pop("execution_state_version", None)

    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Execution state snapshot version must be an integer")
    if version > CURRENT_EXECUTION_STATE_VERSION:
        raise UnsupportedExecutionStateVersionError(
            f"Execution state snapshot version {version} is newer than supported version "
            f"{CURRENT_EXECUTION_STATE_VERSION}"
        )
    if version < LEGACY_EXECUTION_STATE_VERSION:
        raise UnsupportedExecutionStateVersionError(f"Execution state snapshot version {version} is unsupported")

    legacy_execution_snapshot = version == LEGACY_EXECUTION_STATE_VERSION
    execution_effects_persisted = "execution_effects" in payload and not legacy_execution_snapshot
    normalized_payload = _normalize_persisted_effects_in_snapshot(payload)
    return GraphExecutionState.model_validate(
        normalized_payload,
        strict=False,
        context={
            "execution_effects_persisted": execution_effects_persisted,
            "legacy_execution_snapshot": legacy_execution_snapshot,
        },
    )
