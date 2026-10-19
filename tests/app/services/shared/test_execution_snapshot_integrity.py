from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.services.shared.execution_effects import (
    ExecutionEffectsRecorder,
    ExecutionInterface,
)
from invokeai.app.services.shared.execution_state_migration import dump_execution_state, load_execution_state
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph, GraphExecutionState
from invokeai.app.services.shared.invocation_context import InvocationContext


def _context() -> InvocationContext:
    context = object.__new__(InvocationContext)
    context.execution_effects = ExecutionEffectsRecorder()
    context.effects = context.execution_effects
    return context


def _services() -> MagicMock:
    services = MagicMock()
    services.configuration.node_cache_size = 1
    return services


def _apply_flat_for_pair(*, apply_return: bool = True) -> GraphExecutionState:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=["item"]))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
            type="loop_linkage",
        )
    )
    state = GraphExecutionState(graph=graph)

    for_invocation = state.next()
    assert isinstance(for_invocation, ForInvocation)
    for_ref = state.get_execution_ref(for_invocation.id)
    for_context = _context()
    for_context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=for_invocation.id,
        frame_path=for_ref.frame.iteration_path,
        state_id=for_ref.state_id,
        frame_id=for_ref.frame.frame_id,
        workflow_call_depth=for_ref.frame.workflow_call_depth,
    )
    for_context.execution = ExecutionInterface(for_context.execution_effects)
    for_result = for_invocation.invoke_internal_with_effects(for_context, _services())
    state.apply(state.get_execution_ref(for_invocation.id, effect_count=len(for_result.effects)), for_result)

    return_invocation = state.next()
    assert isinstance(return_invocation, ForReturnInvocation)
    return_ref = state.get_execution_ref(return_invocation.id)
    return_context = _context()
    return_context.execution_effects = ExecutionEffectsRecorder(
        source_node_id=return_invocation.id,
        frame_path=return_ref.frame.iteration_path,
        state_id=return_ref.state_id,
        frame_id=return_ref.frame.frame_id,
        workflow_call_depth=return_ref.frame.workflow_call_depth,
    )
    return_context.execution = ExecutionInterface(return_context.execution_effects)
    return_result = return_invocation.invoke_internal_with_effects(return_context, _services())
    if apply_return:
        state.apply(
            state.get_execution_ref(return_invocation.id, effect_count=len(return_result.effects)), return_result
        )
    return state


def _execution_id(state: GraphExecutionState, source_id: str) -> str:
    return next(
        exec_id for exec_id, mapped_source_id in state.prepared_source_mapping.items() if mapped_source_id == source_id
    )


@pytest.mark.parametrize("bucket_mode", ["missing", "empty"])
def test_rejects_tampered_for_result_without_effect_bucket(bucket_mode: str) -> None:
    state = _apply_flat_for_pair()
    for_exec_id = _execution_id(state, "for")
    for_ref = state.get_execution_ref(for_exec_id)
    snapshot = deepcopy(dump_execution_state(state))
    snapshot["results"][for_exec_id]["item"] = "forged"
    if bucket_mode == "missing":
        snapshot["execution_effects"].pop(for_ref.reference_id)
    else:
        snapshot["execution_effects"][for_ref.reference_id] = []

    with pytest.raises(ValueError):
        load_execution_state(snapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_collection", ["forged"]),
        ("final_state", {"values": {"forged": True}}),
    ],
)
def test_rejects_tampered_completed_for_outputs(field: str, value: object) -> None:
    state = _apply_flat_for_pair()
    for_exec_id = _execution_id(state, "for")
    snapshot = deepcopy(dump_execution_state(state))
    snapshot["results"][for_exec_id][field] = value

    with pytest.raises(ValueError, match="For output .* authoritative"):
        load_execution_state(snapshot)


def test_rejects_empty_effect_bucket_for_pending_execution() -> None:
    state = _apply_flat_for_pair(apply_return=False)
    return_exec_id = _execution_id(state, "return")
    return_ref = state.get_execution_ref(return_exec_id)
    snapshot = deepcopy(dump_execution_state(state))
    snapshot["execution_effects"][return_ref.reference_id] = []

    with pytest.raises(ValueError, match="pending execution node"):
        load_execution_state(snapshot)


def test_rejects_effect_bucket_without_executed_marker() -> None:
    state = _apply_flat_for_pair()
    for_exec_id = _execution_id(state, "for")
    snapshot = deepcopy(dump_execution_state(state))
    snapshot["executed"].remove(for_exec_id)

    with pytest.raises(ValueError, match="executed marker"):
        load_execution_state(snapshot)


def test_rejects_unknown_effect_bucket_reference() -> None:
    state = _apply_flat_for_pair()
    snapshot = deepcopy(dump_execution_state(state))
    snapshot["execution_effects"]["unknown-reference"] = []

    with pytest.raises(ValueError, match="unknown execution reference"):
        load_execution_state(snapshot)


@pytest.mark.parametrize("source_id", ["for", "return"])
def test_rejects_persisted_loop_result_without_required_continuation_effect(source_id: str) -> None:
    state = _apply_flat_for_pair()
    exec_id = _execution_id(state, source_id)
    execution_ref = state.get_execution_ref(exec_id)
    snapshot = deepcopy(dump_execution_state(state))
    snapshot["execution_effects"].pop(execution_ref.reference_id)

    with pytest.raises(ValueError, match="must include a continuation effect"):
        load_execution_state(snapshot)


def test_accepts_explicit_legacy_snapshot_without_effect_ledger() -> None:
    state = _apply_flat_for_pair()
    snapshot = dump_execution_state(state)
    snapshot.pop("execution_state_version")
    snapshot.pop("execution_effects")

    restored = load_execution_state(snapshot)

    assert restored.results == state.results
    assert restored.execution_effects == {}
