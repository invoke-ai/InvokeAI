import json
import re

import pytest

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.collections import RangeInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.primitives import IntegerCollectionOutput
from invokeai.app.services.session_queue.session_queue_common import get_session
from invokeai.app.services.shared.execution_effects import (
    AwaitEffect,
    EmitEffect,
    SpawnExecutionEffect,
)
from invokeai.app.services.shared.execution_effects import (
    ExecutionRef as EffectExecutionRef,
)
from invokeai.app.services.shared.execution_state_migration import (
    CURRENT_EXECUTION_STATE_VERSION,
    LEGACY_EXECUTION_STATE_VERSION,
    UnsupportedExecutionStateVersionError,
    dump_execution_state,
    load_execution_state,
)
from invokeai.app.services.shared.graph import (
    Edge,
    EdgeConnection,
    ExecutionFrame,
    ExecutionReference,
    ExecutionToken,
    Graph,
    GraphExecutionState,
    IterateInvocation,
)
from invokeai.app.services.shared.graph_validation import IterateInvocationOutput


def _make_state() -> GraphExecutionState:
    graph = Graph()
    graph.add_node(AddInvocation(id="node-id", a=1, b=2))
    execution_graph = Graph()
    execution_graph.add_node(AddInvocation(id="exec-node", a=1, b=2))
    return GraphExecutionState(
        id="state-id",
        graph=graph,
        execution_graph=execution_graph,
        executed={"node-id"},
        executed_history=["node-id"],
        errors={"node-id": "failure"},
        prepared_source_mapping={"exec-node": "node-id"},
        source_prepared_mapping={"node-id": {"exec-node"}},
        finalized_loop_contexts={("loop", (0, 1))},
        prepared_iteration_paths={"exec-node": (0, 1)},
        ready_order=["exec-node"],
        indegree={"exec-node": 0},
    )


def test_loads_legacy_unwrapped_graph_execution_state() -> None:
    state = _make_state()
    legacy_payload = state.model_dump(mode="json", warnings=False, exclude_none=True)

    restored = load_execution_state(legacy_payload)

    assert restored.model_dump(mode="json", warnings=False, exclude_none=True) == legacy_payload


def test_dumps_and_loads_versioned_execution_state_envelope() -> None:
    state = _make_state()

    snapshot = dump_execution_state(state)
    restored = load_execution_state(snapshot)

    assert snapshot["execution_state_version"] == CURRENT_EXECUTION_STATE_VERSION
    assert "state" not in snapshot
    expected = dict(snapshot)
    expected.pop("execution_state_version")
    restored_snapshot = dump_execution_state(restored)
    restored_snapshot.pop("execution_state_version")
    assert restored_snapshot == expected


def test_loads_versioned_envelope_for_queue_compatibility() -> None:
    state = _make_state()
    raw = dump_execution_state(state)
    raw.pop("execution_state_version")

    envelope = {"version": CURRENT_EXECUTION_STATE_VERSION, "state": raw}
    restored = load_execution_state(envelope)
    restored_from_queue = get_session({"session": json.dumps(envelope)})

    assert restored.id == state.id
    assert restored_from_queue.id == state.id


def test_loads_explicit_legacy_versioned_envelope() -> None:
    state = _make_state()
    raw = state.model_dump(mode="json", warnings=False, exclude_none=True)

    restored = load_execution_state({"version": LEGACY_EXECUTION_STATE_VERSION, "state": raw})

    assert restored.model_dump(mode="json", warnings=False, exclude_none=True) == raw


def test_loads_frozen_legacy_snapshot_without_runtime_ledgers() -> None:
    legacy_payload = {
        "id": "legacy-state",
        "graph": {"id": "legacy-graph", "nodes": {}, "edges": []},
        "execution_graph": {"id": "legacy-execution-graph", "nodes": {}, "edges": []},
        "executed": [],
        "executed_history": [],
        "results": {},
        "errors": {},
        "workflow_call_stack": [],
        "workflow_call_history": [],
        "prepared_source_mapping": {},
        "source_prepared_mapping": {},
    }

    restored = load_execution_state(legacy_payload)

    assert restored.id == "legacy-state"
    assert dump_execution_state(restored)["execution_state_version"] == CURRENT_EXECUTION_STATE_VERSION


def _snapshot_with_persisted_effect(effect: dict[str, object]) -> dict[str, object]:
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    state = GraphExecutionState(graph=graph)
    invocation = state.next()
    assert invocation is not None
    effect["token"]["node_id"] = invocation.id  # type: ignore[index]
    execution_ref = state.get_execution_ref(invocation.id, effect_count=1)
    state.apply(execution_ref, invocation.invoke(None), effects=[effect])
    return dump_execution_state(state)


def test_load_normalizes_supported_effect_kind_alias_to_typed_model() -> None:
    snapshot = _snapshot_with_persisted_effect(
        {
            "kind": "emit",
            "token": {"node_id": "add", "field": "value", "value": 3},
            "value": 3,
        }
    )
    reference_id = next(iter(snapshot["execution_effects"]))
    effect = snapshot["execution_effects"][reference_id][0]
    effect["effect_type"] = effect.pop("kind")

    restored = load_execution_state(snapshot)

    loaded_effect = restored.execution_effects[reference_id][0]
    assert isinstance(loaded_effect, EmitEffect)
    assert loaded_effect.kind == "emit"


def test_load_normalizes_legacy_effect_alias_to_typed_model() -> None:
    snapshot = _snapshot_with_persisted_effect(
        {
            "kind": "emit",
            "token": {"node_id": "add", "field": "value", "value": 3},
            "value": 3,
        }
    )
    snapshot.pop("execution_state_version")
    reference_id = next(iter(snapshot["execution_effects"]))
    effect = snapshot["execution_effects"][reference_id][0]
    effect["effect_type"] = effect.pop("kind")

    restored = load_execution_state(snapshot)

    assert isinstance(restored.execution_effects[reference_id][0], EmitEffect)
    assert restored.execution_effects[reference_id][0].kind == "emit"


@pytest.mark.parametrize(
    ("execution_effects", "error"),
    [
        ([], "execution_effects must be a mapping"),
        ({"reference": {}}, "must be a list"),
        ({"reference": [None]}, "must be a mapping"),
        ({"reference": [{"kind": "unknown"}]}, "unknown execution effect kind 'unknown'"),
        ({"reference": [{"kind": "emit"}]}, "is not a valid emit effect"),
        ({"reference": [{"owner_node_id": "add", "graph": {}}]}, "lifecycle fields"),
    ],
)
def test_load_rejects_malformed_persisted_effects_with_actionable_errors(execution_effects: object, error: str) -> None:
    snapshot = _snapshot_with_persisted_effect(
        {
            "kind": "emit",
            "token": {"node_id": "add", "field": "value", "value": 3},
            "value": 3,
        }
    )
    snapshot["execution_effects"] = execution_effects

    if isinstance(execution_effects, dict):
        reference_id = next(iter(snapshot["execution_effects"]))
        snapshot["execution_effects"] = {reference_id: next(iter(execution_effects.values()))}

    with pytest.raises(ValueError, match=re.escape(error)):
        load_execution_state(snapshot)


def test_rejects_future_execution_state_versions() -> None:
    snapshot = dump_execution_state(_make_state())
    snapshot["execution_state_version"] = CURRENT_EXECUTION_STATE_VERSION + 1

    with pytest.raises(UnsupportedExecutionStateVersionError, match="newer than supported"):
        load_execution_state(snapshot)


def test_round_trips_nullable_execution_token_value() -> None:
    state = _make_state()
    token = ExecutionToken(
        token_id="token-id",
        reference_id="state-id:exec-node",
        owner_node_id="exec-node",
        port="value",
        frame=ExecutionFrame(),
        value=None,
    )
    state.execution_tokens["token-id"] = token

    snapshot = dump_execution_state(state)
    assert snapshot["execution_tokens"] == {}
    restored = load_execution_state(snapshot)

    assert restored.execution_tokens == {}


def test_round_trips_nullable_execution_token_value_in_child_state() -> None:
    state = _make_state()
    child = _make_state()
    child.execution_tokens["token-id"] = ExecutionToken(
        token_id="token-id",
        reference_id="state-id:exec-node",
        owner_node_id="exec-node",
        port="value",
        frame=ExecutionFrame(),
        value=None,
    )
    state.waiting_workflow_call_child_session = child

    snapshot = dump_execution_state(state)
    child_snapshot = snapshot["waiting_workflow_call_child_session"]
    assert child_snapshot["execution_tokens"] == {}
    restored = load_execution_state(snapshot)

    assert restored.waiting_workflow_call_child_session is not None
    assert restored.waiting_workflow_call_child_session.execution_tokens == {}


def test_internal_execution_fields_are_persisted_but_not_publicly_serialized() -> None:
    state = _make_state()
    execution_ref = ExecutionReference(
        reference_id="state-id:exec-node",
        state_id="state-id",
        exec_node_id="exec-node",
        source_node_id="node-id",
        frame=ExecutionFrame(state_id="state-id", frame_id="frame-id"),
    )
    state.execution_refs["exec-node"] = execution_ref
    state.execution_tokens["token-id"] = ExecutionToken(
        token_id="token-id",
        reference_id=execution_ref.reference_id,
        owner_node_id="exec-node",
        port="value",
        frame=execution_ref.frame,
        value=3,
    )

    public_payload = {
        "properties": {
            "execution_refs": {},
            "execution_tokens": {},
            "execution_effects": {},
            "execution_child_dependencies": {},
            "graph": {},
        },
        "required": [
            "execution_refs",
            "execution_tokens",
            "execution_effects",
            "execution_child_dependencies",
            "graph",
        ],
    }
    schema_extra = GraphExecutionState.model_config["json_schema_extra"]
    assert callable(schema_extra)
    schema_extra(public_payload)
    assert "execution_refs" not in public_payload["properties"]
    assert "execution_tokens" not in public_payload["properties"]
    assert "execution_effects" not in public_payload["properties"]
    assert "execution_child_dependencies" not in public_payload["properties"]

    public_model_dump = state.model_dump(mode="json", warnings=False)
    assert {
        "execution_refs",
        "execution_tokens",
        "execution_effects",
        "execution_child_dependencies",
    }.isdisjoint(public_model_dump)

    persisted_payload = dump_execution_state(state)
    assert persisted_payload["execution_refs"] == {}
    assert persisted_payload["execution_tokens"] == {}
    assert persisted_payload["execution_effects"] == {}


def test_round_trips_pending_generic_child_dependency_and_partial_completion() -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="source-call", a=1, b=2))
    state = GraphExecutionState(graph=graph)
    state.execution_graph.add_node(AddInvocation(id="prepared-call", a=1, b=2))
    state.prepared_source_mapping["prepared-call"] = "source-call"
    frame = state.build_workflow_call_frame(exec_node_id="prepared-call", workflow_id="saved-workflow")
    state.begin_waiting_on_workflow_call(frame)
    state.attach_waiting_workflow_call_child_sessions([GraphExecutionState(graph=Graph()) for _ in range(2)])
    state.set_waiting_workflow_call_child_item_ids([20, 10])
    state.record_generic_child_completion(10, {"value": "ten"})

    restored = load_execution_state(dump_execution_state(state))

    dependency = restored._generic_child_dependencies[restored.waiting_workflow_call_execution.id]  # type: ignore[union-attr]
    assert dependency.status == "running"
    assert dependency.enqueue_order == ["20", "10"]
    assert dependency.completed_child_execution_ids == ["10"]
    assert dependency.completions["10"].outputs == {"value": "ten"}


@pytest.mark.parametrize(
    ("terminal_method", "expected_status"),
    [("fail_generic_child", "failed"), ("cancel_generic_child", "canceled")],
)
def test_compacts_terminal_generic_child_dependency_after_recovery(terminal_method: str, expected_status: str) -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="source-call", a=1, b=2))
    state = GraphExecutionState(graph=graph)
    state.execution_graph.add_node(AddInvocation(id="prepared-call", a=1, b=2))
    state.prepared_source_mapping["prepared-call"] = "source-call"
    frame = state.build_workflow_call_frame(exec_node_id="prepared-call", workflow_id="saved-workflow")
    state.begin_waiting_on_workflow_call(frame)
    state.attach_waiting_workflow_call_child_sessions([GraphExecutionState(graph=Graph())])
    state.set_waiting_workflow_call_child_item_ids([20])

    update = getattr(state, terminal_method)(20, "child terminal")
    assert update is not None
    assert update.status == expected_status
    state.end_waiting_on_workflow_call(status="failed", error_message="child terminal")

    snapshot = dump_execution_state(state)

    assert snapshot["execution_child_dependencies"] == {}


def _make_completed_iterate_state(iteration_count: int = 2) -> GraphExecutionState:
    graph = Graph()
    graph.add_node(RangeInvocation(id="range", start=0, stop=iteration_count, step=1))
    graph.add_node(IterateInvocation(id="iterate", collection=list(range(iteration_count))))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="range", field="collection"),
            destination=EdgeConnection(node_id="iterate", field="collection"),
        )
    )
    execution_graph = Graph()
    prepared_source_mapping: dict[str, str] = {"range-exec": "range"}
    source_prepared_mapping = {"range": {"range-exec"}, "iterate": set()}
    prepared_iteration_paths: dict[str, tuple[int, ...]] = {}
    results = {"range-exec": IntegerCollectionOutput(collection=list(range(iteration_count)))}
    executed = {"range-exec"}
    execution_graph.add_node(RangeInvocation(id="range-exec", start=0, stop=iteration_count, step=1))
    for index in range(iteration_count):
        exec_node_id = f"iterate-{index}"
        execution_graph.add_node(
            IterateInvocation(id=exec_node_id, collection=list(range(iteration_count)), index=index)
        )
        prepared_source_mapping[exec_node_id] = "iterate"
        source_prepared_mapping["iterate"].add(exec_node_id)
        prepared_iteration_paths[exec_node_id] = (index,)
        results[exec_node_id] = IterateInvocationOutput(item=index, index=index, total=iteration_count)
        executed.add(exec_node_id)

    state = GraphExecutionState(
        graph=graph,
        execution_graph=execution_graph,
        executed=executed,
        results=results,
        prepared_source_mapping=prepared_source_mapping,
        source_prepared_mapping=source_prepared_mapping,
        prepared_iteration_paths=prepared_iteration_paths,
    )
    for exec_node_id in prepared_source_mapping:
        reference = state.get_execution_ref(exec_node_id)
        iterate_output = results.get(exec_node_id)
        if not isinstance(iterate_output, IterateInvocationOutput):
            continue
        state.execution_tokens[f"{reference.reference_id}:item"] = ExecutionToken(
            token_id=f"{reference.reference_id}:item",
            reference_id=reference.reference_id,
            owner_node_id=exec_node_id,
            port="item",
            frame=reference.frame,
            value=iterate_output.item,
        )
        effects = [
            {
                "kind": "emit",
                "token": {
                    "node_id": exec_node_id,
                    "field": "item",
                    "value": iterate_output.item,
                    "sequence": iterate_output.index,
                },
                "value": iterate_output.item,
            }
        ]
        if iterate_output.index + 1 >= iterate_output.total:
            effects.append(
                {
                    "kind": "close_stream",
                    "token": {
                        "node_id": exec_node_id,
                        "field": "item",
                        "token_kind": "stream_end",
                        "sequence": iteration_count,
                    },
                }
            )
        state.execution_effects[reference.reference_id] = effects
    return state


def test_compact_iterate_ledgers_have_deterministic_record_bound_and_rehydrate() -> None:
    snapshot = dump_execution_state(_make_completed_iterate_state())

    assert snapshot["execution_refs"] == {}
    assert sum(len(effects) for effects in snapshot["execution_effects"].values()) == 3
    assert snapshot["execution_tokens"] == {}
    assert len(json.dumps(snapshot, sort_keys=True, separators=(",", ":"))) < 5000

    restored = load_execution_state(snapshot)

    stream = next(iter(restored._generic_runtime().streams.values()))
    assert stream.values == (0, 1)
    assert stream.closed
    assert restored.is_complete()


def test_compact_iterate_ledger_has_bounded_runtime_metadata_at_100_iterations() -> None:
    snapshot = dump_execution_state(_make_completed_iterate_state(iteration_count=100))

    assert snapshot["execution_refs"] == {}
    assert snapshot["execution_tokens"] == {}
    assert sum(len(effects) for effects in snapshot["execution_effects"].values()) == 101
    assert len(json.dumps(snapshot, sort_keys=True, separators=(",", ":"))) < 100_000


def test_completed_workflow_call_snapshot_does_not_retain_child_graph() -> None:
    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call", workflow_id="saved"))
    parent = GraphExecutionState(graph=graph)
    call = parent.next()
    assert call is not None
    frame = parent.build_workflow_call_frame(call.id, "saved")
    parent.begin_waiting_on_workflow_call(frame)
    child_graph = Graph()
    child_graph.add_node(AddInvocation(id="child-node", a=1, b=2))
    child = GraphExecutionState(graph=child_graph)
    parent.attach_waiting_workflow_call_child_session(child)
    execution_ref = parent._expected_execution_ref(call.id)
    parent.execution_refs[call.id] = execution_ref
    parent.execution_effects[execution_ref.reference_id] = [
        SpawnExecutionEffect(
            parent=EffectExecutionRef(
                execution_node_id=call.id,
                state_id=parent.id,
                frame_path=execution_ref.frame.iteration_path,
                frame_id=execution_ref.frame.frame_id,
                workflow_call_depth=execution_ref.frame.workflow_call_depth,
            ),
            graph=child.graph.model_dump(mode="json"),
            child_execution_id=child.id,
        ),
        AwaitEffect(
            dependency=EffectExecutionRef(
                execution_node_id=call.id,
                state_id=parent.id,
                frame_path=execution_ref.frame.iteration_path,
                frame_id=execution_ref.frame.frame_id,
                workflow_call_depth=execution_ref.frame.workflow_call_depth,
            )
        ),
    ]
    parent.end_waiting_on_workflow_call()
    parent.executed.add(call.id)

    snapshot = dump_execution_state(parent)

    assert "waiting_workflow_call_child_session" not in snapshot
    assert snapshot["workflow_call_history"][0]["child_session_id"] == child.id
    assert execution_ref.reference_id not in snapshot["execution_effects"]

    active_parent = GraphExecutionState(graph=parent.graph.model_copy(deep=True))
    active_call = active_parent.next()
    assert active_call is not None
    active_frame = active_parent.build_workflow_call_frame(active_call.id, "saved")
    active_parent.begin_waiting_on_workflow_call(active_frame)
    active_child = GraphExecutionState(graph=child_graph.model_copy(deep=True))
    active_parent.attach_waiting_workflow_call_child_session(active_child)
    active_ref = active_parent._expected_execution_ref(active_call.id)
    active_parent.execution_refs[active_call.id] = active_ref
    active_parent.execution_effects[active_ref.reference_id] = [
        SpawnExecutionEffect(
            parent=EffectExecutionRef(
                execution_node_id=active_call.id,
                state_id=active_parent.id,
                frame_path=active_ref.frame.iteration_path,
                frame_id=active_ref.frame.frame_id,
                workflow_call_depth=active_ref.frame.workflow_call_depth,
            ),
            graph=active_child.graph.model_dump(mode="json"),
            child_execution_id=active_child.id,
        )
    ]
    active_snapshot = dump_execution_state(active_parent)
    assert active_ref.reference_id in active_snapshot["execution_effects"]
    active_spawn = active_snapshot["execution_effects"][active_ref.reference_id][0]
    assert active_spawn["graph"] == {"child_execution_id": active_child.id}
    assert active_spawn["inputs"] == {}
    assert "waiting_workflow_call_child_session" not in active_snapshot


def test_failed_workflow_call_snapshot_does_not_retain_lifecycle_effects() -> None:
    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call", workflow_id="saved"))
    parent = GraphExecutionState(graph=graph)
    call = parent.next()
    assert call is not None
    frame = parent.build_workflow_call_frame(call.id, "saved")
    parent.begin_waiting_on_workflow_call(frame)
    execution_ref = parent._expected_execution_ref(call.id)
    parent.execution_refs[call.id] = execution_ref
    parent.execution_effects[execution_ref.reference_id] = [
        SpawnExecutionEffect(
            parent=EffectExecutionRef(
                execution_node_id=call.id,
                state_id=parent.id,
                frame_path=execution_ref.frame.iteration_path,
                frame_id=execution_ref.frame.frame_id,
                workflow_call_depth=execution_ref.frame.workflow_call_depth,
            ),
            graph=Graph().model_dump(mode="json"),
            child_execution_id="child-id",
        ),
        AwaitEffect(
            dependency=EffectExecutionRef(
                execution_node_id=call.id,
                state_id=parent.id,
                frame_path=execution_ref.frame.iteration_path,
                frame_id=execution_ref.frame.frame_id,
                workflow_call_depth=execution_ref.frame.workflow_call_depth,
            )
        ),
    ]
    parent.end_waiting_on_workflow_call(status="failed", error_message="child failed")
    parent.set_node_error(call.id, "child failed")

    snapshot = dump_execution_state(parent)

    assert parent.workflow_call_history[0].status == "failed"
    assert execution_ref.reference_id not in snapshot["execution_effects"]
