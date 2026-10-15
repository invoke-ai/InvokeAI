from __future__ import annotations

from unittest.mock import Mock

import pytest

from invokeai.app.invocations.collections import CollectionConcatInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.math import AddInvocation, MultiplyInvocation
from invokeai.app.invocations.primitives import BooleanInvocation
from invokeai.app.services.shared.execution_engine.scheduler import (
    ActivationDependency,
    ExecutionPlan,
    ExecutionScheduler,
)
from invokeai.app.services.shared.execution_state_migration import dump_execution_state, load_execution_state
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    Edge,
    EdgeConnection,
    Graph,
    GraphExecutionState,
    IterateInvocation,
    _ExecutionScheduler,
    _GenericGraphSchedulerAdapter,
)


def _plan() -> ExecutionPlan:
    plan = ExecutionPlan()
    plan.add_node("left", "AddInvocation")
    plan.add_node("right", "AddInvocation")
    plan.add_node("join", "MultiplyInvocation", dependencies=("left", "right"))
    return plan


def test_generic_scheduler_routes_fanout_and_fanin_without_invocation_types() -> None:
    plan = _plan()
    scheduler = ExecutionScheduler(plan)

    scheduler.enqueue("left")
    scheduler.enqueue("right")
    assert scheduler.pop_next() == "left"
    assert scheduler.pop_next() == "right"
    assert scheduler.pop_next() is None

    assert scheduler.complete("left") == ()
    assert scheduler.complete("right") == ("join",)
    assert scheduler.pop_next() == "join"
    assert scheduler.complete("join") == ()


def test_generic_scheduler_honors_opaque_readiness_predicate() -> None:
    plan = ExecutionPlan()
    plan.add_node("source", "Source")
    plan.add_node("blocked", "Work", dependencies=("source",))
    plan.add_node("other", "Work")
    allow_blocked = False
    scheduler = ExecutionScheduler(
        plan,
        ready_order=("Source", "Work"),
        ready_predicate=lambda node_id: node_id != "blocked" or allow_blocked,
    )

    assert scheduler.pop_next() == "source"
    assert scheduler.complete("source") == ()
    assert scheduler.ready_ids == ("other",)
    assert scheduler.pop_next() == "other"
    assert scheduler.pop_next() is None

    allow_blocked = True
    scheduler.rebuild_ready()
    assert scheduler.pop_next() == "blocked"


def test_generic_scheduler_discards_node_with_unmet_prerequisites() -> None:
    plan = ExecutionPlan()
    plan.add_node("prerequisite", "Prerequisite")
    plan.add_node("discarded", "Skipped", dependencies=("prerequisite",))
    plan.add_node("dependent", "Dependent", dependencies=("discarded",))
    scheduler = ExecutionScheduler(plan, ready_order=("Dependent", "Prerequisite"))

    assert scheduler.discard("discarded") == ("dependent",)
    assert scheduler.executed == set()
    assert scheduler.discarded == {"discarded"}
    assert scheduler.pop_next() == "dependent"


def test_generic_plan_round_trips_frame_local_activation_dependencies() -> None:
    plan = ExecutionPlan()
    plan.add_node("if", "If")
    dependency = ActivationDependency(owner_id="if", branch="true_input", frame=(0, "inner"))
    plan.add_node("branch", "Work", dependencies=("if",), activation_dependencies=(dependency,))

    restored = ExecutionPlan.from_snapshot(plan.snapshot())

    assert restored.nodes["branch"].activation_dependencies == (dependency,)


def test_generic_plan_rejects_malformed_activation_dependency_snapshot() -> None:
    plan = ExecutionPlan()
    plan.add_node("if", "If")
    plan.add_node(
        "branch",
        "Work",
        dependencies=("if",),
        activation_dependencies=(ActivationDependency(owner_id="if", branch="true_input"),),
    )
    snapshot = plan.snapshot()
    snapshot["nodes"]["branch"]["activation_dependencies"][0]["frame"] = "not-a-frame"

    with pytest.raises(ValueError, match="activation dependency frame must be a sequence"):
        ExecutionPlan.from_snapshot(snapshot)


def test_graph_state_generic_if_readiness_uses_activation_dependency_records(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(AddInvocation(id="true_value", a=2, b=2))
    graph.add_node(AddInvocation(id="false_value", a=3, b=3))
    graph.add_node(IfInvocation(id="if"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="condition", field="value"),
            destination=EdgeConnection(node_id="if", field="condition"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="true_value", field="value"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="false_value", field="value"),
            destination=EdgeConnection(node_id="if", field="false_input"),
        )
    )

    state = GraphExecutionState(graph=graph)
    condition = state.next()
    assert condition is not None
    state.complete(condition.id, condition.invoke(Mock()))

    while "true_value" not in state.source_prepared_mapping:
        assert state._materializer().prepare(state._get_source_graph_flat()) is not None
    true_exec_id = next(iter(state.source_prepared_mapping["true_value"]))
    plan_node = state._generic_graph_scheduler._scheduler.plan.nodes[true_exec_id]
    assert plan_node.activation_dependencies == (ActivationDependency(owner_id="if", branch="true_input", frame=()),)
    assert "false_value" not in state.source_prepared_mapping

    if_exec_id = next(
        execution_id for execution_id, source_id in state.prepared_source_mapping.items() if source_id == "if"
    )
    assert state._activation_gate(if_exec_id).selected_branch == "true_input"
    assert any(
        token.owner_node_id == if_exec_id
        and token.port == "true_input"
        and token.value == "true_input"
        and token.token_kind == "activation"
        for token in state.execution_tokens.values()
    )
    activation_dependency = plan_node.activation_dependencies[0]
    assert state._is_activation_dependency_satisfied(activation_dependency)
    activation_token_id, activation_token = next(
        (token_id, token)
        for token_id, token in state.execution_tokens.items()
        if token.owner_node_id == if_exec_id and token.token_kind == "activation"
    )
    state.execution_tokens.pop(activation_token_id)
    assert not state._is_activation_dependency_satisfied(activation_dependency)
    assert not state._is_activation_dependency_satisfied(
        ActivationDependency(owner_id="if", branch="true_input", frame=(1,))
    )

    def fail_if_topology_is_consulted(*_: object, **__: object) -> bool:
        raise AssertionError("generic readiness consulted the If topology helper")

    monkeypatch.setattr(GraphExecutionState, "_is_deferred_by_unresolved_if", fail_if_topology_is_consulted)
    state._generic_graph_scheduler._scheduler.rebuild_ready()
    assert state.next() is None

    stale_token = activation_token.model_copy(update={"reference_id": "stale-reference"})
    state.execution_tokens[activation_token_id] = stale_token
    assert not state._is_activation_dependency_satisfied(activation_dependency)
    state.execution_tokens[activation_token_id] = activation_token
    stale_token_id = activation_token.model_copy(update={"token_id": "stale-token-id"})
    state.execution_tokens[activation_token_id] = stale_token_id
    assert not state._is_activation_dependency_satisfied(activation_dependency)
    state.execution_tokens[activation_token_id] = activation_token
    state._generic_graph_scheduler._scheduler.rebuild_ready()
    next_node = state.next()
    assert next_node is not None
    assert state.prepared_source_mapping[next_node.id] == "true_value"


def test_graph_state_rehydrates_legacy_if_without_activation_token() -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(AddInvocation(id="true_value", a=2, b=2))
    graph.add_node(AddInvocation(id="false_value", a=3, b=3))
    graph.add_node(IfInvocation(id="if"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="condition", field="value"),
            destination=EdgeConnection(node_id="if", field="condition"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="true_value", field="value"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="false_value", field="value"),
            destination=EdgeConnection(node_id="if", field="false_input"),
        )
    )

    state = GraphExecutionState(graph=graph)
    condition = state.next()
    assert condition is not None
    state.complete(condition.id, condition.invoke(Mock()))
    next_node = state.next()
    assert next_node is not None
    assert state.prepared_source_mapping[next_node.id] == "true_value"
    snapshot = state.model_dump(mode="python")
    snapshot["execution_tokens"] = {}

    restored = GraphExecutionState.model_validate(snapshot, strict=False)

    restored_if_id = next(
        execution_id for execution_id, source_id in restored.prepared_source_mapping.items() if source_id == "if"
    )
    assert restored._activation_gate(restored_if_id).selected_branch == "true_input"
    assert any(
        token.owner_node_id == restored_if_id and token.token_kind == "activation" and token.port == "true_input"
        for token in restored.execution_tokens.values()
    )
    next_node = restored.next()
    assert next_node is not None
    assert restored.prepared_source_mapping[next_node.id] == "true_value"


def test_graph_state_loads_legacy_if_with_prepared_skipped_branch() -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=False))
    graph.add_node(AddInvocation(id="true_value", a=2, b=2))
    graph.add_node(AddInvocation(id="false_value", a=3, b=3))
    graph.add_node(IfInvocation(id="if"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="condition", field="value"),
            destination=EdgeConnection(node_id="if", field="condition"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="true_value", field="value"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="false_value", field="value"),
            destination=EdgeConnection(node_id="if", field="false_input"),
        )
    )

    state = GraphExecutionState(graph=graph)
    condition = state.next()
    assert condition is not None
    state.complete(condition.id, condition.invoke(Mock()))
    selected_branch = state.next()
    assert selected_branch is not None
    selected_branch_exec_id = selected_branch.id
    assert state.prepared_source_mapping[selected_branch_exec_id] == "false_value"

    # Recreate the prepared/skipped branch projection written by the legacy If scheduler.
    skipped_exec_id = state._materializer().create_execution_node(
        "true_value", [], iteration_path=(), enforce_admission=False
    )[0]
    assert state._get_prepared_exec_metadata(skipped_exec_id).state == "skipped"

    snapshot = dump_execution_state(state)
    snapshot.pop("execution_state_version")
    snapshot.pop("execution_effects")
    if_exec_id = next(
        execution_id for execution_id, source_id in state.prepared_source_mapping.items() if source_id == "if"
    )
    snapshot["execution_graph"]["edges"].append(
        Edge(
            source=EdgeConnection(node_id=skipped_exec_id, field="value"),
            destination=EdgeConnection(node_id=if_exec_id, field="true_input"),
        ).model_dump(mode="json")
    )

    restored = load_execution_state(snapshot)

    assert restored._legacy_snapshot_loaded
    assert restored._get_prepared_exec_metadata(skipped_exec_id).state == "skipped"
    restored_scheduler = restored._scheduler()
    assert isinstance(restored_scheduler, _GenericGraphSchedulerAdapter)
    assert skipped_exec_id in restored_scheduler._scheduler.discarded

    next_node = restored.next()
    assert next_node is not None
    assert next_node.id == selected_branch_exec_id
    assert restored.prepared_source_mapping[next_node.id] == "false_value"


def test_generic_scheduler_discarded_claim_is_not_requeued() -> None:
    plan = ExecutionPlan()
    plan.add_node("source", "Source")
    plan.add_node("dependent", "Dependent", dependencies=("source",))
    scheduler = ExecutionScheduler(plan, ready_order=("Source", "Dependent"))

    assert scheduler.pop_next() == "source"
    assert scheduler.discard("source") == ("dependent",)
    assert scheduler.executed == set()
    scheduler.rebuild_ready()

    assert scheduler.pop_next() == "dependent"
    assert scheduler.pop_next() is None


def test_generic_scheduler_rehydrates_discarded_completion_from_plan_snapshot() -> None:
    plan = ExecutionPlan()
    plan.add_node("discarded", "Skipped")
    plan.add_node("dependent", "Dependent", dependencies=("discarded",))
    scheduler = ExecutionScheduler(plan)
    scheduler.discard("discarded")

    restored = ExecutionScheduler(
        ExecutionPlan.from_snapshot(plan.snapshot()),
        discarded=scheduler.discarded,
    )

    assert restored.executed == set()
    assert restored.discarded == {"discarded"}
    assert restored.pop_next() == "dependent"


def test_generic_scheduler_preserves_legacy_class_drain_and_fifo_order() -> None:
    plan = ExecutionPlan()
    plan.add_node("late", "ZNode", frame=(1,))
    plan.add_node("early", "ZNode", frame=(0,))
    plan.add_node("other", "ANode")
    scheduler = ExecutionScheduler(plan, ready_order=("ZNode", "ANode"))

    for node_id in ("late", "early", "other"):
        scheduler.enqueue(node_id)

    assert [scheduler.pop_next(), scheduler.pop_next(), scheduler.pop_next()] == ["early", "late", "other"]


def test_generic_scheduler_drains_active_class_before_switching() -> None:
    plan = ExecutionPlan()
    plan.add_node("first", "Priority")
    plan.add_node("other", "Other")
    plan.add_node("released", "Priority", dependencies=("first",))
    scheduler = ExecutionScheduler(plan, ready_order=("Priority", "Other"))

    assert scheduler.pop_next() == "first"
    assert scheduler.complete("first") == ("released",)
    assert scheduler.pop_next() == "released"
    assert scheduler.pop_next() == "other"


def test_generic_scheduler_uses_ready_arrival_after_frame_order() -> None:
    plan = ExecutionPlan()
    plan.add_node("q", "Priority")
    plan.add_node("p", "Priority")
    plan.add_node("a", "Work", frame=(0,), dependencies=("p",))
    plan.add_node("b", "Work", frame=(0,), dependencies=("q",))
    scheduler = ExecutionScheduler(plan, ready_order=("Priority", "Work"))

    assert scheduler.pop_next() == "q"
    assert scheduler.complete("q") == ("b",)
    assert scheduler.pop_next() == "p"
    assert scheduler.complete("p") == ("a",)
    assert [scheduler.pop_next(), scheduler.pop_next()] == ["b", "a"]


def test_generic_scheduler_rebuild_preserves_ready_arrival_order() -> None:
    plan = ExecutionPlan()
    plan.add_node("q", "Priority")
    plan.add_node("p", "Priority")
    plan.add_node("a", "Work", frame=(0,), dependencies=("p",))
    plan.add_node("b", "Work", frame=(0,), dependencies=("q",))
    scheduler = ExecutionScheduler(plan, ready_order=("Priority", "Work"))

    assert scheduler.pop_next() == "q"
    scheduler.complete("q")
    assert scheduler.pop_next() == "p"
    scheduler.complete("p")
    scheduler.rebuild_ready()

    assert [scheduler.pop_next(), scheduler.pop_next()] == ["b", "a"]


def test_generic_scheduler_sorts_unlisted_ready_classes_by_name() -> None:
    plan = ExecutionPlan()
    plan.add_node("z", "ZNode")
    plan.add_node("a", "ANode")
    scheduler = ExecutionScheduler(plan, ready_order=("PriorityNode",))

    assert [scheduler.pop_next(), scheduler.pop_next()] == ["a", "z"]


def test_generic_scheduler_rejects_non_closed_executed_projection() -> None:
    plan = ExecutionPlan()
    plan.add_node("parent", "Parent")
    plan.add_node("child", "Child", dependencies=("parent",))

    with pytest.raises(ValueError, match="missing prerequisite"):
        ExecutionScheduler(plan, executed=("child",))


def test_generic_scheduler_does_not_requeue_claimed_work_on_rebuild() -> None:
    scheduler = ExecutionScheduler(_plan())
    assert scheduler.pop_next() == "left"

    scheduler.rebuild_ready()

    assert scheduler.pop_next() == "right"


def test_generic_scheduler_rejects_invalid_completion_without_mutation() -> None:
    plan = _plan()
    scheduler = ExecutionScheduler(plan)
    scheduler.enqueue("left")

    with pytest.raises(KeyError, match="missing"):
        scheduler.complete("missing")
    assert scheduler.pop_next() == "left"

    scheduler.complete("left")
    with pytest.raises(ValueError, match="already completed"):
        scheduler.complete("left")
    assert scheduler.indegree == {"left": 0, "right": 0, "join": 1}


def test_generic_scheduler_rejects_completion_when_durable_indegree_is_missing() -> None:
    scheduler = ExecutionScheduler(_plan())
    scheduler.indegree.pop("join")

    with pytest.raises(KeyError, match="indegree missing"):
        scheduler.complete("left")

    assert scheduler.executed == set()


def test_generic_scheduler_rehydrates_repeated_dependencies() -> None:
    plan = ExecutionPlan()
    plan.add_node("source", "Source")
    plan.add_node("join", "Join", dependencies=("source", "source"))

    scheduler = ExecutionScheduler(plan)
    assert scheduler.indegree["join"] == 2
    scheduler.complete("source")
    assert scheduler.indegree["join"] == 0


def test_generic_scheduler_rejects_malformed_snapshot() -> None:
    with pytest.raises(ValueError, match="node order"):
        ExecutionPlan.from_snapshot({"nodes": {"node": {"node_id": "node", "class_name": "Node", "order": "first"}}})


def test_generic_scheduler_rehydrates_from_durable_projection() -> None:
    plan = _plan()
    scheduler = ExecutionScheduler(plan)
    scheduler.enqueue("left")
    scheduler.complete("left")

    restored = ExecutionScheduler(
        ExecutionPlan.from_snapshot(plan.snapshot()),
        executed=scheduler.executed,
    )
    restored.rebuild_ready()

    assert restored.pop_next() == "right"
    assert restored.indegree["join"] == 1


def test_graph_state_static_dag_matches_generic_scheduler_trace() -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="left", a=1, b=2))
    graph.add_node(AddInvocation(id="right", a=3, b=4))
    graph.add_node(MultiplyInvocation(id="join"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="left", field="value"),
            destination=EdgeConnection(node_id="join", field="a"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="right", field="value"),
            destination=EdgeConnection(node_id="join", field="b"),
        )
    )

    state = GraphExecutionState(graph=graph)
    plan = ExecutionPlan()
    plan.add_node("left", "AddInvocation")
    plan.add_node("right", "AddInvocation")
    plan.add_node("join", "MultiplyInvocation", dependencies=("left", "right"))
    generic = ExecutionScheduler(plan)
    generic.enqueue("left")
    generic.enqueue("right")

    legacy_trace: list[str] = []
    generic_trace: list[str] = []
    while (node := state.next()) is not None:
        legacy_trace.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))
    while (node_id := generic.pop_next()) is not None:
        generic_trace.append(node_id)
        generic.complete(node_id)

    assert legacy_trace == generic_trace
    assert state.is_complete()


def test_graph_state_static_dag_preserves_legacy_fifo_for_released_nodes() -> None:
    graph = Graph()
    for node_id in ("q", "p", "a", "b"):
        graph.add_node(AddInvocation(id=node_id, a=1, b=2))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="q", field="value"),
            destination=EdgeConnection(node_id="b", field="a"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="p", field="value"),
            destination=EdgeConnection(node_id="a", field="a"),
        )
    )

    def run(use_legacy_scheduler: bool) -> tuple[list[str], GraphExecutionState]:
        state = GraphExecutionState(graph=graph.model_copy(deep=True))
        if use_legacy_scheduler:
            object.__setattr__(state, "_execution_scheduler", _ExecutionScheduler(state))
        trace: list[str] = []
        while (node := state.next()) is not None:
            trace.append(state.prepared_source_mapping[node.id])
            state.complete(node.id, node.invoke(Mock()))
        return trace, state

    legacy_trace, legacy_state = run(use_legacy_scheduler=True)
    generic_trace, generic_state = run(use_legacy_scheduler=False)

    assert legacy_trace == generic_trace == ["q", "p", "b", "a"]
    assert legacy_state.is_complete()
    assert generic_state.is_complete()


def test_graph_state_static_dag_delegates_readiness_to_generic_scheduler() -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    state = GraphExecutionState(graph=graph)

    node = state.next()

    assert node is not None
    assert type(state._scheduler()).__name__ == "_GenericGraphSchedulerAdapter"
    state.complete(node.id, node.invoke(Mock()))
    assert state.is_complete()


@pytest.mark.parametrize(
    ("source_collection", "expected_collection"),
    [([], []), ([0, 1, 2, 3], [0, 1, 2, 3]), (list(range(11)), list(range(11)))],
)
def test_graph_state_iterate_collect_uses_generic_scheduler_and_closed_stream(
    source_collection: list[int], expected_collection: list[int]
) -> None:
    graph = Graph()
    graph.add_node(CollectionConcatInvocation(id="source", first=source_collection))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="source", field="collection"),
            destination=EdgeConnection(node_id="iterate", field="collection"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="iterate", field="item"),
            destination=EdgeConnection(node_id="collect", field="item"),
        )
    )

    state = GraphExecutionState(graph=graph)
    trace: list[str] = []
    while (node := state.next()) is not None:
        trace.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))

    assert isinstance(state._scheduler(), _GenericGraphSchedulerAdapter)
    assert trace[0] == "source"
    assert trace[-1] == "collect"
    collect_exec_id = next(iter(state.source_prepared_mapping["collect"]))
    assert state.results[collect_exec_id].collection == expected_collection
    streams = [stream for stream in state._generic_runtime().streams.values() if stream.owner_id == "iterate"]
    assert len(streams) == 1
    assert streams[0].closed
    assert streams[0].values == tuple(expected_collection)
    assert state.is_complete()


def test_graph_state_mixed_if_iterate_collect_keeps_compatibility_scheduler() -> None:
    graph = Graph()
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(CollectionConcatInvocation(id="source", first=[1]))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="source", field="collection"),
            destination=EdgeConnection(node_id="iterate", field="collection"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="iterate", field="item"),
            destination=EdgeConnection(node_id="collect", field="item"),
        )
    )

    state = GraphExecutionState(graph=graph)

    assert not state._can_use_generic_scheduler()
    assert isinstance(state._scheduler(), _ExecutionScheduler)


def test_graph_state_if_uses_generic_activation_routing() -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(AddInvocation(id="true_value", a=2, b=2))
    graph.add_node(AddInvocation(id="false_value", a=3, b=3))
    graph.add_node(IfInvocation(id="if", condition=True))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="condition", field="value"),
            destination=EdgeConnection(node_id="if", field="condition"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="true_value", field="value"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="false_value", field="value"),
            destination=EdgeConnection(node_id="if", field="false_input"),
        )
    )

    state = GraphExecutionState(graph=graph)
    while (node := state.next()) is not None:
        state.complete(node.id, node.invoke(Mock()))

    assert type(state._scheduler()).__name__ == "_GenericGraphSchedulerAdapter"
    completed_sources = {
        source_id for exec_node_id, source_id in state.prepared_source_mapping.items() if exec_node_id in state.results
    }
    assert completed_sources == {
        "condition",
        "true_value",
        "if",
    }


@pytest.mark.parametrize("condition", [False, True])
def test_graph_state_compatibility_if_uses_opaque_activation_routing(
    condition: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=condition))
    graph.add_node(AddInvocation(id="true_value", a=2, b=2))
    graph.add_node(AddInvocation(id="false_value", a=3, b=3))
    graph.add_node(IfInvocation(id="if"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="condition", field="value"),
            destination=EdgeConnection(node_id="if", field="condition"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="true_value", field="value"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="false_value", field="value"),
            destination=EdgeConnection(node_id="if", field="false_input"),
        )
    )

    state = GraphExecutionState(graph=graph)
    state._execution_scheduler = _ExecutionScheduler(state)
    deleted_edges: list[Edge] = []

    def record_deleted_edge(self: GraphExecutionState, edge: Edge) -> None:
        deleted_edges.append(edge)

    monkeypatch.setattr(GraphExecutionState, "_tx_delete_execution_edge", record_deleted_edge)

    trace: list[str] = []
    while (node := state.next()) is not None:
        trace.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))

    selected_value = "true_value" if condition else "false_value"
    assert trace == ["condition", selected_value, "if"]
    assert state.is_complete()
    assert deleted_edges == []


def test_graph_state_if_generic_and_legacy_paths_have_matching_completion() -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=False))
    graph.add_node(AddInvocation(id="true_value", a=2, b=2))
    graph.add_node(AddInvocation(id="false_value", a=3, b=3))
    graph.add_node(IfInvocation(id="if"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="condition", field="value"),
            destination=EdgeConnection(node_id="if", field="condition"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="true_value", field="value"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="false_value", field="value"),
            destination=EdgeConnection(node_id="if", field="false_input"),
        )
    )

    def run(force_legacy: bool) -> tuple[list[str], set[str]]:
        state = GraphExecutionState(graph=graph.model_copy(deep=True))
        if force_legacy:
            object.__setattr__(state, "_execution_scheduler", _ExecutionScheduler(state))
        trace: list[str] = []
        while (node := state.next()) is not None:
            trace.append(state.prepared_source_mapping[node.id])
            state.complete(node.id, node.invoke(Mock()))
        completed_sources = {
            source_id
            for exec_node_id, source_id in state.prepared_source_mapping.items()
            if exec_node_id in state.results
        }
        return trace, completed_sources

    legacy_trace, legacy_completed = run(force_legacy=True)
    generic_trace, generic_completed = run(force_legacy=False)

    assert generic_trace == legacy_trace
    assert generic_completed == legacy_completed == {"condition", "false_value", "if"}


def test_graph_state_if_rehydrates_discarded_unselected_branch() -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=False))
    graph.add_node(AddInvocation(id="true_value", a=2, b=2))
    graph.add_node(AddInvocation(id="false_value", a=3, b=3))
    graph.add_node(IfInvocation(id="if"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="condition", field="value"),
            destination=EdgeConnection(node_id="if", field="condition"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="true_value", field="value"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="false_value", field="value"),
            destination=EdgeConnection(node_id="if", field="false_input"),
        )
    )

    state = GraphExecutionState(graph=graph)
    condition = state.next()
    assert condition is not None
    assert state.prepared_source_mapping[condition.id] == "condition"
    state.complete(condition.id, condition.invoke(Mock()))

    selected_branch = state.next()
    assert selected_branch is not None
    selected_branch_exec_id = selected_branch.id
    assert state.prepared_source_mapping[selected_branch_exec_id] == "false_value"
    assert "true_value" not in state.source_prepared_mapping

    restored = GraphExecutionState.model_validate(state.model_dump(mode="python"), strict=False)

    next_node = restored.next()
    assert next_node is not None
    assert next_node.id == selected_branch_exec_id
    assert restored.prepared_source_mapping[next_node.id] == "false_value"


def test_graph_state_static_dag_rehydrates_generic_scheduler_after_partial_run() -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="first", a=1, b=2))
    graph.add_node(AddInvocation(id="second", b=4))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="first", field="value"),
            destination=EdgeConnection(node_id="second", field="a"),
        )
    )
    state = GraphExecutionState(graph=graph)
    first = state.next()
    assert first is not None
    state.complete(first.id, first.invoke(Mock()))

    restored = GraphExecutionState.model_validate(state.model_dump(mode="python"), strict=False)

    second = restored.next()
    assert second is not None
    assert restored.prepared_source_mapping[second.id] == "second"


@pytest.mark.parametrize("use_legacy_scheduler", [False, True])
def test_graph_state_static_dag_apply_and_rollback_match_scheduler_paths(use_legacy_scheduler: bool) -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    state = GraphExecutionState(graph=graph)
    if use_legacy_scheduler:
        object.__setattr__(state, "_execution_scheduler", _ExecutionScheduler(state))

    node = state.next()
    assert node is not None
    execution_ref = state.get_execution_ref(node.id)
    output = node.invoke(Mock())
    state.apply(execution_ref, output)
    assert {
        state.prepared_source_mapping[node_id] for node_id in state.executed if node_id in state.prepared_source_mapping
    } == {"add"}

    rollback_state = GraphExecutionState(graph=graph.model_copy(deep=True))
    if use_legacy_scheduler:
        object.__setattr__(rollback_state, "_execution_scheduler", _ExecutionScheduler(rollback_state))
    rollback_node = rollback_state.next()
    assert rollback_node is not None
    rollback_ref = rollback_state.get_execution_ref(rollback_node.id)
    original_record_effect_streams = rollback_state._record_effect_streams

    def fail_after_completion(*args: object, **kwargs: object) -> None:
        raise RuntimeError("effect recording failed")

    rollback_state._record_effect_streams = fail_after_completion  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="effect recording failed"):
            rollback_state.apply(rollback_ref, rollback_node.invoke(Mock()))
    finally:
        rollback_state._record_effect_streams = original_record_effect_streams  # type: ignore[method-assign]

    assert rollback_state.executed == set()
    assert rollback_state.results == {}
    retried_node = rollback_state.next()
    assert retried_node is not None
    assert rollback_state.prepared_source_mapping[retried_node.id] == "add"


def test_graph_state_apply_rolls_back_generic_scheduler_transition() -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    state = GraphExecutionState(graph=graph)
    node = state.next()
    assert node is not None
    execution_ref = state.get_execution_ref(node.id)
    original_record_effect_streams = state._record_effect_streams

    def fail_after_completion(*args: object, **kwargs: object) -> None:
        raise RuntimeError("effect recording failed")

    state._record_effect_streams = fail_after_completion  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="effect recording failed"):
            state.apply(execution_ref, node.invoke(Mock()))
    finally:
        state._record_effect_streams = original_record_effect_streams  # type: ignore[method-assign]

    assert state.executed == set()


def test_graph_state_apply_retry_after_if_branch_failure_completes_graph() -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(AddInvocation(id="true_value", a=1, b=2))
    graph.add_node(AddInvocation(id="false_value", a=3, b=4))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AddInvocation(id="consumer", b=10))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="condition", field="value"),
            destination=EdgeConnection(node_id="if", field="condition"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="true_value", field="value"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="false_value", field="value"),
            destination=EdgeConnection(node_id="if", field="false_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="if", field="value"),
            destination=EdgeConnection(node_id="consumer", field="a"),
        )
    )

    state = GraphExecutionState(graph=graph)
    branch_node = None
    while (node := state.next()) is not None:
        source_node_id = state.prepared_source_mapping[node.id]
        if source_node_id == "true_value":
            branch_node = node
            break
        state.complete(node.id, node.invoke(Mock()))
    assert branch_node is not None

    execution_ref = state.get_execution_ref(branch_node.id)
    output = branch_node.invoke(Mock())
    original_record_effect_streams = state._record_effect_streams

    def fail_after_scheduler_mutation(*args: object, **kwargs: object) -> None:
        raise RuntimeError("effect recording failed")

    state._record_effect_streams = fail_after_scheduler_mutation  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="effect recording failed"):
            state.apply(execution_ref, output)
    finally:
        state._record_effect_streams = original_record_effect_streams  # type: ignore[method-assign]

    assert branch_node.id not in state.executed

    state.apply(execution_ref, output)
    while (node := state.next()) is not None:
        state.complete(node.id, node.invoke(Mock()))

    consumer_exec_id = next(iter(state.source_prepared_mapping["consumer"]))
    assert state.results[consumer_exec_id].value == 13
    assert state.is_complete()
