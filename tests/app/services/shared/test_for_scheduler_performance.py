"""Measure scheduler mapping work and optional overhead for trivial loop bodies.

The deterministic scaling regression counts mapping visits during end-to-end execution.
Optional absolute-time benchmarks remain marked `slow`.
"""

import gc
import time
from collections.abc import Callable
from unittest.mock import Mock

import networkx as nx
import pytest

from invokeai.app.invocations.collections import RangeInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.invocations.primitives import BooleanInvocation
from invokeai.app.services.shared.execution_state_migration import dump_execution_state, load_execution_state
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    Graph,
    GraphExecutionState,
    IterateInvocation,
    _ExecutionScheduler,
    _GenericGraphSchedulerAdapter,
)
from tests.test_nodes import AnyTypeTestInvocation, FixedCollectionTestInvocation, create_edge, create_loop_linkage


class _MappingTraversalCounter:
    def __init__(self) -> None:
        self.entries_visited = 0
        self.visits_by_mapping: dict[str, int] = {}

    def record(self, mapping_name: str) -> None:
        self.entries_visited += 1
        self.visits_by_mapping[mapping_name] = self.visits_by_mapping.get(mapping_name, 0) + 1


class _CountingMapping(dict[str, object]):
    """Count mapping accesses and iteration without constraining the scheduler algorithm."""

    def __init__(self, values: dict[str, object], counter: _MappingTraversalCounter, mapping_name: str) -> None:
        super().__init__()
        self.update(values)
        self._counter = counter
        self._mapping_name = mapping_name

    def items(self):  # type: ignore[override]
        for item in super().items():
            self._counter.record(self._mapping_name)
            yield item

    def values(self):  # type: ignore[override]
        for value in super().values():
            self._counter.record(self._mapping_name)
            yield value

    def keys(self):  # type: ignore[override]
        for key in super().keys():
            self._counter.record(self._mapping_name)
            yield key

    def __iter__(self):  # type: ignore[override]
        for key in super().__iter__():
            self._counter.record(self._mapping_name)
            yield key

    def __contains__(self, key: object) -> bool:
        self._counter.record(self._mapping_name)
        return super().__contains__(key)

    def __getitem__(self, key: str) -> object:
        self._counter.record(self._mapping_name)
        return super().__getitem__(key)

    def get(self, key: str, default: object = None) -> object:  # type: ignore[override]
        self._counter.record(self._mapping_name)
        return super().get(key, default)


class _CountingSet(set[str]):
    """Count visits to prepared execution IDs without constraining set operations."""

    def __init__(self, values: set[str], counter: _MappingTraversalCounter, mapping_name: str) -> None:
        super().__init__(values)
        self._counter = counter
        self._mapping_name = mapping_name

    def __iter__(self):  # type: ignore[override]
        for value in super().__iter__():
            self._counter.record(self._mapping_name)
            yield value


def _run_trivial_loop(
    loop_type: str,
    count: int,
    *,
    clock: Callable[[], float] = time.process_time,
    mapping_counter: _MappingTraversalCounter | None = None,
    include_if: bool = False,
    external_condition: bool = False,
    condition_value: bool = True,
) -> float:
    graph = Graph()
    graph.add_node(RangeInvocation(id="range", start=0, stop=count))
    graph.add_node(ForInvocation(id="loop") if loop_type == "for" else IterateInvocation(id="loop"))
    graph.add_edge(create_edge("range", "collection", "loop", "collection"))
    if include_if:
        graph.add_node(AnyTypeTestInvocation(id="true_body"))
        graph.add_node(AnyTypeTestInvocation(id="false_body"))
        graph.add_node(
            IfInvocation(id="if") if external_condition else IfInvocation(id="if", condition=condition_value)
        )
        if external_condition:
            graph.add_node(BooleanInvocation(id="condition", value=condition_value))
            graph.add_edge(create_edge("condition", "value", "if", "condition"))
        graph.add_edge(create_edge("loop", "item", "true_body", "value"))
        graph.add_edge(create_edge("loop", "item", "false_body", "value"))
        graph.add_edge(create_edge("true_body", "value", "if", "true_input"))
        graph.add_edge(create_edge("false_body", "value", "if", "false_input"))
        output_source = "if"
    else:
        graph.add_node(AnyTypeTestInvocation(id="body"))
        graph.add_edge(create_edge("loop", "item", "body", "value"))
        output_source = "body"
    if loop_type == "for":
        graph.add_node(ForReturnInvocation(id="return"))
        graph.add_edge(create_edge(output_source, "value", "return", "output"))
        graph.add_edge(create_loop_linkage("loop", "return"))
    else:
        graph.add_node(CollectInvocation(id="collect"))
        graph.add_edge(create_edge(output_source, "value", "collect", "item"))
    state = GraphExecutionState(graph=graph)
    if mapping_counter is not None:
        scheduler = state._scheduler()
        state.execution_refs = _CountingMapping(state.execution_refs, mapping_counter, "execution_refs")  # type: ignore[assignment]
        state.execution_tokens = _CountingMapping(state.execution_tokens, mapping_counter, "execution_tokens")  # type: ignore[assignment]
        state.execution_effects = _CountingMapping(state.execution_effects, mapping_counter, "execution_effects")  # type: ignore[assignment]
        state.results = _CountingMapping(state.results, mapping_counter, "results")  # type: ignore[assignment]
        if isinstance(scheduler, _GenericGraphSchedulerAdapter):
            scheduler._scheduler.plan.nodes = _CountingMapping(
                scheduler._scheduler.plan.nodes, mapping_counter, "scheduler_plan"
            )
    context = Mock()
    started = clock()
    while (node := state.next()) is not None:
        state.complete(node.id, node.invoke(context))
    assert state.is_complete()
    return clock() - started


def _run_if_free_prepare_workload(
    count: int = 300, *, wrap_prepare: bool = False, force_compatibility_scheduler: bool = False
) -> tuple[float, int]:
    graph = Graph()
    graph.add_node(RangeInvocation(id="range", start=0, stop=count))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_edge(create_edge("range", "collection", "iterate", "collection"))
    previous = "iterate"
    previous_field = "item"
    body_ids = []
    for index in range(30):
        node_id = f"chain_{index}"
        graph.add_node(AnyTypeTestInvocation(id=node_id))
        graph.add_edge(create_edge(previous, previous_field, node_id, "value"))
        previous, previous_field = node_id, "value"
        body_ids.append(node_id)
    collect_ids = []
    for index in range(6):
        node_id = f"collect_{index}"
        graph.add_node(CollectInvocation(id=node_id))
        graph.add_edge(create_edge(previous, previous_field, node_id, "item"))
        collect_ids.append(node_id)

    state = GraphExecutionState(graph=graph)
    if force_compatibility_scheduler:
        state._execution_scheduler = _ExecutionScheduler(state)
    transaction_calls = 0
    run_transaction = state._run_materialization_transaction

    def counted_transaction(operation: Callable[[], object]) -> object:
        nonlocal transaction_calls
        transaction_calls += 1
        return run_transaction(operation)

    state._run_materialization_transaction = counted_transaction  # type: ignore[method-assign]
    if wrap_prepare:
        materializer = state._materializer()
        prepare = materializer.prepare

        def transactional_prepare(base_graph):
            return counted_transaction(lambda: prepare(base_graph))

        materializer.prepare = transactional_prepare  # type: ignore[method-assign]
    started = time.perf_counter()
    trace: list[str] = []
    while (node := state.next()) is not None:
        source_id = state.prepared_source_mapping[node.id]
        trace.append(source_id)
        state.complete(node.id, node.invoke(Mock()))
    elapsed = time.perf_counter() - started

    assert state.is_complete()
    assert not state.has_error()
    assert all(trace.count(node_id) == count for node_id in body_ids)
    assert all(trace.count(node_id) == 1 for node_id in collect_ids)
    assert all(
        state.results[next(iter(state.source_prepared_mapping[node_id]))].collection == list(range(count))
        for node_id in collect_ids
    )
    return elapsed, transaction_calls


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True], ids=["generic", "compatibility"])
def test_if_free_prepare_does_not_open_materialization_transactions(force_compatibility_scheduler: bool) -> None:
    _, transaction_calls = _run_if_free_prepare_workload(force_compatibility_scheduler=force_compatibility_scheduler)

    assert transaction_calls == 0


@pytest.mark.slow
def test_if_free_prepare_workload_performance_sample() -> None:
    measurements: dict[str, list[float]] = {"transaction-wrapped": [], "direct": []}
    for label, wrap_prepare in (("transaction-wrapped", True), ("direct", False)):
        _run_if_free_prepare_workload(wrap_prepare=wrap_prepare)
        for _ in range(3):
            gc.collect()
            elapsed, transaction_calls = _run_if_free_prepare_workload(wrap_prepare=wrap_prepare)
            assert transaction_calls == (42 if wrap_prepare else 0)
            measurements[label].append(elapsed)
    for label, samples in measurements.items():
        print(f"Range(300), Iterate, 30-node chain, 6 Collects {label}: {sorted(samples)[1]:.3f}s median")


def _root_if_collect_graph(groups: int, branch_size: int) -> Graph:
    graph = Graph()
    graph.add_node(RangeInvocation(id="outer_source", start=0, stop=groups))
    graph.add_node(IterateInvocation(id="outer"))
    graph.add_node(FixedCollectionTestInvocation(id="inner_source"))
    graph.add_node(IterateInvocation(id="inner"))
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(CollectInvocation(id="branch_collect"))
    graph.add_node(AnyTypeTestInvocation(id="fallback", value="fallback"))
    graph.add_node(AnyTypeTestInvocation(id="after"))
    graph.add_edge(create_edge("outer_source", "collection", "outer", "collection"))
    graph.add_edge(create_edge("outer", "item", "inner_source", "value"))
    graph.add_edge(create_edge("inner_source", "collection", "inner", "collection"))
    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("fallback", "value", "if", "false_input"))
    previous, previous_field = "inner", "item"
    body_ids = []
    for index in range(branch_size):
        node_id = f"branch_{index}"
        graph.add_node(AnyTypeTestInvocation(id=node_id))
        graph.add_edge(create_edge(previous, previous_field, node_id, "value"))
        previous, previous_field = node_id, "value"
        body_ids.append(node_id)
    graph.add_edge(create_edge(previous, previous_field, "branch_collect", "item"))
    graph.add_edge(create_edge("branch_collect", "collection", "if", "true_input"))
    graph.add_edge(create_edge("if", "value", "after", "value"))
    return graph


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True], ids=["generic", "compatibility"])
def test_root_if_collect_activation_index_rebuilds_after_dump_load(force_compatibility_scheduler: bool) -> None:
    state = GraphExecutionState(graph=_root_if_collect_graph(groups=4, branch_size=4))
    if force_compatibility_scheduler:
        state._execution_scheduler = _ExecutionScheduler(state)

    trace: list[str] = []
    while len(trace) < 12:
        node = state.next()
        assert node is not None
        trace.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))
    assert state._if_activation_dependents_index_complete

    state = load_execution_state(dump_execution_state(state))
    if force_compatibility_scheduler:
        state._execution_scheduler = _ExecutionScheduler(state)
    while (node := state.next()) is not None:
        trace.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))

    assert state.is_complete()
    assert not state.has_error()
    assert trace.count("if") == 4
    assert trace.count("branch_collect") == 4
    after_ids = sorted(state.source_prepared_mapping["after"], key=state._get_iteration_path)
    assert [state.results[node_id].value for node_id in after_ids] == [[0, 1] for _ in range(4)]


def test_root_if_collect_activation_index_rebuilds_after_graph_mutation() -> None:
    state = GraphExecutionState(graph=_root_if_collect_graph(groups=2, branch_size=2))
    state._get_activation_dependents_for_if("if", ())
    assert state._if_activation_dependents_index_complete

    state.add_node(AnyTypeTestInvocation(id="unconnected"))

    assert not state._if_activation_dependents_index_complete
    assert not state._if_activation_active_dependents_by_dependency
    while (node := state.next()) is not None:
        state.complete(node.id, node.invoke(Mock()))
    assert state.is_complete()
    assert state.results[next(iter(state.source_prepared_mapping["unconnected"]))].value is None


def _run_root_if_collect_workload(
    groups: int, branch_size: int, *, force_compatibility_scheduler: bool = False
) -> tuple[int, int, int, int, float, tuple[tuple[tuple[int, ...], int], ...], int, int]:
    graph = _root_if_collect_graph(groups, branch_size)
    body_ids = [f"branch_{index}" for index in range(branch_size)]

    state = GraphExecutionState(graph=graph)
    if force_compatibility_scheduler:
        state._execution_scheduler = _ExecutionScheduler(state)
    scheduler_node_visits = 0
    scheduler_edge_visits = 0
    dependency_index_node_visits = 0
    dependency_index_edge_visits = 0
    discarded_candidates = 0
    candidate_counts: list[tuple[tuple[int, ...], int]] = []
    activation_resolution = False
    original_topological_sort = nx.topological_sort

    def counted_topological_sort(prepared_graph):
        nonlocal scheduler_node_visits, scheduler_edge_visits
        if activation_resolution:
            scheduler_node_visits += prepared_graph.number_of_nodes()
            scheduler_edge_visits += prepared_graph.number_of_edges()
        return original_topological_sort(prepared_graph)

    nx.topological_sort = counted_topological_sort
    original_dependents = state._get_activation_dependents_for_if

    def counted_dependents(source_if_id: str, iteration_path: tuple[int, ...]):
        nonlocal activation_resolution
        activation_resolution = True
        try:
            candidates = original_dependents(source_if_id, iteration_path)
            candidate_counts.append((iteration_path, len(candidates)))
            return candidates
        finally:
            activation_resolution = False

    state._get_activation_dependents_for_if = counted_dependents  # type: ignore[method-assign]
    original_record_dependencies = state._record_activation_dependencies

    def counted_record_dependencies(exec_node_id: str):
        nonlocal dependency_index_node_visits, dependency_index_edge_visits
        dependencies = original_record_dependencies(exec_node_id)
        if activation_resolution:
            dependency_index_node_visits += 1
            dependency_index_edge_visits += len(dependencies)
        return dependencies

    state._record_activation_dependencies = counted_record_dependencies  # type: ignore[method-assign]
    scheduler = state._scheduler()
    discard_method_name = (
        "_discard_rejected_activation_nodes"
        if isinstance(scheduler, _ExecutionScheduler)
        else "_discard_rejected_nodes"
    )
    discard_candidates = getattr(scheduler, discard_method_name)

    def counted_discard_candidates(exec_node_ids):
        nonlocal discarded_candidates
        exec_node_ids = tuple(exec_node_ids)
        discarded_candidates += len(exec_node_ids)
        return discard_candidates(exec_node_ids)

    setattr(scheduler, discard_method_name, counted_discard_candidates)
    try:
        trace: list[str] = []
        started = time.perf_counter()
        while (node := state.next()) is not None:
            source_id = state.prepared_source_mapping[node.id]
            trace.append(source_id)
            state.complete(node.id, node.invoke(Mock()))
        elapsed = time.perf_counter() - started
    finally:
        nx.topological_sort = original_topological_sort

    assert state.is_complete()
    assert not state.has_error()
    assert trace.count("if") == groups
    assert trace.count("branch_collect") == groups
    assert all(trace.count(node_id) == groups * 2 for node_id in body_ids)
    after_ids = sorted(state.source_prepared_mapping["after"], key=state._get_iteration_path)
    assert len(after_ids) == groups
    assert [state.results[node_id].value for node_id in after_ids] == [[0, 1] for _ in range(groups)]
    prepared_nodes = len(state.prepared_source_mapping)
    return (
        scheduler_node_visits,
        scheduler_edge_visits,
        discarded_candidates,
        prepared_nodes,
        elapsed,
        tuple(candidate_counts),
        dependency_index_node_visits,
        dependency_index_edge_visits,
    )


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True], ids=["generic", "compatibility"])
def test_root_if_collect_activation_planning_work_scales_with_prepared_graph(
    force_compatibility_scheduler: bool,
) -> None:
    small = _run_root_if_collect_workload(
        groups=4, branch_size=4, force_compatibility_scheduler=force_compatibility_scheduler
    )
    more_groups = _run_root_if_collect_workload(
        groups=8, branch_size=4, force_compatibility_scheduler=force_compatibility_scheduler
    )
    larger_branch = _run_root_if_collect_workload(
        groups=4, branch_size=8, force_compatibility_scheduler=force_compatibility_scheduler
    )

    # Include the one-time dependency-index build plus any active candidate topo work. Together,
    # these visits must track prepared nodes/edges rather than revisit each completed branch.
    work = (small, more_groups, larger_branch)
    assert all(
        index_nodes + topo_nodes <= prepared_nodes * 2
        for topo_nodes, _, _, prepared_nodes, _, _, index_nodes, _ in work
    ), f"{work=}"
    assert all(
        index_edges + topo_edges <= prepared_nodes * 3
        for _, topo_edges, _, prepared_nodes, _, _, _, index_edges in work
    ), f"{work=}"
    assert all(discarded <= prepared_nodes for _, _, discarded, prepared_nodes, _, _, _, _ in work), f"{work=}"


@pytest.mark.slow
@pytest.mark.parametrize("force_compatibility_scheduler", [False, True], ids=["generic", "compatibility"])
def test_root_if_collect_activation_planning_timing_sample(force_compatibility_scheduler: bool) -> None:
    for groups, branch_size in ((8, 8), (16, 8), (8, 16)):
        samples = [
            _run_root_if_collect_workload(
                groups,
                branch_size,
                force_compatibility_scheduler=force_compatibility_scheduler,
            )[4]
            for _ in range(3)
        ]
        print(
            f"root-If/Collect groups={groups} branch_size={branch_size} "
            f"scheduler={'compatibility' if force_compatibility_scheduler else 'generic'} "
            f"median={sorted(samples)[1]:.3f}s"
        )


@pytest.mark.platform_independent
@pytest.mark.parametrize(
    ("loop_type", "include_if", "external_condition"),
    [
        ("iterate", False, False),
        ("for", False, False),
        ("iterate", True, False),
        ("iterate", True, True),
        ("for", True, False),
        ("for", True, True),
    ],
)
def test_loop_scheduler_mapping_visits_scale_linearly(
    loop_type: str, include_if: bool, external_condition: bool
) -> None:
    """Bound wrapped scheduler-state mapping visits, not preparation retirement or If-frame lookup."""
    visits: dict[int, int] = {}
    visits_by_mapping: dict[int, dict[str, int]] = {}
    for count in (300, 1200):
        counter = _MappingTraversalCounter()
        _run_trivial_loop(
            loop_type,
            count,
            mapping_counter=counter,
            include_if=include_if,
            external_condition=external_condition,
        )
        visits[count] = counter.entries_visited
        visits_by_mapping[count] = counter.visits_by_mapping

    # A small constant number of full mapping passes is acceptable. A pass per iteration is not:
    # it visits growing scheduler state quadratically. For covers durable effect history; Iterate
    # covers result synchronization in the generic scheduler.
    assert all(visits[count] <= count * 128 for count in visits), (
        f"mapping visits for {loop_type}, include_if={include_if}, external_condition={external_condition}: "
        f"{visits}; by mapping: {visits_by_mapping}"
    )
    # Four times as many iterations should not make per-iteration mapping work grow substantially.
    # Per-mapping ratios catch an occasional full scan even when unrelated constant-time lookups
    # mask it in the aggregate. The counters are deterministic; unlike a timing ratio, they cannot
    # fail because the test worker was descheduled or contended.
    mappings = visits_by_mapping[300].keys() | visits_by_mapping[1200].keys()
    for mapping in mappings:
        small_per_node = visits_by_mapping[300].get(mapping, 0) / 300
        large_per_node = visits_by_mapping[1200].get(mapping, 0) / 1200
        assert large_per_node < small_per_node * 1.75, (
            f"{mapping} operations per node for {loop_type}, include_if={include_if}, "
            f"external_condition={external_condition}: "
            f"{small_per_node} at 300 vs {large_per_node} at 1200; all visits: {visits_by_mapping}"
        )


def test_prepared_execution_count_scans_each_source_once() -> None:
    counter = _MappingTraversalCounter()
    state = GraphExecutionState(graph=Graph())
    state.source_prepared_mapping = _CountingMapping(
        {
            "source": _CountingSet({"exec-1", "exec-2", "exec-3"}, counter, "prepared_node_ids"),
        },
        counter,
        "source_prepared_mapping",
    )  # type: ignore[assignment]

    assert state._count_unexecuted_prepared("source") == 3
    assert state._count_unexecuted_prepared("source") == 3
    assert counter.visits_by_mapping == {"source_prepared_mapping": 1, "prepared_node_ids": 3}


@pytest.mark.slow
@pytest.mark.parametrize("loop_type", ["for", "iterate"])
def test_trivial_loop_scheduler_overhead(loop_type: str) -> None:
    elapsed = _run_trivial_loop(loop_type, 600)
    # Broad quiet-machine guard for catastrophic slowdowns; this is not a historical baseline comparison.
    assert elapsed < 4, f"{loop_type}: {elapsed:.3f}s for 600 items"


class _PendingIfScanCounter(set[str]):
    def __init__(self, values: set[str]) -> None:
        super().__init__(values)
        self.items_scanned = 0

    def __iter__(self):
        for value in super().__iter__():
            self.items_scanned += 1
            yield value


def _measure_pending_if_attachment(
    count: int, *, force_compatibility_scheduler: bool = False
) -> tuple[int, int, int, float, float]:
    graph = Graph()
    graph.add_node(RangeInvocation(id="range", start=0, stop=count))
    graph.add_node(IterateInvocation(id="loop"))
    graph.add_node(AnyTypeTestInvocation(id="true_body"))
    graph.add_node(AnyTypeTestInvocation(id="false_body"))
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(create_edge("range", "collection", "loop", "collection"))
    graph.add_edge(create_edge("loop", "item", "true_body", "value"))
    graph.add_edge(create_edge("loop", "item", "false_body", "value"))
    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("true_body", "value", "if", "true_input"))
    graph.add_edge(create_edge("false_body", "value", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "collect", "item"))
    state = GraphExecutionState(graph=graph)
    if force_compatibility_scheduler:
        state._execution_scheduler = _ExecutionScheduler(state)
    state._pending_if_exec_nodes = _PendingIfScanCounter(state._pending_if_exec_nodes)
    materializer = state._materializer()
    original_attach = materializer._attach_pending_if_inputs
    original_prepared_source = materializer._get_ordered_prepared_nodes_for_source
    attach_calls = 0
    attach_seconds = 0.0
    prepared_candidates_scanned = 0
    measuring_attachment = False

    def counted_prepared_source(source_node_id: str) -> list[str]:
        nonlocal prepared_candidates_scanned
        prepared_nodes = original_prepared_source(source_node_id)
        if measuring_attachment:
            prepared_candidates_scanned += len(prepared_nodes)
        return prepared_nodes

    materializer._get_ordered_prepared_nodes_for_source = counted_prepared_source  # type: ignore[method-assign]

    def measured_attach(*, enqueue: bool = True) -> None:
        nonlocal attach_calls, attach_seconds, measuring_attachment
        attach_calls += 1
        started = time.perf_counter()
        measuring_attachment = True
        try:
            original_attach(enqueue=enqueue)
        finally:
            measuring_attachment = False
        attach_seconds += time.perf_counter() - started

    materializer._attach_pending_if_inputs = measured_attach  # type: ignore[method-assign]
    trace: list[str] = []
    started = time.perf_counter()
    while (node := state.next()) is not None:
        trace.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))
    total_seconds = time.perf_counter() - started

    assert state.is_complete()
    assert not state.has_error()
    assert trace.count("true_body") == count
    assert "false_body" not in trace
    assert trace.count("if") == count
    collect_exec_ids = state.source_prepared_mapping["collect"]
    assert len(collect_exec_ids) == 1
    assert state.results[next(iter(collect_exec_ids))].collection == list(range(count))
    return (
        attach_calls,
        state._pending_if_exec_nodes.items_scanned,
        prepared_candidates_scanned,
        total_seconds,
        attach_seconds,
    )


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True], ids=["generic", "compatibility"])
def test_pending_if_attachment_candidate_visits_are_bounded_for_ci(force_compatibility_scheduler: bool) -> None:
    """Keep a deterministic guard for the per-item If attachment sweep in the default CI lane."""
    count = 300
    attach_calls, _, candidates_scanned, _, _ = _measure_pending_if_attachment(
        count, force_compatibility_scheduler=force_compatibility_scheduler
    )
    assert candidates_scanned <= count * 32, (
        f"pending-If attachment revisited {candidates_scanned} prepared candidates "
        f"for {count} items across {attach_calls} sweeps"
    )


@pytest.mark.slow
def test_pending_if_attachment_sweep_scaling() -> None:
    """Measure the reviewed per-item If graph at a larger size, separating attachment from total execution."""
    for count in (1200,):
        attach_calls, scanned, candidates_scanned, total_seconds, attach_seconds = _measure_pending_if_attachment(count)
        assert candidates_scanned <= count * 32, (
            f"pending-If attachment revisited {candidates_scanned} prepared candidates "
            f"for {count} items across {attach_calls} sweeps"
        )
        print(
            f"pending-If n={count} calls={attach_calls} pending_scanned={scanned} "
            f"prepared_candidates_scanned={candidates_scanned} total={total_seconds:.3f}s "
            f"attach={attach_seconds:.3f}s"
        )


def test_completion_predicate_preserves_durable_state() -> None:
    graph = Graph()
    graph.add_node(AnyTypeTestInvocation(id="value", value=1))
    state = GraphExecutionState(graph=graph)
    node = state.next()
    assert node is not None
    state.complete(node.id, node.invoke(Mock()))
    # A restored session may have completed executions without the derived source history yet.
    state.executed.discard("value")
    state.executed_history.clear()
    before = state.model_dump_json()
    assert state.is_complete()
    assert state.model_dump_json() == before
    restored_state = GraphExecutionState.model_validate_json(before)
    # Legacy complete() snapshots intentionally gain their derived execution references on restore.
    restored_before = restored_state.model_dump_json()
    assert restored_state.execution_refs
    assert restored_state.is_complete()
    assert restored_state.model_dump_json() == restored_before


def test_completion_cache_preserves_completed_sources_after_restore() -> None:
    graph = Graph()
    graph.add_node(AnyTypeTestInvocation(id="first", value=1))
    graph.add_node(AnyTypeTestInvocation(id="second", value=2))
    state = GraphExecutionState(graph=graph)

    first = state.next()
    assert first is not None
    state.complete(first.id, first.invoke(Mock()))

    restored_state = GraphExecutionState.model_validate_json(state.model_dump_json())
    second = restored_state.next()
    assert second is not None
    restored_state.complete(second.id, second.invoke(Mock()))

    assert restored_state.is_complete()


def test_prepare_retirement_checks_only_activation_dependent_sources() -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(AnyTypeTestInvocation(id="selected", value=1))
    graph.add_node(AnyTypeTestInvocation(id="unselected", value=2))
    graph.add_node(IfInvocation(id="if"))
    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("selected", "value", "if", "true_input"))
    graph.add_edge(create_edge("unselected", "value", "if", "false_input"))
    unrelated_source_ids = {f"independent_{index}" for index in range(120)}
    for source_id in unrelated_source_ids:
        graph.add_node(AnyTypeTestInvocation(id=source_id, value=source_id))

    state = GraphExecutionState(graph=graph)
    materializer = state._materializer()
    original_candidates = materializer._get_activation_candidate_source_ids
    candidate_sources: set[str] = set()
    candidate_visits = 0
    prepare_scans = 0

    def counted_candidates(source_graph: object) -> tuple[str, ...]:
        nonlocal candidate_visits, prepare_scans
        candidates = original_candidates(source_graph)  # type: ignore[arg-type]
        candidate_sources.update(candidates)
        candidate_visits += len(candidates)
        prepare_scans += 1
        return candidates

    materializer._get_activation_candidate_source_ids = counted_candidates  # type: ignore[method-assign]
    context = Mock()
    while (node := state.next()) is not None:
        state.complete(node.id, node.invoke(context))

    assert state.is_complete()
    assert not state.has_error()
    assert state.results[next(iter(state.source_prepared_mapping["if"]))].value == 1
    assert "unselected" in state.executed
    assert not unrelated_source_ids.intersection(candidate_sources), (
        f"retirement candidates included unrelated sources: {unrelated_source_ids.intersection(candidate_sources)}"
    )
    assert candidate_sources == {"unselected"}
    assert candidate_visits <= 2 * prepare_scans


def _measure_if_chain_retirement_candidates(length: int) -> tuple[int, int, int, list[str]]:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AnyTypeTestInvocation(id="fallback", value="fallback"))
    graph.add_node(AnyTypeTestInvocation(id="sink"))
    previous_node_id: str | None = None
    for index in range(length):
        node_id = f"branch_{index:03d}"
        graph.add_node(AnyTypeTestInvocation(id=node_id, value=True if index == 0 else None))
        if previous_node_id is not None:
            graph.add_edge(create_edge(previous_node_id, "value", node_id, "value"))
        previous_node_id = node_id
    assert previous_node_id is not None
    for source, source_field, destination, destination_field in (
        ("condition", "value", "if", "condition"),
        ("fallback", "value", "if", "false_input"),
        (previous_node_id, "value", "if", "true_input"),
        ("if", "value", "sink", "value"),
    ):
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    state = GraphExecutionState(graph=graph)
    materializer = state._materializer()
    original_candidates = materializer._get_activation_candidate_source_ids
    candidate_visits = 0
    prepare_scans = 0

    def counted_candidates(source_graph: object) -> tuple[str, ...]:
        nonlocal candidate_visits, prepare_scans
        candidates = original_candidates(source_graph)  # type: ignore[arg-type]
        candidate_visits += len(candidates)
        prepare_scans += 1
        return candidates

    materializer._get_activation_candidate_source_ids = counted_candidates  # type: ignore[method-assign]
    trace: list[str] = []
    while (node := state.next()) is not None:
        trace.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))

    assert state.is_complete()
    assert not state.has_error()
    assert "fallback" not in trace
    assert trace.count("if") == 1
    assert trace.count("sink") == 1
    sink_exec_id = next(iter(state.source_prepared_mapping["sink"]))
    assert state.results[sink_exec_id].value is True
    return candidate_visits, prepare_scans, len(trace), trace


def _measure_if_chain_end_to_end_source_inactivity(
    length: int, *, force_compatibility_scheduler: bool = False
) -> tuple[int, int, list[str]]:
    """Count source-inactivity checks while running a serial If branch through the execution interface."""
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AnyTypeTestInvocation(id="fallback", value="fallback"))
    graph.add_node(AnyTypeTestInvocation(id="sink"))
    previous_node_id: str | None = None
    for index in range(length):
        node_id = f"branch_{index:03d}"
        graph.add_node(AnyTypeTestInvocation(id=node_id, value=True if index == 0 else None))
        if previous_node_id is not None:
            graph.add_edge(create_edge(previous_node_id, "value", node_id, "value"))
        previous_node_id = node_id
    assert previous_node_id is not None
    for source, source_field, destination, destination_field in (
        ("condition", "value", "if", "condition"),
        ("fallback", "value", "if", "false_input"),
        (previous_node_id, "value", "if", "true_input"),
        ("if", "value", "sink", "value"),
    ):
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    state = GraphExecutionState(graph=graph)
    if force_compatibility_scheduler:
        state._execution_scheduler = _ExecutionScheduler(state)
    inactive_checks = 0
    completed_cache_builds = 0
    original_is_source_inactive = state._is_source_inactive
    original_get_completed_source_ids_cache = state._get_completed_source_ids_cache

    def counted_is_source_inactive(source_node_id: str, iteration_path: tuple[int, ...] = ()) -> bool:
        nonlocal inactive_checks
        inactive_checks += 1
        return original_is_source_inactive(source_node_id, iteration_path)

    def counted_get_completed_source_ids_cache() -> set[str]:
        nonlocal completed_cache_builds
        completed_cache_builds += 1
        return original_get_completed_source_ids_cache()

    state._is_source_inactive = counted_is_source_inactive  # type: ignore[method-assign]
    state._get_completed_source_ids_cache = counted_get_completed_source_ids_cache  # type: ignore[method-assign]
    trace: list[str] = []
    while (node := state.next()) is not None:
        trace.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))

    assert state.is_complete()
    assert not state.has_error()
    assert "fallback" not in trace
    assert trace.count("if") == 1
    assert trace.count("sink") == 1
    assert all(trace.count(f"branch_{index:03d}") == 1 for index in range(length))
    assert all(source_id in state.executed for source_id in (*trace, "fallback"))
    assert len(state.executed_history) == len(state.graph.nodes)
    assert set(state.executed_history) == set(state.graph.nodes)
    sink_exec_id = next(iter(state.source_prepared_mapping["sink"]))
    assert state.results[sink_exec_id].value is True
    return inactive_checks, completed_cache_builds, trace


def test_if_chain_retirement_candidate_scan_is_bounded() -> None:
    """Bound activation-candidate enumeration, not all end-to-end source-inactivity work."""
    measurements = {length: _measure_if_chain_retirement_candidates(length) for length in (40, 80)}
    for length, (candidate_visits, prepare_scans, _, _) in measurements.items():
        assert candidate_visits <= prepare_scans * 2, (
            f"if-chain length {length}: {candidate_visits} retirement candidates across {prepare_scans} prepare calls"
        )

    small_work = measurements[40][0]
    large_work = measurements[80][0]
    assert large_work <= small_work * 2.25, (
        f"if-chain retirement candidate work grew faster than graph length: 40={small_work}, 80={large_work}"
    )


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True], ids=["generic", "compatibility"])
def test_if_chain_end_to_end_source_inactivity_work_scales_linearly(force_compatibility_scheduler: bool) -> None:
    """Bound source-inactivity checks including completed-source cache rebuilds."""
    measurements = {
        length: _measure_if_chain_end_to_end_source_inactivity(
            length, force_compatibility_scheduler=force_compatibility_scheduler
        )
        for length in (10, 20, 40, 80, 300)
    }
    for length, (inactive_checks, cache_calls, trace) in measurements.items():
        assert inactive_checks > 0
        assert cache_calls >= len(trace)
        # L=10 and L=20 measured at most 26 and 46 checks: two checks per source node bounds both routes.
        assert inactive_checks <= 2 * (length + 4), (
            f"if-chain length {length}: {inactive_checks} inactivity checks for {length + 4} source nodes"
        )


def test_completed_source_cache_only_evicts_for_new_prepared_executions() -> None:
    graph = Graph()
    graph.add_node(AnyTypeTestInvocation(id="source", value=True))
    state = GraphExecutionState(graph=graph)
    state._register_prepared_exec_node("exec-1", "source")
    state._mark_exec_node_executed("exec-1")
    state._mark_source_executed("source")
    completed = state._get_completed_source_ids_cache()
    assert "source" in completed

    state._register_prepared_exec_node("exec-1", "source")
    assert "source" in completed

    state._register_prepared_exec_node("exec-2", "source")
    assert "source" not in completed


def test_completed_source_cache_keeps_prepared_completion_after_source_discard() -> None:
    graph = Graph()
    graph.add_node(AnyTypeTestInvocation(id="source", value=True))
    state = GraphExecutionState(graph=graph)
    state._register_prepared_exec_node("exec-1", "source")
    state._mark_exec_node_executed("exec-1")
    state._mark_source_executed("source")
    completed = state._get_completed_source_ids_cache()
    assert "source" in completed

    state._discard_source_executed("source")

    assert "source" in completed


def test_completed_source_cache_tracks_completion_edits_rollback_and_restore() -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="source", value=True))
    state = GraphExecutionState(graph=graph)

    initial_completed = state._get_completed_source_ids_cache()
    assert "source" not in initial_completed
    state._mark_source_executed("source")
    assert state._get_completed_source_ids_cache() is initial_completed
    assert "source" in initial_completed

    state.add_node(BooleanInvocation(id="added", value=True))
    edited_completed = state._get_completed_source_ids_cache()
    assert edited_completed is not initial_completed
    assert "added" not in edited_completed

    def mark_then_fail() -> None:
        state._mark_source_executed("added")
        raise RuntimeError("rollback cache mutation")

    with pytest.raises(RuntimeError, match="rollback cache mutation"):
        state._run_materialization_transaction(mark_then_fail)
    assert "added" not in state.executed
    assert "added" not in state._get_completed_source_ids_cache()

    restored = load_execution_state(dump_execution_state(state))
    assert "source" in restored._get_completed_source_ids_cache()
