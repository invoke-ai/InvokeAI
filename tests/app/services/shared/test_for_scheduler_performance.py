"""Measure scheduler mapping work and optional overhead for trivial loop bodies.

The deterministic scaling regression counts mapping visits during end-to-end execution.
Optional absolute-time benchmarks remain marked `slow`.
"""

import time
from collections.abc import Callable
from unittest.mock import Mock

import pytest

from invokeai.app.invocations.collections import RangeInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.invocations.primitives import BooleanInvocation
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    Graph,
    GraphExecutionState,
    IterateInvocation,
    _GenericGraphSchedulerAdapter,
)
from tests.test_nodes import AnyTypeTestInvocation, create_edge, create_loop_linkage


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


def _measure_pending_if_attachment(count: int) -> tuple[int, int, int, float, float]:
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


@pytest.mark.slow
def test_pending_if_attachment_sweep_scaling() -> None:
    """Measure the reviewed per-item If graph, separating attachment sweeps from total execution."""
    for count in (300, 1200):
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


def test_if_chain_retirement_candidate_work_scales_linearly() -> None:
    measurements = {length: _measure_if_chain_retirement_candidates(length) for length in (150, 300)}
    for length, (candidate_visits, prepare_scans, _, _) in measurements.items():
        assert candidate_visits <= prepare_scans * 2, (
            f"if-chain length {length}: {candidate_visits} retirement candidates across {prepare_scans} prepare calls"
        )

    small_work = measurements[150][0]
    large_work = measurements[300][0]
    assert large_work <= small_work * 2.25, (
        f"if-chain retirement candidate work grew faster than graph length: 150={small_work}, 300={large_work}"
    )
