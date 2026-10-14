"""Scheduler behavior and overhead for trivial loop bodies.

The scheduler-scaling regression tests count visits to growing scheduler ledgers, so
they remain stable under shared CI load. Optional absolute-time benchmarks remain marked `slow`.
The completion-state tests are not benchmarks.
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


class _ActivationDependencyScanCounter:
    def __init__(self) -> None:
        self.active = False
        self.entries_visited = 0
        self.resolutions = 0
        self.prepared_node_count = 0
        self.states_by_id: dict[int, _ActivationDependencyScanCounter] = {}


class _PreparedSourceCountingMapping(dict[str, str]):
    def __init__(self, values: dict[str, str], counter: _ActivationDependencyScanCounter) -> None:
        super().__init__(values)
        self._counter = counter

    def __iter__(self):  # type: ignore[override]
        for key in super().__iter__():
            if self._counter.active:
                self._counter.entries_visited += 1
            yield key


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
    activation_scan_counter: _ActivationDependencyScanCounter | None = None,
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
    if activation_scan_counter is not None:
        state.prepared_source_mapping = _PreparedSourceCountingMapping(
            state.prepared_source_mapping, activation_scan_counter
        )  # type: ignore[assignment]
        state._prepared_registry()._prepared_source_mapping = state.prepared_source_mapping
        activation_scan_counter.states_by_id[id(state)] = activation_scan_counter
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
    if activation_scan_counter is not None:
        activation_scan_counter.prepared_node_count = len(state.prepared_source_mapping)
    return clock() - started


@pytest.mark.parametrize("loop_type", ["iterate", "for"])
@pytest.mark.parametrize("condition_value", [True, False])
def test_repeated_if_resolution_indexes_activation_dependencies_once(
    monkeypatch: pytest.MonkeyPatch, loop_type: str, condition_value: bool
) -> None:
    counter = _ActivationDependencyScanCounter()
    get_dependents = GraphExecutionState._get_activation_dependents_for_if

    def count_lookup_scans(
        state: GraphExecutionState, source_if_id: str, iteration_path: tuple[int, ...]
    ) -> tuple[str, ...]:
        state_counter = counter.states_by_id.get(id(state))
        if state_counter is None:
            return get_dependents(state, source_if_id, iteration_path)
        state_counter.active = True
        try:
            return get_dependents(state, source_if_id, iteration_path)
        finally:
            state_counter.active = False
            state_counter.resolutions += 1

    monkeypatch.setattr(GraphExecutionState, "_get_activation_dependents_for_if", count_lookup_scans)
    _run_trivial_loop(
        loop_type,
        300,
        include_if=True,
        external_condition=True,
        condition_value=condition_value,
        activation_scan_counter=counter,
    )

    assert counter.resolutions >= 300
    assert counter.entries_visited <= counter.prepared_node_count, (
        f"If resolution scanned {counter.entries_visited} prepared nodes across {counter.resolutions} resolutions; "
        f"session contains {counter.prepared_node_count} prepared nodes"
    )


@pytest.mark.slow
def test_if_activation_dependency_scan_performance_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare 1,200 If resolutions with the former repeated scan and the indexed lookup."""

    original_get_dependents = GraphExecutionState._get_activation_dependents_for_if
    measurements: dict[str, tuple[float, _ActivationDependencyScanCounter]] = {}
    for legacy_scan in (True, False):
        counter = _ActivationDependencyScanCounter()

        def measure_lookup(
            state: GraphExecutionState,
            source_if_id: str,
            iteration_path: tuple[int, ...],
            active_counter: _ActivationDependencyScanCounter = counter,
            use_legacy_scan: bool = legacy_scan,
        ) -> tuple[str, ...]:
            state_counter = active_counter.states_by_id.get(id(state))
            if state_counter is not None:
                if use_legacy_scan:
                    state._if_activation_dependents_index_complete = False
                state_counter.active = True
            try:
                return original_get_dependents(state, source_if_id, iteration_path)
            finally:
                if state_counter is not None:
                    state_counter.active = False
                    state_counter.resolutions += 1

        monkeypatch.setattr(GraphExecutionState, "_get_activation_dependents_for_if", measure_lookup)
        started = time.perf_counter()
        _run_trivial_loop(
            "iterate",
            1200,
            include_if=True,
            external_condition=True,
            condition_value=True,
            activation_scan_counter=counter,
        )
        measurements["legacy repeated scan" if legacy_scan else "indexed"] = (time.perf_counter() - started, counter)

    legacy_seconds, legacy_counter = measurements["legacy repeated scan"]
    indexed_seconds, indexed_counter = measurements["indexed"]
    print(
        "\n1,200-item Iterate/If comparison: "
        f"legacy={legacy_seconds:.3f}s/{legacy_counter.entries_visited} prepared-node visits; "
        f"indexed={indexed_seconds:.3f}s/{indexed_counter.entries_visited} prepared-node visits; "
        f"prepared_nodes={indexed_counter.prepared_node_count}"
    )
    assert legacy_counter.resolutions >= 1200
    assert indexed_counter.resolutions >= 1200
    assert indexed_counter.entries_visited <= indexed_counter.prepared_node_count
    assert legacy_counter.entries_visited > indexed_counter.entries_visited


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
    # Generous headroom for shared CI hosts; the reported quadratic For regression took six seconds.
    assert elapsed < 4, f"{loop_type}: {elapsed:.3f}s for 600 items"


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
