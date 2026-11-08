from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import networkx as nx
import pytest

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.primitives import BooleanInvocation
from invokeai.app.services.shared import graph_if_dependencies
from invokeai.app.services.shared.execution_engine import ActivationDependency
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.graph_validation import _EdgeList
from tests.test_nodes import AnyTypeTestInvocation, create_edge


class _CountingNodes(dict[str, Any]):
    def __init__(self, nodes: dict[str, Any], counts: dict[str, int]) -> None:
        super().__init__(nodes)
        self.counts = counts

    def values(self) -> Iterator[Any]:
        for value in super().values():
            self.counts["nodes"] += 1
            yield value

    def items(self) -> Iterator[tuple[str, Any]]:
        for item in super().items():
            self.counts["nodes"] += 1
            yield item

    def __iter__(self) -> Iterator[str]:
        for node_id in super().__iter__():
            self.counts["nodes"] += 1
            yield node_id


class _CountingEdgeList:
    def __init__(self, edges: list[Any] | tuple[Any, ...], counts: dict[str, int]) -> None:
        self.edges = edges
        self.counts = counts

    def __iter__(self) -> Iterator[Any]:
        for edge in self.edges:
            self.counts["adjacency_edges"] += 1
            yield edge

    def __len__(self) -> int:
        return len(self.edges)


class _CountingAdjacency(dict[str, list[Any]]):
    def __init__(self, edges: dict[str, list[Any]], counts: dict[str, int]) -> None:
        super().__init__(edges)
        self.counts = counts

    def get(self, node_id: str, default: Any = None) -> Any:
        edges = super().get(node_id, default)
        return _CountingEdgeList(edges, self.counts) if isinstance(edges, list) else edges


class _CountingNodeOrder(tuple[str, ...]):
    def __new__(cls, nodes: tuple[str, ...], counts: dict[str, int]):
        instance = super().__new__(cls, nodes)
        instance.counts = counts
        return instance

    def __iter__(self) -> Iterator[str]:
        for node_id in super().__iter__():
            self.counts["traversal_nodes"] += 1
            yield node_id

    def __reversed__(self) -> Iterator[str]:
        for index in range(len(self) - 1, -1, -1):
            self.counts["traversal_nodes"] += 1
            yield tuple.__getitem__(self, index)


def _add_if(graph: Graph, if_id: str, true_source: str, false_source: str) -> None:
    graph.add_node(IfInvocation(id=if_id))
    graph.add_edge(create_edge("condition", "value", if_id, "condition"))
    graph.add_edge(create_edge(true_source, "value", if_id, "true_input"))
    graph.add_edge(create_edge(false_source, "value", if_id, "false_input"))


def _make_graph(if_count: int) -> Graph:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    previous = "condition"
    for index in range(80):
        node_id = f"chain_{index}"
        graph.add_node(AnyTypeTestInvocation(id=node_id))
        graph.add_edge(create_edge(previous, "value", node_id, "value"))
        previous = node_id

    for index in range(if_count):
        true_source = f"true_{index}"
        false_source = f"false_{index}"
        graph.add_node(AnyTypeTestInvocation(id=true_source))
        graph.add_node(AnyTypeTestInvocation(id=false_source))
        graph.add_edge(create_edge(previous, "value", true_source, "value"))
        graph.add_edge(create_edge(previous, "value", false_source, "value"))
        _add_if(graph, f"if_{index}", true_source, false_source)
    return graph


def _make_nested_if_graph(if_count: int) -> Graph:
    graph = Graph()
    graph.add_node(AnyTypeTestInvocation(id="true_leaf"))
    previous = "true_leaf"
    for index in range(if_count):
        if_id = f"if_{index}"
        condition_id = f"condition_{index}"
        false_source_id = f"false_{index}"
        graph.add_node(BooleanInvocation(id=condition_id, value=True))
        graph.add_node(AnyTypeTestInvocation(id=false_source_id))
        graph.add_node(IfInvocation(id=if_id))
        graph.add_edge(create_edge(condition_id, "value", if_id, "condition"))
        graph.add_edge(create_edge(previous, "value", if_id, "true_input"))
        graph.add_edge(create_edge(false_source_id, "value", if_id, "false_input"))
        previous = if_id

    graph.add_node(AnyTypeTestInvocation(id="sink"))
    graph.add_edge(create_edge(previous, "value", "sink", "value"))
    return graph


def _instrument(state: GraphExecutionState, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counts = {
        "nodes": 0,
        "edges": 0,
        "ancestor_nodes": 0,
        "adjacency_edges": 0,
        "traversal_nodes": 0,
        "traversal_edges": 0,
        "owner_mask_entries": 0,
        "owner_relations": 0,
    }
    object.__setattr__(state.graph, "nodes", _CountingNodes(state.graph.nodes, counts))

    original_edge_iter = _EdgeList.__iter__

    def count_edges(edges: _EdgeList):
        for edge in original_edge_iter(edges):
            counts["edges"] += 1
            yield edge

    original_ancestors = nx.ancestors

    def count_ancestors(graph: Any, node_id: Any) -> set[Any]:
        ancestors = original_ancestors(graph, node_id)
        counts["ancestor_nodes"] += len(ancestors) + 1
        return ancestors

    original_topological_sort = nx.topological_sort

    def count_topological_sort(graph: Any) -> Iterator[Any]:
        nodes = tuple(original_topological_sort(graph))
        counts["traversal_nodes"] += len(nodes)
        counts["traversal_edges"] += graph.number_of_edges()
        return iter(nodes)

    original_get_owner_masks = graph_if_dependencies._get_flat_activation_owner_masks_by_source

    def count_owner_analysis(state: GraphExecutionState, analysis: Any) -> dict[str, int]:
        is_new_analysis = analysis.activation_owner_masks_by_source is None
        if is_new_analysis:
            analysis.input_edges = _CountingAdjacency(analysis.input_edges, counts)
            analysis.output_edges = _CountingAdjacency(analysis.output_edges, counts)
        owners_by_source = original_get_owner_masks(state, analysis)
        if is_new_analysis:
            counts["owner_mask_entries"] += len(owners_by_source)
        return owners_by_source

    original_get_dependencies = graph_if_dependencies._get_source_activation_dependencies

    def count_dependencies(
        state: GraphExecutionState, source_node_id: str, iteration_path: tuple[int, ...] = ()
    ) -> tuple[ActivationDependency, ...]:
        dependencies = original_get_dependencies(state, source_node_id, iteration_path)
        counts["owner_relations"] += len(dependencies)
        return dependencies

    original_topological_order = graph_if_dependencies._get_if_topological_order

    def count_topological_order(analysis: Any) -> tuple[str, ...] | None:
        order = original_topological_order(analysis)
        if order is None:
            return None
        if not isinstance(order, _CountingNodeOrder):
            order = _CountingNodeOrder(order, counts)
            analysis.topological_order = order
        return order

    monkeypatch.setattr(_EdgeList, "__iter__", count_edges)
    monkeypatch.setattr(nx, "ancestors", count_ancestors)
    monkeypatch.setattr(nx, "topological_sort", count_topological_sort)
    monkeypatch.setattr(graph_if_dependencies, "_get_flat_activation_owner_masks_by_source", count_owner_analysis)
    monkeypatch.setattr(graph_if_dependencies, "_get_source_activation_dependencies", count_dependencies)
    monkeypatch.setattr(graph_if_dependencies, "_get_if_topological_order", count_topological_order)
    return counts


def _work(counts: dict[str, int]) -> int:
    return sum(counts.values())


def test_if_analysis_graph_traversal_work_is_bounded_and_invalidated_by_graph_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _make_graph(4)
    state = GraphExecutionState(graph=graph)
    counts = _instrument(state, monkeypatch)

    assert state._can_use_fresh_flat_if_activation()
    for index in range(4, 32):
        true_source = f"true_{index}"
        false_source = f"false_{index}"
        if_id = f"if_{index}"
        state.add_node(AnyTypeTestInvocation(id=true_source))
        state.add_node(AnyTypeTestInvocation(id=false_source))
        state.add_node(IfInvocation(id=if_id))
        for source, destination, field in (
            ("chain_79", true_source, "value"),
            ("chain_79", false_source, "value"),
            ("condition", if_id, "condition"),
            (true_source, if_id, "true_input"),
            (false_source, if_id, "false_input"),
        ):
            state.add_edge(create_edge(source, "value", destination, field))

    # The edit APIs must invalidate ownership analysis. Count its rebuild and all dependency lookups.
    counts.update(dict.fromkeys(counts, 0))
    for index in range(32):
        assert state._get_source_activation_dependencies(f"true_{index}", (index,)) == (
            ActivationDependency(owner_id=f"if_{index}", branch="true_input", frame=(index,)),
        )
        assert state._get_source_activation_dependencies(f"false_{index}", (index + 32,)) == (
            ActivationDependency(owner_id=f"if_{index}", branch="false_input", frame=(index + 32,)),
        )
    graph_size = len(graph.nodes) + len(graph.edges)
    graph_traversal_work = _work(counts)
    assert graph_traversal_work <= 8 * graph_size, (
        f"If ownership visited {graph_traversal_work} graph items for {graph_size} nodes and edges"
    )


def test_nested_if_analysis_materializes_gate_frontiers_and_bounds_mask_payload() -> None:
    measurements = []
    for depth in (16, 32, 64, 128):
        graph = _make_nested_if_graph(depth)
        state = GraphExecutionState(graph=graph)
        dependencies = {
            source_id: state._get_source_activation_dependencies(source_id, (7,)) for source_id in graph.nodes
        }
        relation_count = sum(map(len, dependencies.values()))

        analysis = state._if_graph_analysis
        assert analysis is not None
        owner_masks_by_source = analysis.activation_owner_masks_by_source
        assert owner_masks_by_source is not None
        # shortcut: packed masks bound materialized frontier relations, but bit payload remains
        # O(V·I/w); revisit a sparse representation if profiling on real workflows shows it matters.
        mask_payload_bits = sum(mask.bit_length() for mask in owner_masks_by_source.values())
        frontier_iterations = sum(mask.bit_count() for mask in owner_masks_by_source.values())

        assert dependencies["true_leaf"] == (ActivationDependency(owner_id="if_0", branch="true_input", frame=(7,)),)
        assert relation_count <= 3 * depth + 1
        # The reducer performs one loop iteration per retained frontier owner bit.
        assert frontier_iterations == relation_count
        assert len(owner_masks_by_source) <= len(graph.nodes)
        assert all(isinstance(mask, int) for mask in owner_masks_by_source.values())
        # A compact integer bitset still stores through its highest owner bit; that payload grows with
        # graph size × owner count even though only the frontier relations are materialized.
        assert mask_payload_bits <= 2 * len(owner_masks_by_source) * (2 * depth)
        measurements.append((depth, relation_count, mask_payload_bits))

    assert measurements[-1][1] <= 9 * measurements[0][1]
