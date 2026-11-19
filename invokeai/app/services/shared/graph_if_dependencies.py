from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.services.shared.execution_engine.scheduler import ActivationDependency
from invokeai.app.services.shared.graph_validation import (
    COLLECTION_FIELD,
    ITEM_FIELD,
    CollectInvocation,
    Edge,
    IterateInvocation,
    nx,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


@dataclass(slots=True)
class _IfGraphAnalysis:
    source_graph: Any
    nodes_by_id: dict[str, Any]
    if_node_ids: tuple[str, ...]
    input_edges: dict[str, list[Edge]]
    output_edges: dict[str, list[Edge]]
    has_forbidden_activation_node: bool
    has_iterate_or_collect: bool
    topological_order: Optional[tuple[str, ...]] = None
    is_dag: Optional[bool] = None
    ancestors_by_node: dict[str, frozenset[str]] = field(default_factory=dict)
    lazy_if_candidates: dict[str, bool] = field(default_factory=dict)
    activation_owner_order: Optional[tuple[tuple[str, str], ...]] = None
    activation_owner_masks_by_source: Optional[dict[str, int]] = None


def _get_if_graph_analysis(state: "GraphExecutionState") -> _IfGraphAnalysis:
    source_graph = state._get_source_graph_flat()
    analysis = state._if_graph_analysis
    if analysis is not None and analysis.source_graph is source_graph:
        return analysis

    nodes_by_id = dict(state.graph.nodes.items())
    if_node_ids = tuple(node_id for node_id, node in nodes_by_id.items() if isinstance(node, IfInvocation))
    has_forbidden_activation_node = any(
        isinstance(node, (CallSavedWorkflowInvocation, ForInvocation, ForReturnInvocation))
        for node in nodes_by_id.values()
    )
    has_iterate_or_collect = any(
        isinstance(node, (IterateInvocation, CollectInvocation)) for node in nodes_by_id.values()
    )

    input_edges: dict[str, list[Edge]] = {}
    output_edges: dict[str, list[Edge]] = {}
    for edge in state.graph.edges:
        if edge.type != "default":
            continue
        input_edges.setdefault(edge.destination.node_id, []).append(edge)
        output_edges.setdefault(edge.source.node_id, []).append(edge)

    analysis = _IfGraphAnalysis(
        source_graph=source_graph,
        nodes_by_id=nodes_by_id,
        if_node_ids=if_node_ids,
        input_edges=input_edges,
        output_edges=output_edges,
        has_forbidden_activation_node=has_forbidden_activation_node,
        has_iterate_or_collect=has_iterate_or_collect,
    )
    state._if_graph_analysis = analysis
    return analysis


def _get_if_topological_order(analysis: _IfGraphAnalysis) -> Optional[tuple[str, ...]]:
    if analysis.is_dag is None:
        try:
            analysis.topological_order = tuple(nx.topological_sort(analysis.source_graph))
        except nx.NetworkXUnfeasible:
            analysis.is_dag = False
        else:
            analysis.is_dag = True
    return analysis.topological_order


def _get_if_ancestors(analysis: _IfGraphAnalysis, node_id: str) -> frozenset[str]:
    ancestors = analysis.ancestors_by_node.get(node_id)
    if ancestors is None:
        ancestors = frozenset(nx.ancestors(analysis.source_graph, node_id))
        analysis.ancestors_by_node[node_id] = ancestors
    return ancestors


def _get_fresh_if_nodes(state: "GraphExecutionState") -> tuple[IfInvocation, ...]:
    analysis = _get_if_graph_analysis(state)
    if_node_ids = set(analysis.if_node_ids)
    topological_order = _get_if_topological_order(analysis)
    if topological_order is not None:
        return tuple(analysis.nodes_by_id[node_id] for node_id in topological_order if node_id in if_node_ids)
    return tuple(sorted((analysis.nodes_by_id[node_id] for node_id in if_node_ids), key=lambda node: node.id))


def _get_flat_activation_owner_masks_by_source(
    state: "GraphExecutionState", analysis: _IfGraphAnalysis
) -> dict[str, int]:
    cached = analysis.activation_owner_masks_by_source
    if cached is not None:
        return cached

    if_nodes = _get_fresh_if_nodes(state)
    owners = tuple((node.id, branch) for node in if_nodes for branch in ("true_input", "false_input"))
    owner_bits = {owner: 1 << index for index, owner in enumerate(owners)}
    analysis.activation_owner_order = owners
    endpoint_bits = {(node_id, branch): bit for (node_id, branch), bit in owner_bits.items()}
    candidate_bits: dict[str, int] = {}
    extra_candidate_bits: dict[str, int] = {}

    for (if_node_id, branch_field), bit in owner_bits.items():
        direct_sources = {
            edge.source.node_id
            for edge in analysis.input_edges.get(if_node_id, ())
            if edge.destination.field == branch_field
        }
        for source_id in direct_sources:
            candidate_bits[source_id] = candidate_bits.get(source_id, 0) | bit
            if isinstance(analysis.nodes_by_id[source_id], IfInvocation):
                for edge in analysis.output_edges.get(source_id, ()):
                    destination_id = edge.destination.node_id
                    if not isinstance(
                        analysis.nodes_by_id[destination_id], IfInvocation
                    ) and not analysis.output_edges.get(destination_id):
                        extra_candidate_bits[destination_id] = extra_candidate_bits.get(destination_id, 0) | bit

    # Propagate branch roots to their ancestors once for every owner in parallel.
    for node_id in reversed(analysis.topological_order or ()):
        bits = candidate_bits.get(node_id, 0)
        if bits:
            for edge in analysis.input_edges.get(node_id, ()):
                parent_id = edge.source.node_id
                candidate_bits[parent_id] = candidate_bits.get(parent_id, 0) | bits
    for node_id, bits in extra_candidate_bits.items():
        candidate_bits[node_id] = candidate_bits.get(node_id, 0) | bits

    owned_bits: dict[str, int] = {}
    for node_id in reversed(analysis.topological_order or ()):
        bits = candidate_bits.get(node_id, 0)
        for edge in analysis.output_edges.get(node_id, ()):
            destination_id = edge.destination.node_id
            allowed = owned_bits.get(destination_id, 0) | endpoint_bits.get((destination_id, edge.destination.field), 0)
            bits &= allowed
            if not bits:
                break
        if bits:
            owned_bits[node_id] = bits

    # A nested If cannot run until the dependencies gating its own execution are open. Those
    # prerequisites are implied by either branch gate it owns, so retain only the gate frontier.
    owner_prerequisites = {if_node_id: owned_bits.get(if_node_id, 0) for if_node_id in analysis.if_node_ids}
    _reduce_activation_owner_masks_to_frontier(owned_bits, owners, owner_prerequisites)

    analysis.activation_owner_masks_by_source = owned_bits
    return owned_bits


def _reduce_activation_owner_masks_to_frontier(
    owner_masks_by_source: dict[str, int],
    owners: tuple[tuple[str, str], ...],
    owner_prerequisites: dict[str, int],
) -> None:
    """Drop only gates implied by a nested If's own admission dependencies."""
    for source_id, source_mask in tuple(owner_masks_by_source.items()):
        remaining_mask = source_mask
        frontier_mask = 0
        while remaining_mask:
            # Owners follow graph topological order, so nested gates precede the outer gates they imply.
            bit = remaining_mask & -remaining_mask
            owner_id, _branch = owners[bit.bit_length() - 1]
            frontier_mask |= bit
            remaining_mask &= ~bit
            remaining_mask &= ~owner_prerequisites.get(owner_id, 0)
        if frontier_mask:
            owner_masks_by_source[source_id] = frontier_mask
        else:
            del owner_masks_by_source[source_id]


def _can_use_fresh_mixed_if_iterate_collect(state: "GraphExecutionState") -> bool:
    """Admit the bounded per-item Iterate/If/Collect topology."""
    if state._legacy_snapshot_loaded or len(state.graph.nodes) != 6 or len(state.graph.edges) != 7:
        return False

    if_nodes = [node for node in state.graph.nodes.values() if isinstance(node, IfInvocation)]
    iterate_nodes = [node for node in state.graph.nodes.values() if isinstance(node, IterateInvocation)]
    collect_nodes = [node for node in state.graph.nodes.values() if isinstance(node, CollectInvocation)]
    if len(if_nodes) != 1 or len(iterate_nodes) != 1 or len(collect_nodes) != 1:
        return False

    if any(
        isinstance(node, (CallSavedWorkflowInvocation, ForInvocation, ForReturnInvocation))
        for node in state.graph.nodes.values()
    ):
        return False

    if_node = if_nodes[0]
    iterate_node = iterate_nodes[0]
    collect_node = collect_nodes[0]

    collection_edges = state.graph._get_input_edges(iterate_node.id, COLLECTION_FIELD)
    if len(collection_edges) != 1:
        return False
    collection_edge = collection_edges[0]
    collection_source = state.graph.get_node(collection_edge.source.node_id)
    if (
        collection_edge.source.field != COLLECTION_FIELD
        or state.graph._get_input_edges(collection_source.id)
        or state.graph._get_output_edges(collection_source.id) != [collection_edge]
    ):
        return False

    condition_edges = state.graph._get_input_edges(if_node.id, "condition")
    iterate_item_edges = state.graph._get_output_edges(iterate_node.id, ITEM_FIELD)
    if (
        len(condition_edges) != 1
        or len(iterate_item_edges) != 3
        or condition_edges[0].source.node_id != iterate_node.id
    ):
        return False
    if condition_edges[0].source.field != ITEM_FIELD:
        return False

    branch_edges = []
    for branch_field in ("true_input", "false_input"):
        input_edges = state.graph._get_input_edges(if_node.id, branch_field)
        if len(input_edges) != 1:
            return False
        branch_edge = input_edges[0]
        branch_node = state.graph.get_node(branch_edge.source.node_id)
        branch_input_edges = state.graph._get_input_edges(branch_node.id)
        if len(branch_input_edges) != 1 or branch_input_edges[0].source.node_id != iterate_node.id:
            return False
        if branch_input_edges[0].source.field != ITEM_FIELD or state.graph._get_output_edges(branch_node.id) != [
            branch_edge
        ]:
            return False
        branch_edges.append(branch_input_edges[0])

    if len({edge.destination.node_id for edge in branch_edges}) != 2:
        return False

    collect_edges = state.graph._get_input_edges(collect_node.id, "item")
    if len(collect_edges) != 1 or collect_edges[0].source.node_id != if_node.id:
        return False
    if collect_edges[0].source.field != "value" or state.graph._get_output_edges(if_node.id) != [collect_edges[0]]:
        return False
    return not state.graph._get_input_edges(collect_node.id, COLLECTION_FIELD) and not state.graph._get_output_edges(
        collect_node.id
    )


def _can_use_fresh_flat_if_activation(state: "GraphExecutionState") -> bool:
    """Compile dependencies for every fresh acyclic, non-loop If graph."""

    if state._legacy_snapshot_loaded:
        return False

    analysis = _get_if_graph_analysis(state)
    if analysis.has_forbidden_activation_node:
        return False
    if analysis.has_iterate_or_collect:
        return _can_use_fresh_mixed_if_iterate_collect(state)

    return _get_if_topological_order(analysis) is not None


def _is_fresh_lazy_if_candidate(state: "GraphExecutionState", if_node_id: str) -> bool:
    """Return whether this If's unselected branch has no externally visible work."""
    if state._legacy_snapshot_loaded:
        return False

    analysis = _get_if_graph_analysis(state)
    if len(analysis.if_node_ids) != 1 or analysis.if_node_ids[0] != if_node_id:
        return False
    cached = analysis.lazy_if_candidates.get(if_node_id)
    if cached is not None:
        return cached

    condition_edges = [
        edge for edge in analysis.input_edges.get(if_node_id, ()) if edge.destination.field == "condition"
    ]
    if condition_edges and _can_use_fresh_mixed_if_iterate_collect(state):
        analysis.lazy_if_candidates[if_node_id] = True
        return True

    if analysis.has_forbidden_activation_node or analysis.has_iterate_or_collect:
        analysis.lazy_if_candidates[if_node_id] = False
        return False

    if _get_if_topological_order(analysis) is None:
        analysis.lazy_if_candidates[if_node_id] = False
        return False

    condition_sources = {
        node_id
        for edge in condition_edges
        for node_id in (edge.source.node_id, *_get_if_ancestors(analysis, edge.source.node_id))
    }
    branch_sources = {
        branch: {
            node_id
            for edge in analysis.input_edges.get(if_node_id, ())
            if edge.destination.field == branch
            for node_id in (edge.source.node_id, *_get_if_ancestors(analysis, edge.source.node_id))
        }
        for branch in ("true_input", "false_input")
    }
    shared_sources = (branch_sources["true_input"] & branch_sources["false_input"]) | (
        (branch_sources["true_input"] | branch_sources["false_input"]) & condition_sources
    )

    for branch, sources in branch_sources.items():
        private_sources = sources - shared_sources
        for source_id in private_sources:
            for edge in analysis.output_edges.get(source_id, ()):
                if edge.destination.node_id in private_sources:
                    continue
                if edge.destination.node_id == if_node_id and edge.destination.field == branch:
                    continue
                analysis.lazy_if_candidates[if_node_id] = False
                return False
    analysis.lazy_if_candidates[if_node_id] = True
    return True


def _get_fresh_if_branch_sources(state: "GraphExecutionState", if_node_id: str, branch_field: str) -> set[str]:
    cache_key = (if_node_id, branch_field)
    cached = state._if_branch_sources_cache.get(cache_key)
    if cached is not None:
        return set(cached)

    analysis = _get_if_graph_analysis(state)
    direct_sources = {
        edge.source.node_id
        for edge in analysis.input_edges.get(if_node_id, ())
        if edge.destination.field == branch_field
    }
    branch_sources = set(direct_sources)
    for source_node_id in direct_sources:
        branch_sources.update(_get_if_ancestors(analysis, source_node_id))
        source_node = analysis.nodes_by_id[source_node_id]
        if isinstance(source_node, IfInvocation):
            branch_sources.update(
                edge.destination.node_id
                for edge in analysis.output_edges.get(source_node_id, ())
                if not isinstance(analysis.nodes_by_id[edge.destination.node_id], IfInvocation)
                and not analysis.output_edges.get(edge.destination.node_id)
            )

    if not state.graph._get_input_edges(if_node_id, "condition"):
        iterator_graph = state._iterator_graph(analysis.source_graph)
        frame_iterators = {
            ancestor_id
            for ancestor_id in nx.ancestors(iterator_graph, if_node_id)
            if isinstance(analysis.nodes_by_id[ancestor_id], (ForInvocation, IterateInvocation))
        }
        for iterator_id in frame_iterators:
            branch_sources.discard(iterator_id)
            branch_sources.difference_update(_get_if_ancestors(analysis, iterator_id))

    invalid_output_counts: dict[str, int] = {}
    pending = deque()
    for source_node_id in branch_sources:
        invalid_count = sum(
            edge.destination.node_id not in branch_sources
            and not (edge.destination.node_id == if_node_id and edge.destination.field == branch_field)
            for edge in analysis.output_edges.get(source_node_id, ())
        )
        invalid_output_counts[source_node_id] = invalid_count
        if invalid_count:
            pending.append(source_node_id)
    while pending:
        source_node_id = pending.popleft()
        if source_node_id not in branch_sources:
            continue
        branch_sources.remove(source_node_id)
        for edge in analysis.input_edges.get(source_node_id, ()):
            parent_id = edge.source.node_id
            if parent_id in branch_sources:
                invalid_output_counts[parent_id] += 1
                if invalid_output_counts[parent_id] == 1:
                    pending.append(parent_id)
    state._if_branch_sources_cache[cache_key] = frozenset(branch_sources)
    return branch_sources


def _get_source_activation_dependencies(
    state: "GraphExecutionState", source_node_id: str, iteration_path: tuple[int, ...] = ()
) -> tuple[ActivationDependency, ...]:
    key = (source_node_id, iteration_path)
    if key in state._if_activation_dependencies_by_source:
        return state._if_activation_dependencies_by_source[key]

    if not state._can_use_fresh_flat_if_activation():
        return state._if_activation_controller().get_source_dependencies(source_node_id, iteration_path)

    analysis = _get_if_graph_analysis(state)
    if analysis.has_iterate_or_collect:
        owners = tuple(
            (if_node.id, branch_field)
            for if_node in _get_fresh_if_nodes(state)
            for branch_field in ("true_input", "false_input")
            if source_node_id in _get_fresh_if_branch_sources(state, if_node.id, branch_field)
        )
        dependencies = tuple(
            ActivationDependency(owner_id=if_node_id, branch=branch_field, frame=iteration_path)
            for if_node_id, branch_field in owners
        )
    else:
        owner_masks_by_source = _get_flat_activation_owner_masks_by_source(state, analysis)
        owner_order = analysis.activation_owner_order or ()
        owner_mask = owner_masks_by_source.get(source_node_id, 0)
        dependencies_list = []
        while owner_mask:
            bit = owner_mask & -owner_mask
            if_node_id, branch_field = owner_order[bit.bit_length() - 1]
            dependencies_list.append(
                ActivationDependency(owner_id=if_node_id, branch=branch_field, frame=iteration_path)
            )
            owner_mask ^= bit
        dependencies = tuple(dependencies_list)
    state._tx_set_mapping(state._if_activation_dependencies_by_source, key, dependencies)
    return dependencies
