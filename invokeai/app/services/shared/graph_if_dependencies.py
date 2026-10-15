from typing import TYPE_CHECKING

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.services.shared.execution_engine.scheduler import ActivationDependency
from invokeai.app.services.shared.graph_validation import (
    COLLECTION_FIELD,
    ITEM_FIELD,
    CollectInvocation,
    IterateInvocation,
    nx,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


def _get_fresh_if_nodes(state: "GraphExecutionState") -> tuple[IfInvocation, ...]:
    source_graph = state._get_source_graph_flat()
    source_order = {node_id: index for index, node_id in enumerate(nx.topological_sort(source_graph))}
    return tuple(
        sorted(
            (node for node in state.graph.nodes.values() if isinstance(node, IfInvocation)),
            key=lambda node: (source_order.get(node.id, len(source_order)), node.id),
        )
    )


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
    """Admit fresh bounded independent-sibling or nested-If dependency compilation."""

    if state._legacy_snapshot_loaded:
        return False

    if any(
        isinstance(node, (CallSavedWorkflowInvocation, ForInvocation, ForReturnInvocation))
        for node in state.graph.nodes.values()
    ):
        return False
    if any(isinstance(node, (IterateInvocation, CollectInvocation)) for node in state.graph.nodes.values()):
        return _can_use_fresh_mixed_if_iterate_collect(state)

    try:
        source_graph = state._get_source_graph_flat()
        if not nx.is_directed_acyclic_graph(source_graph):
            return False
        if_nodes = _get_fresh_if_nodes(state)
    except nx.NetworkXUnfeasible:
        return False

    if len(if_nodes) != 1:
        if len(if_nodes) not in {2, 3, 4}:
            return False

        if any(edge.type != "default" for edge in state.graph.edges):
            return False

        if_node_ids = {node.id for node in if_nodes}
        nested_edges = [
            edge
            for edge in state.graph.edges
            if edge.source.node_id in if_node_ids and edge.destination.node_id in if_node_ids
        ]
        if not nested_edges:
            expected_input_fields = {"condition", "true_input", "false_input"}
            if any(
                source_if.id != destination_if.id and nx.has_path(source_graph, source_if.id, destination_if.id)
                for source_if in if_nodes
                for destination_if in if_nodes
            ):
                return False
            for if_node in if_nodes:
                input_edges = state.graph._get_input_edges(if_node.id)
                if (
                    len(input_edges) != len(expected_input_fields)
                    or {edge.destination.field for edge in input_edges} != expected_input_fields
                ):
                    return False
                if any(edge.source.node_id in if_node_ids for edge in input_edges):
                    return False
        elif len(if_nodes) in {2, 3, 4} and len(nested_edges) == len(if_nodes) - 1:
            if any(
                edge.source.field != "value" or edge.destination.field not in {"true_input", "false_input"}
                for edge in nested_edges
            ):
                return False

            nested_edge_by_source = {edge.source.node_id: edge for edge in nested_edges}
            nested_destinations = {edge.destination.node_id for edge in nested_edges}
            if len(nested_edge_by_source) != len(nested_edges) or len(nested_destinations) != len(nested_edges):
                return False

            chain_start_ids = if_node_ids - nested_destinations
            chain_end_ids = if_node_ids - set(nested_edge_by_source)
            if len(chain_start_ids) != 1 or len(chain_end_ids) != 1:
                return False

            current_if_id = next(iter(chain_start_ids))
            chain_path = [current_if_id]
            for _ in range(len(nested_edges)):
                nested_edge = nested_edge_by_source.get(current_if_id)
                if nested_edge is None:
                    return False
                next_if_id = nested_edge.destination.node_id
                current_if_id = next_if_id
                chain_path.append(current_if_id)
            if current_if_id not in chain_end_ids:
                return False
            if len(state.graph._get_output_edges(current_if_id)) > 1:
                return False

            middle_if_id = chain_path[-2]
            middle_nested_edge = nested_edge_by_source[middle_if_id]
            for chain_source_id in chain_path[:-1]:
                if chain_source_id == middle_if_id and len(if_nodes) == 3:
                    continue
                nested_edge = nested_edge_by_source[chain_source_id]
                if any(edge.source.node_id == chain_source_id and edge != nested_edge for edge in state.graph.edges):
                    return False
            middle_extra_edges = [
                edge for edge in state.graph.edges if edge.source.node_id == middle_if_id and edge != middle_nested_edge
            ]
            if len(middle_extra_edges) > 1:
                return False
            if middle_extra_edges:
                fanout_edge = middle_extra_edges[0]
                fanout_node = state.graph.get_node(fanout_edge.destination.node_id)
                if (
                    fanout_edge.source.field != "value"
                    or isinstance(fanout_node, IfInvocation)
                    or state.graph._get_output_edges(fanout_node.id)
                ):
                    return False

            expected_input_fields = {"condition", "true_input", "false_input"}
            for if_node in if_nodes:
                input_edges = state.graph._get_input_edges(if_node.id)
                if (
                    len(input_edges) != len(expected_input_fields)
                    or {edge.destination.field for edge in input_edges} != expected_input_fields
                    or any(edge.source.node_id in if_node_ids and edge not in nested_edges for edge in input_edges)
                ):
                    return False
        else:
            return False

    return True


def _get_fresh_if_branch_sources(state: "GraphExecutionState", if_node_id: str, branch_field: str) -> set[str]:
    cache_key = (if_node_id, branch_field)
    cached = state._if_branch_sources_cache.get(cache_key)
    if cached is not None:
        return set(cached)

    source_graph = state._get_source_graph_flat()
    direct_sources = {edge.source.node_id for edge in state.graph._get_input_edges(if_node_id, branch_field)}
    branch_sources = set(direct_sources)
    for source_node_id in direct_sources:
        branch_sources.update(nx.ancestors(source_graph, source_node_id))
        source_node = state.graph.get_node(source_node_id)
        if isinstance(source_node, IfInvocation):
            branch_sources.update(
                edge.destination.node_id
                for edge in state.graph._get_output_edges(source_node_id)
                if not isinstance(state.graph.get_node(edge.destination.node_id), IfInvocation)
                and not state.graph._get_output_edges(edge.destination.node_id)
            )

    changed = True
    while changed:
        changed = False
        for source_node_id in tuple(branch_sources):
            if all(
                edge.destination.node_id in branch_sources
                or (edge.destination.node_id == if_node_id and edge.destination.field == branch_field)
                for edge in state.graph._get_output_edges(source_node_id)
            ):
                continue
            branch_sources.remove(source_node_id)
            changed = True
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

    dependencies = tuple(
        ActivationDependency(owner_id=if_node.id, branch=branch_field, frame=iteration_path)
        for if_node in _get_fresh_if_nodes(state)
        for branch_field in ("true_input", "false_input")
        if source_node_id in _get_fresh_if_branch_sources(state, if_node.id, branch_field)
    )
    state._tx_set_mapping(state._if_activation_dependencies_by_source, key, dependencies)
    return dependencies
