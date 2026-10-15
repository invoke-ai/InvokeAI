"""Bounded generic planning for the supported nested For/Iterate/Collect shape."""

from typing import TYPE_CHECKING

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.services.shared.graph_iterate_planner import (
    _attach_direct_execution_edges,
    _create_direct_execution_node_copy,
    _initialize_direct_execution_node,
)
from invokeai.app.services.shared.graph_models import Edge, EdgeConnection
from invokeai.app.services.shared.graph_validation import (
    COLLECTION_FIELD,
    ITEM_FIELD,
    _SupportedNestedIterateChain,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


def _get_exact_nested_iterate_body(state: "GraphExecutionState"):
    """Return the supported nested stream contract, excluding broader topologies."""
    for_node = next((node for node in state.graph.nodes.values() if isinstance(node, ForInvocation)), None)
    if for_node is None:
        return None
    outer_collection_edges = state.graph._get_input_edges(for_node.id, COLLECTION_FIELD)
    if len(outer_collection_edges) > 1:
        return None
    input_driven_outer = bool(outer_collection_edges)
    expected_node_count = 8 if input_driven_outer else 7
    expected_edge_count = 8 if input_driven_outer else 7
    if not input_driven_outer:
        serial_chain = state.graph._get_supported_for_serial_nested_iterate_chain(
            for_node.id, state._get_source_graph_flat()
        )
        if serial_chain is not None and len(state.graph.nodes) == 9 and len(state.graph.edges) == 9:
            return for_node.id, serial_chain
    if len(state.graph.nodes) != expected_node_count or len(state.graph.edges) != expected_edge_count:
        return None
    if input_driven_outer:
        outer_collection_edge = outer_collection_edges[0]
        producer = state.graph.get_node(outer_collection_edge.source.node_id)
        if (
            outer_collection_edge.destination.node_id != for_node.id
            or outer_collection_edge.source.field != COLLECTION_FIELD
            or isinstance(
                producer,
                (
                    CallSavedWorkflowInvocation,
                    IfInvocation,
                    ForInvocation,
                    ForReturnInvocation,
                ),
            )
            or state.graph._get_input_edges(producer.id)
            or state.graph._get_output_edges(producer.id) != [outer_collection_edge]
        ):
            return None
    nested_body = state.graph._get_supported_for_nested_iterate_body(for_node.id, state._get_source_graph_flat())
    if nested_body is None or len(nested_body.body_path_nodes) != 5:
        return None
    return for_node.id, nested_body


def can_use_nested_iterate_planner(state: "GraphExecutionState") -> bool:
    """Check the exact fresh nested stream topology before generic admission."""
    return not state._legacy_snapshot_loaded and _get_exact_nested_iterate_body(state) is not None


_SUPPORTED_NESTED_ITERATE_SEQUENCE_DEPTHS = range(2, 9)


def get_nested_iterate_sequence_depth(state: "GraphExecutionState") -> int | None:
    """Return the exact supported fresh Iterate-only chain depth, if any."""
    if state._legacy_snapshot_loaded:
        return None
    source_graph = state._get_source_graph_flat()
    for depth in _SUPPORTED_NESTED_ITERATE_SEQUENCE_DEPTHS:
        if state.graph._get_supported_nested_iterate_sequence(source_graph, iterate_count=depth) is not None:
            return depth
    return None


def _outer_iteration_path(state: "GraphExecutionState", prepared_for_id: str) -> tuple[int, ...]:
    path = state._get_iteration_path(prepared_for_id)
    prepared_for = state.execution_graph.get_node(prepared_for_id)
    if isinstance(prepared_for, ForInvocation) and prepared_for.index >= 0:
        return (*state._get_for_parent_iteration_path(prepared_for_id), prepared_for.index)
    return path


def _prepared_at_path(state: "GraphExecutionState", source_node_id: str, path: tuple[int, ...]) -> str | None:
    return state._prepared_registry().get_prepared_id_at_path(source_node_id, path)


def _prepare_nested_iterate_body_for_outer(
    state: "GraphExecutionState", source_for_id: str, prepared_for_id: str
) -> None:
    exact = _get_exact_nested_iterate_body(state)
    if exact is None or exact[0] != source_for_id:
        return
    _, nested_body = exact
    if isinstance(nested_body, _SupportedNestedIterateChain):
        _prepare_serial_nested_iterate_chain_for_outer(state, source_for_id, prepared_for_id, nested_body)
        return
    outer_path = _outer_iteration_path(state, prepared_for_id)
    source_iterate_id = nested_body.iterate_node_id
    source_collect_id = nested_body.collect_node_id
    source_return_id = nested_body.return_node_id
    iterate_input_edge = state.graph._get_input_edges(source_iterate_id, COLLECTION_FIELD)[0]
    preparation_node_id = iterate_input_edge.source.node_id
    preparation_input_edge = state.graph._get_input_edges(preparation_node_id)[0]
    body_node_id = state.graph._get_input_edges(source_collect_id, ITEM_FIELD)[0].source.node_id
    body_input_edge = state.graph._get_input_edges(body_node_id)[0]

    preparation_exec_id = _prepared_at_path(state, preparation_node_id, outer_path)
    if preparation_exec_id is None:
        preparation_node = _create_direct_execution_node_copy(state, preparation_node_id, iteration_path=outer_path)
        attached_preparation_edges = _attach_direct_execution_edges(
            state,
            preparation_node.id,
            [
                Edge(
                    source=EdgeConnection(node_id=prepared_for_id, field=preparation_input_edge.source.field),
                    destination=EdgeConnection(node_id="", field=preparation_input_edge.destination.field),
                )
            ],
        )
        _initialize_direct_execution_node(state, preparation_node.id, attached_preparation_edges)
        return

    if preparation_exec_id not in state.results:
        return
    if _prepared_at_path(state, source_collect_id, outer_path) is not None:
        return

    collection = getattr(state.results[preparation_exec_id], iterate_input_edge.source.field)
    if not isinstance(collection, list):
        raise ValueError("Nested Iterate collection source must produce a list")

    body_exec_ids: list[str] = []
    for index in range(len(collection)):
        iteration_path = (*outer_path, index)
        iterate_node = _create_direct_execution_node_copy(
            state, source_iterate_id, iteration_index=index, iteration_path=iteration_path
        )
        attached_iterate_edges = _attach_direct_execution_edges(
            state,
            iterate_node.id,
            [
                Edge(
                    source=EdgeConnection(node_id=preparation_exec_id, field=iterate_input_edge.source.field),
                    destination=EdgeConnection(node_id="", field=iterate_input_edge.destination.field),
                )
            ],
        )
        _initialize_direct_execution_node(state, iterate_node.id, attached_iterate_edges)

        body_node = _create_direct_execution_node_copy(state, body_node_id, iteration_path=iteration_path)
        attached_body_edges = _attach_direct_execution_edges(
            state,
            body_node.id,
            [
                Edge(
                    source=EdgeConnection(node_id=iterate_node.id, field=body_input_edge.source.field),
                    destination=EdgeConnection(node_id="", field=body_input_edge.destination.field),
                )
            ],
        )
        _initialize_direct_execution_node(state, body_node.id, attached_body_edges)
        body_exec_ids.append(body_node.id)

    if not body_exec_ids:
        state._record_empty_iterate_stream(source_iterate_id, outer_path)

    state._discard_source_executed(source_iterate_id)
    collect_node = _create_direct_execution_node_copy(state, source_collect_id, iteration_path=outer_path)
    collect_item_edge = state.graph._get_input_edges(source_collect_id, ITEM_FIELD)[0]
    attached_collect_edges = _attach_direct_execution_edges(
        state,
        collect_node.id,
        [
            Edge(
                source=EdgeConnection(node_id=body_exec_id, field=collect_item_edge.source.field),
                destination=EdgeConnection(node_id="", field=collect_item_edge.destination.field),
            )
            for body_exec_id in body_exec_ids
        ],
    )
    _initialize_direct_execution_node(state, collect_node.id, attached_collect_edges)

    state._discard_source_executed(source_return_id)
    return_node = _create_direct_execution_node_copy(state, source_return_id, iteration_path=outer_path)
    return_input_edges = state.graph._get_input_edges(source_return_id, "output")
    attached_return_edges = _attach_direct_execution_edges(
        state,
        return_node.id,
        [
            Edge(
                source=EdgeConnection(node_id=collect_node.id, field=COLLECTION_FIELD),
                destination=EdgeConnection(node_id="", field=edge.destination.field),
            )
            for edge in return_input_edges
        ],
    )
    _initialize_direct_execution_node(state, return_node.id, attached_return_edges)


def _prepare_serial_nested_iterate_chain_for_outer(
    state: "GraphExecutionState",
    source_for_id: str,
    prepared_for_id: str,
    nested_body: _SupportedNestedIterateChain,
) -> None:
    """Prepare one exact serial two-level nested iterator body incrementally."""
    outer_path = _outer_iteration_path(state, prepared_for_id)
    first_iterate_id, second_iterate_id = nested_body.iterate_node_ids
    source_collect_id = nested_body.collect_node_id
    source_return_id = nested_body.return_node_id

    first_input_edge = state.graph._get_input_edges(first_iterate_id, COLLECTION_FIELD)[0]
    first_preparation_id = first_input_edge.source.node_id
    first_preparation_input = state.graph._get_input_edges(first_preparation_id)[0]
    second_input_edge = state.graph._get_input_edges(second_iterate_id, COLLECTION_FIELD)[0]
    second_preparation_id = second_input_edge.source.node_id
    second_preparation_input = state.graph._get_input_edges(second_preparation_id)[0]
    body_input = state.graph._get_input_edges(
        state.graph._get_input_edges(source_collect_id, ITEM_FIELD)[0].source.node_id
    )[0]
    body_node_id = body_input.destination.node_id
    collect_item_edge = state.graph._get_input_edges(source_collect_id, ITEM_FIELD)[0]

    def create_copy(
        source_node_id: str,
        iteration_path: tuple[int, ...],
        input_edges: list[Edge],
        iteration_index: int = -1,
    ) -> str:
        node = _create_direct_execution_node_copy(
            state, source_node_id, iteration_index=iteration_index, iteration_path=iteration_path
        )
        attached_edges = _attach_direct_execution_edges(state, node.id, input_edges)
        _initialize_direct_execution_node(state, node.id, attached_edges)
        return node.id

    preparation_exec_id = _prepared_at_path(state, first_preparation_id, outer_path)
    if preparation_exec_id is None:
        create_copy(
            first_preparation_id,
            outer_path,
            [
                Edge(
                    source=EdgeConnection(node_id=prepared_for_id, field=first_preparation_input.source.field),
                    destination=EdgeConnection(node_id="", field=first_preparation_input.destination.field),
                )
            ],
        )
        return
    if preparation_exec_id not in state.results:
        return

    collection = getattr(state.results[preparation_exec_id], first_input_edge.source.field)
    if not isinstance(collection, list):
        raise ValueError("Nested Iterate collection source must produce a list")

    state._discard_source_executed(first_iterate_id)
    for first_index in range(len(collection)):
        first_path = (*outer_path, first_index)
        first_exec_id = _prepared_at_path(state, first_iterate_id, first_path)
        if first_exec_id is None:
            first_exec_id = create_copy(
                first_iterate_id,
                first_path,
                [
                    Edge(
                        source=EdgeConnection(node_id=preparation_exec_id, field=first_input_edge.source.field),
                        destination=EdgeConnection(node_id="", field=first_input_edge.destination.field),
                    )
                ],
                iteration_index=first_index,
            )
        second_preparation_exec_id = _prepared_at_path(state, second_preparation_id, first_path)
        if second_preparation_exec_id is None:
            create_copy(
                second_preparation_id,
                first_path,
                [
                    Edge(
                        source=EdgeConnection(node_id=first_exec_id, field=second_preparation_input.source.field),
                        destination=EdgeConnection(node_id="", field=second_preparation_input.destination.field),
                    )
                ],
            )

    if not collection:
        state._record_empty_iterate_stream(first_iterate_id, outer_path)

    second_preparation_exec_ids = [
        _prepared_at_path(state, second_preparation_id, (*outer_path, first_index))
        for first_index in range(len(collection))
    ]
    if any(exec_id is None or exec_id not in state.results for exec_id in second_preparation_exec_ids):
        return

    body_exec_ids: list[str] = []
    state._discard_source_executed(second_iterate_id)
    for first_index, second_preparation_exec_id in enumerate(second_preparation_exec_ids):
        assert second_preparation_exec_id is not None
        second_path = (*outer_path, first_index)
        second_collection = getattr(state.results[second_preparation_exec_id], second_input_edge.source.field)
        if not isinstance(second_collection, list):
            raise ValueError("Nested Iterate collection source must produce a list")
        for second_index in range(len(second_collection)):
            inner_path = (*second_path, second_index)
            second_exec_id = _prepared_at_path(state, second_iterate_id, inner_path)
            if second_exec_id is None:
                second_exec_id = create_copy(
                    second_iterate_id,
                    inner_path,
                    [
                        Edge(
                            source=EdgeConnection(
                                node_id=second_preparation_exec_id, field=second_input_edge.source.field
                            ),
                            destination=EdgeConnection(node_id="", field=second_input_edge.destination.field),
                        )
                    ],
                    iteration_index=second_index,
                )
            body_exec_id = _prepared_at_path(state, body_node_id, inner_path)
            if body_exec_id is None:
                body_exec_id = create_copy(
                    body_node_id,
                    inner_path,
                    [
                        Edge(
                            source=EdgeConnection(node_id=second_exec_id, field=body_input.source.field),
                            destination=EdgeConnection(node_id="", field=body_input.destination.field),
                        )
                    ],
                )
            body_exec_ids.append(body_exec_id)
        if not second_collection:
            state._record_empty_iterate_stream(second_iterate_id, second_path)

    if _prepared_at_path(state, source_collect_id, outer_path) is not None:
        return
    state._discard_source_executed(source_collect_id)
    collect_node = _create_direct_execution_node_copy(state, source_collect_id, iteration_path=outer_path)
    attached_collect_edges = _attach_direct_execution_edges(
        state,
        collect_node.id,
        [
            Edge(
                source=EdgeConnection(node_id=body_exec_id, field=collect_item_edge.source.field),
                destination=EdgeConnection(node_id="", field=collect_item_edge.destination.field),
            )
            for body_exec_id in body_exec_ids
        ],
    )
    _initialize_direct_execution_node(state, collect_node.id, attached_collect_edges)

    state._discard_source_executed(source_return_id)
    return_node = _create_direct_execution_node_copy(state, source_return_id, iteration_path=outer_path)
    return_input_edges = state.graph._get_input_edges(source_return_id, "output")
    _initialize_direct_execution_node(
        state,
        return_node.id,
        _attach_direct_execution_edges(
            state,
            return_node.id,
            [
                Edge(
                    source=EdgeConnection(node_id=collect_node.id, field=COLLECTION_FIELD),
                    destination=EdgeConnection(node_id="", field=edge.destination.field),
                )
                for edge in return_input_edges
            ],
        ),
    )


def prepare_nested_iterate_bodies(state: "GraphExecutionState") -> None:
    """Prepare the exact nested stream body for every outer For execution path."""
    if not can_use_nested_iterate_planner(state):
        return
    exact = _get_exact_nested_iterate_body(state)
    assert exact is not None
    source_for_id, _ = exact
    for prepared_for_id in tuple(state._prepared_registry().get_prepared_ids(source_for_id)):
        prepared_for = state.execution_graph.get_node(prepared_for_id)
        if isinstance(prepared_for, ForInvocation) and prepared_for.index >= 0:
            _prepare_nested_iterate_body_for_outer(state, source_for_id, prepared_for_id)


def _prepare_nested_iterate_sequence(state: "GraphExecutionState", *, iterate_count: int) -> None:
    nested = state.graph._get_supported_nested_iterate_sequence(
        state._get_source_graph_flat(), iterate_count=iterate_count
    )
    assert nested is not None

    def prepared_at_path(source_node_id: str, path: tuple[int, ...]) -> str | None:
        return state._prepared_registry().get_prepared_id_at_path(source_node_id, path)

    def create_copy(
        source_node_id: str,
        path: tuple[int, ...],
        input_edges: list[Edge],
        iteration_index: int = -1,
    ) -> str:
        node = _create_direct_execution_node_copy(
            state, source_node_id, iteration_index=iteration_index, iteration_path=path
        )
        attached_edges = _attach_direct_execution_edges(state, node.id, input_edges)
        _initialize_direct_execution_node(state, node.id, attached_edges)
        return node.id

    iterate_edges = [
        state.graph._get_input_edges(iterate_id, COLLECTION_FIELD)[0] for iterate_id in nested.iterate_node_ids
    ]
    preparation_inputs = [
        state.graph._get_input_edges(preparation_id)[0] for preparation_id in nested.preparation_node_ids
    ]
    body_input = state.graph._get_input_edges(nested.body_node_id)[0]
    source_exec_id = prepared_at_path(nested.source_node_id, ())
    if source_exec_id is None:
        source_exec_id = create_copy(nested.source_node_id, (), [])
    if source_exec_id not in state.results:
        return

    source_collection = getattr(state.results[source_exec_id], iterate_edges[0].source.field)
    if not isinstance(source_collection, list):
        raise ValueError("Nested Iterate collection source must produce a list")

    all_preparations_ready = True

    def expand(level: int, collection: list[object], parent_exec_id: str, parent_path: tuple[int, ...]) -> None:
        nonlocal all_preparations_ready
        iterate_id = nested.iterate_node_ids[level]
        iterate_input = iterate_edges[level]
        if not collection:
            state._discard_source_executed(iterate_id)
            state._record_empty_iterate_stream(iterate_id, parent_path)
            return
        for index in range(len(collection)):
            path = (*parent_path, index)
            iterate_exec_id = prepared_at_path(iterate_id, path)
            if iterate_exec_id is None:
                iterate_exec_id = create_copy(
                    iterate_id,
                    path,
                    [
                        Edge(
                            source=EdgeConnection(node_id=parent_exec_id, field=iterate_input.source.field),
                            destination=EdgeConnection(node_id="", field=iterate_input.destination.field),
                        )
                    ],
                    iteration_index=index,
                )
            if level == iterate_count - 1:
                if prepared_at_path(nested.body_node_id, path) is None:
                    create_copy(
                        nested.body_node_id,
                        path,
                        [
                            Edge(
                                source=EdgeConnection(node_id=iterate_exec_id, field=body_input.source.field),
                                destination=EdgeConnection(node_id="", field=body_input.destination.field),
                            )
                        ],
                    )
                continue
            preparation_id = nested.preparation_node_ids[level]
            preparation_exec_id = prepared_at_path(preparation_id, path)
            if preparation_exec_id is None:
                preparation_exec_id = create_copy(
                    preparation_id,
                    path,
                    [
                        Edge(
                            source=EdgeConnection(
                                node_id=iterate_exec_id, field=preparation_inputs[level].source.field
                            ),
                            destination=EdgeConnection(node_id="", field=preparation_inputs[level].destination.field),
                        )
                    ],
                )
            if preparation_exec_id not in state.results:
                all_preparations_ready = False
                continue
            next_collection = getattr(state.results[preparation_exec_id], iterate_edges[level + 1].source.field)
            if not isinstance(next_collection, list):
                raise ValueError("Nested Iterate collection source must produce a list")
            expand(level + 1, next_collection, preparation_exec_id, path)

    expand(0, source_collection, source_exec_id, ())
    if all_preparations_ready:
        source_node_ids = [nested.source_node_id]
        for index, iterate_id in enumerate(nested.iterate_node_ids):
            source_node_ids.append(iterate_id)
            if index < len(nested.preparation_node_ids):
                source_node_ids.append(nested.preparation_node_ids[index])
        source_node_ids.append(nested.body_node_id)
        for source_node_id in source_node_ids:
            prepared_ids = state._prepared_registry().get_prepared_ids(source_node_id)
            if (
                not prepared_ids or all(exec_id in state.executed for exec_id in prepared_ids)
            ) and source_node_id not in state.executed:
                state._mark_source_executed(source_node_id)


def prepare_nested_iterate_sequence(state: "GraphExecutionState", *, depth: int | None = None) -> None:
    """Prepare the exact supported nested Iterate chain."""
    if depth is None:
        depth = get_nested_iterate_sequence_depth(state)
    if depth is not None:
        _prepare_nested_iterate_sequence(state, iterate_count=depth)
