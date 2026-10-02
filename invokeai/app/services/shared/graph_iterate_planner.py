"""Direct Iterate/Collect planner helpers for graph execution state."""

from typing import TYPE_CHECKING, Any, Iterable, Optional

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.services.shared.graph_execution_runtime import (
    _BodyIterateCollectFanIn,
    _DirectIterateCollectFanIn,
)
from invokeai.app.services.shared.graph_models import Edge, EdgeConnection
from invokeai.app.services.shared.graph_runtime_records import _ApplyTransaction
from invokeai.app.services.shared.graph_validation import (
    COLLECTION_FIELD,
    ITEM_FIELD,
    CollectInvocation,
    IterateInvocation,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


def _get_direct_iterate_collect_fan_in(
    state: "GraphExecutionState", iterator_ids: list[str], collector_id: str
) -> Optional[_DirectIterateCollectFanIn]:
    branch_count = len(iterator_ids)
    if branch_count not in {2, 3} or len(state.graph.nodes) != 2 * branch_count + 1:
        return None
    if len(state.graph.edges) != 2 * branch_count:
        return None

    item_edges = state.graph._get_input_edges(collector_id, ITEM_FIELD)
    if len(item_edges) != branch_count or state.graph._get_input_edges(collector_id, COLLECTION_FIELD):
        return None
    if {edge.source.node_id for edge in item_edges} != set(iterator_ids) or any(
        edge.destination.field != ITEM_FIELD or edge.source.field != ITEM_FIELD for edge in item_edges
    ):
        return None

    branches: list[tuple[str, str]] = []
    for iterator_id in iterator_ids:
        collection_edges = state.graph._get_input_edges(iterator_id, COLLECTION_FIELD)
        if len(collection_edges) != 1:
            return None
        collection_edge = collection_edges[0]
        if collection_edge.destination.node_id != iterator_id or collection_edge.source.field != COLLECTION_FIELD:
            return None

        source_id = collection_edge.source.node_id
        source = state.graph.get_node(source_id)
        if isinstance(
            source,
            (
                CallSavedWorkflowInvocation,
                ForInvocation,
                ForReturnInvocation,
                IfInvocation,
                IterateInvocation,
                CollectInvocation,
            ),
        ):
            return None
        if state.graph._get_input_edges(source_id) or state.graph._get_output_edges(source_id) != [collection_edge]:
            return None

        iterator_item_edges = state.graph._get_output_edges(iterator_id, ITEM_FIELD)
        if len(iterator_item_edges) != 1 or iterator_item_edges[0] not in item_edges:
            return None
        if state.graph._get_output_edges(iterator_id) != iterator_item_edges:
            return None
        branches.append((source_id, iterator_id))

    branches.sort(key=lambda branch: branch[0])
    return _DirectIterateCollectFanIn(branches=tuple(branches), collector_id=collector_id)


def _get_body_iterate_collect_fan_in(
    state: "GraphExecutionState", iterator_ids: list[str], collector_id: str
) -> Optional[_BodyIterateCollectFanIn]:
    if len(iterator_ids) != 2 or len(state.graph.nodes) != 7 or len(state.graph.edges) != 6:
        return None

    item_edges = state.graph._get_input_edges(collector_id, ITEM_FIELD)
    if len(item_edges) != 2 or state.graph._get_input_edges(collector_id, COLLECTION_FIELD):
        return None
    if any(edge.destination.field != ITEM_FIELD for edge in item_edges):
        return None

    item_edges_by_body = {edge.source.node_id: edge for edge in item_edges}
    branches: list[tuple[str, str, str]] = []
    for iterator_id in iterator_ids:
        collection_edges = state.graph._get_input_edges(iterator_id, COLLECTION_FIELD)
        if len(collection_edges) != 1:
            return None
        collection_edge = collection_edges[0]
        if collection_edge.destination.node_id != iterator_id or collection_edge.source.field != COLLECTION_FIELD:
            return None

        source_id = collection_edge.source.node_id
        source = state.graph.get_node(source_id)
        if isinstance(
            source,
            (
                CallSavedWorkflowInvocation,
                ForInvocation,
                ForReturnInvocation,
                IfInvocation,
                IterateInvocation,
                CollectInvocation,
            ),
        ):
            return None
        if state.graph._get_input_edges(source_id) or state.graph._get_output_edges(source_id) != [collection_edge]:
            return None

        iterator_item_edges = state.graph._get_output_edges(iterator_id, ITEM_FIELD)
        if len(iterator_item_edges) != 1 or iterator_item_edges[0].source.node_id != iterator_id:
            return None
        if state.graph._get_output_edges(iterator_id) != iterator_item_edges:
            return None

        body_edge = iterator_item_edges[0]
        if body_edge.source.field != ITEM_FIELD:
            return None
        body_id = body_edge.destination.node_id
        if body_id in {source_id, iterator_id, collector_id} or body_id not in item_edges_by_body:
            return None
        body = state.graph.get_node(body_id)
        if isinstance(
            body,
            (
                CallSavedWorkflowInvocation,
                ForInvocation,
                ForReturnInvocation,
                IfInvocation,
                IterateInvocation,
                CollectInvocation,
            ),
        ):
            return None
        body_input_edges = state.graph._get_input_edges(body_id)
        if len(body_input_edges) != 1 or body_input_edges[0] != body_edge:
            return None

        collect_item_edge = item_edges_by_body[body_id]
        if state.graph._get_output_edges(body_id) != [collect_item_edge]:
            return None
        branches.append((source_id, iterator_id, body_id))

    if len({source_id for source_id, _iterator_id, _body_id in branches}) != 2:
        return None
    if len({iterator_id for _source_id, iterator_id, _body_id in branches}) != 2:
        return None
    if len({body_id for _source_id, _iterator_id, body_id in branches}) != 2:
        return None

    branches.sort(key=lambda branch: branch[0])
    return _BodyIterateCollectFanIn(branches=tuple(branches), collector_id=collector_id)


def _get_direct_iterate_collect_nodes(
    state: "GraphExecutionState",
) -> Optional[tuple[str, str, str, str, tuple[str, ...]] | _DirectIterateCollectFanIn | _BodyIterateCollectFanIn]:
    """Return bounded fresh stream-planner nodes, including one input-driven source chain."""

    if (
        state._legacy_snapshot_loaded
        or len(state.graph.nodes) < 4
        or len(state.graph.edges) != len(state.graph.nodes) - 1
    ):
        return None

    iterator_ids = [node_id for node_id, node in state.graph.nodes.items() if isinstance(node, IterateInvocation)]
    collector_ids = [node_id for node_id, node in state.graph.nodes.items() if isinstance(node, CollectInvocation)]
    if len(iterator_ids) in {2, 3} and len(collector_ids) == 1:
        if len(iterator_ids) == 2:
            body_fan_in = state._get_body_iterate_collect_fan_in(iterator_ids, collector_ids[0])
            if body_fan_in is not None:
                return body_fan_in
        return state._get_direct_iterate_collect_fan_in(iterator_ids, collector_ids[0])
    if len(iterator_ids) != 1 or len(collector_ids) != 1:
        return None
    iterator_id = iterator_ids[0]
    collector_id = collector_ids[0]

    collection_edges = state.graph._get_input_edges(iterator_id, COLLECTION_FIELD)
    item_edges = state.graph._get_input_edges(collector_id, ITEM_FIELD)
    if len(collection_edges) != 1 or len(item_edges) != 1:
        return None
    collection_edge = collection_edges[0]
    item_edge = item_edges[0]
    if collection_edge.source.field != COLLECTION_FIELD or item_edge.destination.field != ITEM_FIELD:
        return None
    if collection_edge.destination.node_id != iterator_id:
        return None

    body_id = item_edge.source.node_id
    if body_id in {iterator_id, collector_id}:
        return None
    body = state.graph.get_node(body_id)
    if isinstance(body, (ForInvocation, ForReturnInvocation, IfInvocation, IterateInvocation, CollectInvocation)):
        return None
    body_input_edges = state.graph._get_input_edges(body_id)
    if len(body_input_edges) != 1 or body_input_edges[0] != Edge(
        source=EdgeConnection(node_id=iterator_id, field=ITEM_FIELD),
        destination=EdgeConnection(node_id=body_id, field=body_input_edges[0].destination.field),
    ):
        return None
    if body_input_edges[0].source.field != ITEM_FIELD:
        return None

    source_id = collection_edge.source.node_id
    if source_id in {iterator_id, body_id, collector_id}:
        return None
    source = state.graph.get_node(source_id)
    if isinstance(source, (ForInvocation, ForReturnInvocation, IfInvocation, IterateInvocation, CollectInvocation)):
        return None
    source_input_edges = state.graph._get_input_edges(source_id)
    if len(source_input_edges) > 1:
        return None
    for source_input_edge in source_input_edges:
        upstream_id = source_input_edge.source.node_id
        if upstream_id in {source_id, iterator_id, body_id, collector_id}:
            return None
        upstream = state.graph.get_node(upstream_id)
        if isinstance(
            upstream, (ForInvocation, ForReturnInvocation, IfInvocation, IterateInvocation, CollectInvocation)
        ):
            return None
        if state.graph._get_input_edges(upstream_id) or state.graph._get_output_edges(upstream_id) != [
            source_input_edge
        ]:
            return None
    if state.graph._get_output_edges(source_id) != [collection_edge]:
        return None
    if state.graph._get_output_edges(iterator_id) != [body_input_edges[0]]:
        return None
    if state.graph._get_output_edges(body_id) != [item_edge]:
        return None
    if state.graph._get_input_edges(collector_id) != [item_edge]:
        return None

    downstream_edges = state.graph._get_output_edges(collector_id, COLLECTION_FIELD)
    upstream_count = len(source_input_edges)
    if len(downstream_edges) != len(state.graph.nodes) - 4 - upstream_count:
        return None
    downstream_ids: list[str] = []
    for downstream_edge in downstream_edges:
        downstream_id = downstream_edge.destination.node_id
        downstream = state.graph.get_node(downstream_id)
        if downstream_id in {source_id, iterator_id, body_id, collector_id} or isinstance(
            downstream, (ForInvocation, ForReturnInvocation, IfInvocation, IterateInvocation, CollectInvocation)
        ):
            return None
        if state.graph._get_input_edges(downstream_id) != [downstream_edge]:
            return None
        downstream_ids.append(downstream_id)

    return source_id, iterator_id, body_id, collector_id, tuple(downstream_ids)


def _can_use_direct_iterate_collect_planner(state: "GraphExecutionState") -> bool:
    """Use fresh planner ownership only for the bounded direct stream shapes."""

    return state._get_direct_iterate_collect_nodes() is not None


def _create_direct_execution_node_copy(
    state: "GraphExecutionState", source_node_id: str, iteration_index: int = -1, iteration_path: tuple[int, ...] = ()
) -> BaseInvocation:
    source_node = state.graph.get_node(source_node_id)
    new_node = source_node.model_copy(deep=True)
    new_node.id = state._new_execution_node_id()
    if isinstance(new_node, IterateInvocation):
        new_node.index = iteration_index
    if iteration_index >= 0 or isinstance(new_node, CollectInvocation):
        new_node.use_cache = False
    state._tx_add_execution_node(new_node)
    state._add_execution_graph_node(new_node.id)
    state._register_prepared_exec_node(new_node.id, source_node_id)
    state._prepared_registry().set_iteration_path(new_node.id, iteration_path)
    return new_node


def _attach_direct_execution_edges(
    state: "GraphExecutionState", exec_node_id: str, edges: Iterable[Edge]
) -> list[Edge]:
    attached_edges = [
        Edge(
            source=edge.source,
            destination=EdgeConnection(node_id=exec_node_id, field=edge.destination.field),
        )
        for edge in edges
    ]
    state._tx_add_execution_edges(attached_edges)
    state._add_execution_graph_edges(attached_edges)
    return attached_edges


def _initialize_direct_execution_node(
    state: "GraphExecutionState", exec_node_id: str, input_edges: Iterable[Edge], *, project: bool = True
) -> None:
    input_edges = list(input_edges)
    state._tx_set_mapping(
        state.indegree,
        exec_node_id,
        sum(1 for edge in input_edges if edge.source.node_id not in state.executed),
    )
    scheduler = state._scheduler()
    assert state._is_generic_graph_scheduler(scheduler)
    scheduler.register_node(exec_node_id, project=project)
    if state.indegree[exec_node_id] == 0 and exec_node_id not in state.executed:
        if project:
            state._enqueue_if_ready(exec_node_id)
        else:
            scheduler._scheduler.enqueue(exec_node_id)


def _mark_direct_source_empty(state: "GraphExecutionState", source_node_id: str) -> None:
    """Record an empty direct iterator source without entering the legacy materializer."""

    state._tx_set_mapping(state.source_prepared_mapping, source_node_id, set())
    state._reset_unexecuted_prepared(source_node_id)
    state._mark_source_executed(source_node_id)
    if isinstance(state.graph.get_node(source_node_id), IterateInvocation):
        state._record_empty_iterate_stream(source_node_id)


def _prepare_direct_iterate_collect_fan_in_unchecked(
    state: "GraphExecutionState", fan_in: _DirectIterateCollectFanIn
) -> None:
    source_exec_ids: dict[str, str] = {}
    for source_id, _iterator_id in fan_in.branches:
        if source_id not in state.source_prepared_mapping:
            source_node = state._create_direct_execution_node_copy(source_id)
            state._initialize_direct_execution_node(source_node.id, ())
        if source_id in state.executed:
            source_exec_ids[source_id] = sorted(state.source_prepared_mapping[source_id])[0]

    if len(source_exec_ids) != len(fan_in.branches):
        return
    if fan_in.collector_id in state.source_prepared_mapping:
        return

    iterator_exec_ids: list[tuple[str, str]] = []
    collector_item_edges = {
        edge.source.node_id: edge for edge in state.graph._get_input_edges(fan_in.collector_id, ITEM_FIELD)
    }
    for source_id, iterator_id in fan_in.branches:
        collection_edge = state.graph._get_input_edges(iterator_id, COLLECTION_FIELD)[0]
        collection = getattr(state.results[source_exec_ids[source_id]], collection_edge.source.field)
        if not isinstance(collection, list):
            raise ValueError("Direct Iterate collection source must produce a list")

        for index in range(len(collection)):
            iterator_node = state._create_direct_execution_node_copy(iterator_id, index, (index,))
            attached_iterator_edges = state._attach_direct_execution_edges(
                iterator_node.id,
                [
                    Edge(
                        source=EdgeConnection(node_id=source_exec_ids[source_id], field=collection_edge.source.field),
                        destination=EdgeConnection(node_id="", field=collection_edge.destination.field),
                    )
                ],
            )
            state._initialize_direct_execution_node(iterator_node.id, attached_iterator_edges)
            iterator_exec_ids.append((iterator_node.id, iterator_id))

        if not collection:
            state._mark_direct_source_empty(iterator_id)

    collector_node = state._create_direct_execution_node_copy(fan_in.collector_id, iteration_path=())
    collector_edges = [
        Edge(
            source=EdgeConnection(node_id=iterator_exec_id, field=ITEM_FIELD),
            destination=EdgeConnection(node_id="", field=collector_item_edges[iterator_id].destination.field),
        )
        for iterator_exec_id, iterator_id in iterator_exec_ids
    ]
    attached_collector_edges = state._attach_direct_execution_edges(collector_node.id, collector_edges)
    state._initialize_direct_execution_node(collector_node.id, attached_collector_edges)


def _prepare_body_iterate_collect_fan_in_unchecked(
    state: "GraphExecutionState", fan_in: _BodyIterateCollectFanIn
) -> None:
    source_exec_ids: dict[str, str] = {}
    for source_id, _iterator_id, _body_id in fan_in.branches:
        if source_id not in state.source_prepared_mapping:
            source_node = state._create_direct_execution_node_copy(source_id)
            state._initialize_direct_execution_node(source_node.id, ())
        if source_id in state.executed:
            source_exec_ids[source_id] = sorted(state.source_prepared_mapping[source_id])[0]

    if len(source_exec_ids) != len(fan_in.branches):
        return
    if fan_in.collector_id in state.source_prepared_mapping:
        return

    collections: dict[str, list[Any]] = {}
    for source_id, iterator_id, _body_id in fan_in.branches:
        collection_edge = state.graph._get_input_edges(iterator_id, COLLECTION_FIELD)[0]
        collection = getattr(state.results[source_exec_ids[source_id]], collection_edge.source.field)
        if not isinstance(collection, list):
            raise ValueError("Body Iterate collection source must produce a list")
        collections[source_id] = collection

    collector_item_edges = {
        edge.source.node_id: edge for edge in state.graph._get_input_edges(fan_in.collector_id, ITEM_FIELD)
    }
    for source_id, iterator_id, body_id in fan_in.branches:
        collection_edge = state.graph._get_input_edges(iterator_id, COLLECTION_FIELD)[0]
        body_input_edge = state.graph._get_input_edges(body_id)[0]
        collect_item_edge = collector_item_edges[body_id]
        source_exec_id = source_exec_ids[source_id]
        collection = collections[source_id]
        for index in range(len(collection)):
            iterator_node = state._create_direct_execution_node_copy(iterator_id, index, (index,))
            attached_iterator_edges = state._attach_direct_execution_edges(
                iterator_node.id,
                [
                    Edge(
                        source=EdgeConnection(node_id=source_exec_id, field=collection_edge.source.field),
                        destination=EdgeConnection(node_id="", field=collection_edge.destination.field),
                    )
                ],
            )
            state._initialize_direct_execution_node(iterator_node.id, attached_iterator_edges)

            body_node = state._create_direct_execution_node_copy(body_id, iteration_path=(index,))
            attached_body_edges = state._attach_direct_execution_edges(
                body_node.id,
                [
                    Edge(
                        source=EdgeConnection(node_id=iterator_node.id, field=body_input_edge.source.field),
                        destination=EdgeConnection(node_id="", field=body_input_edge.destination.field),
                    )
                ],
            )
            state._initialize_direct_execution_node(body_node.id, attached_body_edges)

        if not collection:
            state._mark_direct_source_empty(iterator_id)
            state._mark_direct_source_empty(body_id)

    collector_node = state._create_direct_execution_node_copy(fan_in.collector_id, iteration_path=())
    collector_edges: list[Edge] = []
    for _source_id, _iterator_id, body_id in fan_in.branches:
        collect_item_edge = collector_item_edges[body_id]
        for body_exec_id in sorted(
            state.source_prepared_mapping.get(body_id, set()),
            key=lambda exec_id: state._get_iteration_path(exec_id),
        ):
            collector_edges.append(
                Edge(
                    source=EdgeConnection(node_id=body_exec_id, field=collect_item_edge.source.field),
                    destination=EdgeConnection(node_id="", field=collect_item_edge.destination.field),
                )
            )
    attached_collector_edges = state._attach_direct_execution_edges(collector_node.id, collector_edges)
    state._initialize_direct_execution_node(collector_node.id, attached_collector_edges)


def _prepare_direct_iterate_collect(state: "GraphExecutionState") -> None:
    """Expand the fresh direct stream shape atomically, without the legacy materializer."""

    if state._apply_transaction is not None:
        state._prepare_direct_iterate_collect_unchecked()
        return

    transaction = _ApplyTransaction()
    object.__setattr__(state, "_apply_transaction", transaction)
    try:
        state._prepare_direct_iterate_collect_unchecked()
    except Exception:
        try:
            transaction.rollback()
        finally:
            state._reset_apply_derived_caches()
            state._rehydrate_ready_queues()
        raise
    finally:
        object.__setattr__(state, "_apply_transaction", None)


def _prepare_direct_iterate_collect_unchecked(state: "GraphExecutionState") -> None:
    """Expand the direct stream shape while the caller owns mutation rollback."""

    node_ids = state._get_direct_iterate_collect_nodes()
    if node_ids is None:
        return
    if isinstance(node_ids, _BodyIterateCollectFanIn):
        state._prepare_body_iterate_collect_fan_in_unchecked(node_ids)
        return
    if isinstance(node_ids, _DirectIterateCollectFanIn):
        state._prepare_direct_iterate_collect_fan_in_unchecked(node_ids)
        return
    source_id, iterator_id, body_id, collector_id, downstream_ids = node_ids

    if source_id not in state.source_prepared_mapping:
        source_input_edges = state.graph._get_input_edges(source_id)
        source_input_exec_ids: dict[str, str] = {}
        for source_input_edge in source_input_edges:
            upstream_id = source_input_edge.source.node_id
            if upstream_id not in state.source_prepared_mapping:
                upstream_node = state._create_direct_execution_node_copy(upstream_id)
                state._initialize_direct_execution_node(upstream_node.id, ())
            source_input_exec_ids[upstream_id] = next(iter(state.source_prepared_mapping[upstream_id]))

        source_node = state._create_direct_execution_node_copy(source_id)
        attached_source_edges = state._attach_direct_execution_edges(
            source_node.id,
            [
                Edge(
                    source=EdgeConnection(
                        node_id=source_input_exec_ids[edge.source.node_id],
                        field=edge.source.field,
                    ),
                    destination=EdgeConnection(node_id="", field=edge.destination.field),
                )
                for edge in source_input_edges
            ],
        )
        state._initialize_direct_execution_node(source_node.id, attached_source_edges)
        return

    if collector_id in state.source_prepared_mapping:
        return

    if source_id not in state.executed:
        return
    source_exec_id = next(iter(state.source_prepared_mapping[source_id]))
    source_output = state.results[source_exec_id]
    collection_edge = state.graph._get_input_edges(iterator_id, COLLECTION_FIELD)[0]
    collection = getattr(source_output, collection_edge.source.field)
    if not isinstance(collection, list):
        raise ValueError("Direct Iterate collection source must produce a list")

    iterator_exec_ids: list[str] = []
    body_exec_ids: list[str] = []
    body_input_edge = state.graph._get_input_edges(body_id)[0]
    collect_item_edge = state.graph._get_input_edges(collector_id, ITEM_FIELD)[0]
    for index in range(len(collection)):
        iterator_node = state._create_direct_execution_node_copy(iterator_id, index, (index,))
        iterator_edges = [
            Edge(
                source=EdgeConnection(node_id=source_exec_id, field=collection_edge.source.field),
                destination=EdgeConnection(node_id="", field=collection_edge.destination.field),
            )
        ]
        attached_iterator_edges = state._attach_direct_execution_edges(iterator_node.id, iterator_edges)
        state._initialize_direct_execution_node(iterator_node.id, attached_iterator_edges, project=False)
        iterator_exec_ids.append(iterator_node.id)

        body_node = state._create_direct_execution_node_copy(body_id, iteration_path=(index,))
        body_edges = [
            Edge(
                source=EdgeConnection(node_id=iterator_node.id, field=body_input_edge.source.field),
                destination=EdgeConnection(node_id="", field=body_input_edge.destination.field),
            )
        ]
        attached_body_edges = state._attach_direct_execution_edges(body_node.id, body_edges)
        state._initialize_direct_execution_node(body_node.id, attached_body_edges, project=False)
        body_exec_ids.append(body_node.id)

    if not iterator_exec_ids:
        state._mark_direct_source_empty(iterator_id)
        state._mark_direct_source_empty(body_id)

    collector_node = state._create_direct_execution_node_copy(collector_id, iteration_path=())
    collector_edges = [
        Edge(
            source=EdgeConnection(node_id=body_exec_id, field=collect_item_edge.source.field),
            destination=EdgeConnection(node_id="", field=collect_item_edge.destination.field),
        )
        for body_exec_id in body_exec_ids
    ]
    attached_collector_edges = state._attach_direct_execution_edges(collector_node.id, collector_edges)
    state._initialize_direct_execution_node(collector_node.id, attached_collector_edges, project=False)

    for downstream_id in downstream_ids:
        downstream_node = state._create_direct_execution_node_copy(downstream_id)
        downstream_edge = state.graph._get_input_edges(downstream_id)[0]
        attached_downstream_edges = state._attach_direct_execution_edges(
            downstream_node.id,
            [
                Edge(
                    source=EdgeConnection(node_id=collector_node.id, field=downstream_edge.source.field),
                    destination=EdgeConnection(node_id="", field=downstream_edge.destination.field),
                )
            ],
        )
        state._initialize_direct_execution_node(downstream_node.id, attached_downstream_edges, project=False)

    state._scheduler()._project_ready_nodes()
