"""Runtime-only helpers for graph execution state."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.invocations.call_saved_workflow import (
    CallSavedWorkflowInvocation,
    is_call_saved_workflow_dynamic_input,
)
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.services.shared.graph_models import Edge
from invokeai.app.services.shared.graph_validation import (
    COLLECTION_FIELD,
    ITEM_FIELD,
    CollectInvocation,
    ForInvocation,
    IterateInvocation,
    copydeep,
    nx,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


@dataclass(frozen=True)
class _DirectIterateCollectFanIn:
    branches: tuple[tuple[str, str], ...]
    collector_id: str


@dataclass(frozen=True)
class _BodyIterateCollectFanIn:
    branches: tuple[tuple[str, str, str], ...]
    collector_id: str


class _ExecutionRuntime:
    """Provides runtime-only helpers such as iteration-path lookup and input hydration."""

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state

    def _get_cached_iteration_path(self, exec_node_id: str) -> Optional[tuple[int, ...]]:
        return self._state._prepared_registry().get_iteration_path(exec_node_id)

    def _get_iteration_source_node_id(self, exec_node_id: str) -> Optional[str]:
        if exec_node_id not in self._state.prepared_source_mapping:
            return None
        return self._state._prepared_registry().get_source_node_id(exec_node_id)

    def _get_ordered_iterator_sources(self, source_node_id: str) -> list[str]:
        iterator_graph = self._state._iterator_graph(self._state.graph.nx_graph())
        iterator_sources = [
            node_id
            for node_id in nx.ancestors(iterator_graph, source_node_id)
            if isinstance(self._state.graph.get_node(node_id), (ForInvocation, IterateInvocation))
        ]

        topo = list(nx.topological_sort(iterator_graph))
        topo_index = {node_id: i for i, node_id in enumerate(topo)}
        iterator_sources.sort(key=lambda node_id: topo_index.get(node_id, 0))
        return iterator_sources

    def _get_iterator_exec_id(
        self, iterator_source_id: str, exec_node_id: str, execution_graph: "nx.DiGraph"
    ) -> Optional[str]:
        prepared = self._state.source_prepared_mapping.get(iterator_source_id)
        if not prepared:
            return None
        return next((pid for pid in prepared if nx.has_path(execution_graph, pid, exec_node_id)), None)

    def _build_iteration_path(self, exec_node_id: str, source_node_id: str) -> tuple[int, ...]:
        iterator_sources = self._get_ordered_iterator_sources(source_node_id)
        execution_graph = self._state.execution_graph.nx_graph()
        path: list[int] = []
        for iterator_source_id in iterator_sources:
            iterator_exec_id = self._get_iterator_exec_id(iterator_source_id, exec_node_id, execution_graph)
            if iterator_exec_id is None:
                continue
            iterator_node = self._state.execution_graph.nodes.get(iterator_exec_id)
            if isinstance(iterator_node, (ForInvocation, IterateInvocation)):
                path.append(iterator_node.index)

        node_obj = self._state.execution_graph.nodes.get(exec_node_id)
        if isinstance(node_obj, (ForInvocation, IterateInvocation)):
            path.append(node_obj.index)

        return tuple(path)

    def _cache_iteration_path(self, exec_node_id: str, iteration_path: tuple[int, ...]) -> tuple[int, ...]:
        self._state._prepared_registry().set_iteration_path(exec_node_id, iteration_path)
        return iteration_path

    def get_iteration_path(self, exec_node_id: str) -> tuple[int, ...]:
        """Best-effort outer->inner iteration indices for an execution node, stopping at collectors."""
        cached = self._get_cached_iteration_path(exec_node_id)
        if cached is not None:
            return cached

        source_node_id = self._get_iteration_source_node_id(exec_node_id)
        if source_node_id is None:
            return self._cache_iteration_path(exec_node_id, ())

        return self._cache_iteration_path(exec_node_id, self._build_iteration_path(exec_node_id, source_node_id))

    def _sort_collect_input_edges(self, input_edges: list[Edge], field_name: str) -> list[Edge]:
        matching_edges = [edge for edge in input_edges if edge.destination.field == field_name]
        fan_in = self._state._get_direct_iterate_collect_nodes()
        if isinstance(fan_in, _DirectIterateCollectFanIn):
            matching_edges.sort(
                key=lambda edge: (
                    self._state.prepared_source_mapping.get(edge.source.node_id, edge.source.node_id),
                    self.get_iteration_path(edge.source.node_id),
                )
            )
        elif isinstance(fan_in, _BodyIterateCollectFanIn):
            body_source_ids = {body_id: source_id for source_id, _iterate_id, body_id in fan_in.branches}
            matching_edges.sort(
                key=lambda edge: (
                    body_source_ids.get(
                        self._state.prepared_source_mapping.get(edge.source.node_id, edge.source.node_id),
                        self._state.prepared_source_mapping.get(edge.source.node_id, edge.source.node_id),
                    ),
                    self.get_iteration_path(edge.source.node_id),
                )
            )
        else:
            matching_edges.sort(key=lambda edge: (self.get_iteration_path(edge.source.node_id), edge.source.node_id))
        return matching_edges

    def _get_copied_result_value(self, edge: Edge) -> Any:
        return copydeep(getattr(self._state.results[edge.source.node_id], edge.source.field))

    def _try_get_copied_result_value(self, edge: Edge) -> tuple[bool, Any]:
        source_output = self._state.results.get(edge.source.node_id)
        if source_output is None:
            return False, None
        return True, copydeep(getattr(source_output, edge.source.field))

    def _build_collect_collection(self, input_edges: list[Edge]) -> list[Any]:
        item_edges = self._sort_collect_input_edges(input_edges, ITEM_FIELD)
        collection_edges = self._sort_collect_input_edges(input_edges, COLLECTION_FIELD)

        output_collection = []
        for edge in collection_edges:
            source_value = self._get_copied_result_value(edge)
            if isinstance(source_value, list):
                output_collection.extend(source_value)
            else:
                output_collection.append(source_value)
        fan_in = self._state._get_direct_iterate_collect_nodes()
        stable_direct_fan_in = isinstance(fan_in, _DirectIterateCollectFanIn)
        stable_body_fan_in = isinstance(fan_in, _BodyIterateCollectFanIn)
        body_source_ids = (
            {body_id: source_id for source_id, _iterate_id, body_id in fan_in.branches} if stable_body_fan_in else {}
        )
        item_values: list[tuple[Any, ...]] = []
        consumed_streams: set[str] = set()
        for edge in item_edges:
            source_node_id = self._state.prepared_source_mapping.get(edge.source.node_id, edge.source.node_id)
            item_source_id = (
                source_node_id
                if stable_direct_fan_in
                else body_source_ids.get(source_node_id, "")
                if stable_body_fan_in
                else ""
            )
            item_exec_id = "" if stable_direct_fan_in or stable_body_fan_in else edge.source.node_id
            stream_info = self._state._stream_for_iterate_edge(edge)
            if stream_info is None:
                item_values.append(
                    (
                        item_source_id,
                        (*self.get_iteration_path(edge.source.node_id), 0),
                        item_exec_id,
                        self._get_copied_result_value(edge),
                    )
                )
                continue
            stream_id, parent_path = stream_info
            stream = self._state._generic_runtime().streams.get(stream_id)
            if stream is None:
                item_values.append(
                    (
                        item_source_id,
                        (*self.get_iteration_path(edge.source.node_id), 0),
                        item_exec_id,
                        self._get_copied_result_value(edge),
                    )
                )
                continue
            if not stream.closed:
                raise RuntimeError(f"Cannot hydrate Collect from open Iterate stream {stream_id}")
            if stream_id in consumed_streams:
                continue
            consumed_streams.add(stream_id)
            item_values.extend(
                (item_source_id, (*parent_path, sequence), item_exec_id, copydeep(value))
                for sequence, value in enumerate(stream.values)
            )
        if stable_direct_fan_in or stable_body_fan_in:
            item_values.sort(key=lambda item: (item[0], item[1]))
        else:
            item_values.sort(key=lambda item: (item[1], item[2]))
        output_collection.extend(value for _source_id, _path, _exec_id, value in item_values)
        return output_collection

    def _set_node_inputs(
        self, node: BaseInvocation, input_edges: list[Edge], allowed_fields: Optional[set[str]] = None
    ) -> None:
        for edge in input_edges:
            if allowed_fields is not None and edge.destination.field not in allowed_fields:
                continue
            if isinstance(node, CallSavedWorkflowInvocation) and is_call_saved_workflow_dynamic_input(
                edge.destination.field
            ):
                continue
            setattr(node, edge.destination.field, self._get_copied_result_value(edge))

    def _prepare_collect_inputs(self, node: "CollectInvocation", input_edges: list[Edge]) -> None:
        node.collection = self._build_collect_collection(input_edges)

    def _prepare_iterate_inputs(self, node: "IterateInvocation", input_edges: list[Edge]) -> None:
        for edge in input_edges:
            if edge.destination.field != COLLECTION_FIELD:
                continue
            source_output = self._state.results[edge.source.node_id]
            object.__setattr__(node, COLLECTION_FIELD, getattr(source_output, edge.source.field))
            return

    def _prepare_if_inputs(self, node: IfInvocation, input_edges: list[Edge]) -> None:
        selected_field = self._state._resolved_if_exec_branches.get(node.id)
        allowed_fields = {"condition", selected_field} if selected_field is not None else {"condition"}

        for edge in input_edges:
            if edge.destination.field not in allowed_fields:
                continue

            found_value, copied_value = self._try_get_copied_result_value(edge)
            if not found_value:
                iteration_path = self._state._get_iteration_path(node.id)
                raise RuntimeError(
                    "IfInvocation selected input edge points at an exec node with no stored result output: "
                    f"if_exec_id={node.id}, source_exec_id={edge.source.node_id}, iteration_path={iteration_path}"
                )

            setattr(node, edge.destination.field, copied_value)

    def _prepare_default_inputs(self, node: BaseInvocation, input_edges: list[Edge]) -> None:
        self._set_node_inputs(node, input_edges)

    def prepare_inputs(self, node: BaseInvocation) -> None:
        input_edges = self._state.execution_graph._get_input_edges(node.id)

        if isinstance(node, IterateInvocation):
            self._prepare_iterate_inputs(node, input_edges)
            return

        if isinstance(node, CollectInvocation):
            self._prepare_collect_inputs(node, input_edges)
            return

        if isinstance(node, IfInvocation):
            self._prepare_if_inputs(node, input_edges)
            return

        self._prepare_default_inputs(node, input_edges)
