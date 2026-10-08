"""Compatibility materialization for graph execution state."""

import itertools
from typing import TYPE_CHECKING, Any, Iterable, Optional

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.invocations.fields import OutputScope
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import (
    LOOP_LINKAGE_FIELD,
    ForInvocation,
    ForInvocationOutput,
    ForReturnInvocation,
    LoopState,
)
from invokeai.app.services.shared import graph_if_runtime
from invokeai.app.services.shared.graph_models import Edge, EdgeConnection
from invokeai.app.services.shared.graph_validation import (
    COLLECTION_FIELD,
    ITEM_FIELD,
    CollectInvocation,
    IterateInvocation,
    _SupportedNestedForBody,
    _SupportedNestedIterateBody,
    _SupportedNestedIterateChain,
    copydeep,
    get_output_field_scope,
    nx,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


class _ExecutionNodeBuilder:
    """Builds concrete execution-graph nodes for a selected runtime planner.

    This class contains low-level copy, edge, input, and iteration mechanics. A
    planner owns whether those mechanics are used for generic or compatibility
    execution.
    """

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state
        self._iteration_axes_by_source: dict[str, tuple[str, ...]] = {}

    def _get_iteration_axes(self, source_node_id: str) -> tuple[str, ...]:
        cached = self._iteration_axes_by_source.get(source_node_id)
        if cached is not None:
            return cached

        axes = self._state._runtime()._get_ordered_iterator_sources(source_node_id)
        if isinstance(self._state.graph.get_node(source_node_id), IterateInvocation):
            axes.append(source_node_id)
        result = tuple(axes)
        self._iteration_axes_by_source[source_node_id] = result
        return result

    def _get_known_iteration_path(
        self,
        iteration_index: int,
        iteration_node_map: list[tuple[str, str]],
        input_edges: Optional[list[Edge]] = None,
    ) -> Optional[tuple[int, ...]]:
        parent_contexts: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
        registry = self._state._prepared_registry()
        input_edges_by_source: dict[str, list[Edge]] = {}
        for edge in input_edges or []:
            input_edges_by_source.setdefault(edge.source.node_id, []).append(edge)
        for source_node_id, prepared_id in iteration_node_map:
            parent_path = registry.get_iteration_path(prepared_id)
            if parent_path is None:
                return None
            source_edges = input_edges_by_source.get(source_node_id)
            if source_edges:
                parent_path = max(
                    (self._get_prepared_edge_iteration_path(edge, prepared_id) for edge in source_edges),
                    key=lambda path: (len(path), path),
                )
            if parent_path:
                parent_contexts.append((parent_path, self._get_iteration_axes(source_node_id)))

        deepest_path = max((path for path, _ in parent_contexts), key=len, default=())
        deepest_axes = max((axes for _, axes in parent_contexts), key=len, default=())
        compatible_parent_contexts = all(
            deepest_path[: len(path)] == path and deepest_axes[: len(axes)] == axes for path, axes in parent_contexts
        )
        if not compatible_parent_contexts:
            return None

        # Materialized iteration boundaries use non-negative indexes; ordinary execution nodes use -1. Keeping this
        # generic allows other scheduler-managed loop nodes to reuse the same path cache.
        if iteration_index >= 0:
            return (*deepest_path, iteration_index)
        return deepest_path

    def _get_iterator_iteration_count(self, node_id: str, iteration_node_map: list[tuple[str, str]]) -> int:
        input_collection_edge = next(iter(self._state.graph._get_input_edges(node_id, COLLECTION_FIELD)))
        input_collection_prepared_node_id = next(
            prepared_id
            for source_id, prepared_id in iteration_node_map
            if source_id == input_collection_edge.source.node_id
        )
        input_collection_output = self._state.results[input_collection_prepared_node_id]
        input_collection = getattr(input_collection_output, input_collection_edge.source.field)
        return len(input_collection)

    def _get_for_iteration_count(self, node_id: str, iteration_node_map: list[tuple[str, str]]) -> int:
        input_collection_edges = self._state.graph._get_input_edges(node_id, COLLECTION_FIELD)
        if len(input_collection_edges) == 0:
            node = self._state.graph.get_node(node_id)
            assert isinstance(node, ForInvocation)
            return len(node.collection)

        input_collection_edge = input_collection_edges[0]
        input_collection_prepared_node_id = next(
            prepared_id
            for source_id, prepared_id in iteration_node_map
            if source_id == input_collection_edge.source.node_id
        )
        input_collection_output = self._state.results[input_collection_prepared_node_id]
        input_collection = getattr(input_collection_output, input_collection_edge.source.field)
        if not isinstance(input_collection, list):
            raise ValueError("For collection input must be a list")
        return len(input_collection)

    def _get_new_node_iterations(
        self, node: BaseInvocation, node_id: str, iteration_node_map: list[tuple[str, str]]
    ) -> list[int]:
        if isinstance(node, IterateInvocation):
            iteration_count = self._get_iterator_iteration_count(node_id, iteration_node_map)
            if iteration_count == 0:
                return []
            return list(range(iteration_count))

        if isinstance(node, ForInvocation):
            iteration_count = self._get_for_iteration_count(node_id, iteration_node_map)
            if iteration_count == 0:
                return []
            return [0]

        return [-1]

    def _build_execution_edges(self, node_id: str, iteration_node_map: list[tuple[str, str]]) -> list[Edge]:
        return self._build_execution_edges_for_fields(node_id, iteration_node_map)

    def _build_execution_edges_for_fields(
        self,
        node_id: str,
        iteration_node_map: list[tuple[str, str]],
        input_fields: Optional[set[str]] = None,
    ) -> list[Edge]:
        input_edges = self._state.graph._get_input_edges(node_id)
        new_edges: list[Edge] = []
        for edge in input_edges:
            if input_fields is not None and edge.destination.field not in input_fields:
                continue
            matching_inputs = [
                prepared_id for source_id, prepared_id in iteration_node_map if source_id == edge.source.node_id
            ]
            for input_node_id in matching_inputs:
                new_edges.append(
                    Edge(
                        source=EdgeConnection(node_id=input_node_id, field=edge.source.field),
                        destination=EdgeConnection(node_id="", field=edge.destination.field),
                    )
                )
        return new_edges

    def _has_unmaterializable_for_final_input(self, node_id: str) -> bool:
        final_for_source_ids = set()
        for edge in self._state.graph._get_input_edges(node_id):
            source_node = self._state.graph.get_node(edge.source.node_id)
            if not isinstance(source_node, ForInvocation):
                continue
            if get_output_field_scope(source_node, edge.source.field) == OutputScope.Final:
                final_for_source_ids.add(edge.source.node_id)

        return any(not self._state._all_for_contexts_finalized(source_for_id) for source_for_id in final_for_source_ids)

    def _create_execution_node_copy(
        self, node: BaseInvocation, node_id: str, iteration_index: int, *, deep_copy: bool = True
    ) -> BaseInvocation:
        new_node = node.model_copy(deep=deep_copy)
        new_node.id = self._state._new_execution_node_id()

        if isinstance(new_node, IterateInvocation):
            new_node.index = iteration_index
        if isinstance(new_node, ForInvocation):
            new_node.index = iteration_index

        # Scheduler-managed iteration boundaries and collectors are cheaper to execute than to hash, especially when
        # their inputs contain large collections. Loop body nodes retain their normal cache behavior.
        if iteration_index >= 0 or isinstance(new_node, CollectInvocation):
            new_node.use_cache = False

        self._state._tx_add_execution_node(new_node)
        self._state._add_execution_graph_node(new_node.id)
        self._state._register_prepared_exec_node(new_node.id, node_id)
        return new_node

    def _create_empty_for_final_output(
        self,
        source_for_id: str,
        node: "ForInvocation",
        iteration_node_map: list[tuple[str, str]],
    ) -> str:
        new_node = self._create_execution_node_copy(node, source_for_id, -1, deep_copy=False)
        assert isinstance(new_node, ForInvocation)
        new_edges = self._build_execution_edges(source_for_id, iteration_node_map)
        iteration_path = self._get_known_iteration_path(
            -1, iteration_node_map, self._state.graph._get_input_edges(source_for_id)
        )
        if iteration_path is not None:
            self._state._prepared_registry().set_iteration_path(new_node.id, iteration_path)
        self._attach_execution_edges(new_node.id, new_edges)
        self._state._runtime().prepare_inputs(new_node)

        initial_state = copydeep(new_node.state or LoopState())
        self._state._tx_set_attr(new_node, "collection", [])
        self._state._tx_set_attr(new_node, "state", initial_state)

        self._state._tx_set_mapping(
            self._state.results,
            new_node.id,
            ForInvocationOutput(
                loop_linkage=LOOP_LINKAGE_FIELD,
                item=None,
                index=-1,
                total=0,
                state=initial_state,
                output_collection=[],
                final_state=initial_state,
            ),
        )
        self._state._mark_exec_node_executed(new_node.id)
        self._state._set_prepared_exec_state(new_node.id, "executed")

        return new_node.id

    def _mark_empty_for_complete(self, source_for_id: str) -> None:
        prepared_for_ids = [
            prepared_id
            for prepared_id in self._state._prepared_registry().get_prepared_ids(source_for_id)
            if isinstance(self._state.execution_graph.get_node(prepared_id), ForInvocation)
            and self._state.execution_graph.get_node(prepared_id).index == -1
        ]
        assert prepared_for_ids, f"Empty For '{source_for_id}' did not create a final execution node"
        for prepared_for_id in prepared_for_ids:
            self._state._mark_loop_context_finalized(source_for_id, prepared_for_id)

        self._state._mark_for_source_complete(source_for_id)

    def create_for_iteration(
        self,
        source_for_id: str,
        iteration_index: int,
        collection: list[Any],
        state: "LoopState",
        iteration_path: tuple[int, ...],
    ) -> str:
        """Prepare a For iteration, taking ownership of the remaining collection.

        The caller clears the completed For node before handing over this list. Bypassing Pydantic assignment
        validation is intentional: validation would rebuild the list and restore quadratic scheduling cost.
        """
        node = self._state.graph.get_node(source_for_id)
        if not isinstance(node, ForInvocation):
            raise TypeError(f"Expected source ForInvocation, got {type(node).__name__}")

        new_node = self._create_execution_node_copy(node, source_for_id, iteration_index, deep_copy=False)
        assert isinstance(new_node, ForInvocation)
        object.__setattr__(new_node, COLLECTION_FIELD, collection)
        self._state._tx_set_attr(new_node, "state", copydeep(state))
        self._state._prepared_registry().set_iteration_path(new_node.id, iteration_path)
        self._initialize_execution_node(new_node.id)
        return new_node.id

    def create_for_body_iteration(self, source_for_id: str, prepared_for_id: str) -> Optional[str]:
        graph = self._state._get_source_graph_flat()
        execution_graph = self._state._get_execution_graph_flat()
        serial_nested_body = self._state.graph._get_supported_for_serial_nested_iterate_chain(source_for_id, graph)
        if serial_nested_body is not None:
            return self._create_serial_nested_iterate_body_iteration(
                source_for_id, prepared_for_id, graph, execution_graph, serial_nested_body
            )
        nested_body = self._state.graph._get_supported_for_nested_iterate_body(source_for_id, graph)
        if nested_body is not None:
            return self._create_nested_iterate_body_iteration(
                source_for_id, prepared_for_id, graph, execution_graph, nested_body
            )
        nested_for_body = self._state.graph._get_supported_for_nested_for_body(source_for_id, graph)
        if nested_for_body is not None:
            return self._create_nested_for_body_iteration(
                source_for_id, prepared_for_id, graph, execution_graph, nested_for_body
            )

        body_path_to_return = self._state.graph._get_for_body_path_to_return(source_for_id, graph)
        if body_path_to_return is None:
            return None

        body_path_nodes, source_return_id = body_path_to_return
        source_to_prepared = {source_for_id: prepared_for_id}
        prepared_return_id: Optional[str] = None

        for source_node_id in nx.topological_sort(graph):
            if source_node_id not in body_path_nodes:
                continue

            node = self._state.graph.get_node(source_node_id)
            new_edges: list[Edge] = []
            for edge in self._state.graph._get_input_edges(source_node_id):
                prepared_source_id = source_to_prepared.get(edge.source.node_id)
                if prepared_source_id is None:
                    prepared_source_id = self.get_iteration_node(
                        edge.source.node_id,
                        graph,
                        execution_graph,
                        [prepared_for_id],
                    )
                if prepared_source_id is None:
                    raise RuntimeError(
                        f"Unable to rematerialize For body input {edge}: no prepared source node is available"
                    )
                new_edges.append(
                    Edge(
                        source=EdgeConnection(node_id=prepared_source_id, field=edge.source.field),
                        destination=EdgeConnection(node_id="", field=edge.destination.field),
                    )
                )

            new_node = self._create_execution_node_copy(node, source_node_id, -1)
            source_to_prepared[source_node_id] = new_node.id
            self._state._prepared_registry().set_iteration_path(
                new_node.id, self._state._get_iteration_path(prepared_for_id)
            )
            self._state._discard_source_executed(source_node_id)
            attached_edges = self._attach_execution_edges(new_node.id, new_edges)
            self._initialize_execution_node(new_node.id, attached_edges)

            if source_node_id == source_return_id:
                prepared_return_id = new_node.id

        return prepared_return_id

    def _is_deferred_nested_for_return(self, node_id: str, graph: "nx.DiGraph") -> bool:
        return any(
            (nested_body := self._state.graph._get_supported_for_nested_for_body(source_for_id, graph)) is not None
            and nested_body.outer_return_id == node_id
            for source_for_id, source_node in self._state.graph.nodes.items()
            if isinstance(source_node, ForInvocation)
        )

    def _get_final_prepared_for_id(self, source_for_id: str, parent_iteration_path: tuple[int, ...]) -> str:
        self._state._get_prepared_for_index()
        assert self._state._final_prepared_for_index is not None
        prepared_for_id = self._state._final_prepared_for_index.get((source_for_id, parent_iteration_path))
        if prepared_for_id is None:
            raise RuntimeError(f"Unable to find finalized nested For '{source_for_id}' for {parent_iteration_path}")
        return prepared_for_id

    def create_nested_for_return(self, inner_for_id: str, prepared_inner_for_id: str) -> Optional[str]:
        graph = self._state._get_source_graph_flat()
        inner_iteration_path = self._state._get_iteration_path(prepared_inner_for_id)
        outer_for_id: Optional[str] = None
        nested_body: Optional[_SupportedNestedForBody] = None
        for source_for_id, source_node in self._state.graph.nodes.items():
            if not isinstance(source_node, ForInvocation):
                continue
            candidate = self._state.graph._get_supported_for_nested_for_body(source_for_id, graph)
            if candidate is not None and inner_for_id in candidate.inner_for_ids:
                outer_for_id = source_for_id
                nested_body = candidate
                break
        if outer_for_id is None or nested_body is None:
            return None

        prepared_inner_for = self._state.execution_graph.get_node(prepared_inner_for_id)
        outer_iteration_path = (
            inner_iteration_path
            if isinstance(prepared_inner_for, ForInvocation) and prepared_inner_for.index == -1
            else inner_iteration_path[:-1]
        )
        prepared_outer_for_id = next(
            (
                prepared_id
                for prepared_id in self._state._prepared_registry().get_prepared_ids(outer_for_id)
                if self._state._get_iteration_path(prepared_id) == outer_iteration_path
            ),
            None,
        )
        if prepared_outer_for_id is None:
            raise RuntimeError("Unable to rematerialize nested ForReturn: owning outer For is unavailable")

        source_return_id = nested_body.outer_return_id
        existing_return_ids = [
            prepared_id
            for prepared_id in self._state._prepared_registry().get_prepared_ids(source_return_id)
            if self._state._get_iteration_path(prepared_id) == outer_iteration_path
        ]
        if len(existing_return_ids) > 1:
            raise RuntimeError(
                f"Multiple nested ForReturn executions exist for {source_return_id} at {outer_iteration_path}"
            )
        if existing_return_ids:
            return existing_return_ids[0]

        continuation_nodes = self._state.graph._get_for_nested_for_continuation_nodes(nested_body)
        prepared_inner_ids: dict[str, str] = {inner_for_id: prepared_inner_for_id}
        if any(
            not self._state._is_loop_context_finalized(inner_id, outer_iteration_path)
            for inner_id in nested_body.inner_for_ids
        ):
            return None
        for inner_id in nested_body.inner_for_ids:
            if inner_id in prepared_inner_ids:
                continue
            prepared_inner_ids[inner_id] = self._get_final_prepared_for_id(inner_id, outer_iteration_path)
        source_to_prepared: dict[str, str] = {
            outer_for_id: prepared_outer_for_id,
            **prepared_inner_ids,
        }
        for source_node_id in nx.topological_sort(graph):
            if source_node_id not in continuation_nodes:
                continue

            new_edges: list[Edge] = []
            for edge in self._state.graph._get_input_edges(source_node_id):
                prepared_source_id = source_to_prepared.get(edge.source.node_id)
                if prepared_source_id is None:
                    prepared_source_id = self.get_iteration_node(
                        edge.source.node_id,
                        graph,
                        self._state._get_execution_graph_flat(),
                        [prepared_outer_for_id],
                    )
                if prepared_source_id is None:
                    raise RuntimeError(
                        f"Unable to rematerialize nested For continuation input {edge}: no prepared source node is available"
                    )
                new_edges.append(
                    Edge(
                        source=EdgeConnection(node_id=prepared_source_id, field=edge.source.field),
                        destination=EdgeConnection(node_id="", field=edge.destination.field),
                    )
                )

            new_node = self._create_execution_node_copy(self._state.graph.get_node(source_node_id), source_node_id, -1)
            source_to_prepared[source_node_id] = new_node.id
            self._state._prepared_registry().set_iteration_path(new_node.id, outer_iteration_path)
            self._state._discard_source_executed(source_node_id)
            attached_edges = self._attach_execution_edges(new_node.id, new_edges)
            self._initialize_execution_node(new_node.id, attached_edges)

        self._state._discard_source_executed(source_return_id)
        return_edges: list[Edge] = []
        for edge in self._state.graph._get_input_edges(source_return_id):
            if edge.destination.field == "output":
                prepared_source_id = source_to_prepared.get(edge.source.node_id)
                source_field = edge.source.field
            elif edge.destination.field == "state":
                prepared_source_id = prepared_outer_for_id
                source_field = edge.source.field
            elif edge.destination.field == "continue_condition":
                prepared_source_id = source_to_prepared.get(edge.source.node_id)
                source_field = edge.source.field
            else:
                raise RuntimeError(f"Unable to rematerialize nested ForReturn input {edge}")
            if prepared_source_id is None:
                raise RuntimeError(f"Unable to rematerialize nested ForReturn input {edge}")
            return_edges.append(
                Edge(
                    source=EdgeConnection(node_id=prepared_source_id, field=source_field),
                    destination=EdgeConnection(node_id="", field=edge.destination.field),
                )
            )

        prepared_return_node = self._create_execution_node_copy(
            self._state.graph.get_node(source_return_id), source_return_id, -1
        )
        self._state._prepared_registry().set_iteration_path(prepared_return_node.id, outer_iteration_path)
        attached_return_edges = self._attach_execution_edges(prepared_return_node.id, return_edges)
        self._initialize_execution_node(prepared_return_node.id, attached_return_edges)
        return prepared_return_node.id

    # Generic For planning uses private aliases. Compatibility callers retain
    # the public methods as an explicit fallback seam.
    _create_for_iteration_generic = create_for_iteration
    _create_for_body_iteration_generic = create_for_body_iteration
    _create_nested_for_return_generic = create_nested_for_return

    def _create_serial_nested_iterate_copy(
        self,
        source_node_id: str,
        input_source_id: str,
        input_source_field: str,
        input_destination_field: str,
        iteration_path: tuple[int, ...],
    ) -> str:
        new_node = self._create_execution_node_copy(self._state.graph.get_node(source_node_id), source_node_id, -1)
        self._state._prepared_registry().set_iteration_path(new_node.id, iteration_path)
        attached_edges = self._attach_execution_edges(
            new_node.id,
            [
                Edge(
                    source=EdgeConnection(node_id=input_source_id, field=input_source_field),
                    destination=EdgeConnection(node_id="", field=input_destination_field),
                )
            ],
        )
        self._initialize_execution_node(new_node.id, attached_edges)
        return new_node.id

    def _get_serial_nested_outer_path(self, prepared_for_id: str) -> tuple[int, ...]:
        prepared_for = self._state.execution_graph.get_node(prepared_for_id)
        iteration_path = self._state._get_iteration_path(prepared_for_id)
        if isinstance(prepared_for, ForInvocation) and prepared_for.index >= 0:
            return (*self._state._get_for_parent_iteration_path(prepared_for_id), prepared_for.index)
        return iteration_path

    def _get_serial_nested_prepared_at_path(
        self, source_node_id: str, iteration_path: tuple[int, ...]
    ) -> Optional[str]:
        matching_ids = [
            prepared_id
            for prepared_id in self._state._prepared_registry().get_prepared_ids(source_node_id)
            if self._state._get_iteration_path(prepared_id) == iteration_path
        ]
        if len(matching_ids) > 1:
            raise RuntimeError(f"Multiple prepared nested nodes exist for {source_node_id} at {iteration_path}")
        return matching_ids[0] if matching_ids else None

    def _create_serial_nested_collect_and_return(
        self,
        source_for_id: str,
        prepared_for_id: str,
        nested_body: _SupportedNestedIterateChain,
        outer_iteration_path: tuple[int, ...],
        body_prepared_ids: list[str],
    ) -> str:
        source_collect_id = nested_body.collect_node_id
        source_return_id = nested_body.return_node_id
        prepared_collect_id = self._get_serial_nested_prepared_at_path(source_collect_id, outer_iteration_path)
        if prepared_collect_id is None:
            collect_item_edge = self._state.graph._get_input_edges(source_collect_id, ITEM_FIELD)[0]
            collect_edges = [
                Edge(
                    source=EdgeConnection(node_id=body_prepared_id, field=collect_item_edge.source.field),
                    destination=EdgeConnection(node_id="", field=collect_item_edge.destination.field),
                )
                for body_prepared_id in body_prepared_ids
            ]
            self._state._discard_source_executed(source_collect_id)
            prepared_collect_node = self._create_execution_node_copy(
                self._state.graph.get_node(source_collect_id), source_collect_id, -1
            )
            self._state._prepared_registry().set_iteration_path(prepared_collect_node.id, outer_iteration_path)
            attached_collect_edges = self._attach_execution_edges(prepared_collect_node.id, collect_edges)
            self._initialize_execution_node(prepared_collect_node.id, attached_collect_edges)
            prepared_collect_id = prepared_collect_node.id

        prepared_return_id = self._get_serial_nested_prepared_at_path(source_return_id, outer_iteration_path)
        if prepared_return_id is not None:
            return prepared_return_id

        self._state._discard_source_executed(source_return_id)
        return_edges: list[Edge] = []
        for edge in self._state.graph._get_input_edges(source_return_id):
            if edge.destination.field == "output":
                prepared_source_id = prepared_collect_id
                source_field = COLLECTION_FIELD
            elif edge.source.node_id == source_for_id:
                prepared_source_id = prepared_for_id
                source_field = edge.source.field
            else:
                raise RuntimeError(f"Unable to rematerialize serial nested ForReturn input {edge}")
            return_edges.append(
                Edge(
                    source=EdgeConnection(node_id=prepared_source_id, field=source_field),
                    destination=EdgeConnection(node_id="", field=edge.destination.field),
                )
            )

        prepared_return_node = self._create_execution_node_copy(
            self._state.graph.get_node(source_return_id), source_return_id, -1
        )
        self._state._prepared_registry().set_iteration_path(prepared_return_node.id, outer_iteration_path)
        attached_return_edges = self._attach_execution_edges(prepared_return_node.id, return_edges)
        self._initialize_execution_node(prepared_return_node.id, attached_return_edges)
        return prepared_return_node.id

    def _create_serial_nested_iterate_body_iteration(
        self,
        source_for_id: str,
        prepared_for_id: str,
        graph: "nx.DiGraph",
        execution_graph: "nx.DiGraph",
        nested_body: _SupportedNestedIterateChain,
    ) -> Optional[str]:
        del execution_graph
        outer_iteration_path = self._get_serial_nested_outer_path(prepared_for_id)
        first_iterate_id, second_iterate_id = nested_body.iterate_node_ids
        first_preparation_id = self._state.graph._get_input_edges(first_iterate_id, COLLECTION_FIELD)[0].source.node_id
        first_preparation_edge = self._state.graph._get_input_edges(first_preparation_id)[0]

        prepared_first_preparation_id = self._get_serial_nested_prepared_at_path(
            first_preparation_id, outer_iteration_path
        )
        if prepared_first_preparation_id is None:
            prepared_first_preparation_id = self._create_serial_nested_iterate_copy(
                first_preparation_id,
                prepared_for_id,
                first_preparation_edge.source.field,
                first_preparation_edge.destination.field,
                outer_iteration_path,
            )
            self._state._discard_source_executed(first_preparation_id)

        if prepared_first_preparation_id not in self._state.results:
            return None

        first_prepared_ids = [
            prepared_id
            for prepared_id in self._state._prepared_registry().get_prepared_ids(first_iterate_id)
            if self._state._get_iteration_path(prepared_id)[: len(outer_iteration_path)] == outer_iteration_path
        ]
        if not first_prepared_ids:
            self._state._discard_source_executed(first_iterate_id)
            first_prepared_ids = self.create_execution_node(
                first_iterate_id,
                [(first_preparation_id, prepared_first_preparation_id)],
                iteration_path=outer_iteration_path,
            )
            if not first_prepared_ids:
                self._state._record_empty_iterate_stream(first_iterate_id, outer_iteration_path)
                return self._create_serial_nested_collect_and_return(
                    source_for_id, prepared_for_id, nested_body, outer_iteration_path, []
                )
            return first_prepared_ids[0]

        second_preparation_id = self._state.graph._get_input_edges(second_iterate_id, COLLECTION_FIELD)[
            0
        ].source.node_id
        second_preparation_edge = self._state.graph._get_input_edges(second_preparation_id)[0]
        prepared_second_preparation_ids: list[str] = []
        for first_prepared_id in first_prepared_ids:
            if first_prepared_id not in self._state.results:
                continue
            prepared_id = self._get_serial_nested_prepared_at_path(
                second_preparation_id, self._state._get_iteration_path(first_prepared_id)
            )
            if prepared_id is not None and prepared_id in self._state.results:
                prepared_second_preparation_ids.append(prepared_id)
        if len(prepared_second_preparation_ids) != len(first_prepared_ids):
            self._state._discard_source_executed(second_preparation_id)
            created_ids: list[str] = []
            for first_prepared_id in first_prepared_ids:
                iteration_path = self._state._get_iteration_path(first_prepared_id)
                if self._get_serial_nested_prepared_at_path(second_preparation_id, iteration_path) is not None:
                    continue
                created_ids.append(
                    self._create_serial_nested_iterate_copy(
                        second_preparation_id,
                        first_prepared_id,
                        second_preparation_edge.source.field,
                        second_preparation_edge.destination.field,
                        iteration_path,
                    )
                )
            return created_ids[0] if created_ids else None

        second_prepared_ids: list[str] = []
        body_prepared_ids: list[str] = []
        body_node_id = self._state.graph._get_input_edges(nested_body.collect_node_id, ITEM_FIELD)[0].source.node_id
        body_input_edge = self._state.graph._get_input_edges(body_node_id)[0]
        self._state._discard_source_executed(second_iterate_id)
        for prepared_second_preparation_id in prepared_second_preparation_ids:
            iteration_path = self._state._get_iteration_path(prepared_second_preparation_id)
            second_ids = [
                prepared_id
                for prepared_id in self._state._prepared_registry().get_prepared_ids(second_iterate_id)
                if self._state._get_iteration_path(prepared_id)[: len(iteration_path)] == iteration_path
                and len(self._state._get_iteration_path(prepared_id)) == len(iteration_path) + 1
            ]
            if not second_ids:
                second_ids = self.create_execution_node(
                    second_iterate_id,
                    [(second_preparation_id, prepared_second_preparation_id)],
                    iteration_path=iteration_path,
                )
            second_prepared_ids.extend(second_ids)
            for prepared_second_id in second_ids:
                prepared_body_id = self._get_serial_nested_prepared_at_path(
                    body_node_id, self._state._get_iteration_path(prepared_second_id)
                )
                if prepared_body_id is None:
                    prepared_body_id = self._create_serial_nested_iterate_copy(
                        body_node_id,
                        prepared_second_id,
                        body_input_edge.source.field,
                        body_input_edge.destination.field,
                        self._state._get_iteration_path(prepared_second_id),
                    )
                body_prepared_ids.append(prepared_body_id)
            if not second_ids:
                self._state._record_empty_iterate_stream(second_iterate_id, iteration_path)

        if not second_prepared_ids:
            self._state._mark_source_node_empty(second_iterate_id, outer_iteration_path)
            self._state._mark_source_node_empty(body_node_id, outer_iteration_path)
        return self._create_serial_nested_collect_and_return(
            source_for_id, prepared_for_id, nested_body, outer_iteration_path, body_prepared_ids
        )

    def _create_nested_for_body_iteration(
        self,
        source_for_id: str,
        prepared_for_id: str,
        graph: "nx.DiGraph",
        execution_graph: "nx.DiGraph",
        nested_body: _SupportedNestedForBody,
    ) -> Optional[str]:
        """Materialize one nested-For body at the owning outer iteration path.

        The source graph is shaped as ``outer For -> inner For(s) -> outer ForReturn``. Existing execution nodes
        are reused at the current path; missing ordinary body nodes are copied, then each inner For is allowed to
        advance independently before the outer return is rematerialized.
        """
        body_path_nodes = nested_body.body_path_nodes
        source_return_id = nested_body.outer_return_id
        prepared_for_node = self._state.execution_graph.get_node(prepared_for_id)
        outer_iteration_path = self._state._get_iteration_path(prepared_for_id)
        if isinstance(prepared_for_node, ForInvocation) and prepared_for_node.index >= 0:
            outer_iteration_path = (
                *self._state._get_for_parent_iteration_path(prepared_for_id),
                prepared_for_node.index,
            )

        source_to_prepared: dict[str, str] = {source_for_id: prepared_for_id}
        for source_node_id in nx.topological_sort(graph):
            if source_node_id not in body_path_nodes or source_node_id in {
                *nested_body.inner_for_ids,
                source_return_id,
            }:
                continue
            if not any(nx.has_path(graph, source_node_id, inner_for_id) for inner_for_id in nested_body.inner_for_ids):
                continue

            existing_prepared_ids = [
                prepared_id
                for prepared_id in self._state._prepared_registry().get_prepared_ids(source_node_id)
                if self._state._get_iteration_path(prepared_id) == outer_iteration_path
            ]
            if len(existing_prepared_ids) == 1:
                source_to_prepared[source_node_id] = existing_prepared_ids[0]
                continue

            new_edges: list[Edge] = []
            for edge in self._state.graph._get_input_edges(source_node_id):
                prepared_source_id = source_to_prepared.get(edge.source.node_id)
                if prepared_source_id is None:
                    prepared_source_id = self.get_iteration_node(
                        edge.source.node_id, graph, execution_graph, [prepared_for_id]
                    )
                if prepared_source_id is None:
                    raise RuntimeError(
                        f"Unable to rematerialize nested For input {edge}: no prepared source node is available"
                    )
                new_edges.append(
                    Edge(
                        source=EdgeConnection(node_id=prepared_source_id, field=edge.source.field),
                        destination=EdgeConnection(node_id="", field=edge.destination.field),
                    )
                )

            new_node = self._create_execution_node_copy(self._state.graph.get_node(source_node_id), source_node_id, -1)
            source_to_prepared[source_node_id] = new_node.id
            self._state._prepared_registry().set_iteration_path(new_node.id, outer_iteration_path)
            self._state._discard_source_executed(source_node_id)
            attached_edges = self._attach_execution_edges(new_node.id, new_edges)
            self._initialize_execution_node(new_node.id, attached_edges)

        for body_node_id in body_path_nodes:
            if body_node_id in nested_body.inner_for_ids or not any(
                nx.has_path(graph, body_node_id, inner_for_id) for inner_for_id in nested_body.inner_for_ids
            ):
                self._state._discard_source_executed(body_node_id)
        self._state._discard_source_executed(source_return_id)

        for source_inner_for_id in nested_body.inner_for_ids:
            existing_prepared_ids = [
                prepared_id
                for prepared_id in self._state._prepared_registry().get_prepared_ids(source_inner_for_id)
                if self._state._get_for_parent_iteration_path(prepared_id) == outer_iteration_path
            ]
            if existing_prepared_ids:
                continue

            inner_input_map: list[tuple[str, str]] = []
            for edge in self._state.graph._get_input_edges(source_inner_for_id):
                prepared_source_id = source_to_prepared.get(edge.source.node_id)
                if prepared_source_id is None:
                    prepared_source_id = self.get_iteration_node(
                        edge.source.node_id, graph, execution_graph, [prepared_for_id]
                    )
                if prepared_source_id is None:
                    raise RuntimeError(
                        f"Unable to rematerialize nested For input {edge}: no prepared source node is available"
                    )
                inner_input_map.append((edge.source.node_id, prepared_source_id))

            if any(prepared_source_id not in self._state.results for _, prepared_source_id in inner_input_map):
                return None

            self._state._discard_source_executed(source_inner_for_id)
            inner_prepared_ids = self.create_execution_node(
                source_inner_for_id, inner_input_map, iteration_path=outer_iteration_path
            )
            if not inner_prepared_ids:
                self._mark_source_node_empty(source_inner_for_id)
            elif all(
                isinstance(self._state.execution_graph.get_node(inner_id), ForInvocation)
                and self._state.execution_graph.get_node(inner_id).index == -1
                for inner_id in inner_prepared_ids
            ):
                self._mark_empty_for_complete(source_inner_for_id)
                for inner_prepared_id in inner_prepared_ids:
                    self._create_nested_for_return_generic(source_inner_for_id, inner_prepared_id)
            else:
                for inner_prepared_id in inner_prepared_ids:
                    self._create_for_body_iteration_generic(
                        source_for_id=source_inner_for_id,
                        prepared_for_id=inner_prepared_id,
                    )

        return None

    def _create_nested_iterate_body_iteration(
        self,
        source_for_id: str,
        prepared_for_id: str,
        graph: "nx.DiGraph",
        execution_graph: "nx.DiGraph",
        nested_body: _SupportedNestedIterateBody,
    ) -> Optional[str]:
        body_path_nodes = nested_body.body_path_nodes
        source_return_id = nested_body.return_node_id
        source_iterate_id = nested_body.iterate_node_id
        source_collect_id = nested_body.collect_node_id
        prepared_for_node = self._state.execution_graph.get_node(prepared_for_id)
        outer_iteration_path = self._state._get_iteration_path(prepared_for_id)
        if isinstance(prepared_for_node, ForInvocation) and prepared_for_node.index >= 0:
            outer_iteration_path = (
                *self._state._get_for_parent_iteration_path(prepared_for_id),
                prepared_for_node.index,
            )
        source_to_prepared: dict[str, str] = {source_for_id: prepared_for_id}
        inner_prepared_by_source: dict[tuple[str, str], str] = {}

        def resolve_outer_input(source_node_id: str) -> Optional[str]:
            prepared_source_id = source_to_prepared.get(source_node_id)
            if prepared_source_id is not None:
                return prepared_source_id
            return self.get_iteration_node(source_node_id, graph, execution_graph, [prepared_for_id])

        def create_body_copy(source_node_id: str, input_resolver, iteration_path: tuple[int, ...]) -> str:
            new_edges: list[Edge] = []
            for edge in self._state.graph._get_input_edges(source_node_id):
                prepared_source_id = input_resolver(edge.source.node_id)
                if prepared_source_id is None:
                    raise RuntimeError(
                        f"Unable to rematerialize For body input {edge}: no prepared source node is available"
                    )
                new_edges.append(
                    Edge(
                        source=EdgeConnection(node_id=prepared_source_id, field=edge.source.field),
                        destination=EdgeConnection(node_id="", field=edge.destination.field),
                    )
                )

            new_node = self._create_execution_node_copy(self._state.graph.get_node(source_node_id), source_node_id, -1)
            self._state._prepared_registry().set_iteration_path(new_node.id, iteration_path)
            attached_edges = self._attach_execution_edges(new_node.id, new_edges)
            self._initialize_execution_node(new_node.id, attached_edges)
            return new_node.id

        def get_existing_body_node(source_node_id: str) -> Optional[str]:
            matching_ids = [
                prepared_id
                for prepared_id in self._state._prepared_registry().get_prepared_ids(source_node_id)
                if self._state._get_iteration_path(prepared_id) == outer_iteration_path
            ]
            if len(matching_ids) == 1:
                return matching_ids[0]
            return None

        for source_node_id in nx.topological_sort(graph):
            if source_node_id not in body_path_nodes or source_node_id in {
                source_iterate_id,
                source_collect_id,
                source_return_id,
            }:
                continue
            if not nx.has_path(graph, source_node_id, source_iterate_id):
                continue
            source_to_prepared[source_node_id] = get_existing_body_node(source_node_id) or create_body_copy(
                source_node_id, resolve_outer_input, outer_iteration_path
            )

        iterate_input_map: list[tuple[str, str]] = []
        for edge in self._state.graph._get_input_edges(source_iterate_id):
            prepared_source_id = resolve_outer_input(edge.source.node_id)
            if prepared_source_id is None:
                raise RuntimeError(
                    f"Unable to rematerialize For body input {edge}: no prepared source node is available"
                )
            iterate_input_map.append((edge.source.node_id, prepared_source_id))

        if any(prepared_source_id not in self._state.results for _, prepared_source_id in iterate_input_map):
            return None

        self._state._discard_source_executed(source_iterate_id)
        inner_prepared_ids = self.create_execution_node(
            source_iterate_id, iterate_input_map, iteration_path=outer_iteration_path
        )
        if not inner_prepared_ids:
            self._mark_source_node_empty(source_iterate_id, outer_iteration_path)

        for inner_prepared_id in inner_prepared_ids:
            inner_iteration_path = self._state._get_iteration_path(inner_prepared_id)
            for source_node_id in nx.topological_sort(graph):
                if source_node_id not in body_path_nodes:
                    continue
                if source_node_id in {source_iterate_id, source_collect_id, source_return_id}:
                    continue
                if nx.has_path(graph, source_node_id, source_iterate_id):
                    continue
                if not nx.has_path(graph, source_iterate_id, source_node_id):
                    continue

                def resolve_inner_input(
                    input_source_node_id: str, current_inner_prepared_id: str = inner_prepared_id
                ) -> Optional[str]:
                    if input_source_node_id == source_iterate_id:
                        return current_inner_prepared_id
                    prepared_source_id = source_to_prepared.get(input_source_node_id)
                    if prepared_source_id is not None:
                        return prepared_source_id
                    prepared_source_id = inner_prepared_by_source.get((input_source_node_id, current_inner_prepared_id))
                    if prepared_source_id is not None:
                        return prepared_source_id
                    return self.get_iteration_node(
                        input_source_node_id, graph, execution_graph, [current_inner_prepared_id]
                    )

                self._state._discard_source_executed(source_node_id)
                prepared_id = create_body_copy(source_node_id, resolve_inner_input, inner_iteration_path)
                inner_prepared_by_source[(source_node_id, inner_prepared_id)] = prepared_id

        if not inner_prepared_ids:
            for source_node_id in body_path_nodes:
                if source_node_id in {source_iterate_id, source_collect_id, source_return_id}:
                    continue
                if nx.has_path(graph, source_iterate_id, source_node_id):
                    self._mark_source_node_empty(source_node_id, outer_iteration_path)

        collect_item_edge = self._state.graph._get_input_edges(source_collect_id, ITEM_FIELD)[0]
        collect_edges: list[Edge] = []
        for inner_prepared_id in inner_prepared_ids:
            prepared_source_id = inner_prepared_by_source.get((collect_item_edge.source.node_id, inner_prepared_id))
            if prepared_source_id is None and collect_item_edge.source.node_id == source_iterate_id:
                prepared_source_id = inner_prepared_id
            if prepared_source_id is None:
                raise RuntimeError(
                    f"Unable to rematerialize For body input {collect_item_edge}: no prepared source node is available"
                )
            collect_edges.append(
                Edge(
                    source=EdgeConnection(node_id=prepared_source_id, field=collect_item_edge.source.field),
                    destination=EdgeConnection(node_id="", field=ITEM_FIELD),
                )
            )

        self._state._discard_source_executed(source_collect_id)
        collect_node = self._state.graph.get_node(source_collect_id)
        prepared_collect_node = self._create_execution_node_copy(collect_node, source_collect_id, -1)
        self._state._prepared_registry().set_iteration_path(prepared_collect_node.id, outer_iteration_path)
        attached_collect_edges = self._attach_execution_edges(prepared_collect_node.id, collect_edges)
        self._initialize_execution_node(prepared_collect_node.id, attached_collect_edges)

        return_edges: list[Edge] = []
        for edge in self._state.graph._get_input_edges(source_return_id):
            if edge.destination.field == "output":
                prepared_source_id = prepared_collect_node.id
                source_field = COLLECTION_FIELD
            else:
                prepared_source_id = resolve_outer_input(edge.source.node_id)
                source_field = edge.source.field
            if prepared_source_id is None:
                raise RuntimeError(
                    f"Unable to rematerialize For body input {edge}: no prepared source node is available"
                )
            return_edges.append(
                Edge(
                    source=EdgeConnection(node_id=prepared_source_id, field=source_field),
                    destination=EdgeConnection(node_id="", field=edge.destination.field),
                )
            )
        self._state._discard_source_executed(source_return_id)
        prepared_return_node = self._create_execution_node_copy(
            self._state.graph.get_node(source_return_id), source_return_id, -1
        )
        self._state._prepared_registry().set_iteration_path(prepared_return_node.id, outer_iteration_path)
        attached_return_edges = self._attach_execution_edges(prepared_return_node.id, return_edges)
        self._initialize_execution_node(prepared_return_node.id, attached_return_edges)
        return prepared_return_node.id

    def _attach_execution_edges(self, exec_node_id: str, new_edges: list[Edge]) -> list[Edge]:
        attached_edges = [
            Edge(
                source=edge.source,
                destination=EdgeConnection(node_id=exec_node_id, field=edge.destination.field),
            )
            for edge in new_edges
        ]
        self._state._tx_add_execution_edges(attached_edges)
        self._state._add_execution_graph_edges(attached_edges)
        return attached_edges

    def _initialize_execution_node(self, exec_node_id: str, input_edges: Optional[list[Edge]] = None) -> None:
        inputs = input_edges if input_edges is not None else self._state.execution_graph._get_input_edges(exec_node_id)
        unmet = sum(1 for edge in inputs if edge.source.node_id not in self._state.executed)
        self._state._tx_set_mapping(self._state.indegree, exec_node_id, unmet)
        scheduler = self._state._scheduler()
        # Resolve a known conditional before registering its generic plan node so
        # generic retirement can release unselected prerequisites without
        # mutating the append-only execution graph.
        if self._state._is_generic_graph_scheduler(scheduler):
            self._state._try_resolve_if_node(exec_node_id)
            if self._state._is_pending_if(exec_node_id):
                self._state._tx_add_set(self._state._pending_if_exec_nodes, exec_node_id)
                return
            scheduler.register_node(exec_node_id)
        else:
            self._state._try_resolve_if_node(exec_node_id, enqueue=False)
            if self._state._is_pending_if(exec_node_id):
                self._state._tx_add_set(self._state._pending_if_exec_nodes, exec_node_id)
                return
        self._state._enqueue_if_ready(exec_node_id)

    def _get_collect_iteration_group_key(self, edge: Edge, sibling_depth: Optional[int] = None) -> tuple[int, ...]:
        path = self._state._get_iteration_path(edge.source.node_id)
        source_node = self._state.execution_graph.get_node(edge.source.node_id)
        if (
            isinstance(source_node, ForInvocation)
            and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
        ):
            return self._state._get_for_parent_iteration_path(edge.source.node_id)
        if edge.destination.field == ITEM_FIELD:
            if isinstance(source_node, ForInvocation) and source_node.index == -1:
                return path
            # Ragged siblings need the deepest path to identify their shared outer group.
            depth = len(path) if sibling_depth is None else sibling_depth
            return path[: max(depth - 1, 0)]
        return path

    def _get_collect_source_iterator_ids(self, source_node_id: str) -> list[str]:
        iterator_node_ids = self.get_node_iterators(source_node_id)
        if isinstance(self._state.graph.get_node(source_node_id), IterateInvocation):
            iterator_node_ids.append(source_node_id)
        return iterator_node_ids

    def _get_ordered_prepared_nodes_for_source(self, source_node_id: str) -> list[str]:
        return sorted(
            self._get_prepared_nodes_for_source(source_node_id),
            key=lambda exec_node_id: (self._state._get_iteration_path(exec_node_id), exec_node_id),
        )

    def _get_ordered_prepared_nodes_for_edge(self, edge: Edge) -> list[str]:
        prepared_nodes = self._get_ordered_prepared_nodes_for_source(edge.source.node_id)
        source_node = self._state.graph.get_node(edge.source.node_id)
        if not (
            isinstance(source_node, ForInvocation)
            and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
        ):
            return prepared_nodes

        final_nodes_by_parent_path: dict[tuple[int, ...], str] = {}
        for prepared_id in prepared_nodes:
            parent_path = self._state._get_for_parent_iteration_path(prepared_id)
            previous_id = final_nodes_by_parent_path.get(parent_path)
            if previous_id is None:
                final_nodes_by_parent_path[parent_path] = prepared_id
                continue
            previous_node = self._state.execution_graph.get_node(previous_id)
            prepared_node = self._state.execution_graph.get_node(prepared_id)
            assert isinstance(previous_node, ForInvocation)
            assert isinstance(prepared_node, ForInvocation)
            if prepared_node.index > previous_node.index:
                final_nodes_by_parent_path[parent_path] = prepared_id

        return [final_nodes_by_parent_path[parent_path] for parent_path in sorted(final_nodes_by_parent_path)]

    def _get_prepared_edge_iteration_path(self, edge: Edge, prepared_id: str) -> tuple[int, ...]:
        source_node = self._state.graph.get_node(edge.source.node_id)
        if (
            isinstance(source_node, ForInvocation)
            and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
        ):
            return self._state._get_for_parent_iteration_path(prepared_id)
        return self._state._get_iteration_path(prepared_id)

    def _get_iterator_input_iteration_paths(self, iterator_node_id: str) -> set[tuple[int, ...]]:
        iteration_paths: set[tuple[int, ...]] = set()
        for edge in self._state.graph._get_input_edges(iterator_node_id, COLLECTION_FIELD):
            source_node_id = edge.source.node_id
            prepared_nodes = self._get_ordered_prepared_nodes_for_source(source_node_id)
            iteration_paths.update(
                self._get_prepared_edge_iteration_path(edge, prepared_id) for prepared_id in prepared_nodes
            )
        return iteration_paths

    def _get_collect_candidate_group_keys(self, edge: Edge) -> set[tuple[int, ...]]:
        source_node_id = edge.source.node_id
        iterator_node_ids = self._get_collect_source_iterator_ids(source_node_id)

        group_depth = len(iterator_node_ids)
        if edge.destination.field == ITEM_FIELD:
            group_depth = max(group_depth - 1, 0)

        group_keys: set[tuple[int, ...]] = set()
        for iterator_node_id in iterator_node_ids:
            prepared_nodes = self._get_ordered_prepared_nodes_for_source(iterator_node_id)
            # Prepared paths use the active group depth. Input paths stay full to preserve scope across collectors.
            if prepared_nodes and group_depth:
                group_keys.update(
                    iteration_path[:group_depth]
                    for prepared_id in prepared_nodes
                    if len(iteration_path := self._state._get_iteration_path(prepared_id)) >= group_depth
                )
            group_keys.update(self._get_iterator_input_iteration_paths(iterator_node_id))

        if group_keys:
            return group_keys
        if group_depth == 0:
            return {()}
        return set()

    def _get_collect_iteration_mapping_groups(
        self, input_edges: list[Edge]
    ) -> list[tuple[tuple[int, ...], list[tuple[str, str]]]]:
        prepared_inputs: list[tuple[Edge, str, str, tuple[int, ...]]] = []
        group_keys: set[tuple[int, ...]] = set()
        for edge in input_edges:
            group_keys.update(self._get_collect_candidate_group_keys(edge))
            prepared_nodes = self._get_ordered_prepared_nodes_for_edge(edge)
            sibling_depth = max(
                (len(self._get_prepared_edge_iteration_path(edge, prepared_id)) for prepared_id in prepared_nodes),
                default=0,
            )
            for prepared_id in prepared_nodes:
                prepared_edge = Edge(
                    source=EdgeConnection(node_id=prepared_id, field=edge.source.field),
                    destination=edge.destination,
                )
                group_key = self._get_collect_iteration_group_key(prepared_edge, sibling_depth)
                group_keys.add(group_key)
                prepared_inputs.append(
                    (
                        prepared_edge,
                        edge.source.node_id,
                        prepared_id,
                        self._get_prepared_edge_iteration_path(edge, prepared_id),
                    )
                )

        if not group_keys:
            group_keys.add(())

        final_group_keys = sorted(
            group_key
            for group_key in group_keys
            if not any(
                group_key != other_group_key and other_group_key[: len(group_key)] == group_key
                for other_group_key in group_keys
            )
        )

        return [
            (
                group_key,
                [
                    (source_node_id, prepared_id)
                    for prepared_edge, source_node_id, prepared_id, iteration_path in prepared_inputs
                    if (
                        prepared_edge.destination.field == ITEM_FIELD
                        and (
                            group_key[: len(iteration_path)] == iteration_path
                            or iteration_path[: len(group_key)] == group_key
                        )
                    )
                    or (
                        prepared_edge.destination.field != ITEM_FIELD
                        and group_key[: len(iteration_path)] == iteration_path
                    )
                ],
            )
            for group_key in final_group_keys
        ]

    def _get_parent_iteration_mappings_without_iterators(
        self, next_node_id: str, input_edges: Optional[list[Edge]] = None
    ) -> list[list[tuple[str, str]]]:
        input_edges = input_edges or self._state.graph._get_input_edges(next_node_id)
        parent_node_ids = list(dict.fromkeys(edge.source.node_id for edge in input_edges))
        parent_prepared_nodes = {
            node_id: list(
                dict.fromkeys(
                    (prepared_id, self._get_prepared_edge_iteration_path(edge, prepared_id))
                    for edge in input_edges
                    if edge.source.node_id == node_id
                    for prepared_id in self._get_ordered_prepared_nodes_for_edge(edge)
                )
            )
            for node_id in parent_node_ids
        }
        all_iteration_paths = {
            iteration_path
            for prepared_nodes in parent_prepared_nodes.values()
            for _prepared_id, iteration_path in prepared_nodes
            if iteration_path != ()
        }
        iteration_paths = sorted(
            iteration_path
            for iteration_path in all_iteration_paths
            if not any(
                iteration_path != other_path and other_path[: len(iteration_path)] == iteration_path
                for other_path in all_iteration_paths
            )
        )
        if not iteration_paths:
            iteration_paths = [()]

        mappings: list[list[tuple[str, str]]] = []
        for iteration_path in iteration_paths:
            mapping: list[tuple[str, str]] = []
            for node_id, prepared_nodes in parent_prepared_nodes.items():
                matching_prepared = next(
                    iter(
                        sorted(
                            (
                                (prepared_id, prepared_path)
                                for prepared_id, prepared_path in prepared_nodes
                                if iteration_path[: len(prepared_path)] == prepared_path
                            ),
                            key=lambda prepared: (-len(prepared[1]), prepared[0]),
                        )
                    ),
                    None,
                )
                if matching_prepared is None:
                    break
                mapping.append((node_id, matching_prepared[0]))
            if len(mapping) == len(parent_node_ids):
                mappings.append(mapping)
        return mappings

    def _mark_source_node_empty(self, source_node_id: str, iteration_path: tuple[int, ...] = ()) -> None:
        self._state._tx_set_mapping(self._state.source_prepared_mapping, source_node_id, set())
        self._state._reset_unexecuted_prepared(source_node_id)
        self._state._mark_source_executed(source_node_id)
        if isinstance(self._state.graph.get_node(source_node_id), IterateInvocation):
            self._state._record_empty_iterate_stream(source_node_id, iteration_path)

    def _index_prepared_nodes_by_iteration_path(
        self, prepared_nodes: set[str], input_edges: list[Edge]
    ) -> dict[tuple[int, ...], list[str]]:
        prepared_nodes_by_iteration_path: dict[tuple[int, ...], list[str]] = {}
        for prepared_id in prepared_nodes:
            iteration_path = self._get_prepared_edge_iteration_path(input_edges[0], prepared_id)
            prepared_nodes_by_iteration_path.setdefault(iteration_path, []).append(prepared_id)
        return prepared_nodes_by_iteration_path

    def _get_target_iteration_path(
        self, source_node_id: str, graph: "nx.DiGraph", prepared_iterator_nodes: tuple[str, ...]
    ) -> Optional[tuple[int, ...]]:
        parent_iterators = self._get_parent_iterator_exec_nodes(source_node_id, graph, list(prepared_iterator_nodes))
        parent_paths = [self._state._get_iteration_path(prepared_id) for prepared_id, _ in parent_iterators]
        if not parent_paths:
            return ()

        target_path = max(parent_paths, key=len)
        if all(target_path[: len(parent_path)] == parent_path for parent_path in parent_paths):
            return target_path
        return None

    def _get_indexed_iteration_node(
        self,
        source_node_id: str,
        graph: "nx.DiGraph",
        prepared_iterator_nodes: tuple[str, ...],
        prepared_nodes_by_iteration_path: dict[tuple[int, ...], list[str]],
    ) -> Optional[str]:
        target_path = self._get_target_iteration_path(source_node_id, graph, prepared_iterator_nodes)
        if target_path is None:
            return None

        for path_length in range(len(target_path), -1, -1):
            candidates = prepared_nodes_by_iteration_path.get(target_path[:path_length], [])
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                return None
        return None

    def _get_parent_iteration_mappings(
        self, next_node_id: str, graph: "nx.DiGraph", input_edges: Optional[list[Edge]] = None
    ) -> Iterable[list[tuple[str, str]]]:
        input_edges = input_edges or self._state.graph._get_input_edges(next_node_id)
        parent_node_ids = list(dict.fromkeys(edge.source.node_id for edge in input_edges))
        iterator_graph = self.iterator_graph(graph)
        iterator_nodes = self.get_node_iterators(next_node_id, iterator_graph)
        if not iterator_nodes:
            return iter(self._get_parent_iteration_mappings_without_iterators(next_node_id, input_edges))

        iterator_nodes_prepared = [
            sorted(self._state.source_prepared_mapping[node_id], key=self._state._get_iteration_path)
            for node_id in iterator_nodes
        ]
        prepared_nodes_by_source = {
            node_id: self._get_prepared_nodes_for_source(node_id) for node_id in parent_node_ids
        }
        prepared_nodes_by_source_and_path = {
            node_id: self._index_prepared_nodes_by_iteration_path(
                prepared_nodes,
                [edge for edge in input_edges if edge.source.node_id == node_id],
            )
            for node_id, prepared_nodes in prepared_nodes_by_source.items()
        }

        def iter_mappings() -> Iterable[list[tuple[str, str]]]:
            execution_graph: Optional["nx.DiGraph"] = None
            for prepared_iterators in itertools.product(*iterator_nodes_prepared):
                mapping: list[tuple[str, str]] = []
                for node_id in parent_node_ids:
                    prepared_id = self._get_indexed_iteration_node(
                        node_id,
                        graph,
                        prepared_iterators,
                        prepared_nodes_by_source_and_path[node_id],
                    )
                    if prepared_id is None:
                        if execution_graph is None:
                            execution_graph = self._state._get_execution_graph_flat()
                        prepared_id = self.get_iteration_node(
                            node_id,
                            graph,
                            execution_graph,
                            list(prepared_iterators),
                            prepared_nodes_by_source[node_id],
                        )
                    if prepared_id is None:
                        break
                    mapping.append((node_id, prepared_id))
                if len(mapping) == len(parent_node_ids):
                    yield mapping

        return iter(iter_mappings())

    def create_execution_node(
        self,
        node_id: str,
        iteration_node_map: list[tuple[str, str]],
        iteration_path: Optional[tuple[int, ...]] = None,
        input_fields: Optional[set[str]] = None,
        enforce_admission: bool = True,
    ) -> list[str]:
        """Prepares an iteration node and connects all edges, returning the new node id"""

        node = self._state.graph.get_node(node_id)
        iteration_indexes = self._get_new_node_iterations(node, node_id, iteration_node_map)
        if not iteration_indexes:
            if isinstance(node, ForInvocation):
                return [self._create_empty_for_final_output(node_id, node, iteration_node_map)]
            return []

        new_edges = self._build_execution_edges_for_fields(node_id, iteration_node_map, input_fields)
        input_edges = self._state.graph._get_input_edges(node_id)
        if input_fields is not None:
            input_edges = [edge for edge in input_edges if edge.destination.field in input_fields]
        new_nodes: list[str] = []
        for iteration_index in iteration_indexes:
            new_node_iteration_path = iteration_path
            if new_node_iteration_path is None:
                new_node_iteration_path = self._get_known_iteration_path(
                    iteration_index, iteration_node_map, input_edges
                )
            elif isinstance(node, (ForInvocation, IterateInvocation)):
                new_node_iteration_path += (iteration_index,)
            if enforce_admission and not self._state._is_source_activation_admitted(
                node_id, new_node_iteration_path or ()
            ):
                continue
            new_node = self._create_execution_node_copy(node, node_id, iteration_index)
            if new_node_iteration_path is not None:
                self._state._prepared_registry().set_iteration_path(new_node.id, new_node_iteration_path)
            attached_edges = self._attach_execution_edges(new_node.id, new_edges)
            self._state._record_activation_dependencies(new_node.id)
            self._initialize_execution_node(new_node.id, attached_edges)
            new_nodes.append(new_node.id)

        return new_nodes

    def _has_admitted_source_mapping(self, node_id: str, graph: "nx.DiGraph") -> bool:
        if not any(isinstance(node, IfInvocation) for node in self._state.graph.nodes.values()):
            return True
        if isinstance(self._state.graph.get_node(node_id), IfInvocation):
            condition_edges = self._state.graph._get_input_edges(node_id, "condition")
            if not condition_edges:
                return any(
                    self._state._is_source_activation_admitted(node_id, iteration_path)
                    for iteration_path in graph_if_runtime._get_if_iteration_paths(
                        self._state,
                        node_id,
                        set(),
                        self._get_prepared_nodes_for_source,
                        self._get_prepared_edge_iteration_path,
                    )
                )
            mappings = self._get_if_condition_iteration_mappings(node_id, graph)
            input_edges = condition_edges
        else:
            mappings = self._get_parent_iteration_mappings(node_id, graph)
            input_edges = self._state.graph._get_input_edges(node_id)
        mappings = list(mappings)
        if isinstance(self._state.graph.get_node(node_id), IfInvocation) and self.get_node_iterators(
            node_id, self.iterator_graph(graph)
        ):
            if not any(mappings):
                iterator_nodes = self.get_node_iterators(node_id, self.iterator_graph(graph))
                if all(
                    iterator_node_id in self._state.executed
                    and not self._state.source_prepared_mapping.get(iterator_node_id)
                    for iterator_node_id in iterator_nodes
                ):
                    return True
                return False
        if not mappings:
            return self._state._is_source_activation_admitted(node_id)
        return any(
            self._state._is_source_activation_admitted(
                node_id, self._get_known_iteration_path(-1, iteration_mapping, input_edges) or ()
            )
            for iteration_mapping in mappings
        )

    def _get_if_condition_iteration_mappings(
        self, node_id: str, graph: "nx.DiGraph"
    ) -> Iterable[list[tuple[str, str]]]:
        condition_edges = self._state.graph._get_input_edges(node_id, "condition")
        iterator_graph = self.iterator_graph(graph)
        iterator_nodes = self.get_node_iterators(node_id, iterator_graph)
        if condition_edges and not iterator_nodes:
            return self._get_parent_iteration_mappings(node_id, graph, input_edges=condition_edges)
        if not iterator_nodes:
            return iter([[]])

        if condition_edges and any(
            edge.source.node_id == iterator_nodes[0]
            or nx.has_path(iterator_graph, iterator_nodes[0], edge.source.node_id)
            for edge in condition_edges
        ):
            return self._get_parent_iteration_mappings(node_id, graph, input_edges=condition_edges)

        # This bounded path derives one frame axis. Leave mixed/nested axes on the existing path because
        # _get_known_iteration_path() intentionally rejects incompatible parent paths.
        if len(iterator_nodes) != 1 or any(
            not any(
                source_edge.source.node_id == iterator_nodes[0]
                or nx.has_path(iterator_graph, iterator_nodes[0], source_edge.source.node_id)
                for source_edge in self._state.graph._get_input_edges(node_id, branch_field)
            )
            for branch_field in ("true_input", "false_input")
        ):
            return iter([[]])

        if any(iterator_node_id not in self._state.source_prepared_mapping for iterator_node_id in iterator_nodes):
            return iter([])
        iterator_nodes_prepared = [
            sorted(self._state.source_prepared_mapping[iterator_node_id], key=self._state._get_iteration_path)
            for iterator_node_id in iterator_nodes
        ]

        if condition_edges:
            condition_mappings = self._get_parent_iteration_mappings(node_id, graph, input_edges=condition_edges)
            return iter(
                condition_mapping + list(zip(iterator_nodes, prepared_iterators, strict=True))
                for condition_mapping, prepared_iterators in zip(
                    condition_mappings, itertools.product(*iterator_nodes_prepared), strict=True
                )
            )

        return iter(
            [
                list(zip(iterator_nodes, prepared_iterators, strict=True))
                for prepared_iterators in itertools.product(*iterator_nodes_prepared)
            ]
        )

    def _is_if_condition_ready(self, node_id: str) -> bool:
        return all(
            edge.source.node_id in self._state.source_prepared_mapping or edge.source.node_id in self._state.executed
            for edge in self._state.graph._get_input_edges(node_id, "condition")
        )

    def _attach_pending_if_inputs(self, *, enqueue: bool = True) -> None:
        """Attach only the selected branch edge to condition-ready If executions."""

        source_graph = self._state._get_source_graph_flat()
        source_order = {node_id: index for index, node_id in enumerate(nx.topological_sort(source_graph))}
        pending_exec_nodes = sorted(
            self._state._pending_if_exec_nodes,
            key=lambda exec_node_id: (
                source_order.get(self._state._prepared_registry().get_source_node_id(exec_node_id), 0),
                self._state._get_iteration_path(exec_node_id),
                exec_node_id,
            ),
        )
        for exec_node_id in pending_exec_nodes:
            self._state._try_resolve_if_node(exec_node_id, enqueue=False)
            selected_field = self._state._resolved_if_exec_branches.get(exec_node_id)
            if selected_field is None:
                continue
            if any(
                edge.destination.field == selected_field
                for edge in self._state.execution_graph._get_input_edges(exec_node_id)
            ):
                if enqueue:
                    self._state._tx_discard_set(self._state._pending_if_exec_nodes, exec_node_id)
                continue

            source_if_id = self._state._prepared_registry().get_source_node_id(exec_node_id)
            target_path = self._state._get_iteration_path(exec_node_id)
            selected_edges = self._state.graph._get_input_edges(source_if_id, selected_field)
            attached_edges: list[Edge] = []
            selected_exec_ids: list[str] = []
            for source_edge in selected_edges:
                if source_edge.source.node_id not in self._state.source_prepared_mapping:
                    continue
                candidates = [
                    prepared_id
                    for prepared_id in self._get_ordered_prepared_nodes_for_edge(source_edge)
                    if self._get_prepared_edge_iteration_path(source_edge, prepared_id) == target_path
                ]
                if not candidates:
                    continue
                selected_exec_ids.append(candidates[0])
                attached_edges.append(
                    Edge(
                        source=EdgeConnection(node_id=candidates[0], field=source_edge.source.field),
                        destination=EdgeConnection(node_id=exec_node_id, field=selected_field),
                    )
                )
            if not attached_edges or any(
                self._state._is_pending_if(candidate_id)
                or (
                    isinstance(self._state.execution_graph.get_node(candidate_id), IfInvocation)
                    and candidate_id not in self._state.executed
                )
                for candidate_id in selected_exec_ids
            ):
                continue

            self._attach_execution_edges(exec_node_id, attached_edges)
            input_edges = self._state.execution_graph._get_input_edges(exec_node_id)
            unmet = sum(1 for edge in input_edges if edge.source.node_id not in self._state.executed)
            self._state._tx_set_mapping(self._state.indegree, exec_node_id, unmet)
            if enqueue:
                self._state._tx_discard_set(self._state._pending_if_exec_nodes, exec_node_id)
            scheduler = self._state._scheduler()
            if self._state._is_generic_graph_scheduler(scheduler):
                scheduler.register_node(exec_node_id)
            if enqueue:
                self._state._enqueue_if_ready(exec_node_id)

    def iterator_graph(self, base: Optional["nx.DiGraph"] = None) -> "nx.DiGraph":
        """Gets a DiGraph with edges to collectors removed so an ancestor search produces all active iterators for any node"""
        g = base.copy() if base is not None else self._state._get_source_graph_flat().copy()
        collectors = (
            n for n in self._state.graph.nodes if isinstance(self._state.graph.get_node(n), CollectInvocation)
        )
        for c in collectors:
            g.remove_edges_from(list(g.in_edges(c)))
        for edge in self._state.graph.edges:
            source_node = self._state.graph.get_node(edge.source.node_id)
            if (
                isinstance(source_node, ForInvocation)
                and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
            ):
                if g.has_edge(edge.source.node_id, edge.destination.node_id):
                    g.remove_edge(edge.source.node_id, edge.destination.node_id)
        return g

    def get_node_iterators(self, node_id: str, it_graph: Optional["nx.DiGraph"] = None) -> list[str]:
        g = it_graph or self.iterator_graph()
        return [
            n
            for n in nx.ancestors(g, node_id)
            if isinstance(self._state.graph.get_node(n), (ForInvocation, IterateInvocation))
        ]

    def _get_prepared_nodes_for_source(self, source_node_id: str) -> set[str]:
        return {
            exec_node_id
            for exec_node_id in self._state.source_prepared_mapping[source_node_id]
            if self._state._get_prepared_exec_metadata(exec_node_id).state != "skipped"
        }

    def _get_parent_iterator_exec_nodes(
        self, source_node_id: str, graph: "nx.DiGraph", prepared_iterator_nodes: list[str]
    ) -> list[tuple[str, str]]:
        iterator_source_node_mapping = [
            (prepared_exec_node_id, self._state.prepared_source_mapping[prepared_exec_node_id])
            for prepared_exec_node_id in prepared_iterator_nodes
        ]
        return [
            iterator_mapping
            for iterator_mapping in iterator_source_node_mapping
            if nx.has_path(graph, iterator_mapping[1], source_node_id)
        ]

    def _matches_parent_iterators(
        self, candidate_exec_node_id: str, parent_iterators: list[tuple[str, str]], execution_graph: "nx.DiGraph"
    ) -> bool:
        return all(
            nx.has_path(execution_graph, parent_iterator_exec_id, candidate_exec_node_id)
            for parent_iterator_exec_id, _ in parent_iterators
        )

    def _get_direct_prepared_iterator_match(
        self,
        prepared_nodes: set[str],
        prepared_iterator_nodes: list[str],
        parent_iterators: list[tuple[str, str]],
        execution_graph: "nx.DiGraph",
    ) -> Optional[str]:
        prepared_iterator = next((node_id for node_id in prepared_iterator_nodes if node_id in prepared_nodes), None)
        if prepared_iterator is None:
            return None
        if self._matches_parent_iterators(prepared_iterator, parent_iterators, execution_graph):
            return prepared_iterator
        return None

    def _find_prepared_node_matching_iterators(
        self, prepared_nodes: set[str], parent_iterators: list[tuple[str, str]], execution_graph: "nx.DiGraph"
    ) -> Optional[str]:
        return next(
            (
                node_id
                for node_id in prepared_nodes
                if self._matches_parent_iterators(node_id, parent_iterators, execution_graph)
            ),
            None,
        )

    def _get_final_for_exec_node(self, prepared_nodes: set[str]) -> Optional[str]:
        prepared_for_nodes = [(node_id, self._state.execution_graph.nodes.get(node_id)) for node_id in prepared_nodes]
        prepared_for_nodes = [
            (node_id, node) for node_id, node in prepared_for_nodes if isinstance(node, ForInvocation)
        ]
        if not prepared_for_nodes:
            return None
        return max(prepared_for_nodes, key=lambda item: item[1].index)[0]

    def get_iteration_node(
        self,
        source_node_id: str,
        graph: "nx.DiGraph",
        execution_graph: "nx.DiGraph",
        prepared_iterator_nodes: list[str],
        prepared_nodes: Optional[set[str]] = None,
    ) -> Optional[str]:
        if prepared_nodes is None:
            prepared_nodes = self._get_prepared_nodes_for_source(source_node_id)
        if len(prepared_nodes) == 1 and not prepared_iterator_nodes:
            return next(iter(prepared_nodes))

        parent_iterators = self._get_parent_iterator_exec_nodes(source_node_id, graph, prepared_iterator_nodes)
        if not parent_iterators and isinstance(self._state.graph.get_node(source_node_id), ForInvocation):
            return self._get_final_for_exec_node(prepared_nodes)
        if len(prepared_nodes) == 1:
            prepared_node_id = next(iter(prepared_nodes))
            if self._matches_parent_iterators(prepared_node_id, parent_iterators, execution_graph):
                return prepared_node_id
            return None

        direct_iterator_match = self._get_direct_prepared_iterator_match(
            prepared_nodes, prepared_iterator_nodes, parent_iterators, execution_graph
        )
        if direct_iterator_match is not None:
            return direct_iterator_match

        return self._find_prepared_node_matching_iterators(prepared_nodes, parent_iterators, execution_graph)

    def prepare(self, base_g: Optional["nx.DiGraph"] = None) -> Optional[str]:
        g = base_g if base_g is not None else self._state._get_source_graph_flat()
        self._attach_pending_if_inputs()
        next_node_id = next(
            (
                node_id
                for node_id in nx.topological_sort(g)
                if node_id not in self._state.source_prepared_mapping
                and node_id not in self._state.executed
                and not (
                    isinstance(self._state.graph.get_node(node_id), ForReturnInvocation)
                    and self._is_deferred_nested_for_return(node_id, g)
                )
                and not self._has_unmaterializable_for_final_input(node_id)
                and (
                    all(
                        edge.source.node_id in self._state.source_prepared_mapping
                        or edge.source.node_id in self._state.executed
                        for edge in self._state.graph._get_input_edges(node_id, "condition")
                    )
                    if isinstance(self._state.graph.get_node(node_id), IfInvocation)
                    else all(
                        source_id in self._state.source_prepared_mapping or source_id in self._state.executed
                        for source_id, _ in g.in_edges(node_id)
                    )
                )
                and (
                    isinstance(self._state.graph.get_node(node_id), IfInvocation)
                    and self._is_if_condition_ready(node_id)
                    or not isinstance(self._state.graph.get_node(node_id), IfInvocation)
                )
                and self._has_admitted_source_mapping(node_id, g)
                and (
                    not isinstance(self._state.graph.get_node(node_id), (ForInvocation, IterateInvocation))
                    or all(source_id in self._state.executed for source_id, _ in g.in_edges(node_id))
                )
                and not any(
                    isinstance(self._state.graph.get_node(ancestor_id), (ForInvocation, IterateInvocation))
                    and ancestor_id not in self._state.executed
                    for ancestor_id in nx.ancestors(g, node_id)
                )
            ),
            None,
        )

        if next_node_id is None:
            return None

        next_node = self._state.graph.get_node(next_node_id)
        new_node_ids: list[str] = []

        if isinstance(next_node, CollectInvocation):
            iteration_mapping_groups = self._get_collect_iteration_mapping_groups(
                self._state.graph._get_input_edges(next_node_id)
            )
            for iteration_path, iteration_mappings in iteration_mapping_groups:
                create_results = self.create_execution_node(next_node_id, iteration_mappings, iteration_path)
                new_node_ids.extend(create_results)
        elif isinstance(next_node, IfInvocation):
            condition_paths = graph_if_runtime._get_if_iteration_paths(
                self._state,
                next_node_id,
                set(),
                self._get_prepared_nodes_for_source,
                self._get_prepared_edge_iteration_path,
            )
            condition_edges = self._state.graph._get_input_edges(next_node_id, "condition")
            if condition_edges:
                iteration_mappings = list(self._get_if_condition_iteration_mappings(next_node_id, g))
                if condition_paths:
                    for iteration_path in condition_paths:
                        matching_mappings = [
                            iteration_mapping
                            for iteration_mapping in iteration_mappings
                            if (
                                (
                                    condition_path := self._get_known_iteration_path(
                                        -1, iteration_mapping, condition_edges
                                    )
                                    or ()
                                )[: len(iteration_path)]
                                == iteration_path
                                or iteration_path[: len(condition_path or ())] == (condition_path or ())
                            )
                        ]
                        for iteration_mapping in matching_mappings:
                            create_results = self.create_execution_node(
                                next_node_id,
                                iteration_mapping,
                                iteration_path,
                                input_fields={"condition"},
                            )
                            new_node_ids.extend(create_results)
                else:
                    for iteration_mapping in iteration_mappings:
                        iteration_path = self._get_known_iteration_path(-1, iteration_mapping, condition_edges)
                        create_results = self.create_execution_node(
                            next_node_id,
                            iteration_mapping,
                            iteration_path,
                            input_fields={"condition"},
                        )
                        new_node_ids.extend(create_results)
            else:
                for iteration_path in condition_paths:
                    create_results = self.create_execution_node(next_node_id, [], iteration_path)
                    new_node_ids.extend(create_results)
        else:
            parent_iterator_nodes = self.get_node_iterators(next_node_id)
            iteration_mappings_iter = (
                self._get_if_condition_iteration_mappings(next_node_id, g)
                if isinstance(next_node, IfInvocation)
                else self._get_parent_iteration_mappings(next_node_id, g)
            )
            for iteration_mappings in iteration_mappings_iter:
                iteration_path = None
                if not parent_iterator_nodes:
                    input_edges = self._state.graph._get_input_edges(next_node_id)
                    iteration_path = max(
                        (
                            self._get_prepared_edge_iteration_path(edge, prepared_id)
                            for source_id, prepared_id in iteration_mappings
                            for edge in input_edges
                            if edge.source.node_id == source_id
                        ),
                        key=lambda path: (len(path), path),
                        default=(),
                    )
                create_results = self.create_execution_node(
                    next_node_id,
                    iteration_mappings,
                    iteration_path,
                    input_fields={"condition"} if isinstance(next_node, IfInvocation) else None,
                )
                new_node_ids.extend(create_results)

        if not new_node_ids:
            # No parent mappings means zero loop contexts, unlike a context with an empty collection.
            self._mark_source_node_empty(next_node_id)
            self._state._invalidate_loop_caches_for_source(next_node_id)
            return next_node_id

        if isinstance(next_node, ForInvocation) and all(
            self._state.execution_graph.get_node(exec_node_id).index == -1 for exec_node_id in new_node_ids
        ):
            self._mark_empty_for_complete(next_node_id)

        return new_node_ids[0]


class _ExecutionMaterializer(_ExecutionNodeBuilder):
    """Compatibility planner facade for unsupported and legacy graphs."""

    pass
