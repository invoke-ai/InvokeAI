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
from invokeai.app.services.shared.execution_engine.primitives import ActivationGate
from invokeai.app.services.shared.graph_iterator_ancestry import _IteratorAncestry
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


class UnsupportedSharedIfCollectBranchesError(ValueError):
    """Raised when one sibling Collect-backed Iterate must serve multiple If owners."""


class _ExecutionNodeBuilder:
    """Builds concrete execution-graph nodes for a selected runtime planner.

    This class contains low-level copy, edge, input, and iteration mechanics. A
    planner owns whether those mechanics are used for generic or compatibility
    execution.
    """

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state
        self._has_if_nodes = any(isinstance(node, IfInvocation) for node in state.graph.nodes.values())
        self._activation_candidate_graph: Optional[nx.DiGraph] = None
        self._activation_candidate_topological_order: tuple[str, ...] = ()
        self._activation_candidate_topological_index: dict[str, int] = {}
        self._activation_sources_by_dependency: dict[tuple[str, str], set[str]] = {}
        self._empty_condition_if_sources: dict[str, tuple[str, ...]] = {}
        self._pending_activation_retirement_candidates: set[str] = set()
        self._activation_retirement_rebuild_required = False
        self._iteration_axes_by_source: dict[str, tuple[str, ...]] = {}
        self._sibling_iterate_if_collect_branches: Optional[dict[str, tuple[str, Optional[str]]]] = None
        self._iteration_relationship_graph: Optional[nx.DiGraph] = None
        self._iteration_relationship_topological_index: Optional[dict[str, int]] = None
        # Keyed by the identity of the cached flat source graph, which every graph edit replaces.
        self._flat_iterator_graph: Optional[tuple["nx.DiGraph", "nx.DiGraph"]] = None
        self._iterator_ancestry_graph: Optional["nx.DiGraph"] = None
        self._iterator_ancestry: Optional[_IteratorAncestry] = None

    def _get_iteration_axes(self, source_node_id: str) -> tuple[str, ...]:
        cached = self._iteration_axes_by_source.get(source_node_id)
        if cached is not None:
            return cached

        axes = self._state._runtime()._get_ordered_iterator_sources(source_node_id)
        if isinstance(self._state.graph.get_node(source_node_id), IterateInvocation):
            collapsed_branch = self._get_sibling_iterate_if_collect_branch(source_node_id)
            if collapsed_branch is not None:
                if_node_id, _ = collapsed_branch
                condition_axes: list[str] = []
                for edge in self._state.graph._get_input_edges(if_node_id, "condition"):
                    condition_source_id = edge.source.node_id
                    condition_axes.extend(self._state._runtime()._get_ordered_iterator_sources(condition_source_id))
                    if isinstance(self._state.graph.get_node(condition_source_id), (ForInvocation, IterateInvocation)):
                        condition_axes.append(condition_source_id)
                axes = list(dict.fromkeys([*condition_axes, *axes]))
            if self._iteration_relationship_graph is not None:
                relationship_ancestors = [
                    iterator_id
                    for iterator_id in nx.ancestors(self._iteration_relationship_graph, source_node_id)
                    if isinstance(self._state.graph.get_node(iterator_id), (ForInvocation, IterateInvocation))
                ]
                relationship_order = self._iteration_relationship_topological_index
                assert relationship_order is not None
                relationship_ancestors.sort(key=relationship_order.__getitem__)
                axes = list(dict.fromkeys([*relationship_ancestors, *axes]))
            axes.append(source_node_id)
        result = tuple(axes)
        self._iteration_axes_by_source[source_node_id] = result
        return result

    def _get_ordered_input_iterator_axes(self, input_edges: list[Edge]) -> tuple[str, ...]:
        if not input_edges:
            return ()

        destination = self._state.graph.get_node(input_edges[0].destination.node_id)
        input_field_order = {field: index for index, field in enumerate(type(destination).model_fields)}
        ordered_edges = sorted(
            input_edges,
            key=lambda edge: (
                input_field_order.get(edge.destination.field, len(input_field_order)),
                edge.source.node_id,
            ),
        )
        axes: list[str] = []
        for edge in ordered_edges:
            for axis_id in self._get_iteration_axes(edge.source.node_id):
                if axis_id not in axes:
                    axes.append(axis_id)
        return tuple(axes)

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
        if compatible_parent_contexts:
            iteration_path = deepest_path
        else:
            # Independent input axes contribute separate path components. Merge their indexes in destination-port
            # order instead of rejecting them as incompatible prefixes.
            ordered_axes = self._get_ordered_input_iterator_axes(input_edges or [])
            axis_indexes: dict[str, int] = {}
            for path, axes in parent_contexts:
                if len(path) > len(axes):
                    return None
                for axis_id, axis_index in zip(axes, path, strict=False):
                    if axis_id not in ordered_axes:
                        return None
                    previous_index = axis_indexes.setdefault(axis_id, axis_index)
                    if previous_index != axis_index:
                        return None
            populated_positions = [index for index, axis_id in enumerate(ordered_axes) if axis_id in axis_indexes]
            if populated_positions != list(range(len(populated_positions))):
                return None
            iteration_path = tuple(axis_indexes[axis_id] for axis_id in ordered_axes[: len(populated_positions)])

        # Materialized iteration boundaries use non-negative indexes; ordinary execution nodes use -1. Keeping this
        # generic allows other scheduler-managed loop nodes to reuse the same path cache.
        if iteration_index >= 0:
            return (*iteration_path, iteration_index)
        return iteration_path

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

    def _ensure_activation_candidate_index(self, graph: "nx.DiGraph") -> None:
        if self._activation_candidate_graph is graph:
            return

        self._activation_candidate_graph = graph
        self._has_if_nodes = any(isinstance(node, IfInvocation) for node in self._state.graph.nodes.values())
        self._activation_candidate_topological_order = ()
        self._activation_candidate_topological_index = {}
        self._activation_sources_by_dependency = {}
        self._empty_condition_if_sources = {}
        self._pending_activation_retirement_candidates.clear()
        if not self._has_if_nodes:
            self._activation_retirement_rebuild_required = False
            return

        self._activation_candidate_topological_order = tuple(nx.topological_sort(graph))
        self._activation_candidate_topological_index = {
            source_node_id: index for index, source_node_id in enumerate(self._activation_candidate_topological_order)
        }
        self._activation_sources_by_dependency = {}
        self._empty_condition_if_sources = {}
        self._pending_activation_retirement_candidates.clear()
        for source_node_id in self._activation_candidate_topological_order:
            for dependency in self._state._get_source_activation_dependencies(source_node_id):
                self._activation_sources_by_dependency.setdefault((dependency.owner_id, dependency.branch), set()).add(
                    source_node_id
                )
        for if_node_id in graph.nodes:
            if not isinstance(self._state.graph.get_node(if_node_id), IfInvocation):
                continue
            condition_edges = self._state.graph._get_input_edges(if_node_id, "condition")
            if len(condition_edges) != 1:
                continue
            condition_source_id = condition_edges[0].source.node_id
            if isinstance(self._state.graph.get_node(condition_source_id), IterateInvocation):
                self._empty_condition_if_sources.setdefault(condition_source_id, ())
                self._empty_condition_if_sources[condition_source_id] += (if_node_id,)

        self._activation_retirement_rebuild_required = True
        self._rebuild_activation_retirement_candidates()

    def _rebuild_activation_retirement_candidates(self) -> None:
        self._pending_activation_retirement_candidates.clear()
        registry = self._state._prepared_registry()
        opposite_branch = {"true_input": "false_input", "false_input": "true_input"}
        for exec_node_id, selected_branch in self._state._resolved_if_exec_branches.items():
            if selected_branch not in opposite_branch:
                continue
            owner_id = registry.get_source_node_id(exec_node_id)
            self._pending_activation_retirement_candidates.update(
                self._activation_sources_by_dependency.get((owner_id, opposite_branch[selected_branch]), ())
            )
        for condition_source_id, if_node_ids in self._empty_condition_if_sources.items():
            if condition_source_id not in self._state.executed or self._state.source_prepared_mapping.get(
                condition_source_id
            ):
                continue
            for if_node_id in if_node_ids:
                self._pending_activation_retirement_candidates.update(
                    self._activation_sources_by_dependency.get((if_node_id, "true_input"), ())
                )
                self._pending_activation_retirement_candidates.update(
                    self._activation_sources_by_dependency.get((if_node_id, "false_input"), ())
                )
        self._activation_retirement_rebuild_required = False

    def _queue_rejected_activation_branch(self, if_node_id: str, selected_branch: str) -> None:
        opposite_branch = {"true_input": "false_input", "false_input": "true_input"}.get(selected_branch)
        if opposite_branch is None:
            return
        self._ensure_activation_candidate_index(self._state._get_source_graph_flat())
        for source_node_id in self._activation_sources_by_dependency.get((if_node_id, opposite_branch), ()):
            self._state._tx_add_set(self._pending_activation_retirement_candidates, source_node_id)

    def _queue_empty_condition_retirements(self, condition_source_id: str) -> None:
        self._ensure_activation_candidate_index(self._state._get_source_graph_flat())
        for if_node_id in self._empty_condition_if_sources.get(condition_source_id, ()):
            for branch in ("true_input", "false_input"):
                for source_node_id in self._activation_sources_by_dependency.get((if_node_id, branch), ()):
                    self._state._tx_add_set(self._pending_activation_retirement_candidates, source_node_id)

    def _get_activation_candidate_source_ids(self, graph: "nx.DiGraph") -> tuple[str, ...]:
        """Return queued retirement candidates in source-graph order."""
        self._ensure_activation_candidate_index(graph)
        if self._activation_retirement_rebuild_required:
            self._rebuild_activation_retirement_candidates()
        return tuple(
            sorted(
                self._pending_activation_retirement_candidates,
                key=self._activation_candidate_topological_index.__getitem__,
            )
        )

    def _has_unmaterializable_for_final_input(self, node_id: str) -> bool:
        final_for_source_ids = set()
        for edge in self._state.graph._get_input_edges(node_id):
            source_node = self._state.graph.get_node(edge.source.node_id)
            if not isinstance(source_node, ForInvocation):
                continue
            if get_output_field_scope(source_node, edge.source.field) == OutputScope.Final:
                final_for_source_ids.add(edge.source.node_id)

        return any(not self._state._all_for_contexts_finalized(source_for_id) for source_for_id in final_for_source_ids)

    def _get_ordered_iterator_axes_for_edges(self, edges: Iterable[Edge], iterator_graph: "nx.DiGraph") -> list[str]:
        ancestry = self._get_iterator_ancestry(iterator_graph)
        return list(dict.fromkeys(axis_id for edge in edges for axis_id in ancestry.get_iterator_axes_for_edge(edge)))

    def _get_iterator_ancestry(self, iterator_graph: "nx.DiGraph") -> _IteratorAncestry:
        if self._iterator_ancestry_graph is not iterator_graph:
            self._iterator_ancestry_graph = iterator_graph
            self._iterator_ancestry = _IteratorAncestry(self._state, iterator_graph=iterator_graph)
        assert self._iterator_ancestry is not None
        return self._iterator_ancestry

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
            input_edges = self._state.graph._get_input_edges(source_node_id)
            if isinstance(node, IfInvocation):
                input_edges = [edge for edge in input_edges if edge.destination.field == "condition"]
            for edge in input_edges:
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

    def _get_collect_source_iterator_ids(
        self, source_node_id: str, iterator_graph: Optional["nx.DiGraph"] = None
    ) -> list[str]:
        iterator_node_ids = self.get_node_iterators(source_node_id, iterator_graph)
        if isinstance(self._state.graph.get_node(source_node_id), IterateInvocation):
            iterator_node_ids.append(source_node_id)
        return iterator_node_ids

    def _get_ordered_prepared_nodes_for_source(self, source_node_id: str) -> list[str]:
        return sorted(
            self._get_prepared_nodes_for_source(source_node_id),
            key=lambda exec_node_id: (self._state._get_iteration_path(exec_node_id), exec_node_id),
        )

    def _prepared_node_has_edge_output(self, edge: Edge, prepared_id: str) -> bool:
        source_node = self._state.graph.get_node(edge.source.node_id)
        if not (
            isinstance(source_node, ForInvocation)
            and get_output_field_scope(source_node, edge.source.field) == OutputScope.Iteration
        ):
            return True
        prepared_node = self._state.execution_graph.get_node(prepared_id)
        return isinstance(prepared_node, ForInvocation) and prepared_node.index >= 0

    def _get_ordered_prepared_nodes_for_edge(self, edge: Edge) -> list[str]:
        prepared_nodes = [
            prepared_id
            for prepared_id in self._get_ordered_prepared_nodes_for_source(edge.source.node_id)
            if self._prepared_node_has_edge_output(edge, prepared_id)
        ]
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
        # A Collect.item edge groups by the outer `group_depth` axes. Read input paths from the current item axis
        # too: an empty stream still carries its parent groups even when it has no prepared items.
        for axis_index, iterator_node_id in enumerate(iterator_node_ids[: group_depth + 1]):
            prepared_nodes = self._get_ordered_prepared_nodes_for_source(iterator_node_id)
            # Only outer group axes contribute prepared paths; the current item axis is collapsed by Collect.
            if axis_index < group_depth and prepared_nodes:
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
        edge = input_edges[0]
        for prepared_id in prepared_nodes:
            if not self._prepared_node_has_edge_output(edge, prepared_id):
                continue
            iteration_path = self._get_prepared_edge_iteration_path(edge, prepared_id)
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

    def _matches_sibling_if_collect_context(
        self, iterator_nodes: list[str], prepared_iterators: tuple[str, ...]
    ) -> bool:
        """Match a branch Collect's sibling iterator to the selected If parent frame."""
        for index, iterator_node_id in enumerate(iterator_nodes):
            collapsed_branch = self._get_sibling_iterate_if_collect_branch(iterator_node_id)
            if collapsed_branch is None:
                continue
            if_node_id, branch_field = collapsed_branch
            branch_path = self._state._get_iteration_path(prepared_iterators[index])
            matching_frame = branch_path[:-1] if branch_path else ()
            matching_frames = (
                (matching_frame,)
                if len(branch_path) > 0
                and graph_if_runtime._has_selected_if_frame_for_preparation(
                    self._state, if_node_id, branch_field, matching_frame
                )
                else ()
            )
            condition_edges = self._state.graph._get_input_edges(if_node_id, "condition")
            condition_iterator_ids = set(
                self._get_ordered_iterator_axes_for_edges(condition_edges, self.iterator_graph())
            )
            for frame in matching_frames:
                if any(
                    self._state._get_iteration_path(prepared_iterators[iterator_nodes.index(condition_iterator_id)])
                    != frame
                    for condition_iterator_id in condition_iterator_ids
                    if condition_iterator_id in iterator_nodes
                ):
                    return False
            if not matching_frames:
                return False
        return True

    def _get_parent_iteration_mappings(
        self,
        next_node_id: str,
        graph: "nx.DiGraph",
        input_edges: Optional[list[Edge]] = None,
        iterator_node_ids: Optional[list[str]] = None,
    ) -> Iterable[list[tuple[str, str]]]:
        input_edges = input_edges or self._state.graph._get_input_edges(next_node_id)
        parent_node_ids = list(dict.fromkeys(edge.source.node_id for edge in input_edges))
        iterator_graph = self.iterator_graph(graph)
        iterator_nodes = (
            iterator_node_ids
            if iterator_node_ids is not None
            else self.get_node_iterators(next_node_id, iterator_graph)
        )
        if isinstance(self._state.graph.get_node(next_node_id), CollectInvocation) and len(input_edges) == 1:
            source_node = self._state.graph.get_node(input_edges[0].source.node_id)
            if isinstance(source_node, IfInvocation):
                condition_edges = self._state.graph._get_input_edges(source_node.id, "condition")
                condition_axes = self._get_ordered_iterator_axes_for_edges(condition_edges, iterator_graph)
                condition_iterator_ids = set(condition_axes)
                if condition_iterator_ids:
                    iterator_nodes = [node_id for node_id in iterator_nodes if node_id in condition_iterator_ids]
        if not iterator_nodes:
            return iter(self._get_parent_iteration_mappings_without_iterators(next_node_id, input_edges))

        iterator_nodes_prepared = [
            sorted(self._state.source_prepared_mapping[node_id], key=self._state._get_iteration_path)
            for node_id in iterator_nodes
        ]
        prepared_nodes_by_source = {
            node_id: {
                prepared_id
                for prepared_id in self._get_prepared_nodes_for_source(node_id)
                if all(
                    self._prepared_node_has_edge_output(edge, prepared_id)
                    for edge in input_edges
                    if edge.source.node_id == node_id
                )
            }
            for node_id in parent_node_ids
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
                if not self._matches_sibling_if_collect_context(iterator_nodes, prepared_iterators):
                    continue
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

    def _has_admitted_source_mapping(
        self, node_id: str, graph: "nx.DiGraph", iterator_ancestry: _IteratorAncestry
    ) -> bool:
        if not self._has_if_nodes:
            return True
        if isinstance(self._state.graph.get_node(node_id), IfInvocation):
            condition_edges = self._state.graph._get_input_edges(node_id, "condition")
            if not condition_edges:
                iteration_paths = graph_if_runtime._get_if_iteration_paths_with_ancestry(
                    self._state,
                    node_id,
                    set(),
                    self._get_prepared_nodes_for_source,
                    self._get_prepared_edge_iteration_path,
                    iterator_ancestry,
                )
                if not iteration_paths:
                    # An empty Iterate giving the If its frames leaves it no executions; preparing it closes it empty.
                    return graph_if_runtime._has_empty_if_frame(self._state, node_id)
                return any(
                    self._state._is_source_activation_admitted(node_id, iteration_path)
                    for iteration_path in iteration_paths
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
                iterator_nodes = self._get_if_frame_iterator_nodes(node_id, self.iterator_graph(graph))
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

    def _get_if_frame_iterator_nodes(self, node_id: str, iterator_graph: "nx.DiGraph") -> list[str]:
        """Return the If's branch iterators, reduced to the innermost one when they are nested.

        Each frame of the innermost nested axis also fixes the outer indexes, so that axis alone maps the If's frames.
        """
        iterator_nodes = self.get_node_iterators(node_id, iterator_graph)
        iterator_ancestry = self._get_iterator_ancestry(iterator_graph)
        if all(
            outer_id in iterator_ancestry.get_ordered_iterator_ancestors(inner_id)
            for outer_id, inner_id in itertools.pairwise(iterator_nodes)
        ):
            return iterator_nodes[-1:]
        return iterator_nodes

    def _get_if_condition_iteration_mappings(
        self, node_id: str, graph: "nx.DiGraph"
    ) -> Iterable[list[tuple[str, str]]]:
        condition_edges = self._state.graph._get_input_edges(node_id, "condition")
        iterator_graph = self.iterator_graph(graph)
        if condition_edges:
            condition_axes = self._get_ordered_iterator_axes_for_edges(condition_edges, iterator_graph)
            condition_iterator_ids = set(condition_axes)
            if condition_iterator_ids:
                iterator_nodes = condition_axes
            else:
                iterator_nodes = self.get_node_iterators(node_id, iterator_graph)
        else:
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
            return self._get_parent_iteration_mappings(
                node_id, graph, input_edges=condition_edges, iterator_node_ids=iterator_nodes
            )

        # This specialization derives one branch axis against a root-frame condition. Mixed or nested condition
        # axes use the general parent-mapping path.
        iterator_ancestry = self._get_iterator_ancestry(iterator_graph)
        iterator_nodes = self._get_if_frame_iterator_nodes(node_id, iterator_graph)
        if len(iterator_nodes) != 1:
            return iter([[]])
        for branch_field in ("true_input", "false_input"):
            branch_edges = self._state.graph._get_input_edges(node_id, branch_field)
            branch_iterator_ids = {
                axis_id for edge in branch_edges for axis_id in iterator_ancestry.get_iterator_axes_for_edge(edge)
            }
            if branch_iterator_ids and not (branch_iterator_ids & set(iterator_nodes)):
                return iter([])

        if any(iterator_node_id not in self._state.source_prepared_mapping for iterator_node_id in iterator_nodes):
            return iter([])
        iterator_nodes_prepared = [
            sorted(self._state.source_prepared_mapping[iterator_node_id], key=self._state._get_iteration_path)
            for iterator_node_id in iterator_nodes
        ]

        if condition_edges:
            condition_mappings = self._get_parent_iteration_mappings(
                node_id, graph, input_edges=condition_edges, iterator_node_ids=iterator_nodes
            )
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
        for edge in self._state.graph._get_input_edges(node_id, "condition"):
            source_node_id = edge.source.node_id
            if source_node_id in self._state.executed and not self._state.source_prepared_mapping.get(source_node_id):
                continue
            prepared_nodes = self._get_ordered_prepared_nodes_for_edge(edge)
            if not prepared_nodes or any(prepared_id not in self._state.results for prepared_id in prepared_nodes):
                return False
        return True

    def _get_if_readiness_ancestors(self, node_id: str, graph: "nx.DiGraph") -> set[str]:
        """Get iterator ancestors that must finish before preparing an If execution.

        A wired condition gates the If through its ancestry. A literal condition's known selected branch supplies the
        value; an iterator only on the unselected branch is gated by the If and must not hold it back. A wired If also
        waits for the iterators in its own ancestry: they give it its frames and are never gated by it, and preparing
        the If before they finish would find no frames and close it empty.
        """
        condition_edges = self._state.graph._get_input_edges(node_id, "condition")
        if condition_edges:
            gating_source_ids = [
                *(edge.source.node_id for edge in condition_edges),
                *self.get_node_iterators(node_id, self.iterator_graph(graph)),
            ]
        else:
            node = self._state.graph.get_node(node_id)
            assert isinstance(node, IfInvocation)
            selected_field = "true_input" if node.condition else "false_input"
            gating_source_ids = [
                edge.source.node_id for edge in self._state.graph._get_input_edges(node_id, selected_field)
            ]
        ancestors: set[str] = set()
        for source_node_id in gating_source_ids:
            ancestors.update(nx.ancestors(graph, source_node_id))
            ancestors.add(source_node_id)
        return ancestors

    def _get_sibling_iterate_if_collect_branches(self) -> dict[str, tuple[str, Optional[str]]]:
        """Find Iterate axes collapsed by branch Collects under an If."""
        if self._sibling_iterate_if_collect_branches is not None:
            return self._sibling_iterate_if_collect_branches

        branches: dict[str, tuple[str, Optional[str]]] = {}
        if not any(isinstance(node, CollectInvocation) for node in self._state.graph.nodes.values()):
            self._sibling_iterate_if_collect_branches = branches
            return branches

        graph = self._state._get_source_graph_flat()
        iterator_graph = self.iterator_graph(graph)
        iterator_ancestry = self._get_iterator_ancestry(iterator_graph)
        relationship_graph = iterator_graph.copy()
        for collect_node in self._state.graph.nodes.values():
            if not isinstance(collect_node, CollectInvocation):
                continue
            for collection_edge in self._state.graph._get_input_edges(collect_node.id, COLLECTION_FIELD):
                relationship_graph.add_edge(collection_edge.source.node_id, collect_node.id)
            item_edges = self._state.graph._get_input_edges(collect_node.id, ITEM_FIELD)
            if len(item_edges) != 1:
                continue
            item_edge = item_edges[0]
            item_source_id = item_edge.source.node_id
            source_node = self._state.graph.get_node(item_source_id)
            if (
                isinstance(source_node, ForInvocation)
                and get_output_field_scope(source_node, item_edge.source.field) == OutputScope.Final
            ):
                continue
            ordered_iterator_ids = list(iterator_ancestry.get_iterator_axes_for_edge(item_edge))
            for parent_iterator_id in ordered_iterator_ids[:-1]:
                relationship_graph.add_edge(parent_iterator_id, collect_node.id)

        for collect_node in self._state.graph.nodes.values():
            if not isinstance(collect_node, CollectInvocation):
                continue
            item_edges = self._state.graph._get_input_edges(collect_node.id, ITEM_FIELD)
            if len(item_edges) != 1:
                continue
            iterator_ids = set(self._get_collect_source_iterator_ids(item_edges[0].source.node_id, iterator_graph))
            if len(iterator_ids) != 2:
                continue

            for output_edge in self._state.graph._get_output_edges(collect_node.id):
                destination = self._state.graph.get_node(output_edge.destination.node_id)
                if not isinstance(destination, IfInvocation) or output_edge.destination.field not in {
                    "true_input",
                    "false_input",
                }:
                    continue

                condition_edges = self._state.graph._get_input_edges(destination.id, "condition")
                condition_iterator_ids = set(self._get_ordered_iterator_axes_for_edges(condition_edges, iterator_graph))
                if not condition_iterator_ids.intersection(iterator_ids):
                    continue

                for iterate_node_id in iterator_ids - condition_iterator_ids:
                    if any(
                        nx.has_path(relationship_graph, condition_iterator_id, iterate_node_id)
                        or nx.has_path(relationship_graph, iterate_node_id, condition_iterator_id)
                        for condition_iterator_id in condition_iterator_ids
                    ):
                        continue
                    branch = (destination.id, output_edge.destination.field)
                    existing_branch = branches.get(iterate_node_id)
                    if existing_branch is not None and existing_branch[0] != branch[0]:
                        raise UnsupportedSharedIfCollectBranchesError(
                            f"Sibling Iterate '{iterate_node_id}' cannot supply Collect-backed inputs to multiple "
                            f"If nodes: '{existing_branch[0]}' and "
                            f"'{branch[0]}.{branch[1]}'"
                        )
                    branches[iterate_node_id] = (
                        branch[0],
                        None if existing_branch is not None and existing_branch[1] != branch[1] else branch[1],
                    )

        self._iteration_relationship_graph = relationship_graph
        # Cached axes may have been queried before this graph established sibling order.
        self._iteration_axes_by_source.clear()
        self._iteration_relationship_topological_index = {
            node_id: index
            for index, node_id in enumerate(nx.lexicographical_topological_sort(relationship_graph, key=str))
        }
        self._sibling_iterate_if_collect_branches = branches
        return branches

    def _get_sibling_iterate_if_collect_branch(self, iterate_node_id: str) -> Optional[tuple[str, Optional[str]]]:
        return self._get_sibling_iterate_if_collect_branches().get(iterate_node_id)

    def _attach_pending_if_inputs(self, *, enqueue: bool = True) -> None:
        """Attach only the selected branch edge to condition-ready If executions."""

        source_graph = self._state._get_source_graph_flat()
        # Prepared branch sources do not change during this sweep. Index them by frame once per source edge;
        # keeping the index local means graph edits, rollback, and restore rebuild it on the next call.
        prepared_exec_by_edge_path: dict[tuple[str, str], dict[tuple[int, ...], str]] = {}
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
                edge_key = (source_edge.source.node_id, source_edge.source.field)
                if edge_key not in prepared_exec_by_edge_path:
                    prepared_exec_by_path: dict[tuple[int, ...], str] = {}
                    for prepared_id in self._get_ordered_prepared_nodes_for_edge(source_edge):
                        prepared_path = self._get_prepared_edge_iteration_path(source_edge, prepared_id)
                        prepared_exec_by_path.setdefault(prepared_path, prepared_id)
                    prepared_exec_by_edge_path[edge_key] = prepared_exec_by_path

                prepared_exec_by_path = prepared_exec_by_edge_path[edge_key]
                selected_exec_id = next(
                    (
                        prepared_exec_by_path[target_path[:path_depth]]
                        for path_depth in range(len(target_path), -1, -1)
                        if target_path[:path_depth] in prepared_exec_by_path
                    ),
                    None,
                )
                if selected_exec_id is None:
                    continue
                selected_exec_ids.append(selected_exec_id)
                attached_edges.append(
                    Edge(
                        source=EdgeConnection(node_id=selected_exec_id, field=source_edge.source.field),
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
        """Gets a DiGraph with collector edges removed for iterator ancestry searches.

        The result for the flat source graph is shared; callers must not mutate it.
        """
        source_graph = self._state._get_source_graph_flat()
        if base is not None and base is not source_graph:
            return self._build_iterator_graph(base)
        if self._flat_iterator_graph is None or self._flat_iterator_graph[0] is not source_graph:
            self._flat_iterator_graph = (source_graph, self._build_iterator_graph(source_graph))
        return self._flat_iterator_graph[1]

    def _build_iterator_graph(self, base: "nx.DiGraph") -> "nx.DiGraph":
        g = base.copy()
        collectors = [
            n
            for n in self._state.graph.nodes
            if isinstance(self._state.graph.get_node(n), CollectInvocation)
            if n in base
        ]
        for c in collectors:
            collection_sources = {
                edge.source.node_id
                for edge in self._state.graph._get_input_edges(c, COLLECTION_FIELD)
                if base.has_edge(edge.source.node_id, c)
            }
            g.remove_edges_from(list(g.in_edges(c)))
            g.add_edges_from((source_id, c) for source_id in collection_sources)
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
        iterator_graph = it_graph if it_graph is not None else self.iterator_graph()
        return self._find_node_iterators(node_id, iterator_graph)

    def _find_node_iterators(self, node_id: str, it_graph: "nx.DiGraph") -> list[str]:
        return self._get_iterator_ancestry(it_graph).get_ordered_iterator_ancestors(node_id)

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

    def _should_defer_sibling_if_collect_iterate(self, node_id: str) -> bool:
        if not isinstance(self._state.graph.get_node(node_id), IterateInvocation):
            return False
        collapsed_branch = self._get_sibling_iterate_if_collect_branch(node_id)
        if collapsed_branch is None:
            return False
        if graph_if_runtime._get_selected_if_frames_for_preparation(self._state, *collapsed_branch):
            return False

        if_node_id, _ = collapsed_branch
        if_exec_ids = self._state.source_prepared_mapping.get(if_node_id, set())
        return not if_exec_ids or not all(exec_id in self._state._resolved_if_exec_branches for exec_id in if_exec_ids)

    def _expand_root_if_for_completed_collect_branch(self, graph: "nx.DiGraph") -> Optional[str]:
        """Align a root-conditioned If frame with each completed selected Collect frame."""
        iterator_graph = self.iterator_graph(graph)
        for if_node_id, node in self._state.graph.nodes.items():
            if not isinstance(node, IfInvocation):
                continue
            condition_edges = self._state.graph._get_input_edges(if_node_id, "condition")
            if (
                not condition_edges
                or any(edge.source.node_id not in iterator_graph for edge in condition_edges)
                or any(
                    self._get_iterator_ancestry(iterator_graph).get_iterator_axes_for_edge(edge)
                    for edge in condition_edges
                )
            ):
                continue

            root_exec_ids = [
                exec_id
                for exec_id in self._state.source_prepared_mapping.get(if_node_id, ())
                if self._state._get_iteration_path(exec_id) == ()
                and self._state._is_pending_if(exec_id)
                and (selected_field := self._state._resolved_if_exec_branches.get(exec_id)) is not None
            ]
            if len(root_exec_ids) != 1:
                continue
            root_exec_id = root_exec_ids[0]
            selected_field = self._state._resolved_if_exec_branches[root_exec_id]
            selected_edges = self._state.graph._get_input_edges(if_node_id, selected_field)
            if not selected_edges or any(
                not isinstance(self._state.graph.get_node(edge.source.node_id), CollectInvocation)
                or edge.source.node_id not in self._state.executed
                or any(
                    prepared_id not in self._state.results
                    for prepared_id in self._state.source_prepared_mapping.get(edge.source.node_id, ())
                )
                for edge in selected_edges
            ):
                continue

            branch_paths = sorted(
                {
                    self._state._get_iteration_path(prepared_id)
                    for edge in selected_edges
                    for prepared_id in self._state.source_prepared_mapping.get(edge.source.node_id, ())
                }
            )
            if not branch_paths or branch_paths[0] == ():
                continue

            existing_paths = {
                self._state._get_iteration_path(exec_id)
                for exec_id in self._state.source_prepared_mapping.get(if_node_id, ())
            }
            if existing_paths != {()}:
                continue

            condition_mapping = [
                (self._state.prepared_source_mapping[edge.source.node_id], edge.source.node_id)
                for edge in self._state.execution_graph._get_input_edges(root_exec_id)
                if edge.destination.field == "condition"
            ]
            if not condition_mapping:
                continue

            return self._state._run_materialization_transaction(
                lambda graph=graph,
                if_node_id=if_node_id,
                root_exec_id=root_exec_id,
                condition_mapping=condition_mapping,
                branch_paths=branch_paths: self._materialize_root_if_for_completed_collect_branch(
                    graph, if_node_id, root_exec_id, condition_mapping, branch_paths
                )
            )

        return None

    def _materialize_root_if_for_completed_collect_branch(
        self,
        graph: "nx.DiGraph",
        if_node_id: str,
        root_exec_id: str,
        condition_mapping: list[tuple[str, str]],
        branch_paths: list[tuple[int, ...]],
    ) -> str:
        """Expand resolved root If state into its completed Collect frames atomically."""
        # The root If already resolved the branch gate so its iterators could run. Reuse that
        # execution as the first selected frame, then create one If execution for every other
        # closed Collect frame. Keep the durable activation token and generic gate frame aligned.
        first_path, *remaining_paths = branch_paths
        self._state._prepared_registry().set_iteration_path(root_exec_id, first_path)
        frame = self._state._get_execution_frame(root_exec_id)
        engine_frame = self._state._engine_frame(first_path)
        generic_runtime = self._state._generic_runtime()
        gate = generic_runtime.gates.get(root_exec_id)
        if gate is not None:
            previous_gate = ActivationGate.model_validate(gate.model_dump(mode="python"))
            generic_runtime.replace_gate(
                ActivationGate(
                    gate_id=gate.gate_id,
                    owner_id=gate.owner_id,
                    frame=engine_frame,
                    branches=gate.branches,
                    status=gate.status,
                    selected_branch=gate.selected_branch,
                )
            )
            self._state._tx_record(lambda gate=previous_gate, runtime=generic_runtime: runtime.replace_gate(gate))
        for token_id, token in tuple(self._state.execution_tokens.items()):
            if token.owner_node_id == root_exec_id and token.token_kind == "activation":
                self._state._tx_set_mapping(
                    self._state.execution_tokens,
                    token_id,
                    token.model_copy(update={"frame": frame}),
                )
        self._state._invalidate_selected_if_frames_for_preparation(if_node_id)
        self._state._prepared_if_exec_indexed_sources.discard(if_node_id)
        for key in tuple(self._state._prepared_if_exec_ids_by_source_and_frame):
            if key[0] == if_node_id:
                self._state._prepared_if_exec_ids_by_source_and_frame.pop(key, None)
        self._state._prepared_if_exec_frames_by_source.pop(if_node_id, None)

        for iteration_path in remaining_paths:
            self.create_execution_node(
                if_node_id,
                condition_mapping,
                iteration_path,
                input_fields={"condition"},
            )

        for consumer_id in graph.successors(if_node_id):
            if isinstance(
                self._state.graph.get_node(consumer_id),
                (IfInvocation, CollectInvocation, ForInvocation, IterateInvocation),
            ):
                continue
            consumer_exec_ids = self._state.source_prepared_mapping.get(consumer_id, ())
            root_consumer_ids = [
                exec_id
                for exec_id in consumer_exec_ids
                if self._state._get_iteration_path(exec_id) == () and exec_id not in self._state.executed
            ]
            if len(root_consumer_ids) != 1:
                continue
            root_consumer_id = root_consumer_ids[0]
            input_edges = self._state.execution_graph._get_input_edges(root_consumer_id)
            if not any(
                self._state.prepared_source_mapping.get(edge.source.node_id) == if_node_id for edge in input_edges
            ):
                continue

            self._state._prepared_registry().set_iteration_path(root_consumer_id, first_path)
            for iteration_path in remaining_paths:
                iteration_mapping = []
                for edge in self._state.graph._get_input_edges(consumer_id):
                    prepared_sources = self._state.source_prepared_mapping.get(edge.source.node_id, ())
                    matching_source = next(
                        (
                            self._state._prepared_registry().get_prepared_id_at_path(
                                edge.source.node_id, iteration_path[:path_depth]
                            )
                            for path_depth in range(len(iteration_path), -1, -1)
                            if self._state._prepared_registry().get_prepared_id_at_path(
                                edge.source.node_id, iteration_path[:path_depth]
                            )
                            is not None
                        ),
                        None,
                    )
                    if matching_source is None and prepared_sources:
                        break
                    if matching_source is not None:
                        iteration_mapping.append((edge.source.node_id, matching_source))
                else:
                    self.create_execution_node(consumer_id, iteration_mapping, iteration_path)

        return root_exec_id

    def _get_unexecuted_root_collect_if_descendants(self, graph: "nx.DiGraph") -> set[str]:
        """Wait to materialize descendants until selected root Collect frames finish their If copies."""
        iterator_graph = self.iterator_graph(graph)
        iterator_ancestry = self._get_iterator_ancestry(iterator_graph)
        deferred_node_ids: set[str] = set()
        for if_node_id, node in self._state.graph.nodes.items():
            if if_node_id not in graph or not isinstance(node, IfInvocation):
                continue
            condition_edges = self._state.graph._get_input_edges(if_node_id, "condition")
            if (
                not condition_edges
                or any(edge.source.node_id not in iterator_graph for edge in condition_edges)
                or any(iterator_ancestry.get_iterator_axes_for_edge(edge) for edge in condition_edges)
            ):
                continue
            prepared_if_ids = self._state.source_prepared_mapping.get(if_node_id, ())
            if not prepared_if_ids or all(exec_id in self._state.executed for exec_id in prepared_if_ids):
                continue
            selected_fields = {
                self._state._resolved_if_exec_branches.get(exec_id)
                for exec_id in prepared_if_ids
                if exec_id in self._state._resolved_if_exec_branches
            }
            if not selected_fields:
                continue
            if any(
                not self._state.graph._get_input_edges(if_node_id, selected_field)
                or any(
                    not isinstance(self._state.graph.get_node(edge.source.node_id), CollectInvocation)
                    for edge in self._state.graph._get_input_edges(if_node_id, selected_field)
                )
                for selected_field in selected_fields
            ):
                continue
            deferred_node_ids.update(nx.descendants(graph, if_node_id))
        return deferred_node_ids

    def prepare(self, base_g: Optional["nx.DiGraph"] = None) -> Optional[str]:
        g = base_g if base_g is not None else self._state._get_source_graph_flat()
        self._get_sibling_iterate_if_collect_branches()
        iterator_ancestry = self._get_iterator_ancestry(self.iterator_graph(g))
        iterator_ancestry.clear_effective_iteration_paths()
        self._attach_pending_if_inputs()
        expanded_if_id = self._expand_root_if_for_completed_collect_branch(g)
        if expanded_if_id is not None:
            return expanded_if_id
        self._attach_pending_if_inputs()
        self._ensure_activation_candidate_index(g)
        deferred_root_collect_if_descendants = self._get_unexecuted_root_collect_if_descendants(g)
        if self._has_if_nodes:
            try:
                for node_id in self._get_activation_candidate_source_ids(g):
                    self._state._tx_discard_set(self._pending_activation_retirement_candidates, node_id)
                    if (
                        node_id not in self._state.source_prepared_mapping
                        and node_id not in self._state.executed
                        and self._state._is_source_inactive(node_id)
                    ):
                        self._mark_source_node_empty(node_id)
                        self._state._invalidate_loop_caches_for_source(node_id)
                        return node_id
            except Exception:
                self._activation_retirement_rebuild_required = True
                raise
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
                and node_id not in deferred_root_collect_if_descendants
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
                and self._has_admitted_source_mapping(node_id, g, iterator_ancestry)
                and not self._should_defer_sibling_if_collect_iterate(node_id)
                and (
                    not isinstance(self._state.graph.get_node(node_id), (ForInvocation, IterateInvocation))
                    or all(source_id in self._state.executed for source_id, _ in g.in_edges(node_id))
                )
                and not any(
                    isinstance(self._state.graph.get_node(ancestor_id), (ForInvocation, IterateInvocation))
                    and ancestor_id not in self._state.executed
                    for ancestor_id in (
                        self._get_if_readiness_ancestors(node_id, g)
                        if isinstance(self._state.graph.get_node(node_id), IfInvocation)
                        else nx.ancestors(g, node_id)
                    )
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
            condition_paths = graph_if_runtime._get_if_iteration_paths_with_ancestry(
                self._state,
                next_node_id,
                set(),
                self._get_prepared_nodes_for_source,
                self._get_prepared_edge_iteration_path,
                iterator_ancestry,
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
                elif not self.get_node_iterators(next_node_id):
                    # Root-frame If nodes may be prepared from their condition while branch sources
                    # are still being materialized; branch inputs attach after the gate resolves.
                    for iteration_mapping in iteration_mappings:
                        create_results = self.create_execution_node(
                            next_node_id,
                            iteration_mapping,
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
                collapsed_branch = (
                    self._get_sibling_iterate_if_collect_branch(next_node_id)
                    if isinstance(next_node, IterateInvocation)
                    else None
                )
                activation_frames = (
                    graph_if_runtime._get_selected_if_frames_for_preparation(self._state, *collapsed_branch)
                    if collapsed_branch is not None
                    else []
                )
                if activation_frames:
                    for activation_frame in activation_frames:
                        new_node_ids.extend(
                            self.create_execution_node(next_node_id, iteration_mappings, activation_frame)
                        )
                elif collapsed_branch is not None:
                    if_node_id, _ = collapsed_branch
                    if_exec_ids = self._state.source_prepared_mapping.get(if_node_id, set())
                    if if_exec_ids and all(
                        exec_id in self._state._resolved_if_exec_branches for exec_id in if_exec_ids
                    ):
                        self._mark_source_node_empty(next_node_id)
                        self._state._invalidate_loop_caches_for_source(next_node_id)
                        return next_node_id
                    # Wait for the owning If to resolve before deriving the branch's parent frames.
                    return None
                else:
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

        if isinstance(next_node, ForInvocation) and any(
            self._state.execution_graph.get_node(exec_node_id).index == -1 for exec_node_id in new_node_ids
        ):
            self._mark_empty_for_complete(next_node_id)

        return new_node_ids[0]


class _ExecutionMaterializer(_ExecutionNodeBuilder):
    """Compatibility planner facade for unsupported and legacy graphs."""

    pass
