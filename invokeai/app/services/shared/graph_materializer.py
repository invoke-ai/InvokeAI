"""Compatibility materialization for graph execution state."""

import itertools
from collections import defaultdict, deque
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
from invokeai.app.services.shared.graph_iterator_ancestry import _get_ordered_input_edges, _IteratorAncestry
from invokeai.app.services.shared.graph_loop_topology import NestedForBody, NestedIterateBody, NestedIterateChain
from invokeai.app.services.shared.graph_models import Edge, EdgeConnection
from invokeai.app.services.shared.graph_validation import (
    COLLECTION_FIELD,
    ITEM_FIELD,
    CollectInvocation,
    IterateInvocation,
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
        self._activation_sources_by_iterator: dict[str, set[str]] = {}
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
        self._collect_group_axes_source_graph: Optional["nx.DiGraph"] = None
        self._collect_group_axes_by_parent: dict[str, tuple[str, ...]] = {}
        self._sibling_iterate_collect_by_parent: dict[str, bool] = {}

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

        ordered_edges = _get_ordered_input_edges(self._state.graph, input_edges)
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
            parent_axes = registry.get_iteration_axes(prepared_id)
            if parent_axes is None:
                parent_axes = self._get_iteration_axes(source_node_id)
            source_edges = input_edges_by_source.get(source_node_id)
            if source_edges:
                parent_path = max(
                    (self._get_prepared_edge_iteration_path(edge, prepared_id) for edge in source_edges),
                    key=lambda path: (len(path), path),
                )
            if parent_path:
                if (
                    source_edges
                    and isinstance(self._state.graph.get_node(source_node_id), ForInvocation)
                    and get_output_field_scope(self._state.graph.get_node(source_node_id), source_edges[0].source.field)
                    == OutputScope.Final
                    and parent_axes
                    and parent_axes[-1] == source_node_id
                ):
                    parent_axes = parent_axes[:-1]
                parent_contexts.append((parent_path, parent_axes))

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
        self._activation_sources_by_iterator = {}
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
        self._activation_sources_by_iterator = {}
        self._empty_condition_if_sources = {}
        self._pending_activation_retirement_candidates.clear()
        for source_node_id in self._activation_candidate_topological_order:
            for dependency in self._state._get_source_activation_dependencies(source_node_id):
                self._activation_sources_by_dependency.setdefault((dependency.owner_id, dependency.branch), set()).add(
                    source_node_id
                )
            for iterator_id in self._state._runtime()._get_ordered_iterator_sources(source_node_id):
                self._activation_sources_by_iterator.setdefault(iterator_id, set()).add(source_node_id)
        for if_node_id in graph.nodes:
            if not isinstance(self._state.graph.get_node(if_node_id), IfInvocation):
                continue
            condition_edges = self._state.graph._get_input_edges(if_node_id, "condition")
            iterator_graph = self.iterator_graph(graph)
            ancestry = self._get_iterator_ancestry(iterator_graph)
            empty_frame_sources = tuple(
                dict.fromkeys(
                    [
                        axis_id
                        for edge in condition_edges
                        if iterator_graph.has_node(edge.source.node_id)
                        for axis_id in ancestry.get_iterator_axes_for_edge(edge)
                    ]
                    + list(ancestry.get_frame_iterators(if_node_id))
                )
            )
            for source_id in empty_frame_sources:
                self._empty_condition_if_sources.setdefault(source_id, ())
                self._empty_condition_if_sources[source_id] += (if_node_id,)

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
            frame = self._state._activation_gate(exec_node_id).frame.iteration_path
            self._queue_selected_empty_iterate_branch_retirements(owner_id, selected_branch, frame)
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

    def _queue_selected_empty_iterate_branch_retirements(
        self, if_node_id: str, selected_branch: str, frame: tuple[int, ...]
    ) -> None:
        """Queue item sources whose selected branch frame has a persisted empty Iterate input."""
        self._ensure_activation_candidate_index(self._state._get_source_graph_flat())
        owner_axes = tuple(self._state._runtime()._get_ordered_iterator_sources(if_node_id))
        if_axes_count = len(owner_axes)
        for source_node_id in self._activation_sources_by_dependency.get((if_node_id, selected_branch), ()):
            source_node = self._state.graph.get_node(source_node_id)
            if isinstance(source_node, (IfInvocation, IterateInvocation, CollectInvocation)):
                continue

            source_axes = self._get_collect_source_iterator_ids(source_node_id)
            if len(source_axes) <= if_axes_count or tuple(source_axes[:if_axes_count]) != owner_axes:
                continue

            iterator_id = source_axes[if_axes_count]
            if self._has_empty_iterate_parent_group(iterator_id, owner_axes, frame):
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
        node = self._state.graph.get_node(node_id)
        input_edges = self._state.graph._get_input_edges(node_id)
        if isinstance(node, IfInvocation) and not self._state.graph._get_input_edges(node_id, "condition"):
            selected_field = "true_input" if node.condition else "false_input"
            input_edges = self._state.graph._get_input_edges(node_id, selected_field)
        for edge in input_edges:
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
        loop_topology = self._state._loop_topology()
        graph = loop_topology.flat_graph
        execution_graph = self._state._get_execution_graph_flat()
        body_shape = loop_topology.for_body_shapes.get(source_for_id)
        if isinstance(body_shape, NestedIterateChain):
            return self._create_serial_nested_iterate_body_iteration(
                source_for_id, prepared_for_id, graph, execution_graph, body_shape
            )
        if isinstance(body_shape, NestedIterateBody):
            return self._create_nested_iterate_body_iteration(
                source_for_id, prepared_for_id, graph, execution_graph, body_shape
            )
        if isinstance(body_shape, NestedForBody):
            return self._create_nested_for_body_iteration(
                source_for_id, prepared_for_id, graph, execution_graph, body_shape
            )

        body_path_to_return = loop_topology.for_body_paths.get(source_for_id)
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

    def _is_deferred_nested_for_return(self, node_id: str) -> bool:
        return any(
            isinstance(body_shape, NestedForBody) and body_shape.outer_return_id == node_id
            for body_shape in self._state._loop_topology().for_body_shapes.values()
        )

    def _get_final_prepared_for_id(self, source_for_id: str, parent_iteration_path: tuple[int, ...]) -> str:
        self._state._get_prepared_for_index()
        assert self._state._final_prepared_for_index is not None
        prepared_for_id = self._state._final_prepared_for_index.get((source_for_id, parent_iteration_path))
        if prepared_for_id is None:
            raise RuntimeError(f"Unable to find finalized nested For '{source_for_id}' for {parent_iteration_path}")
        return prepared_for_id

    def create_nested_for_return(self, inner_for_id: str, prepared_inner_for_id: str) -> Optional[str]:
        loop_topology = self._state._loop_topology()
        graph = loop_topology.flat_graph
        inner_iteration_path = self._state._get_iteration_path(prepared_inner_for_id)
        outer_for_id: Optional[str] = None
        nested_body: Optional[NestedForBody] = None
        for source_for_id, candidate in loop_topology.for_body_shapes.items():
            if isinstance(candidate, NestedForBody) and inner_for_id in candidate.inner_for_ids:
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

        continuation_nodes = nested_body.continuation_nodes
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
        nested_body: NestedIterateChain,
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
        nested_body: NestedIterateChain,
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
        nested_body: NestedForBody,
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
        nested_body: NestedIterateBody,
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

    def _get_collect_group_axes(
        self,
        edge: Edge,
        prepared_id: Optional[str] = None,
        context_deepest_path_depth_by_axes_prefix: Optional[dict[tuple[str, ...], int]] = None,
        max_collected_path_depth: Optional[int] = None,
    ) -> tuple[str, ...]:
        source_node = self._state.graph.get_node(edge.source.node_id)
        if prepared_id is None:
            prepared_ids = self._get_ordered_prepared_nodes_for_edge(edge)
            prepared_id = prepared_ids[0] if prepared_ids else None
        if prepared_id is None:
            return self._get_static_collect_group_axes(edge)

        axes = self._get_prepared_edge_iteration_axes(edge, prepared_id)
        if edge.destination.field != ITEM_FIELD:
            return axes
        if (
            isinstance(source_node, ForInvocation)
            and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
        ):
            return axes
        if max_collected_path_depth is None:
            collection_edges = self._state.graph._get_input_edges(source_node.id, COLLECTION_FIELD)
            max_collected_path_depth = max(
                (
                    len(self._get_prepared_edge_iteration_path(collection_edge, collect_prepared_id))
                    for collection_edge in collection_edges
                    if isinstance(self._state.graph.get_node(collection_edge.source.node_id), CollectInvocation)
                    for collect_prepared_id in self._get_ordered_prepared_nodes_for_edge(collection_edge)
                ),
                default=0,
            )
        if max_collected_path_depth and len(self._state._get_iteration_path(prepared_id)) < max_collected_path_depth:
            return axes
        if isinstance(source_node, IfInvocation):
            ancestry = self._get_iterator_ancestry(self.iterator_graph())
            retained_axes = self._get_selected_for_final_parent_axes(source_node.id, ancestry)
            if retained_axes and self._collect_has_selected_for_final_input(edge.destination.node_id):
                prepared_retained_axes = self._get_selected_for_final_parent_axes(source_node.id, ancestry, prepared_id)
                if prepared_retained_axes:
                    return prepared_retained_axes
                return tuple(axis for axis in axes if axis in retained_axes)
            selected_field = self._state._resolved_if_exec_branches.get(prepared_id)
            condition_edges = self._state.graph._get_input_edges(source_node.id, "condition")
            if (
                condition_edges
                and selected_field is not None
                and not any(ancestry.get_frame_axes_for_edge(condition_edge) for condition_edge in condition_edges)
            ):
                branch_axes, final_axes = graph_if_runtime._get_if_static_frame_analysis(
                    self._state, ancestry, source_node.id
                )
                direct_axes = {
                    axis_id
                    for branch_edge in self._state.graph._get_input_edges(source_node.id, selected_field)
                    for axis_id in ancestry.get_iterator_axes_for_edge(branch_edge)
                }
                selected_axes = branch_axes[selected_field]
                other_field = "false_input" if selected_field == "true_input" else "true_input"
                other_axes = branch_axes[other_field]
                if (
                    other_axes
                    and set(selected_axes) - direct_axes - final_axes[selected_field]
                    and len(selected_axes) > len(other_axes)
                    and selected_axes[: len(other_axes)] == other_axes
                    and selected_axes[: len(axes)] == axes
                ):
                    # Preserve groups only when a root If extends a shared branch-frame prefix.
                    return axes
            if not condition_edges:
                selected_field = graph_if_runtime._get_static_if_selection(self._state, source_node.id)
                if selected_field is not None:
                    branch_axes, _ = graph_if_runtime._get_if_static_frame_analysis(
                        self._state, ancestry, source_node.id
                    )
                    other_field = "false_input" if selected_field == "true_input" else "true_input"
                    selected_axes = branch_axes[selected_field]
                    reopened_axes = set().union(
                        *(
                            graph_if_runtime._get_reopened_source_axes(
                                self._state, ancestry, selected_axis, ancestry.frame_axes_cache
                            )
                            for selected_axis in selected_axes
                        )
                    )
                    if branch_axes[selected_field] and set(branch_axes[other_field]).intersection(reopened_axes):
                        return axes
        for other_edge in self._state.graph._get_input_edges(edge.destination.node_id, ITEM_FIELD):
            if other_edge == edge:
                continue
            other_axes = tuple(self._get_collect_source_iterator_ids(other_edge.source.node_id))
            if len(other_axes) > len(axes) and other_axes[: len(axes)] == axes:
                return axes
        if context_deepest_path_depth_by_axes_prefix is None:
            context_deepest_path_depth_by_axes_prefix = self._get_collect_deepest_path_depth_by_axes_prefix(
                edge.destination.node_id
            )
        context_depth = context_deepest_path_depth_by_axes_prefix.get(axes)
        if context_depth is not None and context_depth > len(self._get_prepared_edge_iteration_path(edge, prepared_id)):
            return axes
        group_depth = max(len(self._get_collect_source_iterator_ids(source_node.id)) - 1, 0)
        if len(axes) <= group_depth:
            return axes
        return axes[:-1]

    def _get_collect_deepest_path_depth_by_axes_prefix(self, collect_node_id: str) -> dict[tuple[str, ...], int]:
        context_depth_by_axes_prefix: dict[tuple[str, ...], int] = {}
        for context_edge in self._state.graph._get_input_edges(collect_node_id, ITEM_FIELD):
            prepared_ids = self._get_ordered_prepared_nodes_for_edge(context_edge)
            if not prepared_ids:
                continue
            prepared_paths = [
                (candidate_id, self._get_prepared_edge_iteration_path(context_edge, candidate_id))
                for candidate_id in prepared_ids
            ]
            deepest_id, deepest_path = max(prepared_paths, key=lambda candidate: (len(candidate[1]), candidate[1]))
            deepest_axes = self._get_prepared_edge_iteration_axes(context_edge, deepest_id)
            for prefix_length in range(len(deepest_axes)):
                prefix = deepest_axes[:prefix_length]
                context_depth_by_axes_prefix[prefix] = max(
                    len(deepest_path), context_depth_by_axes_prefix.get(prefix, 0)
                )
        return context_depth_by_axes_prefix

    def _get_static_collect_group_axes(self, edge: Edge) -> tuple[str, ...]:
        iterator_node_ids = self._get_collect_source_iterator_ids(edge.source.node_id)
        group_depth = len(iterator_node_ids)
        source_node = self._state.graph.get_node(edge.source.node_id)
        if edge.destination.field == ITEM_FIELD:
            if (
                isinstance(source_node, ForInvocation)
                and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
            ):
                return tuple(iterator_node_ids)
            if isinstance(source_node, IfInvocation):
                ancestry = self._get_iterator_ancestry(self.iterator_graph())
                retained_axes = self._get_selected_for_final_parent_axes(source_node.id, ancestry)
                if retained_axes and self._collect_has_selected_for_final_input(edge.destination.node_id):
                    return retained_axes
            group_depth = max(group_depth - 1, 0)
        return tuple(iterator_node_ids[:group_depth])

    def _get_shared_static_collect_group_axes(self, collect_node_id: str) -> tuple[str, ...]:
        self._refresh_collect_group_caches()
        cached = self._collect_group_axes_by_parent.get(collect_node_id)
        if cached is not None:
            return cached

        item_axes = [
            self._get_static_collect_group_axes(item_edge)
            for item_edge in self._state.graph._get_input_edges(collect_node_id, ITEM_FIELD)
        ]
        parent_axes: list[str] = []
        for axis_candidates in zip(*item_axes, strict=False):
            if len(set(axis_candidates)) != 1:
                break
            parent_axes.append(axis_candidates[0])
        result = tuple(parent_axes)
        self._collect_group_axes_by_parent[collect_node_id] = result
        return result

    def _refresh_collect_group_caches(self) -> None:
        source_graph = self._state._get_source_graph_flat()
        if source_graph is self._collect_group_axes_source_graph:
            return
        self._collect_group_axes_source_graph = source_graph
        self._collect_group_axes_by_parent.clear()
        self._sibling_iterate_collect_by_parent.clear()

    def _is_sibling_iterate_collect(self, collect_node_id: str) -> bool:
        self._refresh_collect_group_caches()
        cached = self._sibling_iterate_collect_by_parent.get(collect_node_id)
        if cached is not None:
            return cached

        item_edges = self._state.graph._get_input_edges(collect_node_id, ITEM_FIELD)
        ancestry = self._get_iterator_ancestry(self.iterator_graph())
        iterator_ids = [
            (axes[-1] if axes else None)
            for axes in (self._get_collect_source_iterator_ids(edge.source.node_id) for edge in item_edges)
        ]
        result = (
            len(iterator_ids) > 1
            and not any(self._get_selected_for_final_parent_axes(edge.source.node_id, ancestry) for edge in item_edges)
            and all(iterator_id is not None for iterator_id in iterator_ids)
            and len(set(iterator_ids)) == len(iterator_ids)
            and all(
                isinstance(self._state.graph.get_node(iterator_id), IterateInvocation)
                for iterator_id in iterator_ids
                if iterator_id is not None
            )
        )
        self._sibling_iterate_collect_by_parent[collect_node_id] = result
        return result

    def _is_sibling_collect_reiteration(self, source_node_id: str) -> bool:
        source_axes = self._get_collect_source_iterator_ids(source_node_id)
        if not source_axes or not isinstance(self._state.graph.get_node(source_axes[-1]), IterateInvocation):
            return False
        collection_edges = self._state.graph._get_input_edges(source_axes[-1], COLLECTION_FIELD)
        return (
            len(collection_edges) == 1
            and isinstance(self._state.graph.get_node(collection_edges[0].source.node_id), CollectInvocation)
            and self._is_sibling_iterate_collect(collection_edges[0].source.node_id)
        )

    def _get_prepared_edge_iteration_axes(self, edge: Edge, prepared_id: str) -> tuple[str, ...]:
        """Identify frame axes from the prepared node's real execution ancestors."""
        path = self._get_prepared_edge_iteration_path(edge, prepared_id)
        recorded_axes = self._state._prepared_registry().get_iteration_axes(prepared_id)
        static_axes = tuple(self._get_collect_source_iterator_ids(edge.source.node_id))
        if not path:
            source_node = self._state.graph.get_node(edge.source.node_id)
            if (
                isinstance(source_node, ForInvocation)
                and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
            ):
                return tuple(axis for axis in (static_axes or recorded_axes or ()) if axis != source_node.id)
            if static_axes:
                first_axis = self._state.graph.get_node(static_axes[0])
                if isinstance(first_axis, (IterateInvocation, ForInvocation)):
                    collection_edges = self._state.graph._get_input_edges(static_axes[0], COLLECTION_FIELD)
                    if len(collection_edges) == 1:
                        parent_node = self._state.graph.get_node(collection_edges[0].source.node_id)
                        if isinstance(parent_node, CollectInvocation):
                            parent_axes = self._get_shared_static_collect_group_axes(parent_node.id)
                            if parent_axes:
                                return (*parent_axes, *static_axes)
            return static_axes or (recorded_axes or ())
        if recorded_axes is not None and len(recorded_axes) == len(path):
            return recorded_axes

        if static_axes and len(static_axes) < len(path):
            missing_prefix_depth = len(path) - len(static_axes)
            first_axis_id = static_axes[0]
            first_axis = self._state.graph.get_node(first_axis_id)
            if isinstance(first_axis, (IterateInvocation, ForInvocation)):
                collection_edges = self._state.graph._get_input_edges(first_axis_id, COLLECTION_FIELD)
                if len(collection_edges) == 1:
                    collection_edge = collection_edges[0]
                    parent_node = self._state.graph.get_node(collection_edge.source.node_id)
                    if isinstance(parent_node, CollectInvocation):
                        parent_axes = self._get_shared_static_collect_group_axes(parent_node.id)
                        parent_exec_id = self._state._prepared_registry().get_prepared_id_at_path(
                            parent_node.id, path[:missing_prefix_depth]
                        )
                        if parent_exec_id is not None and len(parent_axes) == missing_prefix_depth:
                            static_axes = (*parent_axes, *static_axes)
                            if len(static_axes) == len(path):
                                return static_axes

        execution_graph = self._state._get_execution_graph_flat()
        pending = deque([(prepared_id, 0)])
        visited = {prepared_id}
        candidates: dict[int, list[tuple[int, str]]] = {}
        while pending:
            exec_id, distance = pending.popleft()
            source_id = self._state.prepared_source_mapping.get(exec_id)
            if source_id is not None:
                source_node = self._state.graph.get_node(source_id)
                prepared_node = self._state.execution_graph.get_node(exec_id)
                if isinstance(source_node, IterateInvocation) or (
                    isinstance(source_node, ForInvocation)
                    and isinstance(prepared_node, ForInvocation)
                    and prepared_node.index >= 0
                ):
                    frame_path = self._state._get_iteration_path(exec_id)
                    if frame_path and len(frame_path) <= len(path) and path[: len(frame_path)] == frame_path:
                        candidates.setdefault(len(frame_path) - 1, []).append((distance, source_id))
            for parent_id in execution_graph.predecessors(exec_id):
                if parent_id not in visited:
                    visited.add(parent_id)
                    pending.append((parent_id, distance + 1))

        edge_axes: Optional[tuple[str, ...]] = None
        axes: list[str] = []
        for axis_index in range(len(path)):
            axis_candidates = candidates.get(axis_index, [])
            preferred_axis = next(
                (axis_id for axis_id in static_axes if any(candidate == axis_id for _, candidate in axis_candidates)),
                None,
            )
            axis_id = preferred_axis or (min(axis_candidates)[1] if axis_candidates else None)
            if axis_id is None:
                if len(static_axes) >= len(path):
                    axis_id = static_axes[axis_index]
                else:
                    if edge_axes is None:
                        iterator_ancestry = _IteratorAncestry(self._state, iterator_graph=self.iterator_graph())
                        edge_axes = (
                            graph_if_runtime._get_edge_frame_axes(
                                self._state, iterator_ancestry, edge, {}, use_runtime_if_selection=True
                            )
                            or ()
                        )
                    axis_id = edge_axes[axis_index] if axis_index < len(edge_axes) else None
            if axis_id is None:
                break
            axes.append(axis_id)
        source_node = self._state.graph.get_node(edge.source.node_id)
        if not axes and isinstance(source_node, CollectInvocation):
            collect_axes = self._get_shared_static_collect_group_axes(source_node.id)
            if len(collect_axes) == len(path):
                return collect_axes
        return tuple(axes)

    def _get_collect_source_iterator_ids(
        self, source_node_id: str, iterator_graph: Optional["nx.DiGraph"] = None
    ) -> list[str]:
        if isinstance(self._state.graph.get_node(source_node_id), IfInvocation):
            iterator_ancestry = self._get_iterator_ancestry(iterator_graph or self.iterator_graph())
            branch_axes_by_field = graph_if_runtime._get_if_runtime_branch_frame_axes(
                self._state, iterator_ancestry, source_node_id
            )
            branch_axes = [branch_axes_by_field[branch] for branch in ("true_input", "false_input")]
            condition_axes = {
                axis_id
                for edge in self._state.graph._get_input_edges(source_node_id, "condition")
                for axis_id in iterator_ancestry.get_frame_axes_for_edge(edge)
            }
            if branch_axes[0] and branch_axes[0] == branch_axes[1] and set(branch_axes[0]) > condition_axes:
                return list(branch_axes[0])

        iterator_graph = iterator_graph or self.iterator_graph()
        iterator_ancestry = self._get_iterator_ancestry(iterator_graph)
        iterator_node_ids = self.get_node_iterators(source_node_id, iterator_graph)
        parent_frame_axes = [
            axis_id
            for iterator_node_id in iterator_node_ids
            for edge in self._state.graph._get_input_edges(iterator_node_id, COLLECTION_FIELD)
            for axis_id in iterator_ancestry.get_frame_axes_for_edge(edge)
        ]
        iterator_node_ids = list(dict.fromkeys([*parent_frame_axes, *iterator_node_ids]))
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

    def _get_iterator_input_iteration_groups(
        self, iterator_node_id: str
    ) -> set[tuple[str, tuple[str, ...], tuple[int, ...]]]:
        iteration_groups: set[tuple[str, tuple[str, ...], tuple[int, ...]]] = set()
        for edge in self._state.graph._get_input_edges(iterator_node_id, COLLECTION_FIELD):
            iteration_groups.update(
                (
                    edge.source.node_id,
                    self._get_prepared_edge_iteration_axes(edge, prepared_id),
                    self._get_prepared_edge_iteration_path(edge, prepared_id),
                )
                for prepared_id in self._get_ordered_prepared_nodes_for_edge(edge)
            )
        return iteration_groups

    def _get_empty_iterate_parent_groups(self, iterator_node_id: str) -> set[tuple[tuple[str, ...], tuple[int, ...]]]:
        """Derive empty parent frames from persisted collection results feeding an Iterate."""
        if not isinstance(self._state.graph.get_node(iterator_node_id), IterateInvocation):
            return set()

        groups: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
        for edge in self._state.graph._get_input_edges(iterator_node_id, COLLECTION_FIELD):
            for prepared_id in self._get_ordered_prepared_nodes_for_edge(edge):
                result = self._state.results.get(prepared_id)
                if (
                    result is None
                    or not isinstance((collection := getattr(result, edge.source.field, None)), list)
                    or collection
                ):
                    continue
                path = self._get_prepared_edge_iteration_path(edge, prepared_id)
                axes = self._get_prepared_edge_iteration_axes(edge, prepared_id)
                if len(axes) == len(path):
                    groups.add((axes, path))
        return groups

    def _has_empty_iterate_parent_group(
        self, iterator_node_id: str, group_axes: tuple[str, ...], group_path: tuple[int, ...]
    ) -> bool:
        return (group_axes, group_path) in self._get_empty_iterate_parent_groups(iterator_node_id)

    def _has_empty_iterate_anchor_for_collect_group(
        self, item_edge: Edge, group_axes: tuple[str, ...], group_path: tuple[int, ...]
    ) -> bool:
        source_axes = self._get_collect_source_iterator_ids(item_edge.source.node_id)
        if len(source_axes) <= len(group_axes) or tuple(source_axes[: len(group_axes)]) != group_axes:
            return False

        iterator_id = source_axes[len(group_axes)]
        return self._has_empty_iterate_parent_group(iterator_id, group_axes, group_path)

    def _has_empty_collect_group_anchor(
        self, collect_id: str, group_axes: tuple[str, ...], group_path: tuple[int, ...]
    ) -> bool:
        collect_inputs = self._state.graph._get_input_edges(collect_id)
        missing_item_edges = [
            edge
            for edge in collect_inputs
            if edge.destination.field == ITEM_FIELD
            and edge.source.node_id not in self._state.source_prepared_mapping
            and edge.source.node_id not in self._state.executed
        ]
        if not missing_item_edges:
            return False
        return self._state._is_source_activation_admitted(collect_id, group_path) and all(
            self._has_empty_iterate_anchor_for_collect_group(edge, group_axes, group_path)
            for edge in missing_item_edges
        )

    def _has_ready_empty_collect_group(self, collect_id: str) -> bool:
        """Allow an empty Collect frame only when its missing item source has a persisted empty Iterate input."""
        groups = self._get_collect_iteration_mapping_groups(self._state.graph._get_input_edges(collect_id))
        active_empty_groups = [
            (group_axes, group_path)
            for group_axes, group_path, iteration_mappings in groups
            if not iteration_mappings and self._state._is_source_activation_admitted(collect_id, group_path)
        ]
        return bool(active_empty_groups) and all(
            self._has_empty_collect_group_anchor(collect_id, group_axes, group_path)
            for group_axes, group_path in active_empty_groups
        )

    def _collect_has_selected_for_final_input(self, collect_id: str) -> bool:
        ancestry = self._get_iterator_ancestry(self.iterator_graph())
        return any(
            self._get_selected_for_final_parent_axes(edge.source.node_id, ancestry)
            for edge in self._state.graph._get_input_edges(collect_id, ITEM_FIELD)
        )

    def _get_selected_if_fields(self, if_node_id: str, prepared_id: Optional[str] = None) -> tuple[str, ...]:
        if prepared_id is not None:
            selected_field = self._state._resolved_if_exec_branches.get(prepared_id)
            if selected_field is not None:
                return (selected_field,)
        selected_field = graph_if_runtime._get_static_if_selection(self._state, if_node_id)
        if selected_field is not None:
            return (selected_field,)
        prepared_ids = self._state.source_prepared_mapping.get(if_node_id, ())
        selected_fields = {self._state._resolved_if_exec_branches.get(prepared_id) for prepared_id in prepared_ids}
        selected_fields.discard(None)
        if not selected_fields or any(
            prepared_id not in self._state._resolved_if_exec_branches for prepared_id in prepared_ids
        ):
            return ("true_input", "false_input")
        return tuple(field for field in ("true_input", "false_input") if field in selected_fields)

    def _get_selected_for_final_parent_axes(
        self, if_node_id: str, ancestry: _IteratorAncestry, prepared_id: Optional[str] = None
    ) -> tuple[str, ...]:
        """Return For-final parent frames retained by this If's selection and condition scope."""
        if not isinstance(self._state.graph.get_node(if_node_id), IfInvocation):
            return ()
        root_selection = (
            graph_if_runtime._get_static_if_selection(self._state, if_node_id)
            if not self._state.graph._get_input_edges(if_node_id, "condition")
            else None
        )
        selected_fields = self._get_selected_if_fields(if_node_id, prepared_id)
        condition_axes = {
            axis_id
            for edge in self._state.graph._get_input_edges(if_node_id, "condition")
            for axis_id in ancestry.get_frame_axes_for_edge(edge)
        }
        pending_edges = [
            (edge, False)
            for branch_field in selected_fields
            for edge in self._state.graph._get_input_edges(if_node_id, branch_field)
        ]
        visited: set[tuple[str, bool]] = set()
        retained_axes: list[str] = []
        while pending_edges:
            edge, nested_condition = pending_edges.pop()
            source_id = edge.source.node_id
            source_node = self._state.graph.get_node(source_id)
            if (
                isinstance(source_node, ForInvocation)
                and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
            ):
                if prepared_id is not None:
                    prepared_path = self._state._get_iteration_path(prepared_id)
                    final_output_paths = {
                        self._get_prepared_edge_iteration_path(edge, final_id)
                        for final_id in self._get_ordered_prepared_nodes_for_edge(edge)
                    }
                    if prepared_path not in final_output_paths:
                        continue
                final_parent_axes = ancestry.get_frame_axes_for_edge(edge)
                for axis_id in final_parent_axes:
                    if (
                        root_selection is not None or nested_condition or axis_id in condition_axes
                    ) and axis_id not in retained_axes:
                        retained_axes.append(axis_id)
                continue
            visit_key = (source_id, nested_condition)
            if visit_key in visited:
                continue
            visited.add(visit_key)
            if isinstance(source_node, IfInvocation):
                nested_fields = self._get_selected_if_fields(source_id)
                pending_edges.extend(
                    (edge, nested_condition or bool(self._state.graph._get_input_edges(source_id, "condition")))
                    for branch_field in nested_fields
                    for edge in self._state.graph._get_input_edges(source_id, branch_field)
                )
                continue
            if isinstance(source_node, (ForInvocation, IterateInvocation, CollectInvocation)):
                continue
            pending_edges.extend(
                (input_edge, nested_condition) for input_edge in self._state.graph._get_input_edges(source_id)
            )
        return tuple(retained_axes)

    def _get_collect_candidate_group_keys(
        self, edge: Edge, group_axes: Optional[tuple[str, ...]] = None
    ) -> set[tuple[tuple[str, ...], tuple[int, ...]]]:
        source_node_id = edge.source.node_id
        iterator_node_ids = self._get_collect_source_iterator_ids(source_node_id)
        group_axes = group_axes if group_axes is not None else self._get_collect_group_axes(edge)
        group_depth = len(group_axes)
        prepared_ids = self._get_ordered_prepared_nodes_for_edge(edge)
        if group_depth == 0:
            if prepared_ids and not self._is_sibling_collect_reiteration(edge.source.node_id):
                return {((), ())}
            group_keys = {((), ())}
            for iterator_node_id in iterator_node_ids:
                for _, input_axes, input_path in self._get_iterator_input_iteration_groups(iterator_node_id):
                    candidate_axes = list(input_axes[: len(input_path)])
                    if input_path and len(candidate_axes) == len(input_path):
                        group_keys.add((tuple(candidate_axes), input_path))
            return group_keys

        group_keys: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
        iterator_input_group_keys: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
        # A Collect.item edge groups by the outer `group_depth` axes. Read input paths from the current item axis
        # too: an empty stream still carries its parent groups even when it has no prepared items.
        for axis_index, iterator_node_id in enumerate(iterator_node_ids[: group_depth + 1]):
            iterator_axes = tuple(self._get_collect_source_iterator_ids(iterator_node_id))
            # Only outer group axes contribute prepared paths; the current item axis is collapsed by Collect.
            if axis_index < group_depth:
                for prepared_id in self._get_ordered_prepared_nodes_for_source(iterator_node_id):
                    iteration_path = self._state._get_iteration_path(prepared_id)[:group_depth]
                    candidate_axes = iterator_axes[: len(iteration_path)]
                    if (
                        iteration_path
                        and len(candidate_axes) == len(iteration_path)
                        and group_axes[: len(candidate_axes)] == candidate_axes
                    ):
                        group_keys.add((candidate_axes, iteration_path))
            for _, input_axes, input_path in self._get_iterator_input_iteration_groups(iterator_node_id):
                candidate_path = input_path[:group_depth]
                candidate_axes = input_axes[: len(candidate_path)]
                if (
                    candidate_path
                    and len(candidate_axes) == len(candidate_path)
                    and group_axes[: len(candidate_axes)] == candidate_axes
                ):
                    candidate_group_key = (candidate_axes, candidate_path)
                    group_keys.add(candidate_group_key)
                    iterator_input_group_keys.add(candidate_group_key)

        prepared_contexts = [
            (
                self._get_prepared_edge_iteration_axes(edge, prepared_id),
                self._get_prepared_edge_iteration_path(edge, prepared_id),
            )
            for prepared_id in prepared_ids
        ]
        if prepared_contexts:
            axes_by_path_prefix: dict[tuple[int, ...], set[tuple[str, ...]]] = {}
            path_prefixes: set[tuple[int, ...]] = set()
            for source_axes, source_path in prepared_contexts:
                source_axis_prefixes = [source_axes[:axes_depth] for axes_depth in range(len(source_axes) + 1)]
                for path_depth in range(len(source_path) + 1):
                    path_prefix = source_path[:path_depth]
                    path_prefixes.add(path_prefix)
                    axes_by_path_prefix.setdefault(path_prefix, set()).update(source_axis_prefixes)
            group_keys = {
                (candidate_axes, candidate_path)
                for candidate_axes, candidate_path in group_keys
                if not candidate_path
                or candidate_path not in path_prefixes
                or candidate_axes in axes_by_path_prefix[candidate_path]
                or (candidate_axes, candidate_path) in iterator_input_group_keys
            }

        return group_keys

    def _get_collect_iteration_mapping_groups(
        self, input_edges: list[Edge]
    ) -> list[tuple[tuple[str, ...], tuple[int, ...], list[tuple[str, str]]]]:
        prepared_inputs: list[tuple[Edge, str, str, tuple[str, ...], tuple[int, ...], bool]] = []
        group_keys: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
        context_deepest_path_depth_by_axes_prefix = (
            self._get_collect_deepest_path_depth_by_axes_prefix(input_edges[0].destination.node_id)
            if input_edges
            else {}
        )
        context_axes_prefixes_with_paths = {
            axes[:prefix_depth] for axes, path in group_keys if path for prefix_depth in range(1, len(axes) + 1)
        }
        root_context_axes: set[tuple[str, ...]] = set()
        max_collected_path_depth_by_output: dict[tuple[str, str, str], int] = {}
        for edge in input_edges:
            prepared_ids = self._get_ordered_prepared_nodes_for_edge(edge)
            depth_key = (edge.source.node_id, edge.source.field, edge.destination.field)
            if depth_key not in max_collected_path_depth_by_output:
                source_node = self._state.graph.get_node(edge.source.node_id)
                if edge.destination.field == ITEM_FIELD and not (
                    isinstance(source_node, ForInvocation)
                    and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
                ):
                    collection_edges = self._state.graph._get_input_edges(edge.source.node_id, COLLECTION_FIELD)
                    max_collected_path_depth_by_output[depth_key] = max(
                        (
                            len(self._get_prepared_edge_iteration_path(collection_edge, collect_prepared_id))
                            for collection_edge in collection_edges
                            if isinstance(self._state.graph.get_node(collection_edge.source.node_id), CollectInvocation)
                            for collect_prepared_id in self._get_ordered_prepared_nodes_for_edge(collection_edge)
                        ),
                        default=0,
                    )
                else:
                    max_collected_path_depth_by_output[depth_key] = 0
            max_collected_path_depth = max_collected_path_depth_by_output[depth_key]
            edge_group_axes = list(
                self._get_collect_group_axes(
                    edge,
                    context_deepest_path_depth_by_axes_prefix=context_deepest_path_depth_by_axes_prefix,
                    max_collected_path_depth=max_collected_path_depth,
                )
            )
            prepared_group_axes_by_id = {
                prepared_id: self._get_collect_group_axes(
                    edge,
                    prepared_id,
                    context_deepest_path_depth_by_axes_prefix,
                    max_collected_path_depth,
                )
                for prepared_id in prepared_ids
            }
            known_axes: list[Optional[str]] = []
            for prepared_group_axes in prepared_group_axes_by_id.values():
                for axis_index, axis_id in enumerate(prepared_group_axes):
                    if axis_index >= len(known_axes):
                        known_axes.extend([None] * (axis_index + 1 - len(known_axes)))
                    if known_axes[axis_index] is None:
                        known_axes[axis_index] = axis_id
            if len(edge_group_axes) < len(known_axes):
                edge_group_axes.extend(axis_id for axis_id in known_axes[len(edge_group_axes) :] if axis_id is not None)
            edge_group_axes = tuple(edge_group_axes)
            candidate_group_keys = self._get_collect_candidate_group_keys(edge, edge_group_axes)
            group_keys.update(candidate_group_keys)
            context_axes_prefixes_with_paths.update(
                group_axes[:prefix_depth]
                for group_axes, group_path in candidate_group_keys
                if group_path
                for prefix_depth in range(1, len(group_axes) + 1)
            )
            for prepared_id in prepared_ids:
                prepared_group_axes = list(prepared_group_axes_by_id[prepared_id])
                prepared_path = self._get_prepared_edge_iteration_path(edge, prepared_id)
                if prepared_path:
                    prepared_group_axes = prepared_group_axes[: len(prepared_path)]
                prepared_group_axes = tuple(prepared_group_axes)
                group_path = prepared_path[: len(prepared_group_axes)]
                group_key = (prepared_group_axes, group_path)
                has_matching_context_group = not prepared_path and any(
                    prepared_group_axes[:prefix_depth] in context_axes_prefixes_with_paths
                    for prefix_depth in range(1, len(prepared_group_axes) + 1)
                )
                if not has_matching_context_group:
                    group_keys.add(group_key)
                if prepared_path:
                    context_axes_prefixes_with_paths.update(
                        prepared_group_axes[:prefix_depth] for prefix_depth in range(1, len(prepared_group_axes) + 1)
                    )
                is_root_source = not self._get_collect_source_iterator_ids(edge.source.node_id)
                is_root_source = is_root_source and not self._get_prepared_edge_iteration_path(edge, prepared_id)
                if not group_path and prepared_group_axes and not is_root_source:
                    root_context_axes.add(prepared_group_axes)
                prepared_inputs.append(
                    (edge, edge.source.node_id, prepared_id, prepared_group_axes, group_path, is_root_source)
                )

        if root_context_axes and context_axes_prefixes_with_paths:
            group_keys = {
                group_key
                for group_key in group_keys
                if not (
                    not group_key[1]
                    and group_key[0] in root_context_axes
                    and any(
                        group_key[0][:prefix_depth] in context_axes_prefixes_with_paths
                        for prefix_depth in range(1, len(group_key[0]) + 1)
                    )
                )
            }
        if not group_keys:
            group_keys.add(((), ()))

        final_group_keys = sorted(
            self._remove_collect_group_prefixes(group_keys), key=lambda group_key: (group_key[1], group_key[0])
        )
        has_root_item_group = any(
            edge.destination.field == ITEM_FIELD and not is_root_source and not group_axes and not group_path
            for edge, _, _, group_axes, group_path, is_root_source in prepared_inputs
        )
        has_non_item_input = any(edge.destination.field != ITEM_FIELD for edge in input_edges)

        # Index prepared inputs by contexts and prefixes so each group visits only compatible candidates.
        prepared_by_context: dict[tuple[tuple[str, ...], tuple[int, ...]], list[int]] = {}
        prepared_by_prefix_context: dict[tuple[tuple[str, ...], tuple[int, ...]], list[int]] = {}
        prepared_root_sources_by_axes: dict[tuple[str, ...], list[int]] = {}
        prepared_root_items_by_axes: dict[tuple[str, ...], list[int]] = {}
        prepared_root_items_by_axes_prefix: dict[tuple[str, ...], list[int]] = {}
        prepared_with_axes: list[int] = []
        for prepared_input_index, (
            edge,
            _,
            _,
            source_axes,
            source_path,
            is_root_source,
        ) in enumerate(prepared_inputs):
            prepared_by_context.setdefault((source_axes, source_path), []).append(prepared_input_index)
            for prefix_depth in range(len(source_axes) + 1):
                axis_prefix = source_axes[:prefix_depth]
                path_prefix = source_path[:prefix_depth]
                prepared_by_prefix_context.setdefault((axis_prefix, path_prefix), []).append(prepared_input_index)
            if source_axes:
                prepared_with_axes.append(prepared_input_index)
            if is_root_source:
                prepared_root_sources_by_axes.setdefault(source_axes, []).append(prepared_input_index)
            if edge.destination.field == ITEM_FIELD and not source_path and source_axes:
                prepared_root_items_by_axes.setdefault(source_axes, []).append(prepared_input_index)
                for prefix_depth in range(len(source_axes) + 1):
                    prepared_root_items_by_axes_prefix.setdefault(source_axes[:prefix_depth], []).append(
                        prepared_input_index
                    )

        def matches_group(
            edge: Edge,
            source_axes: tuple[str, ...],
            source_path: tuple[int, ...],
            group_axes: tuple[str, ...],
            group_path: tuple[int, ...],
            is_root_source: bool,
        ) -> bool:
            if edge.destination.field == ITEM_FIELD and not source_path and source_axes:
                return source_axes[: len(group_axes)] == group_axes or group_axes[: len(source_axes)] == source_axes
            if group_axes == source_axes:
                return group_path == source_path
            if len(group_axes) < len(source_axes) and source_axes[: len(group_axes)] == group_axes:
                return (
                    edge.destination.field != ITEM_FIELD
                    if not group_axes
                    else group_path == source_path[: len(group_axes)]
                )
            if len(source_axes) < len(group_axes) and group_axes[: len(source_axes)] == source_axes:
                if is_root_source:
                    return True
                return (
                    edge.destination.field != ITEM_FIELD
                    if not source_axes
                    else source_path == group_path[: len(source_axes)]
                )
            return False

        iteration_mapping_groups = []
        for group_key in final_group_keys:
            group_axes, group_path = group_key
            candidate_prepared_indices = set(prepared_by_prefix_context.get(group_key, ()))
            for prefix_depth in range(len(group_axes)):
                candidate_axes = group_axes[:prefix_depth]
                candidate_path = group_path[:prefix_depth]
                candidate_prepared_indices.update(prepared_by_context.get((candidate_axes, candidate_path), ()))
                candidate_prepared_indices.update(prepared_root_sources_by_axes.get(candidate_axes, ()))
            if not group_axes and has_non_item_input:
                candidate_prepared_indices.update(prepared_with_axes)
            candidate_prepared_indices.update(prepared_root_items_by_axes_prefix.get(group_axes, ()))
            for prefix_depth in range(len(group_axes) + 1):
                candidate_prepared_indices.update(prepared_root_items_by_axes.get(group_axes[:prefix_depth], ()))
            iteration_mappings = []
            for prepared_input_index in sorted(candidate_prepared_indices):
                edge, source_node_id, prepared_id, source_axes, source_path, is_root_source = prepared_inputs[
                    prepared_input_index
                ]
                if (
                    is_root_source
                    and not group_axes
                    and any(key[0] for key in final_group_keys)
                    and (edge.destination.field == ITEM_FIELD or not has_root_item_group)
                ):
                    continue
                if matches_group(edge, source_axes, source_path, group_axes, group_path, is_root_source):
                    iteration_mappings.append((source_node_id, prepared_id))
            iteration_mapping_groups.append((group_axes, group_path, iteration_mappings))
        has_deeper_group = any(group_path for _, group_path, _ in iteration_mapping_groups)
        return [
            (group_axes, iteration_path, iteration_mappings)
            for group_axes, iteration_path, iteration_mappings in iteration_mapping_groups
            if iteration_mappings or iteration_path or not has_deeper_group
        ]

    @staticmethod
    def _remove_collect_group_prefixes(
        group_keys: set[tuple[tuple[str, ...], tuple[int, ...]]],
    ) -> set[tuple[tuple[str, ...], tuple[int, ...]]]:
        deepest_axes_by_prefix: dict[tuple[tuple[str, ...], tuple[int, ...]], int] = {}
        for axes, path in group_keys:
            for axes_prefix_depth in range(len(axes) + 1):
                axes_prefix = axes[:axes_prefix_depth]
                for path_prefix_depth in range(len(path) + 1):
                    prefix = (axes_prefix, path[:path_prefix_depth])
                    deepest_axes_by_prefix[prefix] = max(len(axes), deepest_axes_by_prefix.get(prefix, 0))
        return {
            group_key
            for group_key in group_keys
            if not group_key[0] or deepest_axes_by_prefix.get(group_key, len(group_key[0])) <= len(group_key[0])
        }

    def _get_parent_iteration_mappings_without_iterators(
        self, next_node_id: str, input_edges: Optional[list[Edge]] = None
    ) -> list[list[tuple[str, str]]]:
        input_edges = input_edges or self._state.graph._get_input_edges(next_node_id)
        empty_if_item_sources = {
            edge.source.node_id
            for edge in input_edges
            if isinstance(self._state.graph.get_node(next_node_id), CollectInvocation)
            and edge.destination.field == ITEM_FIELD
            and isinstance(self._state.graph.get_node(edge.source.node_id), IfInvocation)
            and edge.source.node_id in self._state.executed
            and not self._state.source_prepared_mapping.get(edge.source.node_id)
        }
        parent_node_ids = [
            node_id
            for node_id in dict.fromkeys(edge.source.node_id for edge in input_edges)
            if node_id not in empty_if_item_sources
        ]
        if not parent_node_ids:
            return [[]]
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
        proper_prefixes = {
            iteration_path[:prefix_length]
            for iteration_path in all_iteration_paths
            for prefix_length in range(len(iteration_path))
        }
        iteration_paths = sorted(
            iteration_path for iteration_path in all_iteration_paths if iteration_path not in proper_prefixes
        )
        if not iteration_paths:
            iteration_paths = [()]

        prepared_ids_by_node_and_path: dict[str, dict[tuple[int, ...], str]] = {}
        for node_id, prepared_nodes in parent_prepared_nodes.items():
            prepared_by_path: dict[tuple[int, ...], str] = {}
            for prepared_id, prepared_path in prepared_nodes:
                if prepared_path not in prepared_by_path or prepared_id < prepared_by_path[prepared_path]:
                    prepared_by_path[prepared_path] = prepared_id
            prepared_ids_by_node_and_path[node_id] = prepared_by_path
        mappings: list[list[tuple[str, str]]] = []
        for iteration_path in iteration_paths:
            mapping: list[tuple[str, str]] = []
            for node_id, prepared_by_path in prepared_ids_by_node_and_path.items():
                prepared_id = next(
                    (
                        prepared_by_path[iteration_path[:prefix_length]]
                        for prefix_length in range(len(iteration_path), -1, -1)
                        if iteration_path[:prefix_length] in prepared_by_path
                    ),
                    None,
                )
                if prepared_id is None:
                    break
                mapping.append((node_id, prepared_id))
            if len(mapping) == len(parent_node_ids):
                mappings.append(mapping)
        return mappings

    def _mark_source_node_empty(self, source_node_id: str, iteration_path: tuple[int, ...] = ()) -> None:
        self._state._tx_set_mapping(self._state.source_prepared_mapping, source_node_id, set())
        self._state._reset_unexecuted_prepared(source_node_id)
        self._state._mark_source_executed(source_node_id)
        if isinstance(self._state.graph.get_node(source_node_id), IfInvocation):
            self._queue_empty_if_branch_retirements(source_node_id)
        if isinstance(self._state.graph.get_node(source_node_id), IterateInvocation):
            self._state._record_empty_iterate_stream(source_node_id, iteration_path)
            self._queue_empty_iterator_retirements(source_node_id)

    def _queue_empty_if_branch_retirements(self, if_node_id: str) -> None:
        self._ensure_activation_candidate_index(self._state._get_source_graph_flat())
        for branch in ("true_input", "false_input"):
            for source_node_id in self._activation_sources_by_dependency.get((if_node_id, branch), ()):
                self._state._tx_add_set(self._pending_activation_retirement_candidates, source_node_id)

    def _queue_empty_iterator_retirements(self, iterator_id: str) -> None:
        self._ensure_activation_candidate_index(self._state._get_source_graph_flat())
        for source_node_id in self._activation_sources_by_iterator.get(iterator_id, ()):
            self._state._tx_add_set(self._pending_activation_retirement_candidates, source_node_id)

    def _has_completed_empty_nested_if_branch(self, node_id: str) -> bool:
        node = self._state.graph.get_node(node_id)
        if not isinstance(node, IfInvocation) or self._state.graph._get_input_edges(node_id, "condition"):
            return False
        selected_field = "true_input" if node.condition else "false_input"
        selected_edges = self._state.graph._get_input_edges(node_id, selected_field)
        if bool(selected_edges) and all(
            isinstance(self._state.graph.get_node(edge.source.node_id), IfInvocation)
            and edge.source.node_id in self._state.executed
            and not self._state.source_prepared_mapping.get(edge.source.node_id)
            for edge in selected_edges
        ):
            return True

        # A root If can have no frames when an unselected nested If is itself framed by an empty Iterate.
        unselected_field = "false_input" if node.condition else "true_input"
        unselected_edges = self._state.graph._get_input_edges(node_id, unselected_field)
        return bool(unselected_edges) and all(
            isinstance(self._state.graph.get_node(edge.source.node_id), IfInvocation)
            and not self._state.source_prepared_mapping.get(edge.source.node_id)
            and graph_if_runtime._has_empty_if_frame(self._state, edge.source.node_id)
            for edge in unselected_edges
        )

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
        next_node = self._state.graph.get_node(next_node_id)
        if isinstance(next_node, IterateInvocation) and len(input_edges) == 1:
            collection_edge = input_edges[0]
            if (
                collection_edge.destination.field == COLLECTION_FIELD
                and collection_edge.source.field == COLLECTION_FIELD
                and isinstance(self._state.graph.get_node(collection_edge.source.node_id), CollectInvocation)
            ):
                source_node_id = collection_edge.source.node_id
                return iter(
                    [(source_node_id, prepared_id)]
                    for prepared_id in self._get_ordered_prepared_nodes_for_edge(collection_edge)
                )
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
        iteration_axes: Optional[tuple[str, ...]] = None,
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
                new_node_axes = iteration_axes
                if new_node_axes is None:
                    parent_axes = []
                    for source_node_id, prepared_id in iteration_node_map:
                        axes = self._state._prepared_registry().get_iteration_axes(prepared_id)
                        if axes is None:
                            axes = self._get_iteration_axes(source_node_id)
                        if isinstance(self._state.graph.get_node(source_node_id), ForInvocation) and axes:
                            source_edges = [
                                edge
                                for edge in input_edges
                                if edge.source.node_id == source_node_id
                                and get_output_field_scope(
                                    self._state.graph.get_node(source_node_id), edge.source.field
                                )
                                == OutputScope.Final
                            ]
                            if source_edges and axes[-1] == source_node_id:
                                axes = axes[:-1]
                        parent_axes.append(axes)
                    new_node_axes = max(parent_axes, key=len, default=())
                    if any(new_node_axes[: len(axes)] != axes for axes in parent_axes):
                        new_node_axes = None
                    if new_node_axes is not None:
                        if isinstance(node, (ForInvocation, IterateInvocation)) and iteration_index >= 0:
                            new_node_axes = (*new_node_axes, node_id)
                        if len(new_node_axes) < len(new_node_iteration_path):
                            static_axes = self._get_iteration_axes(node_id)
                            if static_axes[: len(new_node_axes)] == new_node_axes and len(static_axes) >= len(
                                new_node_iteration_path
                            ):
                                new_node_axes = static_axes[: len(new_node_iteration_path)]
                            else:
                                new_node_axes = None
                        elif len(new_node_axes) > len(new_node_iteration_path):
                            new_node_axes = new_node_axes[: len(new_node_iteration_path)]
                if new_node_axes is not None:
                    self._state._prepared_registry().set_iteration_axes(new_node.id, new_node_axes)
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
        if isinstance(self._state.graph.get_node(node_id), CollectInvocation):
            if any(
                edge.destination.field == ITEM_FIELD
                and isinstance(self._state.graph.get_node(edge.source.node_id), IfInvocation)
                and any(
                    self._state._prepared_registry().get_source_node_id(exec_node_id) == edge.source.node_id
                    for exec_node_id in self._state._pending_if_exec_nodes
                )
                for edge in self._state.graph._get_input_edges(node_id)
            ):
                return False
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
                    return self._has_completed_empty_nested_if_branch(node_id) or graph_if_runtime._has_empty_if_frame(
                        self._state, node_id
                    )
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
                # A root condition's frames come from its resolved selected branch. Mapping every open axis here
                # duplicates root results when only the unselected branch carries an axis.
                condition_mappings = self._get_parent_iteration_mappings(
                    node_id, graph, input_edges=condition_edges, iterator_node_ids=[]
                )
                branch_edges = {
                    branch: self._state.graph._get_input_edges(node_id, branch)
                    for branch in ("true_input", "false_input")
                }

                def get_root_condition_mappings() -> Iterable[list[tuple[str, str]]]:
                    for mapping in condition_mappings:
                        condition_exec_id = dict(mapping)[condition_edges[0].source.node_id]
                        if condition_exec_id not in self._state.results:
                            continue
                        condition_value = getattr(
                            self._state.results[condition_exec_id], condition_edges[0].source.field
                        )
                        selected_field = "true_input" if condition_value else "false_input"
                        axes = self._get_ordered_iterator_axes_for_edges(branch_edges[selected_field], iterator_graph)
                        if not axes:
                            yield mapping
                            continue
                        axis_id = axes[-1]
                        prepared_axis_ids = (
                            self._get_prepared_nodes_for_source(axis_id)
                            if axis_id in self._state.source_prepared_mapping
                            else set()
                        )
                        for prepared_axis_id in sorted(prepared_axis_ids, key=self._state._get_iteration_path):
                            yield [*mapping, (axis_id, prepared_axis_id)]

                return get_root_condition_mappings()
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
        if condition_edges:
            if all(
                outer_id in iterator_ancestry.get_ordered_iterator_ancestors(inner_id)
                for outer_id, inner_id in itertools.pairwise(iterator_nodes)
            ):
                iterator_nodes = iterator_nodes[-1:]
        else:
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
            other_field = "false_input" if node.condition else "true_input"
            gating_source_ids = [
                edge.source.node_id for edge in self._state.graph._get_input_edges(node_id, selected_field)
            ]
            iterator_ancestry = self._get_iterator_ancestry(self.iterator_graph(graph))
            branch_axes = graph_if_runtime._get_if_static_frame_analysis(self._state, iterator_ancestry, node_id)[0]
            selected_axes = set(branch_axes[selected_field])
            for axis_id in set(branch_axes[other_field]) - selected_axes:
                if not isinstance(self._state.graph.get_node(axis_id), IterateInvocation):
                    continue
                collection_edges = self._state.graph._get_input_edges(axis_id, COLLECTION_FIELD)
                if graph_if_runtime._get_collects_before_iteration_boundary(self._state, collection_edges):
                    gating_source_ids.append(axis_id)
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
        """Resolve fresh If conditions before admitting branches; attach only the selected value input."""

        source_graph = self._state._get_source_graph_flat()
        self._ensure_activation_candidate_index(source_graph)
        # Prepared branch sources do not change during this sweep. Index them by frame once per source edge;
        # keeping the index local means graph edits, rollback, and restore rebuild it on the next call.
        prepared_exec_by_edge_path: dict[tuple[str, str], dict[tuple[int, ...], str]] = {}
        prepared_exec_by_edge_context: dict[tuple[str, str], dict[tuple[tuple[str, ...], tuple[int, ...]], str]] = {}
        source_order = self._activation_candidate_topological_index
        pending_exec_nodes = sorted(
            self._state._pending_if_exec_nodes,
            key=lambda exec_node_id: (
                source_order.get(self._state._prepared_registry().get_source_node_id(exec_node_id), 0),
                self._state._get_iteration_path(exec_node_id),
                exec_node_id,
            ),
        )
        for exec_node_id in pending_exec_nodes:
            source_if_id = self._state._prepared_registry().get_source_node_id(exec_node_id)
            fresh_if = not self._state._legacy_snapshot_loaded
            self._state._try_resolve_if_node(exec_node_id, enqueue=False)
            if fresh_if:
                selected_field = self._state._resolved_if_exec_branches.get(exec_node_id)
                if selected_field is None:
                    continue
                required_branch_fields = (
                    (selected_field,)
                    if self._state._is_fresh_lazy_if_candidate(source_if_id)
                    else ("true_input", "false_input")
                )
                branch_fields = (selected_field,)
                if any(
                    edge.source.node_id not in self._state.executed
                    for branch_field in required_branch_fields
                    for edge in self._state.graph._get_input_edges(source_if_id, branch_field)
                ):
                    continue
            else:
                selected_field = self._state._resolved_if_exec_branches.get(exec_node_id)
                if selected_field is None:
                    continue
                required_branch_fields = (selected_field,)
                branch_fields = (selected_field,)
            if all(
                not self._state.graph._get_input_edges(source_if_id, branch_field)
                or any(
                    edge.destination.field == branch_field
                    for edge in self._state.execution_graph._get_input_edges(exec_node_id)
                )
                for branch_field in branch_fields
            ):
                if fresh_if:
                    self._state._try_resolve_if_node(exec_node_id, enqueue=False)
                    if exec_node_id not in self._state._resolved_if_exec_branches:
                        continue
                if enqueue:
                    self._state._tx_discard_set(self._state._pending_if_exec_nodes, exec_node_id)
                continue

            target_path = self._state._get_iteration_path(exec_node_id)
            target_axes = self._state._prepared_registry().get_iteration_axes(exec_node_id)
            branch_edges = [
                (branch_field, edge)
                for branch_field in branch_fields
                for edge in self._state.graph._get_input_edges(source_if_id, branch_field)
            ]
            if any(
                isinstance(source_node := self._state.graph.get_node(edge.source.node_id), ForInvocation)
                and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
                and not self._state._all_for_contexts_finalized(source_node.id)
                for _, edge in branch_edges
            ):
                continue
            if (
                selected_field is not None
                and self._state.graph._get_input_edges(source_if_id, selected_field)
                and all(
                    source_edge.source.node_id in self._state.executed
                    and not self._state.source_prepared_mapping.get(source_edge.source.node_id)
                    for source_edge in self._state.graph._get_input_edges(source_if_id, selected_field)
                )
            ):
                # A selected branch can be a completed nested If with no executions. Eagerly running
                # the other branch must not turn that empty selected value into a None result.
                self._mark_source_node_empty(source_if_id)
                self._state._mark_exec_node_executed(exec_node_id)
                self._state._set_prepared_exec_state(exec_node_id, "skipped")
                if enqueue:
                    self._state._tx_discard_set(self._state._pending_if_exec_nodes, exec_node_id)
                continue
            attached_edges: list[Edge] = []
            selected_exec_ids: list[str] = []
            for branch_field, source_edge in branch_edges:
                if source_edge.source.node_id not in self._state.source_prepared_mapping:
                    continue
                edge_key = (source_edge.source.node_id, source_edge.source.field)
                if edge_key not in prepared_exec_by_edge_path:
                    prepared_exec_by_path: dict[tuple[int, ...], str] = {}
                    prepared_exec_by_context: dict[tuple[tuple[str, ...], tuple[int, ...]], str] = {}
                    for prepared_id in self._get_ordered_prepared_nodes_for_edge(source_edge):
                        prepared_path = self._get_prepared_edge_iteration_path(source_edge, prepared_id)
                        prepared_exec_by_path.setdefault(prepared_path, prepared_id)
                        prepared_axes = self._get_prepared_edge_iteration_axes(source_edge, prepared_id)
                        prepared_exec_by_context.setdefault((prepared_axes, prepared_path), prepared_id)
                    prepared_exec_by_edge_path[edge_key] = prepared_exec_by_path
                    prepared_exec_by_edge_context[edge_key] = prepared_exec_by_context

                prepared_exec_by_path = prepared_exec_by_edge_path[edge_key]
                selected_exec_id = None
                if target_axes is not None:
                    selected_exec_id = prepared_exec_by_edge_context[edge_key].get((target_axes, target_path))
                if selected_exec_id is None:
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
                        destination=EdgeConnection(node_id=exec_node_id, field=branch_field),
                    )
                )
            if any(
                self._state._is_pending_if(candidate_id)
                or (
                    isinstance(self._state.execution_graph.get_node(candidate_id), IfInvocation)
                    and candidate_id not in self._state.executed
                )
                for candidate_id in selected_exec_ids
            ):
                continue

            if attached_edges:
                self._attach_execution_edges(exec_node_id, attached_edges)
            if fresh_if and any(
                self._state.graph._get_input_edges(source_if_id, branch_field)
                and not any(
                    edge.destination.field == branch_field
                    for edge in self._state.execution_graph._get_input_edges(exec_node_id)
                )
                for branch_field in branch_fields
            ):
                continue
            if fresh_if:
                self._state._try_resolve_if_node(exec_node_id, enqueue=False)
                if exec_node_id not in self._state._resolved_if_exec_branches:
                    continue
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
            for exec_node_id in self._state.source_prepared_mapping.get(source_node_id, ())
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

    def _should_defer_shared_collect_if(self, node_id: str, iterator_ancestry: _IteratorAncestry) -> bool:
        if not isinstance(self._state.graph.get_node(node_id), IfInvocation):
            return False
        condition_edges = self._state.graph._get_input_edges(node_id, "condition")
        branch_edges = [self._state.graph._get_input_edges(node_id, field) for field in ("true_input", "false_input")]
        if not condition_edges or any(not edges for edges in branch_edges):
            return False

        branch_axes = graph_if_runtime._get_if_runtime_branch_frame_axes(self._state, iterator_ancestry, node_id)
        axes = [set(branch_axes[field]) for field in ("true_input", "false_input")]
        condition_axes = {
            axis_id for edge in condition_edges for axis_id in iterator_ancestry.get_frame_axes_for_edge(edge)
        }
        if axes[0] != axes[1] or not axes[0] > condition_axes:
            return False

        branch_collects = [
            graph_if_runtime._get_collects_before_iteration_boundary(self._state, edges) for edges in branch_edges
        ]
        shared_collects = branch_collects[0] & branch_collects[1]
        return any(source_id not in self._state.executed for source_id in shared_collects)

    def _expand_root_if_for_completed_collect_branch(self, graph: "nx.DiGraph") -> Optional[str]:
        """Align a root If frame with each completed selected nested-If frame."""
        iterator_graph = self.iterator_graph(graph)
        for if_node_id, node in self._state.graph.nodes.items():
            if not isinstance(node, IfInvocation):
                continue
            condition_edges = self._state.graph._get_input_edges(if_node_id, "condition")
            if condition_edges and (
                any(edge.source.node_id not in iterator_graph for edge in condition_edges)
                or any(
                    self._get_iterator_ancestry(iterator_graph).get_iterator_axes_for_edge(edge)
                    for edge in condition_edges
                )
            ):
                continue

            pending_if_exec_ids = [
                exec_id
                for exec_id in self._state.source_prepared_mapping.get(if_node_id, ())
                if self._state._is_pending_if(exec_id)
                and self._state._resolved_if_exec_branches.get(exec_id) is not None
            ]
            if not pending_if_exec_ids:
                continue
            root_exec_id = pending_if_exec_ids[0]
            selected_field = self._state._resolved_if_exec_branches[root_exec_id]
            selected_edges = self._state.graph._get_input_edges(if_node_id, selected_field)
            if not selected_edges or any(
                (
                    isinstance(self._state.graph.get_node(edge.source.node_id), IfInvocation)
                    and (
                        edge.source.node_id not in self._state.executed
                        or any(
                            prepared_id not in self._state.results
                            for prepared_id in self._state.source_prepared_mapping.get(edge.source.node_id, ())
                        )
                    )
                )
                or edge.source.node_id not in self._state.executed
                or any(
                    prepared_id not in self._state.results
                    for prepared_id in self._state.source_prepared_mapping.get(edge.source.node_id, ())
                )
                for edge in selected_edges
            ):
                continue

            branch_paths = graph_if_runtime._get_if_iteration_paths_with_ancestry(
                self._state,
                if_node_id,
                set(),
                self._get_prepared_nodes_for_source,
                self._get_prepared_edge_iteration_path,
                self._get_iterator_ancestry(iterator_graph),
            )
            if not branch_paths or branch_paths[0] == ():
                continue

            prepared_if_ids = self._state.source_prepared_mapping.get(if_node_id, ())
            exec_ids_by_path: dict[tuple[int, ...], list[str]] = {}
            for exec_id in prepared_if_ids:
                exec_ids_by_path.setdefault(self._state._get_iteration_path(exec_id), []).append(exec_id)
            if any(len(exec_ids) > 1 for exec_ids in exec_ids_by_path.values()):
                continue

            # A root condition can be prepared at an outer prefix before its selected branch reveals inner frames.
            # Graph validation keeps the open axes nested, so the integer path prefix has the same axis identity.
            frame_expansions: list[tuple[str, tuple[int, ...], list[tuple[str, str]], list[tuple[int, ...]]]] = []
            existing_paths = set(exec_ids_by_path)
            if any(
                self._state._resolved_if_exec_branches.get(exec_id) != selected_field for exec_id in prepared_if_ids
            ):
                continue
            missing_paths = set(branch_paths) - existing_paths
            if not missing_paths:
                continue
            targets_by_prefix: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
            valid_expansion = True
            for path in missing_paths:
                prefixes = [
                    existing_path
                    for existing_path in existing_paths
                    if len(existing_path) < len(path) and path[: len(existing_path)] == existing_path
                ]
                if not prefixes:
                    valid_expansion = False
                    break
                targets_by_prefix.setdefault(max(prefixes, key=len), []).append(path)
            if not valid_expansion or any(
                existing_path not in targets_by_prefix for existing_path in existing_paths - set(branch_paths)
            ):
                continue
            if any(
                exec_id not in pending_if_exec_ids
                for existing_path in targets_by_prefix
                if existing_path not in branch_paths
                for exec_id in exec_ids_by_path[existing_path]
            ):
                continue
            for existing_path, target_paths in targets_by_prefix.items():
                exec_id = exec_ids_by_path[existing_path][0]
                first_frame_path = existing_path if existing_path in branch_paths else target_paths[0]
                clone_paths = target_paths if existing_path in branch_paths else target_paths[1:]
                condition_mapping = [
                    (self._state.prepared_source_mapping[edge.source.node_id], edge.source.node_id)
                    for edge in self._state.execution_graph._get_input_edges(exec_id)
                    if edge.destination.field == "condition"
                ]
                frame_expansions.append((exec_id, first_frame_path, condition_mapping, clone_paths))
            if not frame_expansions:
                continue
            expanded_paths = {
                path
                for _, first_path, _, remaining_paths in frame_expansions
                for path in (first_path, *remaining_paths)
            }
            if expanded_paths | (existing_paths & set(branch_paths)) != set(branch_paths):
                continue
            return self._state._run_materialization_transaction(
                lambda graph=graph,
                if_node_id=if_node_id,
                root_exec_id=root_exec_id,
                frame_expansions=frame_expansions,
                branch_paths=branch_paths: self._materialize_root_if_for_completed_collect_branch(
                    graph, if_node_id, root_exec_id, frame_expansions, branch_paths
                )
            )

        return None

    def _materialize_root_if_for_completed_collect_branch(
        self,
        graph: "nx.DiGraph",
        if_node_id: str,
        root_exec_id: str,
        frame_expansions: list[tuple[str, tuple[int, ...], list[tuple[str, str]], list[tuple[int, ...]]]],
        branch_paths: list[tuple[int, ...]],
    ) -> str:
        """Expand resolved root If prefixes into their selected nested frames atomically."""
        first_path, *remaining_paths = branch_paths
        generic_runtime = self._state._generic_runtime()
        for exec_id, first_frame_path, condition_mapping, remaining_frame_paths in frame_expansions:
            if self._state._get_iteration_path(exec_id) != first_frame_path:
                self._state._prepared_registry().set_iteration_path(exec_id, first_frame_path)
                frame = self._state._get_execution_frame(exec_id)
                engine_frame = self._state._engine_frame(first_frame_path)
                gate = generic_runtime.gates.get(exec_id)
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
                    self._state._tx_record(
                        lambda gate=previous_gate, runtime=generic_runtime: runtime.replace_gate(gate)
                    )
                for token_id, token in tuple(self._state.execution_tokens.items()):
                    if token.owner_node_id == exec_id and token.token_kind == "activation":
                        self._state._tx_set_mapping(
                            self._state.execution_tokens,
                            token_id,
                            token.model_copy(update={"frame": frame}),
                        )

            for iteration_path in remaining_frame_paths:
                if condition_mapping:
                    self.create_execution_node(
                        if_node_id,
                        condition_mapping,
                        iteration_path,
                        input_fields={"condition"},
                    )
                else:
                    self.create_execution_node(if_node_id, [], iteration_path)

        self._state._invalidate_selected_if_frames_for_preparation(if_node_id)
        self._state._prepared_if_exec_indexed_sources.discard(if_node_id)
        for key in tuple(self._state._prepared_if_exec_ids_by_source_and_frame):
            if key[0] == if_node_id:
                self._state._prepared_if_exec_ids_by_source_and_frame.pop(key, None)
        self._state._prepared_if_exec_frames_by_source.pop(if_node_id, None)

        for consumer_id in graph.successors(if_node_id):
            if isinstance(
                self._state.graph.get_node(consumer_id),
                (IfInvocation, ForInvocation, IterateInvocation),
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

            def get_iteration_mapping(
                iteration_path: tuple[int, ...], consumer_source_id: str = consumer_id
            ) -> Optional[list[tuple[str, str]]]:
                mapping_by_source: dict[str, str] = {}
                for edge in self._state.graph._get_input_edges(consumer_source_id):
                    if edge.source.node_id in mapping_by_source:
                        continue
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
                        return None
                    if matching_source is not None:
                        mapping_by_source[edge.source.node_id] = matching_source
                return list(mapping_by_source.items())

            root_iteration_mapping = get_iteration_mapping(first_path)
            if root_iteration_mapping is None:
                continue
            self._state._prepared_registry().set_iteration_path(root_consumer_id, first_path)
            scheduler = self._state._scheduler()
            scheduler.remove_from_ready_queues(root_consumer_id)
            old_input_edges = self._state.execution_graph._get_input_edges(root_consumer_id)
            execution_graph_flat = self._state.__pydantic_private__["_runtime_state"].execution_graph_flat
            for edge in old_input_edges:
                self._state._tx_delete_execution_edge(edge)
                if (
                    execution_graph_flat is not None
                    and edge.type == "default"
                    and execution_graph_flat.has_edge(edge.source.node_id, root_consumer_id)
                    and not any(
                        candidate.type == "default"
                        and candidate.source.node_id == edge.source.node_id
                        and candidate.destination.node_id == root_consumer_id
                        for candidate in self._state.execution_graph.edges
                    )
                ):
                    execution_graph_flat.remove_edge(edge.source.node_id, root_consumer_id)
            attached_edges = self._attach_execution_edges(
                root_consumer_id,
                self._build_execution_edges(consumer_id, root_iteration_mapping),
            )
            self._initialize_execution_node(root_consumer_id, attached_edges)
            for iteration_path in remaining_paths:
                iteration_mapping = get_iteration_mapping(iteration_path)
                if iteration_mapping is not None:
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
                    and self._is_deferred_nested_for_return(node_id)
                )
                and (
                    isinstance(self._state.graph.get_node(node_id), IfInvocation)
                    and bool(self._state.graph._get_input_edges(node_id, "condition"))
                    or not self._has_unmaterializable_for_final_input(node_id)
                )
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
                    or isinstance(self._state.graph.get_node(node_id), CollectInvocation)
                    and self._has_ready_empty_collect_group(node_id)
                )
                and (
                    isinstance(self._state.graph.get_node(node_id), IfInvocation)
                    and self._is_if_condition_ready(node_id)
                    or not isinstance(self._state.graph.get_node(node_id), IfInvocation)
                )
                and not self._should_defer_shared_collect_if(node_id, iterator_ancestry)
                and not (
                    isinstance(self._state.graph.get_node(node_id), IfInvocation)
                    and graph_if_runtime._has_unresolved_if_frame_axes(self._state, node_id, iterator_ancestry)
                    and not self._has_completed_empty_nested_if_branch(node_id)
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
            collect_inputs = self._state.graph._get_input_edges(next_node_id)
            nested_literal_sources = [
                edge.source.node_id
                for edge in collect_inputs
                if edge.destination.field == ITEM_FIELD
                and isinstance(self._state.graph.get_node(edge.source.node_id), IfInvocation)
                and not self._state.graph._get_input_edges(edge.source.node_id, "condition")
                and any(
                    isinstance(self._state.graph.get_node(branch_edge.source.node_id), IfInvocation)
                    for branch in ("true_input", "false_input")
                    for branch_edge in self._state.graph._get_input_edges(edge.source.node_id, branch)
                )
            ]
            prepared_nested_ids = [
                prepared_id
                for source_id in dict.fromkeys(nested_literal_sources)
                for prepared_id in self._get_ordered_prepared_nodes_for_source(source_id)
            ]
            nested_literal_source_ids = set(nested_literal_sources)
            iterator_ancestry = self._get_iterator_ancestry(self.iterator_graph())
            prepared_ids_by_edge: dict[tuple[str, str, str, str], list[str]] = {}
            prepared_paths_by_edge: dict[tuple[str, str, str, str], dict[str, tuple[int, ...]]] = {}
            selected_parent_axes_by_source: dict[str, tuple[str, ...]] = {}
            selected_source_axes_by_source: dict[str, tuple[str, ...]] = {}
            static_source_axes_by_source: dict[str, tuple[str, ...]] = {}
            root_item_by_edge: dict[tuple[str, str, str, str], bool] = {}

            def input_edge_key(edge: Edge) -> tuple[str, str, str, str]:
                return (
                    edge.source.node_id,
                    edge.source.field,
                    edge.destination.node_id,
                    edge.destination.field,
                )

            def get_prepared_ids(edge: Edge) -> list[str]:
                key = input_edge_key(edge)
                if key not in prepared_ids_by_edge:
                    prepared_ids_by_edge[key] = self._get_ordered_prepared_nodes_for_edge(edge)
                return prepared_ids_by_edge[key]

            def get_prepared_path(edge: Edge, prepared_id: str) -> tuple[int, ...]:
                key = input_edge_key(edge)
                paths = prepared_paths_by_edge.setdefault(key, {})
                if prepared_id not in paths:
                    paths[prepared_id] = self._get_prepared_edge_iteration_path(edge, prepared_id)
                return paths[prepared_id]

            def get_selected_parent_axes(source_id: str) -> tuple[str, ...]:
                if source_id not in selected_parent_axes_by_source:
                    selected_parent_axes_by_source[source_id] = self._get_selected_for_final_parent_axes(
                        source_id, iterator_ancestry
                    )
                return selected_parent_axes_by_source[source_id]

            def get_static_source_axes(source_id: str) -> tuple[str, ...]:
                if source_id not in static_source_axes_by_source:
                    static_source_axes_by_source[source_id] = tuple(self._get_collect_source_iterator_ids(source_id))
                return static_source_axes_by_source[source_id]

            def get_selected_source_axes(source_id: str) -> tuple[str, ...]:
                if source_id not in selected_source_axes_by_source:
                    selected_axes = (
                        get_selected_parent_axes(source_id)
                        if isinstance(self._state.graph.get_node(source_id), IfInvocation)
                        else ()
                    )
                    selected_source_axes_by_source[source_id] = selected_axes or get_static_source_axes(source_id)
                return selected_source_axes_by_source[source_id]

            def is_root_item_source(edge: Edge) -> bool:
                key = input_edge_key(edge)
                if key not in root_item_by_edge:
                    prepared_ids = get_prepared_ids(edge)
                    root_item_by_edge[key] = (
                        edge.destination.field == ITEM_FIELD
                        and not get_static_source_axes(edge.source.node_id)
                        and bool(prepared_ids)
                        and all(not get_prepared_path(edge, prepared_id) for prepared_id in prepared_ids)
                    )
                return root_item_by_edge[key]

            item_edges = [edge for edge in collect_inputs if edge.destination.field == ITEM_FIELD]
            selected_axis_prefixes = {
                axes[:prefix_depth]
                for source_id in (edge.source.node_id for edge in item_edges)
                for axes in (get_selected_parent_axes(source_id),)
                for prefix_depth in range(len(axes))
            }

            prepared_item_groups: set[tuple[tuple[str, ...], tuple[int, ...]]] = set()
            for input_edge in item_edges:
                source_id = input_edge.source.node_id
                if is_root_item_source(input_edge):
                    continue
                source_axes = get_selected_source_axes(source_id)
                if source_id in nested_literal_source_ids:
                    selected_for_axes = get_selected_parent_axes(source_id)
                    enclosing_selected_for_axes = source_axes in selected_axis_prefixes
                    group_axes = selected_for_axes or (
                        source_axes if source_axes and enclosing_selected_for_axes else source_axes[:-1]
                    )
                    prepared_item_groups.update(
                        (group_axes[: len(path)], path[: len(group_axes)])
                        for candidate_id in get_prepared_ids(input_edge)
                        if len(path := get_prepared_path(input_edge, candidate_id)) <= len(source_axes)
                    )
                else:
                    prepared_item_groups.update(
                        (group_axes, path)
                        for group_axes, path, _ in self._get_collect_iteration_mapping_groups([input_edge])
                    )
            prepared_item_groups = self._remove_collect_group_prefixes(prepared_item_groups)
            if (
                self._collect_has_selected_for_final_input(next_node_id)
                and len(prepared_nested_ids) > 1
                and len(prepared_item_groups) > 1
            ):
                group_keys_by_depths: dict[tuple[int, int], set[tuple[tuple[str, ...], tuple[int, ...]]]] = defaultdict(
                    set
                )
                for group_key in prepared_item_groups:
                    group_keys_by_depths[(len(group_key[0]), len(group_key[1]))].add(group_key)
                groups_by_candidate_prefix: dict[
                    tuple[int, int],
                    dict[tuple[tuple[str, ...], tuple[int, ...]], list[tuple[tuple[str, ...], tuple[int, ...]]]],
                ] = {}

                def get_candidate_ancestor_groups(
                    candidate_axes: tuple[str, ...], candidate_path: tuple[int, ...]
                ) -> list[tuple[tuple[str, ...], tuple[int, ...]]]:
                    prefix_depths = (len(candidate_axes), len(candidate_path))
                    index = groups_by_candidate_prefix.get(prefix_depths)
                    if index is None:
                        index = defaultdict(list)
                        for group_axes, group_path in prepared_item_groups:
                            if len(group_axes) >= prefix_depths[0] and len(group_path) >= prefix_depths[1]:
                                index[(group_axes[: prefix_depths[0]], group_path[: prefix_depths[1]])].append(
                                    (group_axes, group_path)
                                )
                        groups_by_candidate_prefix[prefix_depths] = index
                    return index.get((candidate_axes, candidate_path), [])

                grouped_item_mappings: list[
                    Optional[dict[tuple[tuple[str, ...], tuple[int, ...]], list[tuple[str, str]]]]
                ] = []
                special_item_candidates_by_edge: dict[
                    tuple[str, str, str, str], dict[tuple[tuple[str, ...], tuple[int, ...]], list[str]]
                ] = {}
                for input_edge in collect_inputs:
                    source_id = input_edge.source.node_id
                    is_root_item = is_root_item_source(input_edge)
                    if (
                        input_edge.destination.field != ITEM_FIELD
                        or source_id in nested_literal_source_ids
                        or is_root_item
                    ):
                        grouped_item_mappings.append(None)
                        source_axes = get_selected_source_axes(source_id)
                        candidates_by_group: dict[tuple[tuple[str, ...], tuple[int, ...]], list[str]] = defaultdict(
                            list
                        )
                        for candidate_id in get_prepared_ids(input_edge):
                            candidate_path = get_prepared_path(input_edge, candidate_id)
                            candidate_axes = source_axes[: len(candidate_path)]
                            candidate_groups = set(get_candidate_ancestor_groups(candidate_axes, candidate_path))
                            for group_depths, groups_at_depth in group_keys_by_depths.items():
                                group_axes_depth, group_path_depth = group_depths
                                if group_axes_depth < len(candidate_axes) and group_path_depth <= len(candidate_path):
                                    group_key = (candidate_axes[:group_axes_depth], candidate_path[:group_path_depth])
                                    if group_key in groups_at_depth:
                                        candidate_groups.add(group_key)
                            for group_axes, iteration_path in candidate_groups:
                                candidate_is_ancestor = (
                                    len(candidate_axes) <= len(group_axes)
                                    and group_axes[: len(candidate_axes)] == candidate_axes
                                    and iteration_path[: len(candidate_path)] == candidate_path
                                )
                                candidate_is_descendant = (
                                    len(candidate_axes) > len(group_axes)
                                    and candidate_axes[: len(group_axes)] == group_axes
                                    and candidate_path[: len(iteration_path)] == iteration_path
                                )
                                if candidate_is_ancestor or (
                                    input_edge.destination.field == ITEM_FIELD and candidate_is_descendant
                                ):
                                    candidates_by_group[(group_axes, iteration_path)].append(candidate_id)
                        special_item_candidates_by_edge[input_edge_key(input_edge)] = candidates_by_group
                        continue
                    grouped_item_mappings.append(
                        {
                            (candidate_axes, iteration_path): iteration_mappings
                            for candidate_axes, iteration_path, iteration_mappings in (
                                self._get_collect_iteration_mapping_groups([input_edge])
                            )
                        }
                    )
                has_nonroot_prepared_group = any(axes for axes, _ in prepared_item_groups)
                for group_axes, iteration_path in sorted(prepared_item_groups):
                    iteration_mapping: list[tuple[str, str]] = []
                    for input_edge, item_mappings in zip(collect_inputs, grouped_item_mappings, strict=True):
                        source_id = input_edge.source.node_id
                        if item_mappings is not None:
                            iteration_mapping.extend(item_mappings.get((group_axes, iteration_path), ()))
                            continue

                        source_axes = get_selected_source_axes(source_id)
                        is_root_item = is_root_item_source(input_edge)
                        if is_root_item and not group_axes and has_nonroot_prepared_group:
                            continue
                        # A selected nested branch stays out of a separate root item group.
                        if (
                            input_edge.destination.field == ITEM_FIELD
                            and not group_axes
                            and get_selected_parent_axes(source_id)
                        ):
                            continue
                        iteration_mapping.extend(
                            (source_id, candidate_id)
                            for candidate_id in special_item_candidates_by_edge.get(input_edge_key(input_edge), {}).get(
                                (group_axes, iteration_path), ()
                            )
                        )
                    new_node_ids.extend(
                        self.create_execution_node(
                            next_node_id,
                            iteration_mapping,
                            iteration_path,
                            iteration_axes=group_axes,
                        )
                    )
            else:
                iteration_mapping_groups = self._get_collect_iteration_mapping_groups(collect_inputs)
                for group_axes, iteration_path, iteration_mappings in iteration_mapping_groups:
                    create_results = self.create_execution_node(
                        next_node_id, iteration_mappings, iteration_path, iteration_axes=group_axes
                    )
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
                    condition_axes = tuple(
                        self._get_ordered_iterator_axes_for_edges(condition_edges, self.iterator_graph(g))
                    )
                    branch_edges = [
                        edge
                        for branch in ("true_input", "false_input")
                        for edge in self._state.graph._get_input_edges(next_node_id, branch)
                    ]
                    branch_axes = [
                        tuple(self._get_collect_source_iterator_ids(edge.source.node_id)) for edge in branch_edges
                    ]
                    condition_frames_are_exact = (
                        bool(condition_axes)
                        and any(axes == condition_axes for axes in branch_axes)
                        and all(not axes or axes == condition_axes for axes in branch_axes)
                    )
                    if condition_frames_are_exact:
                        for iteration_mapping in iteration_mappings:
                            condition_exec_id = dict(iteration_mapping).get(condition_edges[0].source.node_id)
                            if condition_exec_id is None:
                                continue
                            condition_path = self._get_prepared_edge_iteration_path(
                                condition_edges[0], condition_exec_id
                            )
                            axes = self._state._prepared_registry().get_iteration_axes(condition_exec_id)
                            if axes is None:
                                axes = self._get_prepared_edge_iteration_axes(condition_edges[0], condition_exec_id)
                            new_node_ids.extend(
                                self.create_execution_node(
                                    next_node_id,
                                    iteration_mapping,
                                    condition_path,
                                    input_fields={"condition"},
                                    iteration_axes=axes,
                                )
                            )
                    else:
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
                            condition_exec_ids = self._state.source_prepared_mapping.get(
                                condition_edges[0].source.node_id, ()
                            )
                            condition_exec_id = next(iter(condition_exec_ids), None)
                            has_unresolved_nested_if = False
                            if condition_exec_id in self._state.results:
                                selected_field = (
                                    "true_input"
                                    if getattr(self._state.results[condition_exec_id], condition_edges[0].source.field)
                                    else "false_input"
                                )
                                has_unresolved_nested_if = any(
                                    isinstance(self._state.graph.get_node(edge.source.node_id), IfInvocation)
                                    and self._state.graph._get_input_edges(edge.source.node_id, "condition")
                                    and not self._state.source_prepared_mapping.get(edge.source.node_id)
                                    for edge in self._state.graph._get_input_edges(next_node_id, selected_field)
                                )
                            if not matching_mappings and not iteration_path and not has_unresolved_nested_if:
                                matching_mappings = list(
                                    self._get_parent_iteration_mappings(
                                        next_node_id, g, input_edges=condition_edges, iterator_node_ids=[]
                                    )
                                )
                            for iteration_mapping in matching_mappings:
                                create_results = self.create_execution_node(
                                    next_node_id,
                                    iteration_mapping,
                                    iteration_path,
                                    input_fields={"condition"},
                                )
                                new_node_ids.extend(create_results)
                elif not self.get_node_iterators(
                    next_node_id
                ) and not graph_if_runtime._has_empty_selected_for_final_frame(
                    self._state, iterator_ancestry, next_node_id
                ):
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
                selected_field = "true_input" if next_node.condition else "false_input"
                selected_edges = self._state.graph._get_input_edges(next_node_id, selected_field)
                selected_empty_nested = bool(selected_edges) and all(
                    isinstance(self._state.graph.get_node(edge.source.node_id), IfInvocation)
                    and edge.source.node_id in self._state.executed
                    and not self._state.source_prepared_mapping.get(edge.source.node_id)
                    for edge in selected_edges
                )
                if selected_empty_nested:
                    create_results = self.create_execution_node(next_node_id, [], ())
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
