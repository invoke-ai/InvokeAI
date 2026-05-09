# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import copy
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, Literal, Optional, Type, TypeVar, Union, get_args, get_origin

import networkx as nx
from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    PrivateAttr,
    ValidationError,
    field_validator,
)
from pydantic.fields import Field
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

# Importing * is bad karma but needed here for node detection
from invokeai.app.invocations import *  # noqa: F401 F403
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationRegistry,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import Input, InputField, OutputField, UIType
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import uuid_string

# in 3.10 this would be "from types import NoneType"
NoneType = type(None)

# Port name constants
ITEM_FIELD = "item"
COLLECTION_FIELD = "collection"


class EdgeConnection(BaseModel):
    node_id: str = Field(description="The id of the node for this edge connection")
    field: str = Field(description="The field for this connection")

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and getattr(other, "node_id", None) == self.node_id
            and getattr(other, "field", None) == self.field
        )

    def __hash__(self):
        return hash(f"{self.node_id}.{self.field}")


class Edge(BaseModel):
    source: EdgeConnection = Field(description="The connection for the edge's from node and field")
    destination: EdgeConnection = Field(description="The connection for the edge's to node and field")

    def __str__(self):
        return f"{self.source.node_id}.{self.source.field} -> {self.destination.node_id}.{self.destination.field}"


PreparedExecState = Literal["pending", "ready", "executed", "skipped"]


@dataclass
class _PreparedExecNodeMetadata:
    """Cached metadata for a materialized execution node."""

    source_node_id: str
    iteration_path: Optional[tuple[int, ...]] = None
    state: PreparedExecState = "pending"


class _PreparedExecRegistry:
    """Tracks prepared execution nodes and their relationship to source graph nodes."""

    def __init__(
        self,
        prepared_source_mapping: dict[str, str],
        source_prepared_mapping: dict[str, set[str]],
        metadata: dict[str, _PreparedExecNodeMetadata],
    ) -> None:
        self._prepared_source_mapping = prepared_source_mapping
        self._source_prepared_mapping = source_prepared_mapping
        self._metadata = metadata

    def register(self, exec_node_id: str, source_node_id: str) -> None:
        self._prepared_source_mapping[exec_node_id] = source_node_id
        self._metadata[exec_node_id] = _PreparedExecNodeMetadata(source_node_id=source_node_id)
        if source_node_id not in self._source_prepared_mapping:
            self._source_prepared_mapping[source_node_id] = set()
        self._source_prepared_mapping[source_node_id].add(exec_node_id)

    def get_metadata(self, exec_node_id: str) -> _PreparedExecNodeMetadata:
        metadata = self._metadata.get(exec_node_id)
        if metadata is None:
            metadata = _PreparedExecNodeMetadata(source_node_id=self._prepared_source_mapping[exec_node_id])
            self._metadata[exec_node_id] = metadata
        return metadata

    def get_source_node_id(self, exec_node_id: str) -> str:
        metadata = self._metadata.get(exec_node_id)
        if metadata is not None:
            return metadata.source_node_id
        return self._prepared_source_mapping[exec_node_id]

    def get_prepared_ids(self, source_node_id: str) -> set[str]:
        return self._source_prepared_mapping.get(source_node_id, set())

    def set_state(self, exec_node_id: str, state: PreparedExecState) -> None:
        self.get_metadata(exec_node_id).state = state

    def get_iteration_path(self, exec_node_id: str) -> Optional[tuple[int, ...]]:
        metadata = self._metadata.get(exec_node_id)
        return metadata.iteration_path if metadata is not None else None

    def set_iteration_path(self, exec_node_id: str, iteration_path: tuple[int, ...]) -> None:
        self.get_metadata(exec_node_id).iteration_path = iteration_path


class _IfBranchScheduler:
    """Applies lazy `If` semantics by deferring, releasing, and skipping branch-local exec nodes."""

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state

    def _get_branch_input_sources(self, if_node_id: str, branch_field: str) -> set[str]:
        return {e.source.node_id for e in self._state.graph._get_input_edges(if_node_id, branch_field)}

    def _expand_with_ancestors(self, node_ids: set[str]) -> set[str]:
        expanded = set(node_ids)
        source_graph = self._state.graph.nx_graph_flat()
        for node_id in list(expanded):
            expanded.update(nx.ancestors(source_graph, node_id))
        return expanded

    def _node_outputs_stay_in_branch(
        self, node_id: str, if_node_id: str, branch_field: str, branch_nodes: set[str]
    ) -> bool:
        output_edges = self._state.graph._get_output_edges(node_id)
        return all(
            edge.destination.node_id in branch_nodes
            or (edge.destination.node_id == if_node_id and edge.destination.field == branch_field)
            for edge in output_edges
        )

    def _prune_nonexclusive_branch_nodes(
        self, if_node_id: str, branch_field: str, candidate_nodes: set[str]
    ) -> set[str]:
        exclusive_nodes = set(candidate_nodes)
        changed = True
        while changed:
            changed = False
            for node_id in list(exclusive_nodes):
                if self._node_outputs_stay_in_branch(node_id, if_node_id, branch_field, exclusive_nodes):
                    continue
                exclusive_nodes.remove(node_id)
                changed = True
        return exclusive_nodes

    def _get_matching_prepared_if_ids(self, if_node_id: str, iteration_path: tuple[int, ...]) -> list[str]:
        prepared_if_ids = self._state._prepared_registry().get_prepared_ids(if_node_id)
        return [pid for pid in prepared_if_ids if self._state._get_iteration_path(pid) == iteration_path]

    def _has_unresolved_matching_if(self, if_node_id: str, iteration_path: tuple[int, ...]) -> bool:
        matching_prepared_if_ids = self._get_matching_prepared_if_ids(if_node_id, iteration_path)
        if not matching_prepared_if_ids:
            return True
        return not all(pid in self._state._resolved_if_exec_branches for pid in matching_prepared_if_ids)

    def _apply_condition_inputs(self, exec_node_id: str, node: IfInvocation) -> bool:
        return self._state._apply_if_condition_inputs(exec_node_id, node)

    def _get_selected_branch_fields(self, node: IfInvocation) -> tuple[str, str]:
        selected_field = "true_input" if node.condition else "false_input"
        unselected_field = "false_input" if node.condition else "true_input"
        return selected_field, unselected_field

    def _prune_unselected_if_inputs(self, exec_node_id: str, unselected_field: str) -> None:
        for edge in self._state.execution_graph._get_input_edges(exec_node_id, unselected_field):
            if edge.source.node_id not in self._state.executed:
                if self._state.indegree[exec_node_id] == 0:
                    raise RuntimeError(f"indegree underflow for {exec_node_id} when pruning {unselected_field}")
                self._state.indegree[exec_node_id] -= 1
            self._state.execution_graph.delete_edge(edge)

    def _apply_branch_resolution(
        self,
        exec_node_id: str,
        iteration_path: tuple[int, ...],
        exclusive_sources: dict[str, set[str]],
        selected_field: str,
        unselected_field: str,
    ) -> None:
        # This iterates over the stable prepared-source mapping while mutating per-exec runtime state such as ready
        # queues, execution state, and prepared metadata. Branch resolution never adds or removes prepared exec nodes.
        for prepared_id, prepared_source in self._state.prepared_source_mapping.items():
            if prepared_id in self._state.executed:
                continue
            if self._state._get_iteration_path(prepared_id) != iteration_path:
                continue
            if prepared_source in exclusive_sources[selected_field]:
                self._state._enqueue_if_ready(prepared_id)
            elif prepared_source in exclusive_sources[unselected_field]:
                self.mark_exec_node_skipped(prepared_id)

    def get_branch_exclusive_sources(self, if_node_id: str) -> dict[str, set[str]]:
        cached = self._state._if_branch_exclusive_sources.get(if_node_id)
        if cached is not None:
            return cached

        branch_sources: dict[str, set[str]] = {}
        for branch_field in ("true_input", "false_input"):
            direct_inputs = self._get_branch_input_sources(if_node_id, branch_field)
            candidate_nodes = self._expand_with_ancestors(direct_inputs)
            branch_sources[branch_field] = self._prune_nonexclusive_branch_nodes(
                if_node_id, branch_field, candidate_nodes
            )

        self._state._if_branch_exclusive_sources[if_node_id] = branch_sources
        return branch_sources

    def is_deferred_by_unresolved_if(self, exec_node_id: str) -> bool:
        source_node_id = self._state._prepared_registry().get_source_node_id(exec_node_id)
        iteration_path = self._state._get_iteration_path(exec_node_id)

        for source_if_id, source_if_node in self._state.graph.nodes.items():
            if not isinstance(source_if_node, IfInvocation):
                continue

            branches = self.get_branch_exclusive_sources(source_if_id)
            if source_node_id not in branches["true_input"] and source_node_id not in branches["false_input"]:
                continue

            if self._has_unresolved_matching_if(source_if_id, iteration_path):
                return True
        return False

    def mark_exec_node_skipped(self, exec_node_id: str) -> None:
        state = self._state._get_prepared_exec_metadata(exec_node_id).state
        if state in ("executed", "skipped"):
            return

        self._state._remove_from_ready_queues(exec_node_id)
        self._state._set_prepared_exec_state(exec_node_id, "skipped")
        self._state.executed.add(exec_node_id)

        registry = self._state._prepared_registry()
        source_node_id = registry.get_source_node_id(exec_node_id)
        prepared_nodes = registry.get_prepared_ids(source_node_id)
        if all(n in self._state.executed for n in prepared_nodes):
            if source_node_id not in self._state.executed:
                self._state.executed.add(source_node_id)
                self._state.executed_history.append(source_node_id)

    def try_resolve_if_node(self, exec_node_id: str) -> None:
        if exec_node_id in self._state._resolved_if_exec_branches:
            return
        node = self._state.execution_graph.get_node(exec_node_id)
        if not isinstance(node, IfInvocation):
            return

        if not self._apply_condition_inputs(exec_node_id, node):
            return

        selected_field, unselected_field = self._get_selected_branch_fields(node)
        self._state._resolved_if_exec_branches[exec_node_id] = selected_field

        source_if_node_id = self._state._prepared_registry().get_source_node_id(exec_node_id)
        exclusive_sources = self.get_branch_exclusive_sources(source_if_node_id)

        iteration_path = self._state._get_iteration_path(exec_node_id)
        self._prune_unselected_if_inputs(exec_node_id, unselected_field)
        self._apply_branch_resolution(exec_node_id, iteration_path, exclusive_sources, selected_field, unselected_field)
        self._state._enqueue_if_ready(exec_node_id)


class _ExecutionMaterializer:
    """Expands source-graph nodes into concrete execution-graph nodes for the current runtime state.

    `GraphExecutionState.next()` calls into this helper when no prepared exec node is ready. The materializer chooses
    the next source node that can be expanded, creates the corresponding exec nodes in the execution graph, wires their
    inputs, and initializes their scheduler state.
    """

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state

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

    def _get_new_node_iterations(
        self, node: BaseInvocation, node_id: str, iteration_node_map: list[tuple[str, str]]
    ) -> list[int]:
        if not isinstance(node, IterateInvocation):
            return [-1]

        iteration_count = self._get_iterator_iteration_count(node_id, iteration_node_map)
        if iteration_count == 0:
            return []
        return list(range(iteration_count))

    def _build_execution_edges(self, node_id: str, iteration_node_map: list[tuple[str, str]]) -> list[Edge]:
        input_edges = self._state.graph._get_input_edges(node_id)
        new_edges: list[Edge] = []
        for edge in input_edges:
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

    def _create_execution_node_copy(self, node: BaseInvocation, node_id: str, iteration_index: int) -> BaseInvocation:
        new_node = node.model_copy(deep=True)
        new_node.id = uuid_string()

        if isinstance(new_node, IterateInvocation):
            new_node.index = iteration_index

        self._state.execution_graph.add_node(new_node)
        self._state._register_prepared_exec_node(new_node.id, node_id)
        return new_node

    def _attach_execution_edges(self, exec_node_id: str, new_edges: list[Edge]) -> None:
        for edge in new_edges:
            self._state.execution_graph.add_edge(
                Edge(
                    source=edge.source,
                    destination=EdgeConnection(node_id=exec_node_id, field=edge.destination.field),
                )
            )

    def _initialize_execution_node(self, exec_node_id: str) -> None:
        inputs = self._state.execution_graph._get_input_edges(exec_node_id)
        unmet = sum(1 for edge in inputs if edge.source.node_id not in self._state.executed)
        self._state.indegree[exec_node_id] = unmet
        self._state._try_resolve_if_node(exec_node_id)
        self._state._enqueue_if_ready(exec_node_id)

    def _get_collect_iteration_mappings(self, parent_node_ids: list[str]) -> list[tuple[str, str]]:
        all_iteration_mappings: list[tuple[str, str]] = []
        for source_node_id in parent_node_ids:
            prepared_nodes = self._get_prepared_nodes_for_source(source_node_id)
            all_iteration_mappings.extend((source_node_id, prepared_id) for prepared_id in prepared_nodes)
        return all_iteration_mappings

    def _get_parent_iteration_mappings(self, next_node_id: str, graph: nx.DiGraph) -> list[list[tuple[str, str]]]:
        parent_node_ids = [source_id for source_id, _ in graph.in_edges(next_node_id)]
        iterator_graph = self.iterator_graph(graph)
        iterator_nodes = self.get_node_iterators(next_node_id, iterator_graph)
        iterator_nodes_prepared = [list(self._state.source_prepared_mapping[node_id]) for node_id in iterator_nodes]
        iterator_node_prepared_combinations = list(itertools.product(*iterator_nodes_prepared))

        execution_graph = self._state.execution_graph.nx_graph_flat()
        prepared_parent_mappings = [
            [
                (node_id, self.get_iteration_node(node_id, graph, execution_graph, prepared_iterators))
                for node_id in parent_node_ids
            ]
            for prepared_iterators in iterator_node_prepared_combinations
        ]
        return [
            mapping
            for mapping in prepared_parent_mappings
            if all(prepared_id is not None for _, prepared_id in mapping)
        ]

    def create_execution_node(self, node_id: str, iteration_node_map: list[tuple[str, str]]) -> list[str]:
        """Prepares an iteration node and connects all edges, returning the new node id"""

        node = self._state.graph.get_node(node_id)
        iteration_indexes = self._get_new_node_iterations(node, node_id, iteration_node_map)
        if not iteration_indexes:
            return []

        new_edges = self._build_execution_edges(node_id, iteration_node_map)
        new_nodes: list[str] = []
        for iteration_index in iteration_indexes:
            new_node = self._create_execution_node_copy(node, node_id, iteration_index)
            self._attach_execution_edges(new_node.id, new_edges)
            self._initialize_execution_node(new_node.id)
            new_nodes.append(new_node.id)

        return new_nodes

    def iterator_graph(self, base: Optional[nx.DiGraph] = None) -> nx.DiGraph:
        """Gets a DiGraph with edges to collectors removed so an ancestor search produces all active iterators for any node"""
        g = base.copy() if base is not None else self._state.graph.nx_graph_flat()
        collectors = (
            n for n in self._state.graph.nodes if isinstance(self._state.graph.get_node(n), CollectInvocation)
        )
        for c in collectors:
            g.remove_edges_from(list(g.in_edges(c)))
        return g

    def get_node_iterators(self, node_id: str, it_graph: Optional[nx.DiGraph] = None) -> list[str]:
        g = it_graph or self.iterator_graph()
        return [n for n in nx.ancestors(g, node_id) if isinstance(self._state.graph.get_node(n), IterateInvocation)]

    def _get_prepared_nodes_for_source(self, source_node_id: str) -> set[str]:
        return {
            exec_node_id
            for exec_node_id in self._state.source_prepared_mapping[source_node_id]
            if self._state._get_prepared_exec_metadata(exec_node_id).state != "skipped"
        }

    def _get_parent_iterator_exec_nodes(
        self, source_node_id: str, graph: nx.DiGraph, prepared_iterator_nodes: list[str]
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
        self, candidate_exec_node_id: str, parent_iterators: list[tuple[str, str]], execution_graph: nx.DiGraph
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
        execution_graph: nx.DiGraph,
    ) -> Optional[str]:
        prepared_iterator = next((node_id for node_id in prepared_nodes if node_id in prepared_iterator_nodes), None)
        if prepared_iterator is None:
            return None
        if self._matches_parent_iterators(prepared_iterator, parent_iterators, execution_graph):
            return prepared_iterator
        return None

    def _find_prepared_node_matching_iterators(
        self, prepared_nodes: set[str], parent_iterators: list[tuple[str, str]], execution_graph: nx.DiGraph
    ) -> Optional[str]:
        return next(
            (
                node_id
                for node_id in prepared_nodes
                if self._matches_parent_iterators(node_id, parent_iterators, execution_graph)
            ),
            None,
        )

    def get_iteration_node(
        self,
        source_node_id: str,
        graph: nx.DiGraph,
        execution_graph: nx.DiGraph,
        prepared_iterator_nodes: list[str],
    ) -> Optional[str]:
        prepared_nodes = self._get_prepared_nodes_for_source(source_node_id)
        if len(prepared_nodes) == 1 and not prepared_iterator_nodes:
            return next(iter(prepared_nodes))

        parent_iterators = self._get_parent_iterator_exec_nodes(source_node_id, graph, prepared_iterator_nodes)
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

    def prepare(self, base_g: Optional[nx.DiGraph] = None) -> Optional[str]:
        g = base_g or self._state.graph.nx_graph_flat()
        next_node_id = next(
            (
                node_id
                for node_id in nx.topological_sort(g)
                if node_id not in self._state.source_prepared_mapping
                and (
                    not isinstance(self._state.graph.get_node(node_id), IterateInvocation)
                    or all(source_id in self._state.executed for source_id, _ in g.in_edges(node_id))
                )
                and not any(
                    isinstance(self._state.graph.get_node(ancestor_id), IterateInvocation)
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
            next_node_parents = [source_id for source_id, _ in g.in_edges(next_node_id)]
            create_results = self.create_execution_node(
                next_node_id, self._get_collect_iteration_mappings(next_node_parents)
            )
            if create_results is not None:
                new_node_ids.extend(create_results)
        else:
            for iteration_mappings in self._get_parent_iteration_mappings(next_node_id, g):
                create_results = self.create_execution_node(next_node_id, iteration_mappings)
                if create_results is not None:
                    new_node_ids.extend(create_results)

        return next(iter(new_node_ids), None)


class _ExecutionScheduler:
    """Owns ready-queue ordering and indegree-driven execution transitions."""

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state

    def _validate_exec_node_ready_state(self, exec_node_id: str) -> None:
        if exec_node_id not in self._state.execution_graph.nodes:
            raise KeyError(f"exec node {exec_node_id} missing from execution_graph")
        if exec_node_id not in self._state.indegree:
            raise KeyError(f"indegree missing for exec node {exec_node_id}")

    def _should_skip_ready_enqueue(self, exec_node_id: str) -> bool:
        return (
            self._state.indegree[exec_node_id] != 0
            or exec_node_id in self._state.executed
            or self._state._is_deferred_by_unresolved_if(exec_node_id)
        )

    def _get_ready_queue(self, exec_node_id: str) -> Deque[str]:
        node_obj = self._state.execution_graph.nodes[exec_node_id]
        return self.queue_for(self._state._type_key(node_obj))

    def _insert_ready_node(self, queue: Deque[str], exec_node_id: str) -> None:
        exec_node_path = self._state._get_iteration_path(exec_node_id)
        for i, existing in enumerate(queue):
            if self._state._get_iteration_path(existing) > exec_node_path:
                queue.insert(i, exec_node_id)
                return
        queue.append(exec_node_id)

    def _record_completed_node(self, exec_node_id: str, output: BaseInvocationOutput) -> None:
        self._state._set_prepared_exec_state(exec_node_id, "executed")
        self._state.executed.add(exec_node_id)
        self._state.results[exec_node_id] = output

    def _mark_source_node_complete(self, exec_node_id: str) -> None:
        registry = self._state._prepared_registry()
        source_node_id = registry.get_source_node_id(exec_node_id)
        prepared_nodes = registry.get_prepared_ids(source_node_id)
        if all(node_id in self._state.executed for node_id in prepared_nodes):
            self._state.executed.add(source_node_id)
            self._state.executed_history.append(source_node_id)

    def _decrement_child_indegree(self, child_exec_node_id: str, parent_exec_node_id: str) -> None:
        if child_exec_node_id not in self._state.indegree:
            raise KeyError(f"indegree missing for exec node {child_exec_node_id}")
        if self._state.indegree[child_exec_node_id] == 0:
            raise RuntimeError(f"indegree underflow for {child_exec_node_id} from parent {parent_exec_node_id}")
        self._state.indegree[child_exec_node_id] -= 1

    def _release_downstream_nodes(self, exec_node_id: str) -> None:
        for edge in self._state.execution_graph._get_output_edges(exec_node_id):
            child = edge.destination.node_id
            self._decrement_child_indegree(child, exec_node_id)
            self._state._try_resolve_if_node(child)
            if self._state.indegree[child] == 0:
                self.enqueue_if_ready(child)

    def queue_for(self, cls_name: str) -> Deque[str]:
        q = self._state._ready_queues.get(cls_name)
        if q is None:
            q = deque()
            self._state._ready_queues[cls_name] = q
        return q

    def remove_from_ready_queues(self, exec_node_id: str) -> None:
        for q in self._state._ready_queues.values():
            try:
                q.remove(exec_node_id)
            except ValueError:
                continue

    def enqueue_if_ready(self, exec_node_id: str) -> None:
        """Push exec_node_id to its class queue if unmet inputs == 0."""
        self._validate_exec_node_ready_state(exec_node_id)
        if self._should_skip_ready_enqueue(exec_node_id):
            return
        queue = self._get_ready_queue(exec_node_id)
        if exec_node_id in queue:
            return
        self._state._set_prepared_exec_state(exec_node_id, "ready")
        self._insert_ready_node(queue, exec_node_id)

    def get_next_node(self) -> Optional[BaseInvocation]:
        """Gets the next ready node: FIFO within class, drain class before switching."""
        while True:
            if self._state._active_class:
                q = self._state._ready_queues.get(self._state._active_class)
                while q:
                    exec_node_id = q.popleft()
                    if exec_node_id not in self._state.executed:
                        return self._state.execution_graph.nodes[exec_node_id]
                self._state._active_class = None
                continue

            seen = set(self._state.ready_order)
            next_class = next(
                (cls_name for cls_name in self._state.ready_order if self._state._ready_queues.get(cls_name)),
                None,
            )
            if next_class is None:
                next_class = next(
                    (
                        cls_name
                        for cls_name in sorted(k for k in self._state._ready_queues.keys() if k not in seen)
                        if self._state._ready_queues[cls_name]
                    ),
                    None,
                )
            if next_class is None:
                return None

            self._state._active_class = next_class

    def complete(self, exec_node_id: str, output: BaseInvocationOutput) -> None:
        if exec_node_id not in self._state.execution_graph.nodes:
            return

        self._record_completed_node(exec_node_id, output)
        self._mark_source_node_complete(exec_node_id)
        self._release_downstream_nodes(exec_node_id)


class _ExecutionRuntime:
    """Provides runtime-only helpers such as iteration-path lookup and input hydration."""

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state

    def _get_cached_iteration_path(self, exec_node_id: str) -> Optional[tuple[int, ...]]:
        registry = self._state._prepared_registry()
        metadata_iteration_path = registry.get_iteration_path(exec_node_id)
        if metadata_iteration_path is not None:
            return metadata_iteration_path

        return self._state._iteration_path_cache.get(exec_node_id)

    def _get_iteration_source_node_id(self, exec_node_id: str) -> Optional[str]:
        if exec_node_id not in self._state.prepared_source_mapping:
            return None
        return self._state._prepared_registry().get_source_node_id(exec_node_id)

    def _get_ordered_iterator_sources(self, source_node_id: str) -> list[str]:
        iterator_graph = self._state._iterator_graph(self._state.graph.nx_graph())
        iterator_sources = [
            node_id
            for node_id in nx.ancestors(iterator_graph, source_node_id)
            if isinstance(self._state.graph.get_node(node_id), IterateInvocation)
        ]

        topo = list(nx.topological_sort(iterator_graph))
        topo_index = {node_id: i for i, node_id in enumerate(topo)}
        iterator_sources.sort(key=lambda node_id: topo_index.get(node_id, 0))
        return iterator_sources

    def _get_iterator_exec_id(
        self, iterator_source_id: str, exec_node_id: str, execution_graph: nx.DiGraph
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
            if isinstance(iterator_node, IterateInvocation):
                path.append(iterator_node.index)

        node_obj = self._state.execution_graph.nodes.get(exec_node_id)
        if isinstance(node_obj, IterateInvocation):
            path.append(node_obj.index)

        return tuple(path)

    def _cache_iteration_path(self, exec_node_id: str, iteration_path: tuple[int, ...]) -> tuple[int, ...]:
        self._state._iteration_path_cache[exec_node_id] = iteration_path
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
        output_collection.extend(self._get_copied_result_value(edge) for edge in item_edges)
        return output_collection

    def _set_node_inputs(
        self, node: BaseInvocation, input_edges: list[Edge], allowed_fields: Optional[set[str]] = None
    ) -> None:
        for edge in input_edges:
            if allowed_fields is not None and edge.destination.field not in allowed_fields:
                continue
            setattr(node, edge.destination.field, self._get_copied_result_value(edge))

    def _prepare_collect_inputs(self, node: "CollectInvocation", input_edges: list[Edge]) -> None:
        node.collection = self._build_collect_collection(input_edges)

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

        if isinstance(node, CollectInvocation):
            self._prepare_collect_inputs(node, input_edges)
            return

        if isinstance(node, IfInvocation):
            self._prepare_if_inputs(node, input_edges)
            return

        self._prepare_default_inputs(node, input_edges)


def get_output_field_type(node: BaseInvocation, field: str) -> Any:
    # TODO(psyche): This is awkward - if field_info is None, it means the field is not defined in the output, which
    # really should raise. The consumers of this utility expect it to never raise, and return None instead. Fixing this
    # would require some fairly significant changes and I don't want risk breaking anything.
    try:
        invocation_class = type(node)
        invocation_output_class = invocation_class.get_output_annotation()
        field_info = invocation_output_class.model_fields.get(field)
        assert field_info is not None, f"Output field '{field}' not found in {invocation_output_class.get_type()}"
        output_field_type = field_info.annotation
        return output_field_type
    except Exception:
        return None


def get_input_field_type(node: BaseInvocation, field: str) -> Any:
    # TODO(psyche): This is awkward - if field_info is None, it means the field is not defined in the output, which
    # really should raise. The consumers of this utility expect it to never raise, and return None instead. Fixing this
    # would require some fairly significant changes and I don't want risk breaking anything.
    try:
        invocation_class = type(node)
        field_info = invocation_class.model_fields.get(field)
        assert field_info is not None, f"Input field '{field}' not found in {invocation_class.get_type()}"
        input_field_type = field_info.annotation
        return input_field_type
    except Exception:
        return None


def is_union_subtype(t1, t2):
    t1_args = get_args(t1)
    t2_args = get_args(t2)
    if not t1_args:
        # t1 is a single type
        return t1 in t2_args
    else:
        # t1 is a Union, check that all of its types are in t2_args
        return all(arg in t2_args for arg in t1_args)


def is_list_or_contains_list(t):
    t_args = get_args(t)

    # If the type is a List
    if get_origin(t) is list:
        return True

    # If the type is a Union
    elif t_args:
        # Check if any of the types in the Union is a List
        for arg in t_args:
            if get_origin(arg) is list:
                return True
    return False


def is_any(t: Any) -> bool:
    return t == Any or Any in get_args(t)


def extract_collection_item_types(t: Any) -> set[Any]:
    """Extracts list item types from a collection annotation, including unions containing list branches."""
    if is_any(t):
        return {Any}

    if get_origin(t) is list:
        return {arg for arg in get_args(t) if arg != NoneType}

    item_types: set[Any] = set()
    for arg in get_args(t):
        if is_any(arg):
            item_types.add(Any)
        elif get_origin(arg) is list:
            item_types.update(item_arg for item_arg in get_args(arg) if item_arg != NoneType)
    return item_types


def are_connection_types_compatible(from_type: Any, to_type: Any) -> bool:
    if not from_type or not to_type:
        return False

    # Ports are compatible
    if from_type == to_type or is_any(from_type) or is_any(to_type):
        return True

    if from_type in get_args(to_type):
        return True

    if to_type in get_args(from_type):
        return True

    # allow int -> float, pydantic will cast for us
    if from_type is int and to_type is float:
        return True

    # allow int|float -> str, pydantic will cast for us
    if (from_type is int or from_type is float) and to_type is str:
        return True

    # Prefer issubclass when both are real classes
    try:
        if isinstance(from_type, type) and isinstance(to_type, type):
            return issubclass(from_type, to_type)
    except TypeError:
        pass

    # Union-to-Union (or Union-to-non-Union) handling
    return is_union_subtype(from_type, to_type)


def are_connections_compatible(
    from_node: BaseInvocation, from_field: str, to_node: BaseInvocation, to_field: str
) -> bool:
    """Determines if a connection between fields of two nodes is compatible."""

    # TODO: handle iterators and collectors
    from_type = get_output_field_type(from_node, from_field)
    to_type = get_input_field_type(to_node, to_field)

    return are_connection_types_compatible(from_type, to_type)


T = TypeVar("T")


def copydeep(obj: T) -> T:
    """Deep-copies an object. If it is a pydantic model, use the model's copy method."""
    if isinstance(obj, BaseModel):
        return obj.model_copy(deep=True)
    return copy.deepcopy(obj)


class NodeAlreadyInGraphError(ValueError):
    pass


class InvalidEdgeError(ValueError):
    pass


class NodeNotFoundError(ValueError):
    pass


class NodeAlreadyExecutedError(ValueError):
    pass


class DuplicateNodeIdError(ValueError):
    pass


class NodeFieldNotFoundError(ValueError):
    pass


class NodeIdMismatchError(ValueError):
    pass


class CyclicalGraphError(ValueError):
    pass


class UnknownGraphValidationError(ValueError):
    pass


class NodeInputError(ValueError):
    """Raised when a node fails preparation. This occurs when a node's inputs are being set from its incomers, but an
    input fails validation.

    Attributes:
        node: The node that failed preparation. Note: only successfully set fields will be accurate. Review the error to
            determine which field caused the failure.
    """

    def __init__(self, node: BaseInvocation, e: ValidationError):
        self.original_error = e
        self.node = node
        # When preparing a node, we set each input one-at-a-time. We may thus safely assume that the first error
        # represents the first input that failed.
        self.failed_input = loc_to_dot_sep(e.errors()[0]["loc"])
        super().__init__(f"Node {node.id} has invalid incoming input for {self.failed_input}")


def loc_to_dot_sep(loc: tuple[Union[str, int], ...]) -> str:
    """Helper to pretty-print pydantic error locations as dot-separated strings.
    Taken from https://docs.pydantic.dev/latest/errors/errors/#customize-error-messages
    """
    path = ""
    for i, x in enumerate(loc):
        if isinstance(x, str):
            if i > 0:
                path += "."
            path += x
        else:
            path += f"[{x}]"
    return path


@invocation_output("iterate_output")
class IterateInvocationOutput(BaseInvocationOutput):
    """Used to connect iteration outputs. Will be expanded to a specific output."""

    item: Any = OutputField(
        description="The item being iterated over", title="Collection Item", ui_type=UIType._CollectionItem
    )
    index: int = OutputField(description="The index of the item", title="Index")
    total: int = OutputField(description="The total number of items", title="Total")


# TODO: Fill this out and move to invocations
@invocation("iterate", version="1.1.0")
class IterateInvocation(BaseInvocation):
    """Iterates over a list of items"""

    collection: list[Any] = InputField(
        description="The list of items to iterate over", default=[], ui_type=UIType._Collection
    )
    index: int = InputField(description="The index, will be provided on executed iterators", default=0, ui_hidden=True)

    def invoke(self, context: InvocationContext) -> IterateInvocationOutput:
        """Produces the outputs as values"""
        return IterateInvocationOutput(item=self.collection[self.index], index=self.index, total=len(self.collection))


@invocation_output("collect_output")
class CollectInvocationOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(
        description="The collection of input items", title="Collection", ui_type=UIType._Collection
    )


@invocation("collect", version="1.1.0")
class CollectInvocation(BaseInvocation):
    """Collects values into a collection"""

    item: Optional[Any] = InputField(
        default=None,
        description="The item to collect (all inputs must be of the same type)",
        ui_type=UIType._CollectionItem,
        title="Collection Item",
        input=Input.Connection,
    )
    collection: list[Any] = InputField(
        description="An optional collection to append to",
        default=[],
        ui_type=UIType._Collection,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> CollectInvocationOutput:
        """Invoke with provided services and return outputs."""
        return CollectInvocationOutput(collection=copy.copy(self.collection))


class AnyInvocation(BaseInvocation):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        def validate_invocation(v: Any) -> "AnyInvocation":
            return InvocationRegistry.get_invocation_typeadapter().validate_python(v)

        return core_schema.no_info_plain_validator_function(validate_invocation)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Nodes are too powerful, we have to make our own OpenAPI schema manually
        # No but really, because the schema is dynamic depending on loaded nodes, we need to generate it manually
        oneOf: list[dict[str, str]] = []
        names = [i.__name__ for i in InvocationRegistry.get_invocation_classes()]
        for name in sorted(names):
            oneOf.append({"$ref": f"#/components/schemas/{name}"})
        return {"oneOf": oneOf}


class AnyInvocationOutput(BaseInvocationOutput):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler):
        def validate_invocation_output(v: Any) -> "AnyInvocationOutput":
            return InvocationRegistry.get_output_typeadapter().validate_python(v)

        return core_schema.no_info_plain_validator_function(validate_invocation_output)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Nodes are too powerful, we have to make our own OpenAPI schema manually
        # No but really, because the schema is dynamic depending on loaded nodes, we need to generate it manually

        oneOf: list[dict[str, str]] = []
        names = [i.__name__ for i in InvocationRegistry.get_output_classes()]
        for name in sorted(names):
            oneOf.append({"$ref": f"#/components/schemas/{name}"})
        return {"oneOf": oneOf}


class Graph(BaseModel):
    """A validated invocation graph made of nodes and typed edges."""

    id: str = Field(description="The id of this graph", default_factory=uuid_string)
    # TODO: use a list (and never use dict in a BaseModel) because pydantic/fastapi hates me
    nodes: dict[str, AnyInvocation] = Field(description="The nodes in this graph", default_factory=dict)
    edges: list[Edge] = Field(
        description="The connections between nodes and their fields in this graph",
        default_factory=list,
    )

    def add_node(self, node: BaseInvocation) -> None:
        """Adds a node to a graph

        :raises NodeAlreadyInGraphError: the node is already present in the graph.
        """

        if node.id in self.nodes:
            raise NodeAlreadyInGraphError()

        self.nodes[node.id] = node

    def delete_node(self, node_id: str) -> None:
        """Deletes a node from a graph"""

        try:
            # Delete edges for this node
            input_edges = self._get_input_edges(node_id)
            output_edges = self._get_output_edges(node_id)

            for edge in input_edges:
                self.delete_edge(edge)

            for edge in output_edges:
                self.delete_edge(edge)

            del self.nodes[node_id]

        except NodeNotFoundError:
            pass  # Ignore, not doesn't exist (should this throw?)

    def add_edge(self, edge: Edge) -> None:
        """Adds an edge to a graph

        :raises InvalidEdgeError: the provided edge is invalid.
        """

        self._validate_edge(edge)
        if edge not in self.edges:
            self.edges.append(edge)
        else:
            raise InvalidEdgeError()

    def delete_edge(self, edge: Edge) -> None:
        """Deletes an edge from a graph"""

        try:
            self.edges.remove(edge)
        except ValueError:
            pass

    def _validate_unique_node_ids(self) -> None:
        node_ids = [n.id for n in self.nodes.values()]
        seen = set()
        duplicate_node_ids = {nid for nid in node_ids if (nid in seen) or seen.add(nid)}
        if duplicate_node_ids:
            raise DuplicateNodeIdError(f"Node ids must be unique, found duplicates {duplicate_node_ids}")

    def _validate_node_id_mapping(self) -> None:
        for node_dict_id, node in self.nodes.items():
            if node_dict_id != node.id:
                raise NodeIdMismatchError(f"Node ids must match, got {node_dict_id} and {node.id}")

    def _validate_edge_nodes_and_fields(self) -> None:
        for edge in self.edges:
            source_node = self.nodes.get(edge.source.node_id, None)
            if source_node is None:
                raise NodeNotFoundError(f"Edge source node {edge.source.node_id} does not exist in the graph")

            destination_node = self.nodes.get(edge.destination.node_id, None)
            if destination_node is None:
                raise NodeNotFoundError(f"Edge destination node {edge.destination.node_id} does not exist in the graph")

            if edge.source.field not in source_node.get_output_annotation().model_fields:
                raise NodeFieldNotFoundError(
                    f"Edge source field {edge.source.field} does not exist in node {edge.source.node_id}"
                )

            if edge.destination.field not in type(destination_node).model_fields:
                raise NodeFieldNotFoundError(
                    f"Edge destination field {edge.destination.field} does not exist in node {edge.destination.node_id}"
                )

    def _validate_graph_is_acyclic(self) -> None:
        graph = self.nx_graph_flat()
        if not nx.is_directed_acyclic_graph(graph):
            raise CyclicalGraphError("Graph contains cycles")

    def _validate_edge_type_compatibility(self) -> None:
        for edge in self.edges:
            if not are_connections_compatible(
                self.get_node(edge.source.node_id),
                edge.source.field,
                self.get_node(edge.destination.node_id),
                edge.destination.field,
            ):
                raise InvalidEdgeError(f"Edge source and target types do not match ({edge})")

    def _validate_special_nodes(self) -> None:
        # TODO: may need to validate all iterators & collectors in subgraphs so edge connections in parent graphs will be available
        for node in self.nodes.values():
            if isinstance(node, IterateInvocation):
                err = self._is_iterator_connection_valid(node.id)
                if err is not None:
                    raise InvalidEdgeError(f"Invalid iterator node ({node.id}): {err}")
            if isinstance(node, CollectInvocation):
                err = self._is_collector_connection_valid(node.id)
                if err is not None:
                    raise InvalidEdgeError(f"Invalid collector node ({node.id}): {err}")

    def validate_self(self) -> None:
        """
        Validates the graph.

        Raises an exception if the graph is invalid:
        - `DuplicateNodeIdError`
        - `NodeIdMismatchError`
        - `InvalidSubGraphError`
        - `NodeNotFoundError`
        - `NodeFieldNotFoundError`
        - `CyclicalGraphError`
        - `InvalidEdgeError`
        """

        self._validate_unique_node_ids()
        self._validate_node_id_mapping()
        self._validate_edge_nodes_and_fields()
        self._validate_graph_is_acyclic()
        self._validate_edge_type_compatibility()
        self._validate_special_nodes()
        return None

    def is_valid(self) -> bool:
        """
        Checks if the graph is valid.

        Raises `UnknownGraphValidationError` if there is a problem validating the graph (not a validation error).
        """
        try:
            self.validate_self()
            return True
        except (
            DuplicateNodeIdError,
            NodeIdMismatchError,
            NodeNotFoundError,
            NodeFieldNotFoundError,
            CyclicalGraphError,
            InvalidEdgeError,
        ):
            return False
        except Exception as e:
            raise UnknownGraphValidationError(f"Problem validating graph {e}") from e

    def _is_destination_field_Any(self, edge: Edge) -> bool:
        """Checks if the destination field for an edge is of type typing.Any"""
        return get_input_field_type(self.get_node(edge.destination.node_id), edge.destination.field) == Any

    def _is_destination_field_list_of_Any(self, edge: Edge) -> bool:
        """Checks if the destination field for an edge is of type typing.Any"""
        return get_input_field_type(self.get_node(edge.destination.node_id), edge.destination.field) == list[Any]

    def _get_edge_nodes(self, edge: Edge) -> tuple[BaseInvocation, BaseInvocation]:
        try:
            return self.get_node(edge.source.node_id), self.get_node(edge.destination.node_id)
        except NodeNotFoundError:
            raise InvalidEdgeError(f"One or both nodes don't exist ({edge})")

    def _validate_edge_destination_uniqueness(self, edge: Edge, destination_node: BaseInvocation) -> None:
        input_edges = self._get_input_edges(edge.destination.node_id, edge.destination.field)
        if len(input_edges) > 0 and (
            not isinstance(destination_node, CollectInvocation) or edge.destination.field != ITEM_FIELD
        ):
            raise InvalidEdgeError(f"Edge already exists ({edge})")

    def _validate_edge_would_not_create_cycle(self, edge: Edge) -> None:
        graph = self.nx_graph_flat()
        graph.add_edge(edge.source.node_id, edge.destination.node_id)
        if not nx.is_directed_acyclic_graph(graph):
            raise InvalidEdgeError(f"Edge creates a cycle in the graph ({edge})")

    def _validate_edge_field_compatibility(
        self, edge: Edge, source_node: BaseInvocation, destination_node: BaseInvocation
    ) -> None:
        if not are_connections_compatible(source_node, edge.source.field, destination_node, edge.destination.field):
            raise InvalidEdgeError(f"Field types are incompatible ({edge})")

    def _validate_iterator_edge_rules(
        self, edge: Edge, source_node: BaseInvocation, destination_node: BaseInvocation
    ) -> None:
        if isinstance(destination_node, IterateInvocation) and edge.destination.field == COLLECTION_FIELD:
            err = self._is_iterator_connection_valid(edge.destination.node_id, new_input=edge.source)
            if err is not None:
                raise InvalidEdgeError(f"Iterator input type does not match iterator output type ({edge}): {err}")

        if isinstance(source_node, IterateInvocation) and edge.source.field == ITEM_FIELD:
            err = self._is_iterator_connection_valid(edge.source.node_id, new_output=edge.destination)
            if err is not None:
                raise InvalidEdgeError(f"Iterator output type does not match iterator input type ({edge}): {err}")

    def _validate_collector_edge_rules(
        self, edge: Edge, source_node: BaseInvocation, destination_node: BaseInvocation
    ) -> None:
        if isinstance(destination_node, CollectInvocation) and edge.destination.field in (ITEM_FIELD, COLLECTION_FIELD):
            err = self._is_collector_connection_valid(
                edge.destination.node_id, new_input=edge.source, new_input_field=edge.destination.field
            )
            if err is not None:
                raise InvalidEdgeError(f"Collector output type does not match collector input type ({edge}): {err}")

        if (
            isinstance(source_node, CollectInvocation)
            and edge.source.field == COLLECTION_FIELD
            and not self._is_destination_field_list_of_Any(edge)
            and not self._is_destination_field_Any(edge)
        ):
            err = self._is_collector_connection_valid(edge.source.node_id, new_output=edge.destination)
            if err is not None:
                raise InvalidEdgeError(f"Collector input type does not match collector output type ({edge}): {err}")

    def _validate_edge(self, edge: Edge):
        """Validates that a new edge doesn't create a cycle in the graph"""
        source_node, destination_node = self._get_edge_nodes(edge)
        self._validate_edge_destination_uniqueness(edge, destination_node)
        self._validate_edge_would_not_create_cycle(edge)
        self._validate_edge_field_compatibility(edge, source_node, destination_node)
        self._validate_iterator_edge_rules(edge, source_node, destination_node)
        self._validate_collector_edge_rules(edge, source_node, destination_node)

    def has_node(self, node_id: str) -> bool:
        """Determines whether or not a node exists in the graph."""
        try:
            _ = self.get_node(node_id)
            return True
        except NodeNotFoundError:
            return False

    def get_node(self, node_id: str) -> BaseInvocation:
        """Gets a node from the graph."""
        try:
            return self.nodes[node_id]
        except KeyError as e:
            raise NodeNotFoundError(f"Node {node_id} not found in graph") from e

    def update_node(self, node_id: str, new_node: BaseInvocation) -> None:
        """Updates a node in the graph."""
        node = self.nodes[node_id]

        # Ensure the node type matches the new node
        if type(node) is not type(new_node):
            raise TypeError(f"Node {node_id} is type {type(node)} but new node is type {type(new_node)}")

        # Ensure the new id is either the same or is not in the graph
        if new_node.id != node.id and self.has_node(new_node.id):
            raise NodeAlreadyInGraphError(f"Node with id {new_node.id} already exists in graph")

        # Set the new node in the graph
        self.nodes[new_node.id] = new_node
        if new_node.id != node.id:
            input_edges = self._get_input_edges(node_id)
            output_edges = self._get_output_edges(node_id)

            # Delete node and all edges
            self.delete_node(node_id)

            # Create new edges for each input and output
            for edge in input_edges:
                self.add_edge(
                    Edge(
                        source=edge.source,
                        destination=EdgeConnection(node_id=new_node.id, field=edge.destination.field),
                    )
                )

            for edge in output_edges:
                self.add_edge(
                    Edge(
                        source=EdgeConnection(node_id=new_node.id, field=edge.source.field),
                        destination=edge.destination,
                    )
                )

    def _get_input_edges(self, node_id: str, field: Optional[str] = None) -> list[Edge]:
        """Gets all input edges for a node. If field is provided, only edges to that field are returned."""

        edges = [e for e in self.edges if e.destination.node_id == node_id]

        if field is None:
            return edges

        filtered_edges = [e for e in edges if e.destination.field == field]

        return filtered_edges

    def _get_output_edges(self, node_id: str, field: Optional[str] = None) -> list[Edge]:
        """Gets all output edges for a node. If field is provided, only edges from that field are returned."""
        edges = [e for e in self.edges if e.source.node_id == node_id]

        if field is None:
            return edges

        filtered_edges = [e for e in edges if e.source.field == field]

        return filtered_edges

    def _is_iterator_connection_valid(
        self,
        node_id: str,
        new_input: Optional[EdgeConnection] = None,
        new_output: Optional[EdgeConnection] = None,
    ) -> str | None:
        inputs = [e.source for e in self._get_input_edges(node_id, COLLECTION_FIELD)]
        outputs = [e.destination for e in self._get_output_edges(node_id, ITEM_FIELD)]

        if new_input is not None:
            inputs.append(new_input)
        if new_output is not None:
            outputs.append(new_output)

        return self._validate_iterator_connections(inputs, outputs)

    def _validate_iterator_connections(self, inputs: list[EdgeConnection], outputs: list[EdgeConnection]) -> str | None:
        presence_error = self._validate_iterator_input_presence(inputs)
        if presence_error is not None:
            return presence_error

        input_node = self.get_node(inputs[0].node_id)
        input_field_type = get_output_field_type(input_node, inputs[0].field)
        output_field_types = self._get_iterator_output_field_types(outputs)

        input_type_error = self._validate_iterator_input_type(input_field_type)
        if input_type_error is not None:
            return input_type_error

        output_type_error = self._validate_iterator_output_types(input_field_type, output_field_types)
        if output_type_error is not None:
            return output_type_error

        return self._validate_iterator_collector_input(input_node, output_field_types)

    def _validate_iterator_input_presence(self, inputs: list[EdgeConnection]) -> str | None:
        if len(inputs) == 0:
            return "Iterator must have a collection input edge"
        if len(inputs) > 1:
            return "Iterator may only have one input edge"
        return None

    def _get_iterator_output_field_types(self, outputs: list[EdgeConnection]) -> list[Any]:
        return [get_input_field_type(self.get_node(e.node_id), e.field) for e in outputs]

    def _validate_iterator_input_type(self, input_field_type: Any) -> str | None:
        if get_origin(input_field_type) is not list:
            return "Iterator input must be a collection"
        return None

    def _validate_iterator_output_types(self, input_field_type: Any, output_field_types: list[Any]) -> str | None:
        input_field_item_type = get_args(input_field_type)[0]
        if not all(are_connection_types_compatible(input_field_item_type, t) for t in output_field_types):
            return "Iterator outputs must connect to an input with a matching type"
        return None

    def _validate_iterator_collector_input(
        self, input_node: BaseInvocation, output_field_types: list[Any]
    ) -> str | None:
        if not isinstance(input_node, CollectInvocation):
            return None

        input_root_type = self._get_collector_input_root_type(input_node.id)
        if input_root_type is None:
            return "Iterator input collector must have at least one item or collection input edge"
        if not all(are_connection_types_compatible(input_root_type, t) for t in output_field_types):
            return "Iterator collection type must match all iterator output types"
        return None

    def _resolve_collector_input_types(self, node_id: str, visited: Optional[set[str]] = None) -> set[Any]:
        """Resolves possible item types for a collector's inputs, recursively following chained collectors."""
        visited = visited or set()
        if node_id in visited:
            return set()
        visited.add(node_id)

        input_types: set[Any] = set()

        for edge in self._get_input_edges(node_id, ITEM_FIELD):
            input_field_type = get_output_field_type(self.get_node(edge.source.node_id), edge.source.field)
            resolved_types = [input_field_type] if get_origin(input_field_type) is None else get_args(input_field_type)
            input_types.update(t for t in resolved_types if t != NoneType)

        for edge in self._get_input_edges(node_id, COLLECTION_FIELD):
            source_node = self.get_node(edge.source.node_id)
            if isinstance(source_node, CollectInvocation) and edge.source.field == COLLECTION_FIELD:
                input_types.update(self._resolve_collector_input_types(source_node.id, visited.copy()))
                continue

            input_field_type = get_output_field_type(source_node, edge.source.field)
            input_types.update(extract_collection_item_types(input_field_type))

        return input_types

    def _get_type_tree_root_types(self, input_types: set[Any]) -> list[Any]:
        type_tree = nx.DiGraph()
        type_tree.add_nodes_from(input_types)
        type_tree.add_edges_from([e for e in itertools.permutations(input_types, 2) if issubclass(e[1], e[0])])
        type_degrees = type_tree.in_degree(type_tree.nodes)
        return [t[0] for t in type_degrees if t[1] == 0]  # type: ignore

    def _get_collector_input_root_type(self, node_id: str) -> Any | None:
        input_types = self._resolve_collector_input_types(node_id)
        non_any_input_types = {t for t in input_types if t != Any}
        if len(non_any_input_types) == 0 and Any in input_types:
            return Any
        if len(non_any_input_types) == 0:
            return None

        root_types = self._get_type_tree_root_types(non_any_input_types)
        if len(root_types) != 1:
            return Any
        return root_types[0]

    def _get_collector_connections(
        self,
        node_id: str,
        new_input: Optional[EdgeConnection] = None,
        new_input_field: Optional[str] = None,
        new_output: Optional[EdgeConnection] = None,
    ) -> tuple[list[EdgeConnection], list[EdgeConnection], list[EdgeConnection]]:
        item_inputs = [e.source for e in self._get_input_edges(node_id, ITEM_FIELD)]
        collection_inputs = [e.source for e in self._get_input_edges(node_id, COLLECTION_FIELD)]
        outputs = [e.destination for e in self._get_output_edges(node_id, COLLECTION_FIELD)]

        if new_input is not None:
            field = new_input_field or ITEM_FIELD
            if field == ITEM_FIELD:
                item_inputs.append(new_input)
            elif field == COLLECTION_FIELD:
                collection_inputs.append(new_input)

        if new_output is not None:
            outputs.append(new_output)

        return item_inputs, collection_inputs, outputs

    def _get_collector_port_types(
        self,
        item_inputs: list[EdgeConnection],
        collection_inputs: list[EdgeConnection],
        outputs: list[EdgeConnection],
    ) -> tuple[list[Any], list[Any], list[Any]]:
        item_input_field_types = [get_output_field_type(self.get_node(e.node_id), e.field) for e in item_inputs]
        collection_input_field_types = [
            get_output_field_type(self.get_node(e.node_id), e.field) for e in collection_inputs
        ]
        output_field_types = [get_input_field_type(self.get_node(e.node_id), e.field) for e in outputs]
        return item_input_field_types, collection_input_field_types, output_field_types

    def _resolve_item_input_types(self, item_input_field_types: list[Any]) -> set[Any]:
        return {
            resolved_type
            for input_field_type in item_input_field_types
            for resolved_type in (
                [input_field_type] if get_origin(input_field_type) is None else get_args(input_field_type)
            )
            if resolved_type != NoneType
        }

    def _resolve_collection_input_types(
        self, collection_inputs: list[EdgeConnection], collection_input_field_types: list[Any]
    ) -> set[Any]:
        input_field_types: set[Any] = set()
        for input_conn, input_field_type in zip(collection_inputs, collection_input_field_types, strict=False):
            source_node = self.get_node(input_conn.node_id)
            if isinstance(source_node, CollectInvocation) and input_conn.field == COLLECTION_FIELD:
                input_field_types.update(self._resolve_collector_input_types(source_node.id))
                continue
            input_field_types.update(extract_collection_item_types(input_field_type))
        return input_field_types

    def _validate_collector_collection_inputs(self, collection_input_field_types: list[Any]) -> str | None:
        if not all((is_list_or_contains_list(t) or is_any(t) for t in collection_input_field_types)):
            return "Collector collection input must be a collection"
        return None

    def _get_collector_input_root_type_from_resolved_types(
        self, input_field_types: set[Any]
    ) -> tuple[bool, Any | None]:
        non_any_input_field_types = {t for t in input_field_types if t != Any}
        root_types = self._get_type_tree_root_types(non_any_input_field_types)
        if len(root_types) > 1:
            return True, None
        return False, root_types[0] if len(root_types) == 1 else None

    def _validate_collector_output_types(
        self, output_field_types: list[Any], input_root_type: Any | None
    ) -> str | None:
        if not all(is_list_or_contains_list(t) or is_any(t) for t in output_field_types):
            return "Collector output must connect to a collection input"

        if input_root_type is not None:
            if not all(
                is_any(t)
                or is_union_subtype(input_root_type, get_args(t)[0])
                or issubclass(input_root_type, get_args(t)[0])
                for t in output_field_types
            ):
                return "Collector outputs must connect to a collection input with a matching type"
        elif any(not is_any(t) and get_args(t)[0] != Any for t in output_field_types):
            return "Collector outputs must connect to a collection input with a matching type"

        return None

    def _validate_downstream_collector_outputs(
        self, outputs: list[EdgeConnection], input_root_type: Any | None
    ) -> str | None:
        for output in outputs:
            output_node = self.get_node(output.node_id)
            if not isinstance(output_node, CollectInvocation) or output.field != COLLECTION_FIELD:
                continue
            output_root_type = self._get_collector_input_root_type(output_node.id)
            if output_root_type is None:
                continue
            if input_root_type is None:
                if output_root_type != Any:
                    return "Collector outputs must connect to a collection input with a matching type"
                continue
            if not are_connection_types_compatible(input_root_type, output_root_type):
                return "Collector outputs must connect to a collection input with a matching type"
        return None

    def _is_collector_connection_valid(
        self,
        node_id: str,
        new_input: Optional[EdgeConnection] = None,
        new_input_field: Optional[str] = None,
        new_output: Optional[EdgeConnection] = None,
    ) -> str | None:
        item_inputs, collection_inputs, outputs = self._get_collector_connections(
            node_id, new_input=new_input, new_input_field=new_input_field, new_output=new_output
        )

        if len(item_inputs) == 0 and len(collection_inputs) == 0:
            return "Collector must have at least one item or collection input edge"

        item_input_field_types, collection_input_field_types, output_field_types = self._get_collector_port_types(
            item_inputs, collection_inputs, outputs
        )

        collection_input_error = self._validate_collector_collection_inputs(collection_input_field_types)
        if collection_input_error is not None:
            return collection_input_error

        input_field_types = self._resolve_item_input_types(item_input_field_types)
        input_field_types.update(self._resolve_collection_input_types(collection_inputs, collection_input_field_types))

        has_multiple_root_types, input_root_type = self._get_collector_input_root_type_from_resolved_types(
            input_field_types
        )
        if has_multiple_root_types:
            return "Collector input collection items must be of a single type"

        output_type_error = self._validate_collector_output_types(output_field_types, input_root_type)
        if output_type_error is not None:
            return output_type_error

        downstream_output_error = self._validate_downstream_collector_outputs(outputs, input_root_type)
        if downstream_output_error is not None:
            return downstream_output_error

        return None

    def nx_graph(self) -> nx.DiGraph:
        """Returns a NetworkX DiGraph representing the layout of this graph"""
        # TODO: Cache this?
        g = nx.DiGraph()
        g.add_nodes_from(list(self.nodes.keys()))
        g.add_edges_from({(e.source.node_id, e.destination.node_id) for e in self.edges})
        return g

    def nx_graph_flat(self, nx_graph: Optional[nx.DiGraph] = None) -> nx.DiGraph:
        """Returns a flattened NetworkX DiGraph, including all subgraphs (but not with iterations expanded)"""
        g = nx_graph or nx.DiGraph()

        # Add all nodes from this graph except graph/iteration nodes
        g.add_nodes_from([n.id for n in self.nodes.values()])

        unique_edges = {(e.source.node_id, e.destination.node_id) for e in self.edges}
        g.add_edges_from(unique_edges)
        return g


class GraphExecutionState(BaseModel):
    """Tracks source-graph expansion, execution progress, and runtime results."""

    id: str = Field(description="The id of the execution state", default_factory=uuid_string)
    # TODO: Store a reference to the graph instead of the actual graph?
    graph: Graph = Field(description="The graph being executed")

    # The graph of materialized nodes
    execution_graph: Graph = Field(
        description="The expanded graph of activated and executed nodes",
        default_factory=Graph,
    )

    # Nodes that have been executed
    executed: set[str] = Field(description="The set of node ids that have been executed", default_factory=set)
    executed_history: list[str] = Field(
        description="The list of node ids that have been executed, in order of execution",
        default_factory=list,
    )

    # The results of executed nodes
    results: dict[str, AnyInvocationOutput] = Field(description="The results of node executions", default_factory=dict)

    # Errors raised when executing nodes
    errors: dict[str, str] = Field(description="Errors raised when executing nodes", default_factory=dict)

    # Map of prepared/executed nodes to their original nodes
    prepared_source_mapping: dict[str, str] = Field(
        description="The map of prepared nodes to original graph nodes",
        default_factory=dict,
    )

    # Map of original nodes to prepared nodes
    source_prepared_mapping: dict[str, set[str]] = Field(
        description="The map of original graph nodes to prepared nodes",
        default_factory=dict,
    )
    # Ready queues grouped by node class name (internal only)
    _ready_queues: dict[str, Deque[str]] = PrivateAttr(default_factory=dict)
    # Current class being drained; stays until its queue empties
    _active_class: Optional[str] = PrivateAttr(default=None)
    # Optional priority; others follow in name order
    ready_order: list[str] = Field(default_factory=list)
    indegree: dict[str, int] = Field(default_factory=dict, description="Remaining unmet input count for exec nodes")
    _iteration_path_cache: dict[str, tuple[int, ...]] = PrivateAttr(default_factory=dict)
    _if_branch_exclusive_sources: dict[str, dict[str, set[str]]] = PrivateAttr(default_factory=dict)
    _resolved_if_exec_branches: dict[str, str] = PrivateAttr(default_factory=dict)
    _prepared_exec_metadata: dict[str, _PreparedExecNodeMetadata] = PrivateAttr(default_factory=dict)
    _prepared_exec_registry: Optional[_PreparedExecRegistry] = PrivateAttr(default=None)
    _if_branch_scheduler: Optional[_IfBranchScheduler] = PrivateAttr(default=None)
    _execution_materializer: Optional[_ExecutionMaterializer] = PrivateAttr(default=None)
    _execution_scheduler: Optional[_ExecutionScheduler] = PrivateAttr(default=None)
    _execution_runtime: Optional[_ExecutionRuntime] = PrivateAttr(default=None)

    def _type_key(self, node_obj: BaseInvocation) -> str:
        return node_obj.__class__.__name__

    def _prepared_registry(self) -> _PreparedExecRegistry:
        if self._prepared_exec_registry is None:
            self._prepared_exec_registry = _PreparedExecRegistry(
                prepared_source_mapping=self.prepared_source_mapping,
                source_prepared_mapping=self.source_prepared_mapping,
                metadata=self._prepared_exec_metadata,
            )
        return self._prepared_exec_registry

    def _if_scheduler(self) -> _IfBranchScheduler:
        if self._if_branch_scheduler is None:
            self._if_branch_scheduler = _IfBranchScheduler(self)
        return self._if_branch_scheduler

    def _materializer(self) -> _ExecutionMaterializer:
        if self._execution_materializer is None:
            self._execution_materializer = _ExecutionMaterializer(self)
        return self._execution_materializer

    def _scheduler(self) -> _ExecutionScheduler:
        if self._execution_scheduler is None:
            self._execution_scheduler = _ExecutionScheduler(self)
        return self._execution_scheduler

    def _runtime(self) -> _ExecutionRuntime:
        if self._execution_runtime is None:
            self._execution_runtime = _ExecutionRuntime(self)
        return self._execution_runtime

    def _register_prepared_exec_node(self, exec_node_id: str, source_node_id: str) -> None:
        self._prepared_registry().register(exec_node_id, source_node_id)

    def _get_prepared_exec_metadata(self, exec_node_id: str) -> _PreparedExecNodeMetadata:
        return self._prepared_registry().get_metadata(exec_node_id)

    def _set_prepared_exec_state(self, exec_node_id: str, state: PreparedExecState) -> None:
        self._prepared_registry().set_state(exec_node_id, state)

    def _get_iteration_path(self, exec_node_id: str) -> tuple[int, ...]:
        return self._runtime().get_iteration_path(exec_node_id)

    def _queue_for(self, cls_name: str) -> Deque[str]:
        return self._scheduler().queue_for(cls_name)

    def _is_deferred_by_unresolved_if(self, exec_node_id: str) -> bool:
        return self._if_scheduler().is_deferred_by_unresolved_if(exec_node_id)

    def _remove_from_ready_queues(self, exec_node_id: str) -> None:
        self._scheduler().remove_from_ready_queues(exec_node_id)

    def _try_resolve_if_node(self, exec_node_id: str) -> None:
        self._if_scheduler().try_resolve_if_node(exec_node_id)

    def set_ready_order(self, order: Iterable[Type[BaseInvocation] | str]) -> None:
        names: list[str] = []
        for x in order:
            names.append(x.__name__ if hasattr(x, "__name__") else str(x))
        self.ready_order = names

    def _enqueue_if_ready(self, nid: str) -> None:
        self._scheduler().enqueue_if_ready(nid)

    def _prepare_until_node_ready(self) -> Optional[BaseInvocation]:
        base_graph = self.graph.nx_graph_flat()
        prepared_id = self._materializer().prepare(base_graph)
        next_node: Optional[BaseInvocation] = None

        while prepared_id is not None:
            prepared_id = self._materializer().prepare(base_graph)
            if next_node is None:
                next_node = self._get_next_node()

        return next_node

    def _reset_runtime_caches(self) -> None:
        self._ready_queues = {}
        self._active_class = None
        self._iteration_path_cache = {}
        self._if_branch_exclusive_sources = {}
        self._resolved_if_exec_branches = {}
        self._prepared_exec_metadata = {}
        self._prepared_exec_registry = None
        self._if_branch_scheduler = None
        self._execution_materializer = None
        self._execution_scheduler = None
        self._execution_runtime = None

    def _rehydrate_prepared_exec_metadata(self) -> None:
        registry = self._prepared_registry()
        for exec_node_id, source_node_id in self.prepared_source_mapping.items():
            metadata = registry.get_metadata(exec_node_id)
            metadata.source_node_id = source_node_id
            metadata.iteration_path = self._get_iteration_path(exec_node_id)
            if exec_node_id in self.executed:
                metadata.state = "executed" if exec_node_id in self.results else "skipped"
            elif self.indegree.get(exec_node_id) == 0:
                metadata.state = "ready"
            else:
                metadata.state = "pending"

    def _apply_if_condition_inputs(self, exec_node_id: str, node: IfInvocation) -> bool:
        condition_edges = self.execution_graph._get_input_edges(exec_node_id, "condition")
        if any(edge.source.node_id not in self.executed for edge in condition_edges):
            return False

        for edge in condition_edges:
            setattr(
                node,
                edge.destination.field,
                copydeep(getattr(self.results[edge.source.node_id], edge.source.field)),
            )
        return True

    def _rehydrate_resolved_if_exec_branches(self) -> None:
        for exec_node_id, node in self.execution_graph.nodes.items():
            if not isinstance(node, IfInvocation):
                continue

            if not self._apply_if_condition_inputs(exec_node_id, node):
                continue

            self._resolved_if_exec_branches[exec_node_id] = "true_input" if node.condition else "false_input"

    def _rehydrate_ready_queues(self) -> None:
        execution_graph = self.execution_graph.nx_graph_flat()
        for exec_node_id in nx.topological_sort(execution_graph):
            if exec_node_id in self.executed:
                continue
            if self.indegree.get(exec_node_id) != 0:
                continue
            self._enqueue_if_ready(exec_node_id)

    def _rehydrate_runtime_state(self) -> None:
        self._reset_runtime_caches()
        self._rehydrate_prepared_exec_metadata()
        self._rehydrate_resolved_if_exec_branches()
        self._rehydrate_ready_queues()

    def model_post_init(self, __context: Any) -> None:
        self._rehydrate_runtime_state()

    model_config = ConfigDict(
        json_schema_extra={
            "required": [
                "id",
                "graph",
                "execution_graph",
                "executed",
                "executed_history",
                "results",
                "errors",
                "prepared_source_mapping",
                "source_prepared_mapping",
            ]
        }
    )

    @field_validator("graph")
    def graph_is_valid(cls, v: Graph):
        """Validates that the graph is valid"""
        v.validate_self()
        return v

    def next(self) -> Optional[BaseInvocation]:
        """Gets the next node ready to execute."""

        # TODO: enable multiple nodes to execute simultaneously by tracking currently executing nodes
        #       possibly with a timeout?

        # If there are no prepared nodes, prepare some nodes
        next_node = self._get_next_node()
        if next_node is None:
            next_node = self._prepare_until_node_ready()

        # Get values from edges
        if next_node is not None:
            try:
                self._prepare_inputs(next_node)
            except ValidationError as e:
                raise NodeInputError(next_node, e)

        # If next is still none, there's no next node, return None
        return next_node

    def complete(self, node_id: str, output: BaseInvocationOutput) -> None:
        """Marks a node as complete"""
        self._scheduler().complete(node_id, output)

    def set_node_error(self, node_id: str, error: str):
        """Marks a node as errored"""
        self.errors[node_id] = error

    def is_complete(self) -> bool:
        """Returns true if the graph is complete"""
        node_ids = set(self.graph.nx_graph_flat().nodes)
        return self.has_error() or all((k in self.executed for k in node_ids))

    def has_error(self) -> bool:
        """Returns true if the graph has any errors"""
        return len(self.errors) > 0

    def _create_execution_node(self, node_id: str, iteration_node_map: list[tuple[str, str]]) -> list[str]:
        return self._materializer().create_execution_node(node_id, iteration_node_map)

    def _iterator_graph(self, base: Optional[nx.DiGraph] = None) -> nx.DiGraph:
        return self._materializer().iterator_graph(base)

    def _get_node_iterators(self, node_id: str, it_graph: Optional[nx.DiGraph] = None) -> list[str]:
        return self._materializer().get_node_iterators(node_id, it_graph)

    def _prepare(self, base_g: Optional[nx.DiGraph] = None) -> Optional[str]:
        return self._materializer().prepare(base_g)

    def _get_iteration_node(
        self,
        source_node_id: str,
        graph: nx.DiGraph,
        execution_graph: nx.DiGraph,
        prepared_iterator_nodes: list[str],
    ) -> Optional[str]:
        return self._materializer().get_iteration_node(source_node_id, graph, execution_graph, prepared_iterator_nodes)

    def _get_next_node(self) -> Optional[BaseInvocation]:
        return self._scheduler().get_next_node()

    def _prepare_inputs(self, node: BaseInvocation):
        self._runtime().prepare_inputs(node)

    # TODO: Add API for modifying underlying graph that checks if the change will be valid given the current execution state
    def _is_edge_valid(self, edge: Edge) -> bool:
        try:
            self.graph._validate_edge(edge)
        except InvalidEdgeError:
            return False

        # Invalid if destination has already been prepared or executed
        if edge.destination.node_id in self.source_prepared_mapping:
            return False

        # Otherwise, the edge is valid
        return True

    def _is_node_updatable(self, node_id: str) -> bool:
        # The node is updatable as long as it hasn't been prepared or executed
        return node_id not in self.source_prepared_mapping

    def add_node(self, node: BaseInvocation) -> None:
        self.graph.add_node(node)

    def update_node(self, node_id: str, new_node: BaseInvocation) -> None:
        if not self._is_node_updatable(node_id):
            raise NodeAlreadyExecutedError(
                f"Node {node_id} has already been prepared or executed and cannot be updated"
            )
        self.graph.update_node(node_id, new_node)

    def delete_node(self, node_id: str) -> None:
        if not self._is_node_updatable(node_id):
            raise NodeAlreadyExecutedError(
                f"Node {node_id} has already been prepared or executed and cannot be deleted"
            )
        self.graph.delete_node(node_id)

    def add_edge(self, edge: Edge) -> None:
        if not self._is_node_updatable(edge.destination.node_id):
            raise NodeAlreadyExecutedError(
                f"Destination node {edge.destination.node_id} has already been prepared or executed and cannot be linked to"
            )
        self.graph.add_edge(edge)

    def delete_edge(self, edge: Edge) -> None:
        if not self._is_node_updatable(edge.destination.node_id):
            raise NodeAlreadyExecutedError(
                f"Destination node {edge.destination.node_id} has already been prepared or executed and cannot have a source edge deleted"
            )
        self.graph.delete_edge(edge)
