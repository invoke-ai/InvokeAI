from typing import TYPE_CHECKING, Optional

from invokeai.app.invocations.fields import OutputScope
from invokeai.app.invocations.loops import ForInvocation
from invokeai.app.services.shared.graph_models import Edge
from invokeai.app.services.shared.graph_validation import IterateInvocation, get_output_field_scope, nx

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


class _IteratorAncestry:
    """Own the ordered open For/Iterate axes for one source-graph traversal."""

    def __init__(
        self,
        state: "GraphExecutionState",
        source_graph: Optional["nx.DiGraph"] = None,
        *,
        iterator_graph: Optional["nx.DiGraph"] = None,
    ) -> None:
        self._state = state
        self._source_graph = source_graph
        self._iterator_graph = iterator_graph
        self._has_iterator_ancestor: dict[str, bool] | None = None
        self._ordered_iterator_ancestors: dict[str, tuple[str, ...]] = {}
        self._frame_iterators: dict[str, tuple[str, ...]] = {}
        self.effective_iteration_paths_by_edge: dict[tuple[str, str], set[tuple[int, ...]]] = {}

    def clear_effective_iteration_paths(self) -> None:
        """Invalidate prepared-path results when execution materialization advances."""
        self.effective_iteration_paths_by_edge.clear()

    def has_iterator_ancestor(self, node_id: str) -> bool:
        if self._has_iterator_ancestor is None:
            iterator_graph = self._iterator_graph
            if iterator_graph is None:
                source_graph = self._source_graph
                if source_graph is None:
                    source_graph = self._state._get_source_graph_flat()
                iterator_graph = self._state._iterator_graph(source_graph)
            has_iterator_ancestor = dict.fromkeys(iterator_graph, False)
            for source_node_id in nx.topological_sort(iterator_graph):
                has_iterator_ancestor[source_node_id] = any(
                    has_iterator_ancestor[ancestor_id]
                    or isinstance(self._state.graph.get_node(ancestor_id), (ForInvocation, IterateInvocation))
                    for ancestor_id in iterator_graph.predecessors(source_node_id)
                )
            self._iterator_graph = iterator_graph
            self._has_iterator_ancestor = has_iterator_ancestor

        if node_id not in self._has_iterator_ancestor:
            raise nx.NetworkXError(f"The node {node_id} is not in the digraph.")
        return self._has_iterator_ancestor[node_id]

    def get_ordered_iterator_ancestors(self, node_id: str) -> list[str]:
        """Return open iterator axes in outer-to-inner, input-port order."""
        self.has_iterator_ancestor(node_id)
        cached = self._ordered_iterator_ancestors.get(node_id)
        if cached is not None:
            return list(cached)

        assert self._iterator_graph is not None
        ordered: list[str] = []
        visited: set[str] = set()
        pending: list[tuple[str, bool]] = [(node_id, False)]
        while pending:
            current_node_id, exiting = pending.pop()
            if exiting:
                if current_node_id != node_id and isinstance(
                    self._state.graph.get_node(current_node_id), (ForInvocation, IterateInvocation)
                ):
                    ordered.append(current_node_id)
                continue
            if current_node_id in visited:
                continue
            visited.add(current_node_id)
            pending.append((current_node_id, True))
            current_node = self._state.graph.get_node(current_node_id)
            input_fields = list(type(current_node).model_fields)
            field_order = {field: index for index, field in enumerate(input_fields)}
            input_edges = sorted(
                self._state.graph._get_input_edges(current_node_id),
                key=lambda edge: (field_order.get(edge.destination.field, len(field_order)), edge.source.node_id),
            )
            for edge in reversed(input_edges):
                source_node_id = edge.source.node_id
                if self._iterator_graph.has_edge(source_node_id, current_node_id):
                    pending.append((source_node_id, False))

        cached = tuple(ordered)
        self._ordered_iterator_ancestors[node_id] = cached
        return list(cached)

    def get_iterator_ancestors_for_edge(self, edge: "Edge") -> frozenset[str]:
        """Return iterator axes carried by an edge, excluding closed For output axes."""
        ancestors = self.get_iterator_axes_for_edge(edge)
        return frozenset(ancestors)

    def get_iterator_axes_for_edge(self, edge: "Edge") -> tuple[str, ...]:
        """Return ordered axes carried by an edge, excluding closed For output axes."""
        source_node_id = edge.source.node_id
        ancestors = self.get_ordered_iterator_ancestors(source_node_id)
        source_node = self._state.graph.get_node(source_node_id)
        if isinstance(source_node, (ForInvocation, IterateInvocation)):
            if not (
                isinstance(source_node, ForInvocation)
                and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
            ):
                ancestors.append(source_node_id)
        return tuple(ancestors)

    def get_frame_iterators(self, node_id: str) -> tuple[str, ...]:
        """Return the iterators whose frames are open at a node, outer to inner in input-port order.

        Unlike ``get_ordered_iterator_ancestors``, a For's final output is followed into the For's parent frame: the
        For's own axis is closed there, but the axes the For runs under stay open.
        """
        cached = self._frame_iterators.get(node_id)
        if cached is not None:
            return cached
        self.has_iterator_ancestor(node_id)
        assert self._iterator_graph is not None

        ordered: list[str] = []
        visited: set[tuple[str, bool]] = set()
        # (node, exiting, opens): ``opens`` is False when the node is reached only through a For's final output.
        pending: list[tuple[str, bool, bool]] = [(node_id, False, False)]
        while pending:
            current_node_id, exiting, opens = pending.pop()
            if exiting:
                if opens and current_node_id not in ordered:
                    ordered.append(current_node_id)
                continue
            if (current_node_id, opens) in visited:
                continue
            visited.add((current_node_id, opens))
            pending.append((current_node_id, True, opens))
            current_node = self._state.graph.get_node(current_node_id)
            field_order = {field: index for index, field in enumerate(type(current_node).model_fields)}
            input_edges = sorted(
                self._state.graph._get_input_edges(current_node_id),
                key=lambda edge: (field_order.get(edge.destination.field, len(field_order)), edge.source.node_id),
            )
            for edge in reversed(input_edges):
                source_node_id = edge.source.node_id
                source_node = self._state.graph.get_node(source_node_id)
                if self._iterator_graph.has_edge(source_node_id, current_node_id):
                    pending.append((source_node_id, False, isinstance(source_node, (ForInvocation, IterateInvocation))))
                elif (
                    isinstance(source_node, ForInvocation)
                    and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
                ):
                    pending.append((source_node_id, False, False))

        result = tuple(ordered)
        self._frame_iterators[node_id] = result
        return result

    def get_frame_axes_for_edge(self, edge: "Edge") -> tuple[str, ...]:
        """Return the iterator frames open on an edge, following For final outputs into their parent frame."""
        axes = list(self.get_frame_iterators(edge.source.node_id))
        source_node = self._state.graph.get_node(edge.source.node_id)
        if isinstance(source_node, (ForInvocation, IterateInvocation)) and not (
            isinstance(source_node, ForInvocation)
            and get_output_field_scope(source_node, edge.source.field) == OutputScope.Final
        ):
            axes.append(edge.source.node_id)
        return tuple(axes)
