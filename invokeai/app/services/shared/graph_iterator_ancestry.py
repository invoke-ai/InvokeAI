from typing import TYPE_CHECKING, Optional

from invokeai.app.invocations.loops import ForInvocation
from invokeai.app.services.shared.graph_validation import IterateInvocation, nx

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


class _IteratorAncestry:
    """Lazily index strict iterator ancestry for one source-graph traversal."""

    def __init__(self, state: "GraphExecutionState", source_graph: Optional["nx.DiGraph"] = None) -> None:
        self._state = state
        self._source_graph = source_graph
        self._iterator_graph: Optional["nx.DiGraph"] = None
        self._has_iterator_ancestor: dict[str, bool] | None = None
        self._iterator_ancestors: dict[str, frozenset[str]] = {}

    def has_iterator_ancestor(self, node_id: str) -> bool:
        if self._has_iterator_ancestor is None:
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

    def get_iterator_ancestors(self, node_id: str) -> frozenset[str]:
        self.has_iterator_ancestor(node_id)
        if node_id not in self._iterator_ancestors:
            assert self._iterator_graph is not None
            ancestors = {
                ancestor_id
                for ancestor_id in nx.ancestors(self._iterator_graph, node_id)
                if isinstance(self._state.graph.get_node(ancestor_id), (ForInvocation, IterateInvocation))
            }
            if isinstance(self._state.graph.get_node(node_id), (ForInvocation, IterateInvocation)):
                ancestors.add(node_id)
            self._iterator_ancestors[node_id] = frozenset(ancestors)
        return self._iterator_ancestors[node_id]
