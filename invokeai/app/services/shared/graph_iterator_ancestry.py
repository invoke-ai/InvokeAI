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
        self._has_iterator_ancestor: dict[str, bool] | None = None

    def has_iterator_ancestor(self, node_id: str) -> bool:
        if self._has_iterator_ancestor is None:
            source_graph = self._source_graph
            if source_graph is None:
                source_graph = self._state._get_source_graph_flat()
            iterator_graph = self._state._iterator_graph(source_graph)
            has_iterator_ancestor = dict.fromkeys(iterator_graph, False)
            for source_node_id in nx.topological_sort(iterator_graph):
                if has_iterator_ancestor[source_node_id] or isinstance(
                    self._state.graph.get_node(source_node_id), (ForInvocation, IterateInvocation)
                ):
                    for successor_id in iterator_graph.successors(source_node_id):
                        has_iterator_ancestor[successor_id] = True
            self._has_iterator_ancestor = has_iterator_ancestor

        if node_id not in self._has_iterator_ancestor:
            raise nx.NetworkXError(f"The node {node_id} is not in the digraph.")
        return self._has_iterator_ancestor[node_id]
