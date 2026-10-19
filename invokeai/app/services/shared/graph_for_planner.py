"""Generic-owned For expansion seam.

The generic scheduler owns continuation admission and transition selection. This
planner owns admitted For expansion while sharing low-level execution-node
construction mechanics with the compatibility planner.
"""

from typing import TYPE_CHECKING, Any, Optional

from invokeai.app.invocations.loops import LoopState
from invokeai.app.services.shared.graph_materializer import _ExecutionNodeBuilder

if TYPE_CHECKING:
    import networkx as nx

    from invokeai.app.services.shared.graph import GraphExecutionState


class _GenericForPlanner:
    """Prepare admitted For contexts without calling compatibility entry points."""

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state
        self._builder = _ExecutionNodeBuilder(state)

    def prepare(self, base_graph: Any) -> Optional[str]:
        """Prepare one admitted source node without constructing compatibility facade."""

        return self._builder.prepare(base_graph)

    def iterator_graph(self, base: Optional["nx.DiGraph"] = None) -> "nx.DiGraph":
        return self._builder.iterator_graph(base)

    def create_for_iteration(
        self,
        source_for_id: str,
        iteration_index: int,
        collection: list[Any],
        state: LoopState,
        iteration_path: tuple[int, ...],
    ) -> str:
        return self._builder._create_for_iteration_generic(
            source_for_id=source_for_id,
            iteration_index=iteration_index,
            collection=collection,
            state=state,
            iteration_path=iteration_path,
        )

    def create_for_body_iteration(self, source_for_id: str, prepared_for_id: str) -> Optional[str]:
        return self._builder._create_for_body_iteration_generic(
            source_for_id=source_for_id,
            prepared_for_id=prepared_for_id,
        )

    def create_nested_for_return(self, inner_for_id: str, prepared_inner_for_id: str) -> Optional[str]:
        return self._builder._create_nested_for_return_generic(
            inner_for_id=inner_for_id,
            prepared_inner_for_id=prepared_inner_for_id,
        )
