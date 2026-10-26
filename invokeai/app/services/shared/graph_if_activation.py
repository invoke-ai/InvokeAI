from typing import TYPE_CHECKING, Optional

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.services.shared.execution_engine.scheduler import ActivationDependency
from invokeai.app.services.shared.graph_validation import ForInvocation, IterateInvocation, nx

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


class _IfActivationController:
    """Compile activation dependencies for source nodes on compatibility paths.

    Unlike the legacy compiler, this controller does not lower or cache a
    branch projection. It derives frame-scoped branch requirements from the
    current source graph for the fallback scheduler.
    """

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state
        self._branch_sources_cache: dict[tuple[str, str], frozenset[str]] = {}
        self._if_node_ids: Optional[tuple[str, ...]] = None
        self._branch_memberships_by_source: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {}
        self._source_level_selected_fields: dict[str, Optional[str]] = {}

    def _branch_sources(self, if_node_id: str, branch_field: str, source_graph: "nx.DiGraph") -> set[str]:
        cache_key = (if_node_id, branch_field)
        cached = self._branch_sources_cache.get(cache_key)
        if cached is not None:
            return set(cached)

        direct_sources = {edge.source.node_id for edge in self._state.graph._get_input_edges(if_node_id, branch_field)}
        branch_sources = set(direct_sources)
        for source_node_id in direct_sources:
            branch_sources.update(nx.ancestors(source_graph, source_node_id))
        # An iterator still open at the If gives the If its frames, whichever branch is selected, so it and its
        # ancestors run regardless of the condition.
        if any(
            isinstance(self._state.graph.nodes.get(node_id), (ForInvocation, IterateInvocation))
            for node_id in branch_sources
        ):
            for iterator_id in self._state._runtime()._iterator_ancestry.get_frame_iterators(if_node_id):
                branch_sources.discard(iterator_id)
                branch_sources.difference_update(nx.ancestors(source_graph, iterator_id))
        # A For's loop body feeds its ForReturn, which reaches the For only through the loop linkage. The body belongs
        # to the branch whenever its For does, and leaves it with the For. Fors are visited outermost first, so a gated
        # For brings in a For nested in its body together with that body. Two Fors stay ungated. One whose final
        # output enters the If directly is attached after it finalizes. With a wired condition the For and its body
        # can still be gated once that condition resolves. One nested in an ungated For's body is expanded by that loop,
        # which needs it to run. (A For enclosing the If is a frame iterator of the If and was removed above.)
        for_by_return_id: dict[str, str] = {}
        if any(isinstance(self._state.graph.nodes.get(node_id), ForInvocation) for node_id in branch_sources):
            return_id_by_for = {
                for_id: return_id for return_id, for_id in self._state._loop_topology().for_id_by_return_id.items()
            }
            body_ids_by_for = {
                for_id: nx.ancestors(source_graph, return_id) & nx.descendants(source_graph, for_id)
                for for_id, return_id in return_id_by_for.items()
            }
            for for_id in (node_id for node_id in nx.topological_sort(source_graph) if node_id in return_id_by_for):
                if (
                    for_id not in branch_sources
                    or for_id in direct_sources
                    and branch_field == self._get_source_level_selected_field(if_node_id)
                    or any(
                        for_id in body_ids_by_for[enclosing_id] and enclosing_id not in for_by_return_id.values()
                        for enclosing_id in body_ids_by_for
                    )
                ):
                    continue
                for_by_return_id[return_id_by_for[for_id]] = for_id
                branch_sources.update(body_ids_by_for[for_id])
                branch_sources.add(return_id_by_for[for_id])

        changed = True
        while changed:
            changed = False
            for source_node_id in tuple(branch_sources):
                linked_for_id = for_by_return_id.get(source_node_id)
                if all(
                    edge.destination.node_id in branch_sources
                    or (edge.destination.node_id == if_node_id and edge.destination.field == branch_field)
                    for edge in self._state.graph._get_output_edges(source_node_id)
                ) and (linked_for_id is None or linked_for_id in branch_sources):
                    continue
                branch_sources.remove(source_node_id)
                changed = True
        self._branch_sources_cache[cache_key] = frozenset(branch_sources)
        return branch_sources

    def _get_if_node_ids(self) -> tuple[str, ...]:
        if self._if_node_ids is None:
            self._if_node_ids = tuple(
                node_id for node_id, node in self._state.graph.nodes.items() if isinstance(node, IfInvocation)
            )
        return self._if_node_ids

    def _get_source_level_selected_field(self, if_node_id: str) -> Optional[str]:
        if if_node_id not in self._source_level_selected_fields:
            if_node = self._state.graph.nodes[if_node_id]
            selected_field = None
            if not self._state.graph._get_input_edges(if_node_id, "condition"):
                selected_field = "true_input" if if_node.condition else "false_input"
            self._source_level_selected_fields[if_node_id] = selected_field
        return self._source_level_selected_fields[if_node_id]

    def _branch_memberships(self, source_node_id: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return each If with the branch fields whose exclusive sources include this source."""
        cached = self._branch_memberships_by_source.get(source_node_id)
        if cached is not None:
            return cached

        source_graph = self._state._get_source_graph_flat()
        memberships = []
        for if_node_id in self._get_if_node_ids():
            fields = tuple(
                branch_field
                for branch_field in ("true_input", "false_input")
                if source_node_id in self._branch_sources(if_node_id, branch_field, source_graph)
            )
            if not fields:
                continue
            selected_field = self._get_source_level_selected_field(if_node_id)
            if selected_field is not None and selected_field in fields:
                continue
            memberships.append((if_node_id, fields))
        memberships = tuple(memberships)
        self._branch_memberships_by_source[source_node_id] = memberships
        return memberships

    def get_source_dependencies(
        self, source_node_id: str, iteration_path: tuple[int, ...] = ()
    ) -> tuple[ActivationDependency, ...]:
        """Return branch requirements before a source node is materialized."""

        return tuple(
            ActivationDependency(owner_id=if_node_id, branch=fields[0], frame=iteration_path)
            for if_node_id, fields in self._branch_memberships(source_node_id)
            if len(fields) == 1
        )
