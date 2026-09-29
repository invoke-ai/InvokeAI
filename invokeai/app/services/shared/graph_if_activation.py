from typing import TYPE_CHECKING

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.services.shared.execution_engine.scheduler import ActivationDependency
from invokeai.app.services.shared.graph_validation import nx

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


class _IfActivationController:
    """Own activation admission for concrete execution nodes.

    Unlike the legacy compiler, this controller does not lower or cache a
    branch projection. It answers one readiness question for one prepared
    execution node from the current source graph and durable execution frame.
    """

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state
        self._branch_sources_cache: dict[tuple[str, str], frozenset[str]] = {}

    def _branch_sources(self, if_node_id: str, branch_field: str, source_graph: "nx.DiGraph") -> set[str]:
        cache_key = (if_node_id, branch_field)
        cached = self._branch_sources_cache.get(cache_key)
        if cached is not None:
            return set(cached)

        direct_sources = {edge.source.node_id for edge in self._state.graph._get_input_edges(if_node_id, branch_field)}
        branch_sources = set(direct_sources)
        for source_node_id in direct_sources:
            branch_sources.update(nx.ancestors(source_graph, source_node_id))

        changed = True
        while changed:
            changed = False
            for source_node_id in tuple(branch_sources):
                if all(
                    edge.destination.node_id in branch_sources
                    or (edge.destination.node_id == if_node_id and edge.destination.field == branch_field)
                    for edge in self._state.graph._get_output_edges(source_node_id)
                ):
                    continue
                branch_sources.remove(source_node_id)
                changed = True
        self._branch_sources_cache[cache_key] = frozenset(branch_sources)
        return branch_sources

    def get_dependencies(self, exec_node_id: str) -> tuple[ActivationDependency, ...]:
        """Return activation requirements for one concrete prepared node."""

        source_node_id = self._state._prepared_registry().get_source_node_id(exec_node_id)
        if not any(isinstance(node, IfInvocation) for node in self._state.graph.nodes.values()):
            return ()
        return self.get_source_dependencies(source_node_id, self._state._get_iteration_path(exec_node_id))

    def get_source_dependencies(
        self, source_node_id: str, iteration_path: tuple[int, ...] = ()
    ) -> tuple[ActivationDependency, ...]:
        """Return branch requirements before a source node is materialized."""

        if not any(isinstance(node, IfInvocation) for node in self._state.graph.nodes.values()):
            return ()
        source_graph = self._state._get_source_graph_flat()
        dependencies: list[ActivationDependency] = []
        for source_if_id, source_if_node in self._state.graph.nodes.items():
            if not isinstance(source_if_node, IfInvocation):
                continue

            matching_fields = tuple(
                branch_field
                for branch_field in ("true_input", "false_input")
                if source_node_id in self._branch_sources(source_if_id, branch_field, source_graph)
            )
            if len(matching_fields) != 1:
                continue
            dependencies.append(
                ActivationDependency(
                    owner_id=source_if_id,
                    branch=matching_fields[0],
                    frame=iteration_path,
                )
            )
        return tuple(dependencies)

    def is_source_admitted(self, source_node_id: str, iteration_path: tuple[int, ...] = ()) -> bool:
        dependencies = self.get_source_dependencies(source_node_id, iteration_path)
        return bool(
            not dependencies
            or all(self._state._is_activation_dependency_satisfied(dependency) for dependency in dependencies)
        )

    def is_source_rejected(self, source_node_id: str, iteration_path: tuple[int, ...] = ()) -> bool:
        return any(
            self._state._is_activation_dependency_rejected(dependency)
            for dependency in self.get_source_dependencies(source_node_id, iteration_path)
        )

    def is_source_inactive(self, source_node_id: str, iteration_path: tuple[int, ...] = ()) -> bool:
        """Return whether a source belongs exclusively to a rejected If branch."""

        if iteration_path:
            return self.is_source_rejected(source_node_id, iteration_path)

        source_graph = self._state._get_source_graph_flat()
        relevant_if_ids = {
            if_node_id
            for if_node_id, if_node in self._state.graph.nodes.items()
            if isinstance(if_node, IfInvocation)
            and any(
                source_node_id in self._branch_sources(if_node_id, branch_field, source_graph)
                for branch_field in ("true_input", "false_input")
            )
        }
        if not relevant_if_ids:
            return False

        frames: set[tuple[int, ...]] = set()
        for if_node_id in relevant_if_ids:
            frames.update(
                self._state._get_iteration_path(exec_node_id)
                for exec_node_id in self._state._prepared_registry().get_prepared_ids(if_node_id)
            )
        if not frames:
            return False

        for frame in frames:
            dependencies = self.get_source_dependencies(source_node_id, frame)
            if dependencies and not any(
                self._state._is_activation_dependency_rejected(dependency) for dependency in dependencies
            ):
                return False
        return True
