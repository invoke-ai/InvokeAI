from typing import TYPE_CHECKING, Optional

from invokeai.app.invocations.fields import OutputScope
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.services.shared.execution_engine.scheduler import ActivationDependency
from invokeai.app.services.shared.graph_validation import (
    ITEM_FIELD,
    CollectInvocation,
    ForInvocation,
    IterateInvocation,
    get_output_field_scope,
    nx,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState
    from invokeai.app.services.shared.graph_models import Edge


class UnsupportedIfFramesError(ValueError):
    """An If whose condition and branch iterate over unrelated collections has no defined set of frames."""


class _IfActivationController:
    """Own activation admission for concrete execution nodes.

    Unlike the legacy compiler, this controller does not lower or cache a
    branch projection. It answers one readiness question for one prepared
    execution node from the current source graph and durable execution frame.
    """

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state
        self._branch_sources_cache: dict[tuple[str, str], frozenset[str]] = {}
        self._branch_framed_cache: dict[str, bool] = {}
        self._frame_depths: Optional[dict[str, int]] = None

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
                    frame=self._get_owner_frame(source_node_id, source_if_id, matching_fields[0], iteration_path),
                )
            )
        return tuple(dependencies)

    def _get_owner_frame(
        self, source_node_id: str, if_node_id: str, branch_field: str, iteration_path: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Return the part of a branch source's frame that the If shares.

        A Collect between a branch source and its If closes the source's innermost frames, so the source runs deeper
        than the If that admits it. Only the leading indexes up to the branch's depth at the If are shared; the rest
        belong to closed frames and must not be compared with If frames from other iterators, which reuse the same
        indexes.
        """

        if not iteration_path or not any(
            isinstance(self._state.graph.nodes.get(node_id), (ForInvocation, IterateInvocation))
            for node_id in (*nx.ancestors(self._state._get_source_graph_flat(), source_node_id), source_node_id)
        ):
            return iteration_path
        branch_depth = max(
            (self._get_edge_depth(edge) for edge in self._state.graph._get_input_edges(if_node_id, branch_field)),
            default=0,
        )
        return iteration_path[:branch_depth]

    def _get_edge_depth(self, edge: "Edge", depths: Optional[dict[str, int]] = None) -> int:
        """Return the frame depth of the value an edge carries; a For's final output is in the For's parent frame."""

        source_node = self._state.graph.get_node(edge.source.node_id)
        depth = (depths if depths is not None else self._get_frame_depths())[edge.source.node_id]
        if isinstance(source_node, ForInvocation) and get_output_field_scope(source_node, edge.source.field) == (
            OutputScope.Final
        ):
            return max(depth - 1, 0)
        return depth

    def _get_frame_depths(self) -> dict[str, int]:
        """Return the length of the iteration path each source node executes at.

        An Iterate or For runs once per item, one level below its input. A Collect's item input closes one level.
        Other nodes run at the deepest level among their inputs.
        """

        if self._frame_depths is not None:
            return self._frame_depths
        depths: dict[str, int] = {}
        for node_id in nx.topological_sort(self._state._get_source_graph_flat()):
            node = self._state.graph.get_node(node_id)
            input_edges = [edge for edge in self._state.graph._get_input_edges(node_id) if edge.type == "default"]
            if isinstance(node, CollectInvocation):
                depth = max(
                    (
                        self._get_edge_depth(edge, depths) - (1 if edge.destination.field == ITEM_FIELD else 0)
                        for edge in input_edges
                    ),
                    default=0,
                )
            else:
                depth = max((self._get_edge_depth(edge, depths) for edge in input_edges), default=0)
                if isinstance(node, (ForInvocation, IterateInvocation)):
                    depth += 1
            depths[node_id] = max(depth, 0)
        self._frame_depths = depths
        return depths

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

    def get_condition_verdict(self, dependency: ActivationDependency) -> Optional[tuple[bool, bool]]:
        """Return ``(satisfied, rejected)`` for a dependency no If execution can gate, else ``None``.

        Such a dependency is decided by the If's condition value in every frame related to the dependency's frame:
        a parent-frame condition decides each item below it, and a parent-frame source is selected when any item
        frame selects it.
        """

        if not self._is_decided_by_condition(dependency.owner_id, dependency.frame):
            return None
        if_node = self._state.graph.get_node(dependency.owner_id)
        assert isinstance(if_node, IfInvocation)
        condition_edges = self._state.graph._get_input_edges(dependency.owner_id, "condition")
        if not condition_edges:
            selected = "true_input" if if_node.condition else "false_input"
            return selected == dependency.branch, selected != dependency.branch

        condition_edge = condition_edges[0]
        condition_source_id = condition_edge.source.node_id
        if condition_source_id not in self._state.source_prepared_mapping:
            return False, False
        materializer = self._state._materializer()
        selected_branches: set[str] = set()
        is_known = True
        for prepared_id in materializer._get_prepared_nodes_for_source(condition_source_id):
            condition_path = materializer._get_prepared_edge_iteration_path(condition_edge, prepared_id)
            shared_depth = min(len(condition_path), len(dependency.frame))
            if condition_path[:shared_depth] != dependency.frame[:shared_depth]:
                continue
            result = self._state.results.get(prepared_id)
            if result is None:
                is_known = False
                continue
            selected_branches.add("true_input" if getattr(result, condition_edge.source.field) else "false_input")
        satisfied = dependency.branch in selected_branches
        return satisfied, is_known and not satisfied

    def _is_decided_by_condition(self, if_node_id: str, frame: tuple[int, ...]) -> bool:
        """Return whether no If execution can gate a branch source running at ``frame``.

        Either the If takes its frames from the branch it selects, or its executions all run in frames below
        ``frame``, as when a per-item condition selects between values computed once in the parent frame.
        """

        if self._is_branch_framed(if_node_id) or self._has_empty_condition(if_node_id):
            return True
        owner_frames = self._state._prepared_if_exec_frames(if_node_id)
        return any(owner_frame[: len(frame)] == frame for owner_frame in owner_frames) and not any(
            frame[: len(owner_frame)] == owner_frame for owner_frame in owner_frames
        )

    def _has_empty_condition(self, if_node_id: str) -> bool:
        """Return whether the condition ran in no frame, so the If never executes and selects neither branch."""

        condition_edges = self._state.graph._get_input_edges(if_node_id, "condition")
        if not condition_edges:
            return False
        condition_source_id = condition_edges[0].source.node_id
        return (
            condition_source_id in self._state.executed
            and not self._state._materializer()._get_prepared_nodes_for_source(condition_source_id)
        )

    def _is_branch_framed(self, if_node_id: str) -> bool:
        """Return whether the If's frames depend on which branch it selects.

        That holds when an iterator's frame stays open into one branch input but not the other, and the condition
        does not carry it. Such an If cannot be materialized until its selected branch runs.
        """

        cached = self._branch_framed_cache.get(if_node_id)
        if cached is not None:
            return cached
        materializer = self._state._materializer()
        source_graph = self._state._get_source_graph_flat()
        condition_iterator_ids: list[str] = []
        branch_iterator_ids: list[str] = []
        for iterator_id in nx.ancestors(source_graph, if_node_id):
            if not isinstance(self._state.graph.get_node(iterator_id), (ForInvocation, IterateInvocation)):
                continue
            if materializer._is_iterator_frame_open_at(iterator_id, if_node_id, {"condition"}):
                condition_iterator_ids.append(iterator_id)
            elif materializer._is_iterator_frame_open_at(
                iterator_id, if_node_id, {"true_input"}
            ) != materializer._is_iterator_frame_open_at(iterator_id, if_node_id, {"false_input"}):
                branch_iterator_ids.append(iterator_id)
        for branch_iterator_id in branch_iterator_ids:
            for condition_iterator_id in condition_iterator_ids:
                if not nx.has_path(source_graph, condition_iterator_id, branch_iterator_id) and not nx.has_path(
                    source_graph, branch_iterator_id, condition_iterator_id
                ):
                    raise UnsupportedIfFramesError(
                        f"If node {if_node_id} combines a condition iterated by {condition_iterator_id} with a branch "
                        f"that may carry frames of the unrelated {branch_iterator_id}, which is not supported"
                    )
        result = bool(branch_iterator_ids)
        self._branch_framed_cache[if_node_id] = result
        return result

    def is_source_inactive(self, source_node_id: str, iteration_path: tuple[int, ...] = ()) -> bool:
        """Return whether a source belongs exclusively to a rejected If branch."""

        if self.is_source_rejected(source_node_id, iteration_path):
            return True
        if iteration_path:
            return False

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
