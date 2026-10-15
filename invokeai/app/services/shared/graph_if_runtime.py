"""If activation and selected-branch context helpers operating on graph execution state."""

from typing import TYPE_CHECKING, Callable, Optional

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.services.shared.execution_engine.primitives import ActivationGate
from invokeai.app.services.shared.execution_engine.scheduler import ActivationDependency
from invokeai.app.services.shared.graph_iterator_ancestry import _IteratorAncestry
from invokeai.app.services.shared.graph_models import Edge, ExecutionToken
from invokeai.app.services.shared.graph_validation import (
    COLLECTION_FIELD,
    ITEM_FIELD,
    CollectInvocation,
    ForInvocation,
    IterateInvocation,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


class UnsupportedIfIterationAxesError(ValueError):
    """Raised when an If condition and its branches cannot share a supported open-axis scope."""


def validate_if_iteration_axes(state: "GraphExecutionState") -> None:
    """Reject unrelated condition and branch axes using graph topology alone."""
    iterator_ancestry = _IteratorAncestry(state)
    for node_id, node in state.graph.nodes.items():
        if not isinstance(node, IfInvocation):
            continue
        condition_edges = state.graph._get_input_edges(node_id, "condition")
        if not condition_edges:
            continue
        condition_iterators = iterator_ancestry.get_iterator_ancestors_for_edge(condition_edges[0])
        for branch_field in ("true_input", "false_input"):
            for edge in state.graph._get_input_edges(node_id, branch_field):
                branch_iterators = iterator_ancestry.get_iterator_ancestors_for_edge(edge)
                branch_source = state.graph.get_node(edge.source.node_id)
                if (
                    condition_iterators
                    and condition_iterators < branch_iterators
                    and not isinstance(branch_source, CollectInvocation)
                ):
                    raise UnsupportedIfIterationAxesError(
                        f"If node '{node_id}' cannot select branch input '{branch_field}' from deeper open iterator "
                        f"axes without a Collect boundary: condition {sorted(condition_iterators)}, "
                        f"branch {sorted(branch_iterators)}"
                    )
                shared_iterators = condition_iterators.intersection(branch_iterators)
                nested_axes = condition_iterators.issubset(branch_iterators) or branch_iterators.issubset(
                    condition_iterators
                )
                if condition_iterators and branch_iterators and (not shared_iterators or not nested_axes):
                    raise UnsupportedIfIterationAxesError(
                        f"If node '{node_id}' cannot select branch input '{branch_field}' from "
                        f"unrelated open iterator axes: condition {sorted(condition_iterators)}, "
                        f"branch {sorted(branch_iterators)}"
                    )


def _get_effective_iteration_paths_for_edge(
    state: "GraphExecutionState",
    edge: Edge,
    visited_source_ids: set[str],
    get_prepared_nodes_for_source: Callable[[str], set[str]],
    get_prepared_edge_iteration_path: Callable[[Edge, str], tuple[int, ...]],
    iterator_ancestry: _IteratorAncestry,
    effective_paths_by_edge: Optional[dict[tuple[str, str], set[tuple[int, ...]]]] = None,
) -> set[tuple[int, ...]]:
    cache = effective_paths_by_edge if effective_paths_by_edge is not None else {}
    pending: list[tuple[Edge, bool]] = [(edge, False)]

    while pending:
        current_edge, exiting_source = pending.pop()
        current_source_id = current_edge.source.node_id
        cache_key = (current_source_id, current_edge.source.field)
        if cache_key in cache:
            continue

        if exiting_source:
            input_edges = state.graph._get_input_edges(current_source_id)
            if not input_edges:
                cache[cache_key] = {()}
            else:
                cache[cache_key] = set().union(
                    *(cache[(input_edge.source.node_id, input_edge.source.field)] for input_edge in input_edges)
                )
            continue

        if current_source_id in visited_source_ids:
            cache[cache_key] = set()
            continue
        prepared_nodes: set[str] = set()
        if current_source_id in state.source_prepared_mapping:
            prepared_nodes = get_prepared_nodes_for_source(current_source_id)
            if not prepared_nodes and current_source_id in state.executed:
                cache[cache_key] = set()
                continue
        if prepared_nodes:
            cache[cache_key] = {
                get_prepared_edge_iteration_path(current_edge, prepared_id) for prepared_id in prepared_nodes
            }
            continue

        source_node = state.graph.get_node(current_source_id)
        if isinstance(source_node, CollectInvocation) and current_edge.source.field == COLLECTION_FIELD:
            input_edges = state.graph._get_input_edges(current_source_id)
            collapsed_paths: set[tuple[int, ...]] = set()
            # Collect.item collapses its final open axis even before its own execution frame exists.
            for input_edge in input_edges:
                input_paths = _get_effective_iteration_paths_for_edge(
                    state,
                    input_edge,
                    visited_source_ids,
                    get_prepared_nodes_for_source,
                    get_prepared_edge_iteration_path,
                    iterator_ancestry,
                    cache,
                )
                group_depth = len(iterator_ancestry.get_iterator_axes_for_edge(input_edge))
                if input_edge.destination.field == ITEM_FIELD:
                    group_depth = max(group_depth - 1, 0)
                collapsed_paths.update(path[:group_depth] for path in input_paths)
            cache[cache_key] = collapsed_paths or ({()} if not input_edges else set())
            continue
        if isinstance(source_node, IfInvocation):
            cache[cache_key] = set(
                _get_if_iteration_paths_with_ancestry(
                    state,
                    current_source_id,
                    visited_source_ids,
                    get_prepared_nodes_for_source,
                    get_prepared_edge_iteration_path,
                    iterator_ancestry,
                )
            )
            continue

        input_edges = state.graph._get_input_edges(current_source_id)
        if not input_edges:
            cache[cache_key] = {()}
            continue
        pending.append((current_edge, True))
        pending.extend((input_edge, False) for input_edge in reversed(input_edges))

    return cache[(edge.source.node_id, edge.source.field)]


def _get_if_iteration_paths_with_ancestry(
    state: "GraphExecutionState",
    node_id: str,
    visited_source_ids: set[str],
    get_prepared_nodes_for_source: Callable[[str], set[str]],
    get_prepared_edge_iteration_path: Callable[[Edge, str], tuple[int, ...]],
    iterator_ancestry: _IteratorAncestry,
) -> list[tuple[int, ...]]:
    """Infer selected branch paths using one ancestry context for all recursion."""
    effective_paths_by_edge = iterator_ancestry.effective_iteration_paths_by_edge
    node = state.graph.get_node(node_id)
    assert isinstance(node, IfInvocation)
    next_visited_source_ids = {*visited_source_ids, node_id}
    condition_edges = state.graph._get_input_edges(node_id, "condition")

    condition_selections: list[tuple[Optional[tuple[int, ...]], str]] = []
    if condition_edges:
        condition_edge = condition_edges[0]
        if condition_edge.source.node_id not in state.source_prepared_mapping:
            return []
        for prepared_id in get_prepared_nodes_for_source(condition_edge.source.node_id):
            if prepared_id not in state.results:
                return []
            condition_value = getattr(state.results[prepared_id], condition_edge.source.field)
            condition_path = get_prepared_edge_iteration_path(condition_edge, prepared_id)
            selected_field = "true_input" if condition_value else "false_input"
            condition_selections.append((condition_path, selected_field))
    else:
        selected_field = "true_input" if node.condition else "false_input"
        condition_selections.append((None, selected_field))

    paths: set[tuple[int, ...]] = set()
    has_selected_edges = False
    has_unresolved_selected_source = False
    for condition_path, selected_field in condition_selections:
        selected_edges = state.graph._get_input_edges(node_id, selected_field)
        has_selected_edges = has_selected_edges or bool(selected_edges)
        has_unresolved_selected_source = has_unresolved_selected_source or any(
            edge.source.node_id not in state.source_prepared_mapping and edge.source.node_id not in state.executed
            for edge in selected_edges
        )
        if selected_edges:
            branch_paths = set().union(
                *(
                    _get_effective_iteration_paths_for_edge(
                        state,
                        edge,
                        next_visited_source_ids,
                        get_prepared_nodes_for_source,
                        get_prepared_edge_iteration_path,
                        iterator_ancestry,
                        effective_paths_by_edge,
                    )
                    for edge in selected_edges
                )
            )
        else:
            branch_paths = {condition_path or ()}

        for branch_path in branch_paths:
            if condition_path is None:
                paths.add(branch_path)
            elif condition_path[: len(branch_path)] == branch_path:
                paths.add(condition_path)
            elif branch_path[: len(condition_path)] == condition_path:
                paths.add(branch_path)

    if not condition_edges:
        frame_iterators = iterator_ancestry.get_ordered_iterator_ancestors(node_id)
        if frame_iterators:
            frame_paths: set[tuple[int, ...]] = set()
            for iterator_id in reversed(frame_iterators):
                if iterator_id not in state.source_prepared_mapping:
                    if iterator_id not in state.executed:
                        return []
                    continue
                prepared_ids = get_prepared_nodes_for_source(iterator_id)
                iterator_edges = state.graph._get_output_edges(iterator_id, ITEM_FIELD)
                if prepared_ids and iterator_edges:
                    frame_paths = {
                        get_prepared_edge_iteration_path(iterator_edges[0], prepared_id) for prepared_id in prepared_ids
                    }
                    break
            if not frame_paths:
                return []
            paths = {
                frame_path if frame_path[: len(branch_path)] == branch_path else branch_path
                for branch_path in paths
                for frame_path in frame_paths
                if frame_path[: len(branch_path)] == branch_path or branch_path[: len(frame_path)] == frame_path
            }

    if not paths:
        if not has_selected_edges:
            return [()]
        if not condition_edges and has_unresolved_selected_source:
            if not iterator_ancestry.has_iterator_ancestor(node_id):
                return [()]
        return []
    return sorted(path for path in paths if not any(path != other and other[: len(path)] == path for other in paths))


def _record_activation_dependencies(
    state: "GraphExecutionState", exec_node_id: str
) -> tuple[ActivationDependency, ...]:
    dependencies = state._if_activation_dependencies_by_exec.get(exec_node_id)
    if dependencies is not None:
        state._index_activation_dependencies(exec_node_id, dependencies)
        return dependencies
    source_node_id = state._prepared_registry().get_source_node_id(exec_node_id)
    if any(isinstance(node, IfInvocation) for node in state.graph.nodes.values()):
        dependencies = _get_source_activation_dependencies_for_preparation(
            state, source_node_id, state._get_iteration_path(exec_node_id)
        )
    else:
        dependencies = ()
    state._tx_set_mapping(state._if_activation_dependencies_by_exec, exec_node_id, dependencies)
    state._index_activation_dependencies(exec_node_id, dependencies)
    return dependencies


def _get_source_activation_dependencies_for_preparation(
    state: "GraphExecutionState", source_node_id: str, iteration_path: tuple[int, ...]
) -> tuple[ActivationDependency, ...]:
    dependencies = state._get_source_activation_dependencies(source_node_id, iteration_path)
    if not dependencies:
        return dependencies

    runtime = state._runtime()
    source_axes = runtime._get_ordered_iterator_sources(source_node_id)
    # Materialization can prepend a condition axis to a branch iterator, so a dependent node's path may
    # follow input-port order rather than its graph-ancestry order. Use the path's construction order when
    # translating axis coordinates into the selected If frame. Collect nodes can collapse a trailing axis,
    # so only axes represented by this path participate.
    input_axes = state._materializer()._get_ordered_input_iterator_axes(state.graph._get_input_edges(source_node_id))
    if input_axes:
        source_axes = list(input_axes[: len(iteration_path)])
    resolved_dependencies = []
    for dependency in dependencies:
        owner_axes = runtime._get_ordered_iterator_sources(dependency.owner_id)
        shared_axis_indices = [
            (owner_axes.index(axis_id), source_axes.index(axis_id)) for axis_id in owner_axes if axis_id in source_axes
        ]
        candidate_frames = state._prepared_if_exec_frames(dependency.owner_id)
        matching_frames = [
            frame
            for frame in candidate_frames
            if all(
                owner_index < len(frame)
                and source_index < len(iteration_path)
                and frame[owner_index] == iteration_path[source_index]
                for owner_index, source_index in shared_axis_indices
            )
        ]
        if not matching_frames:
            resolved_dependencies.append(dependency)
            continue

        frame_dependencies = [
            ActivationDependency(owner_id=dependency.owner_id, branch=dependency.branch, frame=frame)
            for frame in matching_frames
        ]
        resolved_dependencies.append(
            next(
                (candidate for candidate in frame_dependencies if state._is_activation_dependency_satisfied(candidate)),
                frame_dependencies[0],
            )
        )
    return tuple(resolved_dependencies)


def _get_selected_if_frames_for_preparation(
    state: "GraphExecutionState", if_node_id: str, branch_field: Optional[str]
) -> tuple[tuple[int, ...], ...]:
    """Return only resolved If frames that selected a Collect-backed branch."""
    cache_key = (if_node_id, branch_field)
    cached = state._selected_if_frames_for_preparation.get(cache_key)
    if cached is None:
        selected_frames = tuple(
            state._get_iteration_path(exec_node_id)
            for exec_node_id in state.source_prepared_mapping.get(if_node_id, ())
            if branch_field is None or state._resolved_if_exec_branches.get(exec_node_id) == branch_field
        )
        cached = (selected_frames, frozenset(selected_frames))
        state._selected_if_frames_for_preparation[cache_key] = cached
    return cached[0]


def _has_selected_if_frame_for_preparation(
    state: "GraphExecutionState", if_node_id: str, branch_field: Optional[str], frame: tuple[int, ...]
) -> bool:
    """Test one selected frame without scanning every frame owned by the If."""
    _get_selected_if_frames_for_preparation(state, if_node_id, branch_field)
    return frame in state._selected_if_frames_for_preparation[(if_node_id, branch_field)][1]


def _is_source_activation_admitted(
    state: "GraphExecutionState", source_node_id: str, iteration_path: tuple[int, ...] = ()
) -> bool:
    dependencies = _get_source_activation_dependencies_for_preparation(state, source_node_id, iteration_path)
    return not dependencies or all(state._is_activation_dependency_satisfied(dependency) for dependency in dependencies)


def _is_activation_axis_complete(state: "GraphExecutionState", axis_source_id: str) -> bool:
    axis_source = state.graph.nodes.get(axis_source_id)
    if isinstance(axis_source, ForInvocation):
        return state._all_for_contexts_finalized(axis_source_id)
    if isinstance(axis_source, IterateInvocation):
        streams = [stream for stream in state._generic_runtime().streams.values() if stream.owner_id == axis_source_id]
        if streams:
            return all(stream.closed for stream in streams)
        prepared_ids = state.source_prepared_mapping.get(axis_source_id, ())
        return (
            axis_source_id in state.executed
            or bool(prepared_ids)
            and all(exec_node_id in state.executed for exec_node_id in prepared_ids)
        )
    return True


def _is_source_inactive(
    state: "GraphExecutionState", source_node_id: str, iteration_path: tuple[int, ...] = ()
) -> bool:
    dependencies = state._get_source_activation_dependencies(source_node_id, iteration_path)
    if not dependencies:
        return False
    if iteration_path:
        prepared_dependencies = _get_source_activation_dependencies_for_preparation(
            state, source_node_id, iteration_path
        )
        return any(state._is_activation_dependency_rejected(dependency) for dependency in prepared_dependencies)

    frames = {frame for dependency in dependencies for frame in state._prepared_if_exec_frames(dependency.owner_id)}
    if not frames:
        return all(_is_empty_if_condition(state, dependency) for dependency in dependencies)

    source_axes = set(state._runtime()._get_ordered_iterator_sources(source_node_id))
    for dependency in dependencies:
        for axis_source_id in state._runtime()._get_ordered_iterator_sources(dependency.owner_id):
            if axis_source_id not in source_axes and not _is_activation_axis_complete(state, axis_source_id):
                return False

    return all(
        (prepared_dependencies := _get_source_activation_dependencies_for_preparation(state, source_node_id, frame))
        and any(state._is_activation_dependency_rejected(dependency) for dependency in prepared_dependencies)
        for frame in frames
    )


def _is_empty_if_condition(state: "GraphExecutionState", dependency: ActivationDependency) -> bool:
    """Recognize a branch whose If never materialized because its Iterate was empty."""

    owner = state.graph.nodes.get(dependency.owner_id)
    if not isinstance(owner, IfInvocation):
        return False
    condition_edges = state.graph._get_input_edges(dependency.owner_id, "condition")
    if len(condition_edges) != 1:
        return False
    condition_source_id = condition_edges[0].source.node_id
    return (
        isinstance(state.graph.get_node(condition_source_id), IterateInvocation)
        and condition_source_id in state.executed
        and not state.source_prepared_mapping.get(condition_source_id)
    )


def _activation_gate(state: "GraphExecutionState", exec_node_id: str) -> ActivationGate:
    return state._generic_runtime().register_gate(
        gate_id=exec_node_id,
        owner_id=exec_node_id,
        frame=state._engine_frame(state._get_iteration_path(exec_node_id)),
        branches=("true_input", "false_input"),
    )


def _record_compatibility_activation_token(
    state: "GraphExecutionState", exec_node_id: str, selected_field: str
) -> None:
    execution_ref = state._expected_execution_ref(exec_node_id)
    activation_token_id = f"{execution_ref.reference_id}:activation:{selected_field}"
    state._tx_set_mapping(
        state.execution_tokens,
        activation_token_id,
        ExecutionToken(
            token_id=activation_token_id,
            reference_id=execution_ref.reference_id,
            owner_node_id=exec_node_id,
            port=selected_field,
            frame=execution_ref.frame,
            value=selected_field,
            token_kind="activation",
        ),
    )


def _is_activation_dependency_satisfied(state: "GraphExecutionState", dependency: ActivationDependency) -> bool:
    """Check one opaque plan requirement against durable gate and token state."""
    matching_gate_ids = state._prepared_if_exec_ids(dependency.owner_id, dependency.frame)
    if not matching_gate_ids:
        return False

    for gate_id in matching_gate_ids:
        expected_ref = state._expected_execution_ref(gate_id)
        gate = state._activation_gate(gate_id)
        if (
            gate.frame.state_id != expected_ref.frame.state_id
            or gate.frame.frame_id != expected_ref.frame.frame_id
            or gate.frame.iteration_path != expected_ref.frame.iteration_path
            or gate.frame.workflow_call_depth != expected_ref.frame.workflow_call_depth
            or gate.frame.iteration_path != dependency.frame
        ):
            return False
        if not gate.is_active(dependency.branch, owner_id=gate_id, frame=gate.frame):
            return False
        token = state.execution_tokens.get(f"{expected_ref.reference_id}:activation:{dependency.branch}")
        if token is None or not (
            token.token_id == f"{expected_ref.reference_id}:activation:{dependency.branch}"
            and token.reference_id == expected_ref.reference_id
            and token.owner_node_id == gate_id
            and token.token_kind == "activation"
            and token.port == dependency.branch
            and token.value == dependency.branch
            and token.frame.state_id == expected_ref.frame.state_id
            and token.frame.frame_id == expected_ref.frame.frame_id
            and token.frame.iteration_path == expected_ref.frame.iteration_path
            and token.frame.workflow_call_depth == expected_ref.frame.workflow_call_depth
        ):
            return False
    return True


def _is_activation_dependency_rejected(state: "GraphExecutionState", dependency: ActivationDependency) -> bool:
    """Return whether a resolved gate explicitly selected another branch."""

    matching_gate_ids = state._prepared_if_exec_ids(dependency.owner_id, dependency.frame)
    for gate_id in matching_gate_ids:
        gate = state._activation_gate(gate_id)
        expected_frame = state._expected_execution_ref(gate_id).frame
        if (
            gate.frame.state_id != expected_frame.state_id
            or gate.frame.frame_id != expected_frame.frame_id
            or gate.frame.iteration_path != expected_frame.iteration_path
            or gate.frame.workflow_call_depth != expected_frame.workflow_call_depth
        ):
            return False
        if gate.resolved and gate.selected_branch != dependency.branch:
            return True
    return False


def _resolve_activation_gate(state: "GraphExecutionState", exec_node_id: str, branch: str) -> bool:
    runtime = state._generic_runtime()
    gate = state._activation_gate(exec_node_id)
    if gate.resolved and gate.selected_branch == branch:
        return False
    previous = ActivationGate.model_validate(gate.model_dump(mode="python"))
    changed = runtime.resolve_gate(
        exec_node_id,
        exec_node_id,
        gate.frame,
        branch,
    )
    if changed:
        state._completed_source_ids_cache = None
        state._tx_record(lambda: runtime.replace_gate(previous))
        materializer = state.__pydantic_private__["_runtime_state"].execution_materializer
        if materializer is not None:
            source_if_id = state._prepared_registry().get_source_node_id(exec_node_id)
            materializer._queue_rejected_activation_branch(source_if_id, branch)
    return changed


def _is_deferred_by_unresolved_if(state: "GraphExecutionState", exec_node_id: str) -> bool:
    dependencies = state._get_activation_dependencies(exec_node_id)
    if not dependencies or any(state._is_activation_dependency_rejected(dependency) for dependency in dependencies):
        return False
    return not all(state._is_activation_dependency_satisfied(dependency) for dependency in dependencies)


def _has_rejected_activation_dependency(state: "GraphExecutionState", exec_node_id: str) -> bool:
    return any(
        state._is_activation_dependency_rejected(dependency)
        for dependency in state._get_activation_dependencies(exec_node_id)
    )


def _get_activation_dependencies(state: "GraphExecutionState", exec_node_id: str) -> tuple[ActivationDependency, ...]:
    return state._record_activation_dependencies(exec_node_id)
