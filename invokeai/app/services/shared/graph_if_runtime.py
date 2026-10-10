"""If activation and selected-branch context helpers operating on graph execution state."""

from typing import TYPE_CHECKING, Callable, Optional

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.services.shared.execution_engine.primitives import ActivationGate
from invokeai.app.services.shared.execution_engine.scheduler import ActivationDependency
from invokeai.app.services.shared.graph_models import Edge, ExecutionToken
from invokeai.app.services.shared.graph_validation import ForInvocation, IterateInvocation, nx

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


def _get_effective_iteration_paths_for_edge(
    state: "GraphExecutionState",
    edge: Edge,
    visited_source_ids: set[str],
    get_prepared_nodes_for_source: Callable[[str], set[str]],
    get_prepared_edge_iteration_path: Callable[[Edge, str], tuple[int, ...]],
) -> set[tuple[int, ...]]:
    source_node_id = edge.source.node_id
    prepared_nodes: set[str] = set()
    if source_node_id in state.source_prepared_mapping:
        prepared_nodes = get_prepared_nodes_for_source(source_node_id)
        if not prepared_nodes and source_node_id in state.executed:
            return set()
    if prepared_nodes:
        return {get_prepared_edge_iteration_path(edge, prepared_id) for prepared_id in prepared_nodes}
    if source_node_id in visited_source_ids:
        return set()

    source_node = state.graph.get_node(source_node_id)
    if isinstance(source_node, IfInvocation):
        return set(
            _get_if_iteration_paths(
                state,
                source_node_id,
                visited_source_ids,
                get_prepared_nodes_for_source,
                get_prepared_edge_iteration_path,
            )
        )

    input_edges = state.graph._get_input_edges(source_node_id)
    if not input_edges:
        return {()}
    next_visited_source_ids = {*visited_source_ids, source_node_id}
    return set().union(
        *(
            _get_effective_iteration_paths_for_edge(
                state,
                input_edge,
                next_visited_source_ids,
                get_prepared_nodes_for_source,
                get_prepared_edge_iteration_path,
            )
            for input_edge in input_edges
        )
    )


def _get_if_iteration_paths(
    state: "GraphExecutionState",
    node_id: str,
    visited_source_ids: set[str],
    get_prepared_nodes_for_source: Callable[[str], set[str]],
    get_prepared_edge_iteration_path: Callable[[Edge, str], tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Infer the selected branch's effective iteration paths for an If output."""
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

    if not paths:
        if not has_selected_edges:
            return [()]
        if not condition_edges and has_unresolved_selected_source:
            iterator_graph = state._iterator_graph(state._get_source_graph_flat())
            has_iterator_ancestor = any(
                isinstance(state.graph.get_node(ancestor_id), (ForInvocation, IterateInvocation))
                for ancestor_id in nx.ancestors(iterator_graph, node_id)
            )
            if not has_iterator_ancestor:
                return [()]
        return []
    return sorted(path for path in paths if not any(path != other and other[: len(path)] == path for other in paths))


def _record_activation_dependencies(
    state: "GraphExecutionState", exec_node_id: str
) -> tuple[ActivationDependency, ...]:
    dependencies = state._if_activation_dependencies_by_exec.get(exec_node_id)
    if dependencies is not None:
        return dependencies
    source_node_id = state._prepared_registry().get_source_node_id(exec_node_id)
    if any(isinstance(node, IfInvocation) for node in state.graph.nodes.values()):
        dependencies = state._get_source_activation_dependencies(
            source_node_id, state._get_iteration_path(exec_node_id)
        )
    else:
        dependencies = ()
    state._tx_set_mapping(state._if_activation_dependencies_by_exec, exec_node_id, dependencies)
    return dependencies


def _is_source_activation_admitted(
    state: "GraphExecutionState", source_node_id: str, iteration_path: tuple[int, ...] = ()
) -> bool:
    dependencies = state._get_source_activation_dependencies(source_node_id, iteration_path)
    return not dependencies or all(state._is_activation_dependency_satisfied(dependency) for dependency in dependencies)


def _is_source_inactive(
    state: "GraphExecutionState", source_node_id: str, iteration_path: tuple[int, ...] = ()
) -> bool:
    if not state._can_use_fresh_flat_if_activation():
        return state._if_activation_controller().is_source_inactive(source_node_id, iteration_path)

    dependencies = state._get_source_activation_dependencies(source_node_id, iteration_path)
    if not dependencies:
        return False
    if iteration_path:
        return any(state._is_activation_dependency_rejected(dependency) for dependency in dependencies)

    frames = {
        iteration_path
        for dependency in dependencies
        for iteration_path in state._prepared_if_exec_frames(dependency.owner_id)
    }
    if not frames:
        return all(_is_empty_if_condition(state, dependency) for dependency in dependencies)
    return all(
        state._get_source_activation_dependencies(source_node_id, frame)
        and any(
            state._is_activation_dependency_rejected(frame_dependency)
            for frame_dependency in state._get_source_activation_dependencies(source_node_id, frame)
        )
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
