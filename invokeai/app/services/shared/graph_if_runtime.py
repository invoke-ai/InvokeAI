"""If activation and selected-branch context helpers operating on graph execution state."""

from typing import TYPE_CHECKING, Callable, Optional

from invokeai.app.invocations.fields import OutputScope
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
    get_output_field_scope,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState


class UnsupportedIfIterationAxesError(ValueError):
    """Raised when an If condition and its branches cannot share a supported open-axis scope."""


def validate_if_iteration_axes(state: "GraphExecutionState") -> None:
    """Reject unrelated condition and branch axes using graph topology alone."""
    iterator_ancestry = _IteratorAncestry(state)
    collect_axes_cache: dict[str, Optional[tuple[str, ...]]] = {}
    for node_id, node in state.graph.nodes.items():
        if not isinstance(node, IfInvocation):
            continue
        condition_edges = state.graph._get_input_edges(node_id, "condition")
        if not condition_edges:
            _validate_frame_axes(state, iterator_ancestry, node_id, condition_edges, collect_axes_cache)
            continue
        condition_iterators = iterator_ancestry.get_iterator_ancestors_for_edge(condition_edges[0])
        branch_edges = {
            branch_field: state.graph._get_input_edges(node_id, branch_field)
            for branch_field in ("true_input", "false_input")
        }
        broadcast_axes = None
        if all(len(edges) == 1 for edges in branch_edges.values()):
            true_axes = iterator_ancestry.get_iterator_ancestors_for_edge(branch_edges["true_input"][0])
            false_axes = iterator_ancestry.get_iterator_ancestors_for_edge(branch_edges["false_input"][0])
            if true_axes == false_axes and condition_iterators < true_axes:
                broadcast_axes = true_axes
        for branch_field in ("true_input", "false_input"):
            for edge in state.graph._get_input_edges(node_id, branch_field):
                branch_iterators = iterator_ancestry.get_iterator_ancestors_for_edge(edge)
                branch_source = state.graph.get_node(edge.source.node_id)
                if (
                    condition_iterators
                    and condition_iterators < branch_iterators
                    and branch_iterators != broadcast_axes
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
        _validate_frame_axes(state, iterator_ancestry, node_id, condition_edges, collect_axes_cache)


def _validate_frame_axes(
    state: "GraphExecutionState",
    iterator_ancestry: _IteratorAncestry,
    node_id: str,
    condition_edges: list[Edge],
    collect_axes_cache: dict[str, Optional[tuple[str, ...]]],
) -> None:
    """Reject frame sets an If cannot run in.

    The If's frames come from the iterators open at it, including one reached through a For's final output. Two of
    them that are unrelated (neither stays open inside the other) form a cross product, which only the input deciding
    the If's frames can carry: the condition, or the known selected branch of a literal condition. A literal condition
    whose selected branch carries one of the two axes also runs when the other comes only from the unselected branch
    and either:
    - the unselected one reaches the If only through a For's final output, so it adds no frames and the If runs in
      the selected branch's frames; or
    - the selected axis, or an iterator enclosing it, reopens the unselected one one-to-one (see
      `_get_reopened_source_axes`), so the selected frames pair with it by index. Selecting the upstream axis over its
      reopening is not supported.
    A root-frame condition cannot select between branches with different deeper axes.
    """
    if condition_edges:
        _validate_collect_branch_axes(state, iterator_ancestry, node_id, condition_edges[0])
    frame_iterators = iterator_ancestry.get_frame_iterators(node_id)
    if condition_edges:
        deciding_edges = condition_edges
    else:
        node = state.graph.get_node(node_id)
        assert isinstance(node, IfInvocation)
        deciding_edges = state.graph._get_input_edges(node_id, "true_input" if node.condition else "false_input")
    deciding_axes = {axis_id for edge in deciding_edges for axis_id in iterator_ancestry.get_frame_axes_for_edge(edge)}
    for index, axis_id in enumerate(frame_iterators):
        for other_axis_id in frame_iterators[index + 1 :]:
            if axis_id in iterator_ancestry.get_frame_iterators(
                other_axis_id
            ) or other_axis_id in iterator_ancestry.get_frame_iterators(axis_id):
                continue
            if axis_id in deciding_axes and other_axis_id in deciding_axes:
                continue
            if not condition_edges and (axis_id in deciding_axes) != (other_axis_id in deciding_axes):
                selected_id, unselected_id = (
                    (axis_id, other_axis_id) if axis_id in deciding_axes else (other_axis_id, axis_id)
                )
                if unselected_id not in iterator_ancestry.get_ordered_iterator_ancestors(node_id):
                    continue
                if unselected_id in _get_reopened_source_axes(
                    state, iterator_ancestry, selected_id, collect_axes_cache
                ):
                    continue
            raise UnsupportedIfIterationAxesError(
                f"If node '{node_id}' cannot run in the frames of unrelated open iterator axes: "
                f"{sorted((axis_id, other_axis_id))}"
            )
    if len(frame_iterators) < 2 or not condition_edges or iterator_ancestry.get_frame_axes_for_edge(condition_edges[0]):
        return
    branch_axes = {
        branch_field: {
            axis_id
            for edge in state.graph._get_input_edges(node_id, branch_field)
            for axis_id in iterator_ancestry.get_frame_axes_for_edge(edge)
        }
        for branch_field in ("true_input", "false_input")
    }
    if branch_axes["true_input"] != branch_axes["false_input"]:
        raise UnsupportedIfIterationAxesError(
            f"If node '{node_id}' cannot select between branches with different deeper open iterator axes against a "
            f"root-frame condition: true {sorted(branch_axes['true_input'])}, "
            f"false {sorted(branch_axes['false_input'])}"
        )


def _get_reopened_source_axes(
    state: "GraphExecutionState",
    iterator_ancestry: _IteratorAncestry,
    axis_id: str,
    collect_axes_cache: dict[str, Optional[tuple[str, ...]]],
) -> set[str]:
    """Return the axes whose frames an axis, or an iterator enclosing it, reopens one-to-one.

    An Iterate reopens a Collect's item axes when its collection comes straight from that Collect, the Collect
    gathers exactly one item per frame of them (a single item input and nothing appended through `collection`), and
    those axes form one nested chain. Each item then becomes one frame of the Iterate, in the item frames' order, so a
    selected axis can pair by index with those axes, and with the axes they in turn reopen. A node between the Collect
    and the Iterate, extra collected items, or an item crossing unrelated axes can break that pairing, and the If
    would then pair frames of unequal axes by index.
    """
    sources: set[str] = set()
    visited: set[str] = set()
    pending = [axis_id]
    while pending:
        current_id = pending.pop()
        for reopened_id in (current_id, *iterator_ancestry.get_frame_iterators(current_id)):
            if reopened_id in visited:
                continue
            visited.add(reopened_id)
            if not isinstance(state.graph.get_node(reopened_id), IterateInvocation):
                continue
            # Graph validation gives every Iterate exactly one collection input.
            [collection_edge] = state.graph._get_input_edges(reopened_id, COLLECTION_FIELD)
            collect_id = collection_edge.source.node_id
            collect = state.graph.get_node(collect_id)
            if not isinstance(collect, CollectInvocation) or collect.collection:
                continue
            collect_inputs = state.graph._get_input_edges(collect_id)
            if len(collect_inputs) != 1 or collect_inputs[0].destination.field != ITEM_FIELD:
                continue
            item_axes = _get_edge_frame_axes(state, iterator_ancestry, collect_inputs[0], collect_axes_cache)
            if item_axes is None or not _is_nested_chain(iterator_ancestry, item_axes):
                continue
            sources.update(item_axes)
            pending.extend(item_axes)
    return sources


def _validate_collect_branch_axes(
    state: "GraphExecutionState", iterator_ancestry: _IteratorAncestry, node_id: str, condition_edge: Edge
) -> None:
    """Reject a branch Collect that keeps axes open below a per-item condition unless the other branch shares them."""
    condition_axes = set(iterator_ancestry.get_frame_axes_for_edge(condition_edge))
    if not condition_axes:
        return
    branch_axes: dict[str, set[str]] = {}
    for branch_field in ("true_input", "false_input"):
        axes: set[str] = set()
        for edge in state.graph._get_input_edges(node_id, branch_field):
            if isinstance(state.graph.get_node(edge.source.node_id), CollectInvocation):
                axes.update(_get_collect_kept_axes(state, iterator_ancestry, edge.source.node_id))
            else:
                axes.update(iterator_ancestry.get_frame_axes_for_edge(edge))
        branch_axes[branch_field] = axes
    for branch_field, other_field in (("true_input", "false_input"), ("false_input", "true_input")):
        axes = branch_axes[branch_field]
        if axes > condition_axes and axes != branch_axes[other_field]:
            raise UnsupportedIfIterationAxesError(
                f"If node '{node_id}' cannot select branch input '{branch_field}' from deeper open iterator axes "
                f"kept by a Collect: condition {sorted(condition_axes)}, branch {sorted(axes)}"
            )


def _get_collect_kept_axes(
    state: "GraphExecutionState", iterator_ancestry: _IteratorAncestry, collect_id: str
) -> set[str]:
    """Return the axes a Collect's output stays in: its item's axes except the innermost one, which it closes."""
    kept: set[str] = set()
    for edge in state.graph._get_input_edges(collect_id):
        kept.update(_without_closed_axis(state, edge, iterator_ancestry.get_frame_axes_for_edge(edge)))
    return kept


def _without_closed_axis(state: "GraphExecutionState", edge: Edge, axes: tuple[str, ...]) -> tuple[str, ...]:
    """Return the axes a Collect input keeps: an item closes its innermost axis unless it is a For's final output."""
    source = state.graph.get_node(edge.source.node_id)
    is_for_final = isinstance(source, ForInvocation) and (
        get_output_field_scope(source, edge.source.field) == OutputScope.Final
    )
    if edge.destination.field == ITEM_FIELD and not is_for_final:
        return axes[:-1]
    return axes


def _is_nested_chain(iterator_ancestry: _IteratorAncestry, axes: tuple[str, ...]) -> bool:
    """Return whether each axis encloses the next one."""
    return all(
        outer_id in iterator_ancestry.get_frame_iterators(inner_id)
        for outer_id, inner_id in zip(axes, axes[1:], strict=False)
    )


def _get_edge_frame_axes(
    state: "GraphExecutionState",
    iterator_ancestry: _IteratorAncestry,
    edge: Edge,
    collect_axes_cache: dict[str, Optional[tuple[str, ...]]],
) -> Optional[tuple[str, ...]]:
    """Return the frames open on an edge, outer to inner, following a Collect source into the axes it keeps.

    Iterator ancestry stops at a Collect, so a Collect's output would otherwise look like it runs in no frames. A
    Collect whose kept axes are not one nested chain, or that collects such a Collect, has no frame order a reopen could
    pair with, so it yields None. Results are cached per Collect for one validation run and resolved with an explicit
    stack, so a lattice or a long chain of Collects costs one pass per node.
    """
    if not isinstance(state.graph.get_node(edge.source.node_id), CollectInvocation):
        return iterator_ancestry.get_frame_axes_for_edge(edge)
    pending = [edge.source.node_id]
    while pending:
        collect_id = pending[-1]
        if collect_id in collect_axes_cache:
            pending.pop()
            continue
        input_edges = state.graph._get_input_edges(collect_id)
        unresolved = [
            input_edge.source.node_id
            for input_edge in input_edges
            if isinstance(state.graph.get_node(input_edge.source.node_id), CollectInvocation)
            and input_edge.source.node_id not in collect_axes_cache
        ]
        if unresolved:
            pending.extend(unresolved)
            continue
        pending.pop()
        kept: Optional[set[str]] = set()
        for input_edge in input_edges:
            source_id = input_edge.source.node_id
            if isinstance(state.graph.get_node(source_id), CollectInvocation):
                input_axes = collect_axes_cache[source_id]
            else:
                input_axes = iterator_ancestry.get_frame_axes_for_edge(input_edge)
            if input_axes is None:
                kept = None
                break
            kept.update(_without_closed_axis(state, input_edge, input_axes))
        ordered = None
        if kept is not None:
            ordered = tuple(
                sorted(kept, key=lambda axis_id: (len(iterator_ancestry.get_frame_iterators(axis_id)), axis_id))
            )
            if not _is_nested_chain(iterator_ancestry, ordered):
                ordered = None
        collect_axes_cache[collect_id] = ordered
    return collect_axes_cache[edge.source.node_id]


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
                input_source = state.graph.get_node(input_edge.source.node_id)
                # A For's final output already excludes the For's own axis, so the Collect closes nothing more.
                is_for_final = isinstance(input_source, ForInvocation) and (
                    get_output_field_scope(input_source, input_edge.source.field) == OutputScope.Final
                )
                if input_edge.destination.field == ITEM_FIELD and not is_for_final:
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

        if (
            not branch_paths
            and condition_path is not None
            and any(
                edge.source.node_id not in state.source_prepared_mapping and edge.source.node_id not in state.executed
                for edge in selected_edges
            )
        ):
            # The selected branch is not materialized yet; the If still runs in the condition's frame.
            branch_paths = {condition_path}

        for branch_path in branch_paths:
            if condition_path is None:
                paths.add(branch_path)
            elif condition_path[: len(branch_path)] == branch_path:
                paths.add(condition_path)
            elif branch_path[: len(condition_path)] == condition_path:
                paths.add(branch_path)

    # The If runs once per frame of the innermost iterator open at it; validation keeps those iterators nested. An
    # iterator that has not run leaves the frames unknown, and one with no items leaves the If no frames. An iterator
    # reaching the If only through a For's final output does not add frames here.
    frame_iterators = iterator_ancestry.get_ordered_iterator_ancestors(node_id)
    if frame_iterators:
        innermost_id = frame_iterators[-1]
        if innermost_id not in state.executed and innermost_id not in state.source_prepared_mapping:
            return []
        prepared_ids = (
            get_prepared_nodes_for_source(innermost_id) if innermost_id in state.source_prepared_mapping else set()
        )
        iterator_edges = state.graph._get_output_edges(innermost_id, ITEM_FIELD)
        if not prepared_ids:
            return []
        frame_paths = (
            {get_prepared_edge_iteration_path(iterator_edges[0], prepared_id) for prepared_id in prepared_ids}
            if iterator_edges
            else {state._get_iteration_path(prepared_id) for prepared_id in prepared_ids}
        )
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
        return all(_has_empty_if_frame(state, dependency.owner_id) for dependency in dependencies)

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


def _has_empty_if_frame(state: "GraphExecutionState", if_node_id: str) -> bool:
    """Recognize an If that has no executions because an Iterate open at it had no items at all."""

    if not isinstance(state.graph.nodes.get(if_node_id), IfInvocation):
        return False
    return any(
        isinstance(state.graph.get_node(iterator_id), IterateInvocation)
        and iterator_id in state.executed
        and not state.source_prepared_mapping.get(iterator_id)
        for iterator_id in state._runtime()._iterator_ancestry.get_frame_iterators(if_node_id)
    )


def _activation_gate(state: "GraphExecutionState", exec_node_id: str) -> ActivationGate:
    runtime = state._generic_runtime()
    is_new_gate = exec_node_id not in runtime.gates
    gate = runtime.register_gate(
        gate_id=exec_node_id,
        owner_id=exec_node_id,
        frame=state._engine_frame(state._get_iteration_path(exec_node_id)),
        branches=("true_input", "false_input"),
    )
    if is_new_gate:
        state._tx_record(lambda runtime=runtime, gate_id=exec_node_id: runtime.remove_gate(gate_id))
    return gate


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
