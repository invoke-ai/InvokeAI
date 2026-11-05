"""Small semantic regressions distilled from the execution-engine fuzzer stalls."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest

from invokeai.app.services.shared.execution_state_migration import dump_execution_state, load_execution_state
from invokeai.app.services.shared.graph import (
    GraphExecutionState,
    _ExecutionScheduler,
    _GenericGraphSchedulerAdapter,
)
from tests.app.services.shared.test_execution_engine_fuzz import build


def _run_seed(
    seed: int,
    *,
    compatibility: bool,
    checkpoint: int | None,
    checkpoint_observer: Callable[[GraphExecutionState], None] | None = None,
    completion_observer: Callable[[GraphExecutionState], None] | None = None,
) -> dict[str, list[tuple[tuple[int, ...], Any]]]:
    _, graph_factory = build(seed)
    state = GraphExecutionState(graph=graph_factory())
    if compatibility:
        state._execution_scheduler = _ExecutionScheduler(state)
    route_type = type(state._scheduler())

    for step in range(500):
        if checkpoint == step:
            state = load_execution_state(json.loads(json.dumps(dump_execution_state(state))))
            state._execution_scheduler = (
                _ExecutionScheduler(state)
                if route_type is _ExecutionScheduler
                else _GenericGraphSchedulerAdapter(state)
            )
            if checkpoint_observer is not None:
                checkpoint_observer(state)
        node = state.next()
        if node is None:
            break
        state.complete(node.id, node.invoke(Mock()))

    assert state.is_complete(), f"seed={seed} route={route_type.__name__} stopped before source completion"
    if completion_observer is not None:
        completion_observer(state)
    outputs: dict[str, list[tuple[tuple[int, ...], Any]]] = {}
    for source_id in ("if", "out", "out2"):
        executions = []
        for exec_id, prepared_source_id in state.prepared_source_mapping.items():
            if prepared_source_id != source_id or exec_id not in state.results:
                continue
            result = state.results[exec_id]
            value = getattr(result, "collection", getattr(result, "value", None))
            executions.append((state._get_iteration_path(exec_id), value))
        outputs[source_id] = sorted(executions)
    return outputs


_ROOT_IF_BRANCH_ITEMS: dict[int, list[tuple[tuple[int, ...], list[list[str]]]]] = {
    # These are the values collected by the selected nested branch for each root If frame.
    2394: [((0, 0), [["c"], [], ["b", "c"]]), ((0, 1), [["c", "b"]])],
    1728: [((0, 0), []), ((0, 1), [["", "a"]]), ((0, 2), [[]])],
    2775: [
        ((0, 0), [[""]]),
        ((0, 1), [["c"], ["", "a", "a"], ["c"]]),
        ((1,), []),
        ((2, 0), [["c"], ["b"]]),
        ((2, 1), [["a"]]),
    ],
}
_ROOT_IF_OTHER_COLLECT_VALUES: dict[int, Any] = {2394: "S", 1728: ["Fu"], 2775: "S"}
_ROOT_IF_PARTIAL_EXPANSION_CHECKPOINTS = {2394: 22, 1728: 24, 2775: 43}


def _expected_root_if_outputs(seed: int) -> list[tuple[tuple[int, ...], str]]:
    _, graph_factory = build(seed)
    graph = graph_factory()
    condition_edge = graph._get_input_edges("if", "condition")[0]
    condition = graph.get_node(condition_edge.source.node_id).value
    selected_field = "true_input" if condition else "false_input"
    selected_branch_edge = graph._get_input_edges("if", selected_field)[0]
    marker = graph.get_node(selected_branch_edge.source.node_id).marker
    return [
        (path, f"{marker}{[f'x{item}' for item in branch_items]}") for path, branch_items in _ROOT_IF_BRANCH_ITEMS[seed]
    ]


def _assert_out_collect_uses_matching_if_frames(
    state: GraphExecutionState, expected_paths: list[tuple[int, ...]]
) -> None:
    if_exec_ids_by_path: dict[tuple[int, ...], list[str]] = {}
    for exec_id in state.source_prepared_mapping["if"]:
        if_exec_ids_by_path.setdefault(state._get_iteration_path(exec_id), []).append(exec_id)
    assert set(if_exec_ids_by_path) == set(expected_paths)
    assert all(len(exec_ids) == 1 for exec_ids in if_exec_ids_by_path.values())

    out_exec_ids = state.source_prepared_mapping["out"]
    assert sorted(state._get_iteration_path(exec_id) for exec_id in out_exec_ids) == sorted(expected_paths)
    for out_exec_id in out_exec_ids:
        out_path = state._get_iteration_path(out_exec_id)
        if_edges = [
            edge
            for edge in state.execution_graph._get_input_edges(out_exec_id)
            if edge.destination.field == "item" and state.prepared_source_mapping.get(edge.source.node_id) == "if"
        ]
        assert len(if_edges) == 1
        assert if_edges[0].source.node_id == if_exec_ids_by_path[out_path][0]


@pytest.mark.parametrize("compatibility", [False, True], ids=["auto", "compatibility"])
@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "partial-expansion-checkpoint"])
@pytest.mark.parametrize("seed", [2394, 1728, 2775], ids=lambda seed: f"seed-{seed}")
def test_root_if_expansion_reframes_reused_collect_consumers(compatibility: bool, resume: bool, seed: int) -> None:
    expected_if = _expected_root_if_outputs(seed)
    expected_paths = [path for path, _ in expected_if]

    def check_partial_expansion(state: GraphExecutionState) -> None:
        _assert_out_collect_uses_matching_if_frames(state, expected_paths)
        completed_if_paths = [
            state._get_iteration_path(exec_id)
            for exec_id in state.source_prepared_mapping["if"]
            if exec_id in state.results
        ]
        assert 0 < len(completed_if_paths) < len(expected_paths)

    outputs = _run_seed(
        seed,
        compatibility=compatibility,
        checkpoint=_ROOT_IF_PARTIAL_EXPANSION_CHECKPOINTS[seed] if resume else None,
        checkpoint_observer=check_partial_expansion if resume else None,
        completion_observer=lambda state: _assert_out_collect_uses_matching_if_frames(state, expected_paths),
    )

    assert outputs["if"] == expected_if
    expected_out_value = _ROOT_IF_OTHER_COLLECT_VALUES[seed]
    assert outputs["out"] == [(path, [expected_out_value, value]) for path, value in expected_if]


@pytest.mark.parametrize("compatibility", [False, True], ids=["auto", "compatibility"])
@pytest.mark.parametrize("checkpoint", [None, 5], ids=["fresh", "resume-after-iteration-preparation"])
def test_root_condition_selects_each_collected_frame(compatibility: bool, checkpoint: int | None) -> None:
    outputs = _run_seed(3, compatibility=compatibility, checkpoint=checkpoint)

    assert outputs["if"] == [
        ((0,), "T:[\"x['']\", \"x['c', 'a']\"]"),
        ((1,), "T:['x[]', \"x['', 'b']\"]"),
    ]


@pytest.mark.parametrize("compatibility", [False, True], ids=["auto", "compatibility"])
@pytest.mark.parametrize("checkpoint", [3, 5], ids=["before-empty-parent-input", "after-empty-parent-input"])
def test_per_item_if_closes_empty_selected_collect_frame(compatibility: bool, checkpoint: int | None) -> None:
    def check_empty_parent_anchor(state: GraphExecutionState) -> None:
        results = [
            (state._get_iteration_path(exec_id), state.results[exec_id].collection)
            for exec_id in state.source_prepared_mapping["cc0"]
            if exec_id in state.results
        ]
        assert sorted(results) == [((0,), [["b", "a"]]), ((1,), [])]

    outputs = _run_seed(
        233,
        compatibility=compatibility,
        checkpoint=checkpoint,
        checkpoint_observer=check_empty_parent_anchor if checkpoint == 5 else None,
    )

    assert outputs["if"] == [((0,), "T:[\"x[['b', 'a']]\"]"), ((1,), [[]])]
    expected_items = ["S:[\"x['b', 'a']\"]", "T:[\"x[['b', 'a']]\"]", "S:[]", [[]]]
    assert len(outputs["out"]) == 1 and outputs["out"][0][0] == ()
    assert sorted(outputs["out"][0][1], key=repr) == sorted(expected_items, key=repr)
    assert outputs["out2"] == [((), [outputs["out"][0][1]])]


@pytest.mark.parametrize("compatibility", [False, True], ids=["auto", "compatibility"])
@pytest.mark.parametrize("checkpoint", [None, 5, 12], ids=["fresh", "resumed", "resumed-after-parent-collections"])
def test_nested_per_item_if_retires_empty_selected_collect_frame(compatibility: bool, checkpoint: int | None) -> None:
    def check_empty_parent_anchor(state: GraphExecutionState) -> None:
        results = [
            (state._get_iteration_path(exec_id), state.results[exec_id].collection)
            for exec_id in state.source_prepared_mapping["cc1"]
            if exec_id in state.results
        ]
        assert sorted(results) == [((0, 0), []), ((0, 1), [["a"], ["c", "b"], []])]
        assert [
            (state._get_iteration_path(exec_id), state.results[exec_id].value)
            for exec_id in state.source_prepared_mapping["sx"]
            if exec_id in state.results
        ] == [((0,), "np")]

    outputs = _run_seed(
        2057,
        compatibility=compatibility,
        checkpoint=checkpoint,
        checkpoint_observer=check_empty_parent_anchor if checkpoint == 12 else None,
    )

    assert outputs["if"] == [
        ((0, 0), [[]]),
        ((0, 1), [["x['a']", "x['c', 'b']", "x[]"]]),
    ]
    assert outputs["out"][0] == ((), ["np"])
    expected_nested_items = [[[]], [["x['a']", "x['c', 'b']", "x[]"]]]
    assert outputs["out"][1][0] == (0,)
    assert sorted(outputs["out"][1][1], key=repr) == sorted(expected_nested_items, key=repr)
    assert outputs["out2"][0][0] == ()
    assert sorted(outputs["out2"][0][1], key=repr) == sorted([["np"], outputs["out"][1][1]], key=repr)


@pytest.mark.parametrize("compatibility", [False, True], ids=["auto", "compatibility"])
@pytest.mark.parametrize("checkpoint", [None, 5], ids=["fresh", "resumed"])
def test_empty_if_frames_retire_and_finish_downstream_collects(compatibility: bool, checkpoint: int | None) -> None:
    outputs = _run_seed(2575, compatibility=compatibility, checkpoint=checkpoint)

    assert outputs["out"] == [((), ["np"]), ((0,), []), ((1,), [])]


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (880, [((), ["S:[]"])]),
        (992, [((0,), [["Fu"]])]),
    ],
    ids=["empty-iterate-branch", "empty-iterate-nested-collect"],
)
@pytest.mark.parametrize("compatibility", [False, True], ids=["auto", "compatibility"])
@pytest.mark.parametrize("checkpoint", [None, 5], ids=["fresh", "resumed"])
def test_empty_iterate_frames_retire_before_downstream_collect(
    seed: int,
    expected: list[tuple[tuple[int, ...], Any]],
    compatibility: bool,
    checkpoint: int | None,
) -> None:
    outputs = _run_seed(seed, compatibility=compatibility, checkpoint=checkpoint)

    assert outputs["out"] == expected


@pytest.mark.parametrize("compatibility", [False, True], ids=["auto", "compatibility"])
def test_nested_if_branch_completion_reaches_downstream_collect(compatibility: bool) -> None:
    outputs = _run_seed(573, compatibility=compatibility, checkpoint=4)

    # The selected true branch comes from an empty Iterate; the J input is unselected.
    assert outputs["if"] == []
    assert outputs["out"] == [((), [])]


@pytest.mark.parametrize("compatibility", [False, True], ids=["auto", "compatibility"])
@pytest.mark.parametrize("checkpoint", [None, 5], ids=["fresh", "resumed"])
def test_root_condition_expands_partial_frames_from_completed_branch(
    compatibility: bool, checkpoint: int | None
) -> None:
    outputs = _run_seed(1331, compatibility=compatibility, checkpoint=checkpoint)

    assert outputs["if"] == [
        ((0, 0), "T:bT:[\"x['c']\", \"x['', 'c']\"]"),
        ((1, 0), "T:bT:[\"x['a', 'a']\", \"x['b', 'a', 'c']\", \"x['b']\"]"),
    ]


@pytest.mark.parametrize("compatibility", [False, True], ids=["auto", "compatibility"])
@pytest.mark.parametrize("checkpoint", [None, 15], ids=["fresh", "blocked-frame-resume"])
def test_nested_if_activation_uses_matching_axis_frame(compatibility: bool, checkpoint: int | None) -> None:
    outputs = _run_seed(1836, compatibility=compatibility, checkpoint=checkpoint)
    selected_value = "bT:[\"F['c']\"]"

    assert outputs == {
        "if": [((0, 0), selected_value)],
        "out": [((0,), ["S", selected_value]), ((1,), ["S"])],
        "out2": [((), [["S", selected_value], ["S"]])],
    }
