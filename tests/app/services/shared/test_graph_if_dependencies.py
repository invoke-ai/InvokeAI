from __future__ import annotations

from typing import Any

import pytest

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.primitives import BooleanInvocation
from invokeai.app.services.shared import graph as graph_module
from invokeai.app.services.shared import graph_if_dependencies
from invokeai.app.services.shared.execution_engine import ActivationDependency
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph, GraphExecutionState
from tests.app.services.shared.import_test_utils import assert_module_imports_without_graph


def _edge(source: str, source_field: str, destination: str, destination_field: str) -> Edge:
    return Edge(
        source=EdgeConnection(node_id=source, field=source_field),
        destination=EdgeConnection(node_id=destination, field=destination_field),
    )


def _flat_if_graph() -> Graph:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(AddInvocation(id="true_branch", a=2, b=3))
    graph.add_node(AddInvocation(id="false_branch", a=10, b=20))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AddInvocation(id="sink", b=1))
    graph.add_edge(_edge("condition", "value", "if", "condition"))
    graph.add_edge(_edge("true_branch", "value", "if", "true_input"))
    graph.add_edge(_edge("false_branch", "value", "if", "false_input"))
    graph.add_edge(_edge("if", "value", "sink", "a"))
    return graph


def _nested_if_graph() -> Graph:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="outer_condition", value=True))
    graph.add_node(BooleanInvocation(id="inner_condition", value=False))
    graph.add_node(AddInvocation(id="inner_true", a=2, b=2))
    graph.add_node(AddInvocation(id="inner_false", a=3, b=3))
    graph.add_node(AddInvocation(id="outer_false", a=10, b=0))
    graph.add_node(IfInvocation(id="inner_if"))
    graph.add_node(IfInvocation(id="outer_if"))
    graph.add_node(AddInvocation(id="sink", b=1))
    graph.add_edge(_edge("outer_condition", "value", "outer_if", "condition"))
    graph.add_edge(_edge("inner_condition", "value", "inner_if", "condition"))
    graph.add_edge(_edge("inner_true", "value", "inner_if", "true_input"))
    graph.add_edge(_edge("inner_false", "value", "inner_if", "false_input"))
    graph.add_edge(_edge("inner_if", "value", "outer_if", "true_input"))
    graph.add_edge(_edge("outer_false", "value", "outer_if", "false_input"))
    graph.add_edge(_edge("outer_if", "value", "sink", "a"))
    return graph


def _sibling_if_graph(*, first_condition: bool = True, second_condition: bool = True) -> Graph:
    graph = Graph()
    graph.add_node(AddInvocation(id="shared", a=1, b=1))
    graph.add_node(BooleanInvocation(id="first_condition", value=first_condition))
    graph.add_node(BooleanInvocation(id="second_condition", value=second_condition))
    graph.add_node(AddInvocation(id="first_true", b=3))
    graph.add_node(AddInvocation(id="first_false", a=10, b=1))
    graph.add_node(AddInvocation(id="second_true", b=4))
    graph.add_node(AddInvocation(id="second_false", a=20, b=1))
    graph.add_node(IfInvocation(id="first_if"))
    graph.add_node(IfInvocation(id="second_if"))
    graph.add_node(AddInvocation(id="first_sink", b=100))
    graph.add_node(AddInvocation(id="second_sink", b=200))
    graph.add_edge(_edge("shared", "value", "first_true", "a"))
    graph.add_edge(_edge("shared", "value", "second_true", "a"))
    graph.add_edge(_edge("first_condition", "value", "first_if", "condition"))
    graph.add_edge(_edge("first_true", "value", "first_if", "true_input"))
    graph.add_edge(_edge("first_false", "value", "first_if", "false_input"))
    graph.add_edge(_edge("second_condition", "value", "second_if", "condition"))
    graph.add_edge(_edge("second_true", "value", "second_if", "true_input"))
    graph.add_edge(_edge("second_false", "value", "second_if", "false_input"))
    graph.add_edge(_edge("first_if", "value", "first_sink", "a"))
    graph.add_edge(_edge("second_if", "value", "second_sink", "a"))
    return graph


def _sibling_if_graph_with_count(count: int) -> Graph:
    graph = _sibling_if_graph()
    names = ("first", "second", "third", "fourth", "fifth")
    assert 2 <= count <= len(names)
    for index in range(3, count + 1):
        name = names[index - 1]
        graph.add_node(BooleanInvocation(id=f"{name}_condition", value=index % 2 == 1))
        graph.add_node(AddInvocation(id=f"{name}_true", b=index + 2))
        graph.add_node(AddInvocation(id=f"{name}_false", a=index * 10, b=1))
        graph.add_node(IfInvocation(id=f"{name}_if"))
        graph.add_node(AddInvocation(id=f"{name}_sink", b=index * 100))
        graph.add_edge(_edge("shared", "value", f"{name}_true", "a"))
        graph.add_edge(_edge(f"{name}_condition", "value", f"{name}_if", "condition"))
        graph.add_edge(_edge(f"{name}_true", "value", f"{name}_if", "true_input"))
        graph.add_edge(_edge(f"{name}_false", "value", f"{name}_if", "false_input"))
        graph.add_edge(_edge(f"{name}_if", "value", f"{name}_sink", "a"))
    return graph


def _three_nested_if_graph(*, middle_branch: str = "true_input", outer_branch: str = "true_input") -> Graph:
    assert middle_branch in {"true_input", "false_input"}
    assert outer_branch in {"true_input", "false_input"}
    middle_other_branch = "false_input" if middle_branch == "true_input" else "true_input"
    outer_other_branch = "false_input" if outer_branch == "true_input" else "true_input"
    middle_other_id = "middle_false" if middle_branch == "true_input" else "middle_true"
    outer_other_id = "outer_false" if outer_branch == "true_input" else "outer_true"
    graph = Graph()
    graph.add_node(BooleanInvocation(id="outer_condition", value=True))
    graph.add_node(BooleanInvocation(id="middle_condition", value=False))
    graph.add_node(BooleanInvocation(id="inner_condition", value=True))
    graph.add_node(AddInvocation(id="inner_true", a=2, b=2))
    graph.add_node(AddInvocation(id="inner_false", a=3, b=3))
    graph.add_node(AddInvocation(id=middle_other_id, a=10, b=0))
    graph.add_node(AddInvocation(id=outer_other_id, a=20, b=0))
    graph.add_node(IfInvocation(id="inner_if"))
    graph.add_node(IfInvocation(id="middle_if"))
    graph.add_node(IfInvocation(id="outer_if"))
    graph.add_node(AddInvocation(id="sink", b=1))
    graph.add_edge(_edge("inner_condition", "value", "inner_if", "condition"))
    graph.add_edge(_edge("inner_true", "value", "inner_if", "true_input"))
    graph.add_edge(_edge("inner_false", "value", "inner_if", "false_input"))
    graph.add_edge(_edge("middle_condition", "value", "middle_if", "condition"))
    graph.add_edge(_edge("inner_if", "value", "middle_if", middle_branch))
    graph.add_edge(_edge(middle_other_id, "value", "middle_if", middle_other_branch))
    graph.add_edge(_edge("outer_condition", "value", "outer_if", "condition"))
    graph.add_edge(_edge("middle_if", "value", "outer_if", outer_branch))
    graph.add_edge(_edge(outer_other_id, "value", "outer_if", outer_other_branch))
    graph.add_edge(_edge("outer_if", "value", "sink", "a"))
    return graph


def _four_nested_if_graph() -> Graph:
    graph = _three_nested_if_graph()
    graph.delete_node("sink")
    graph.add_node(BooleanInvocation(id="deepest_condition", value=True))
    graph.add_node(AddInvocation(id="deepest_false", a=30, b=0))
    graph.add_node(IfInvocation(id="deepest_if"))
    graph.add_node(AddInvocation(id="sink", b=1))
    graph.add_edge(_edge("deepest_condition", "value", "deepest_if", "condition"))
    graph.add_edge(_edge("outer_if", "value", "deepest_if", "true_input"))
    graph.add_edge(_edge("deepest_false", "value", "deepest_if", "false_input"))
    graph.add_edge(_edge("deepest_if", "value", "sink", "a"))
    return graph


def _five_nested_if_graph() -> Graph:
    graph = _four_nested_if_graph()
    graph.delete_node("sink")
    graph.add_node(BooleanInvocation(id="fifth_condition", value=True))
    graph.add_node(AddInvocation(id="fifth_false", a=40, b=0))
    graph.add_node(IfInvocation(id="fifth_if"))
    graph.add_node(AddInvocation(id="sink", b=1))
    graph.add_edge(_edge("fifth_condition", "value", "fifth_if", "condition"))
    graph.add_edge(_edge("deepest_if", "value", "fifth_if", "true_input"))
    graph.add_edge(_edge("fifth_false", "value", "fifth_if", "false_input"))
    graph.add_edge(_edge("fifth_if", "value", "sink", "a"))
    return graph


def _four_nested_if_fanout_graph() -> Graph:
    graph = _four_nested_if_graph()
    graph.add_node(AddInvocation(id="outer_side_consumer", b=1))
    graph.add_edge(_edge("outer_if", "value", "outer_side_consumer", "a"))
    return graph


def _four_nested_if_terminal_fanout_graph() -> Graph:
    graph = _four_nested_if_graph()
    graph.add_node(AddInvocation(id="deepest_side_consumer", b=1))
    graph.add_edge(_edge("deepest_if", "value", "deepest_side_consumer", "a"))
    return graph


def _three_nested_if_fanout_graph() -> Graph:
    graph = _three_nested_if_graph()
    graph.add_node(AddInvocation(id="middle_side_consumer", b=1))
    graph.add_edge(_edge("middle_if", "value", "middle_side_consumer", "a"))
    return graph


def _three_nested_if_extra_fanout_graph() -> Graph:
    graph = _three_nested_if_fanout_graph()
    graph.add_node(AddInvocation(id="middle_side_consumer_2", b=2))
    graph.add_edge(_edge("middle_if", "value", "middle_side_consumer_2", "a"))
    return graph


def _three_nested_if_non_leaf_fanout_graph() -> Graph:
    graph = _three_nested_if_fanout_graph()
    graph.add_node(AddInvocation(id="middle_side_tail", b=2))
    graph.add_edge(_edge("middle_side_consumer", "value", "middle_side_tail", "a"))
    return graph


def _three_nested_if_inner_fanout_graph() -> Graph:
    graph = _three_nested_if_fanout_graph()
    graph.add_node(AddInvocation(id="inner_side_consumer", b=2))
    graph.add_edge(_edge("inner_if", "value", "inner_side_consumer", "a"))
    return graph


def _indirectly_connected_if_graph() -> Graph:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="first_condition", value=True))
    graph.add_node(BooleanInvocation(id="second_condition", value=True))
    graph.add_node(AddInvocation(id="first_true", a=2, b=3))
    graph.add_node(AddInvocation(id="first_false", a=10, b=1))
    graph.add_node(IfInvocation(id="first_if"))
    graph.add_node(AddInvocation(id="bridge", b=4))
    graph.add_node(AddInvocation(id="second_false", a=20, b=1))
    graph.add_node(IfInvocation(id="second_if"))
    graph.add_node(AddInvocation(id="first_sink", b=100))
    graph.add_node(AddInvocation(id="second_sink", b=200))
    graph.add_edge(_edge("first_condition", "value", "first_if", "condition"))
    graph.add_edge(_edge("first_true", "value", "first_if", "true_input"))
    graph.add_edge(_edge("first_false", "value", "first_if", "false_input"))
    graph.add_edge(_edge("first_if", "value", "bridge", "a"))
    graph.add_edge(_edge("bridge", "value", "second_if", "true_input"))
    graph.add_edge(_edge("second_condition", "value", "second_if", "condition"))
    graph.add_edge(_edge("second_false", "value", "second_if", "false_input"))
    graph.add_edge(_edge("first_if", "value", "first_sink", "a"))
    graph.add_edge(_edge("second_if", "value", "second_sink", "a"))
    return graph


def test_leaf_import_does_not_import_graph() -> None:
    assert_module_imports_without_graph("invokeai.app.services.shared.graph_if_dependencies")


def test_graph_methods_delegate_to_leaf_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    class State:
        __pydantic_private__ = {"_runtime_state": type("Runtime", (), {"fresh_flat_if_activation": None})()}

    state = State()
    marker = object()
    monkeypatch.setattr(graph_if_dependencies, "_get_fresh_if_nodes", lambda actual_state: (actual_state, marker))
    monkeypatch.setattr(graph_if_dependencies, "_can_use_fresh_flat_if_activation", lambda actual_state: marker)
    monkeypatch.setattr(
        graph_if_dependencies,
        "_get_fresh_if_branch_sources",
        lambda actual_state, if_node_id, branch_field: {actual_state, if_node_id, branch_field},
    )
    monkeypatch.setattr(
        graph_if_dependencies,
        "_get_source_activation_dependencies",
        lambda actual_state, source_node_id, iteration_path=(): (actual_state, source_node_id, iteration_path),
    )

    assert GraphExecutionState._get_fresh_if_nodes(state) == (state, marker)
    assert GraphExecutionState._can_use_fresh_flat_if_activation(state) is marker
    assert GraphExecutionState._get_fresh_if_branch_sources(state, "if", "true_input") == {
        state,
        "if",
        "true_input",
    }
    assert GraphExecutionState._get_source_activation_dependencies(state, "source", (2,)) == (
        state,
        "source",
        (2,),
    )


def test_fresh_flat_if_eligibility_is_cached_per_state(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def can_use_fresh_flat_if_activation(_state: GraphExecutionState) -> bool:
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(graph_if_dependencies, "_can_use_fresh_flat_if_activation", can_use_fresh_flat_if_activation)
    state = GraphExecutionState(graph=_flat_if_graph())

    assert state._can_use_fresh_flat_if_activation()
    assert state._can_use_fresh_flat_if_activation()
    assert calls == 1

    copied_state = state.model_copy(update={"graph": Graph()})
    assert copied_state._can_use_fresh_flat_if_activation()
    assert calls == 2


def test_fresh_flat_if_eligibility_is_invalidated_when_graph_changes() -> None:
    state = GraphExecutionState(graph=_flat_if_graph())

    assert state._can_use_fresh_flat_if_activation()

    state.add_node(CallSavedWorkflowInvocation(id="call"))

    assert not state._can_use_fresh_flat_if_activation()


def test_replacing_graph_invalidates_source_graph_and_if_eligibility_caches() -> None:
    state = GraphExecutionState(graph=_flat_if_graph())
    assert state._can_use_fresh_flat_if_activation()

    copied_state = state.model_copy(update={"graph": _indirectly_connected_if_graph()})

    assert not copied_state._can_use_fresh_flat_if_activation()


def test_flat_and_nested_dependencies_preserve_values_and_order() -> None:
    flat_state = GraphExecutionState(graph=_flat_if_graph())
    assert flat_state._get_source_activation_dependencies("true_branch") == (
        ActivationDependency(owner_id="if", branch="true_input", frame=()),
    )
    assert flat_state._get_source_activation_dependencies("false_branch", (3,)) == (
        ActivationDependency(owner_id="if", branch="false_input", frame=(3,)),
    )

    nested_state = GraphExecutionState(graph=_nested_if_graph())
    assert nested_state._get_source_activation_dependencies("inner_true") == (
        ActivationDependency(owner_id="inner_if", branch="true_input", frame=()),
        ActivationDependency(owner_id="outer_if", branch="true_input", frame=()),
    )
    assert nested_state._get_source_activation_dependencies("inner_false", (4,)) == (
        ActivationDependency(owner_id="inner_if", branch="false_input", frame=(4,)),
        ActivationDependency(owner_id="outer_if", branch="true_input", frame=(4,)),
    )


def test_ineligible_shape_uses_controller_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _flat_if_graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call", workflow_id="saved-workflow"))
    state = GraphExecutionState(graph=graph)
    expected = (ActivationDependency(owner_id="fallback", branch="true_input", frame=(7,)),)
    calls: list[tuple[str, tuple[int, ...]]] = []

    def get_source_dependencies(
        _controller: Any, source_node_id: str, iteration_path: tuple[int, ...]
    ) -> tuple[ActivationDependency, ...]:
        calls.append((source_node_id, iteration_path))
        return expected

    monkeypatch.setattr(graph_module._IfActivationController, "get_source_dependencies", get_source_dependencies)
    assert not state._can_use_fresh_flat_if_activation()
    assert state._get_source_activation_dependencies("true_branch", (7,)) == expected
    assert calls == [("true_branch", (7,))]


def test_branch_cache_uses_graph_facade_nx_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    state = GraphExecutionState(graph=_flat_if_graph())
    source_graph = state._get_source_graph_flat()
    calls: list[tuple[str, str]] = []

    class _NetworkX:
        @staticmethod
        def topological_sort(graph: Any) -> list[str]:
            calls.append(("topological_sort", ""))
            return list(graph.nodes)

        @staticmethod
        def ancestors(_graph: Any, node_id: str) -> set[str]:
            calls.append(("ancestors", node_id))
            return set()

        @staticmethod
        def is_directed_acyclic_graph(_graph: Any) -> bool:
            calls.append(("is_directed_acyclic_graph", ""))
            return True

    monkeypatch.setattr(graph_module, "nx", _NetworkX())
    assert graph_if_dependencies._get_fresh_if_nodes(state)[0].id == "if"
    assert graph_if_dependencies._get_fresh_if_branch_sources(state, "if", "true_input") == {"true_branch"}
    assert graph_if_dependencies._get_fresh_if_branch_sources(state, "if", "true_input") == {"true_branch"}
    assert source_graph is state._source_graph_flat
    assert calls == [("topological_sort", ""), ("ancestors", "true_branch")]


def test_branch_dependency_cache_is_frame_local_and_graph_edits_invalidate() -> None:
    state = GraphExecutionState(graph=_flat_if_graph())
    first = state._get_source_activation_dependencies("true_branch", (1,))
    second = state._get_source_activation_dependencies("true_branch", (2,))
    assert first[0].frame == (1,)
    assert second[0].frame == (2,)
    assert set(state._if_activation_dependencies_by_source) == {
        ("true_branch", (1,)),
        ("true_branch", (2,)),
    }

    state.add_node(AddInvocation(id="true_consumer", b=1))
    state.add_edge(_edge("true_branch", "value", "true_consumer", "a"))
    assert state._get_source_activation_dependencies("true_branch") == ()


def test_sibling_if_dependencies_are_owner_local_and_frame_local() -> None:
    state = GraphExecutionState(graph=_sibling_if_graph(first_condition=True, second_condition=False))

    assert state._can_use_fresh_flat_if_activation()
    assert state._get_source_activation_dependencies("first_true", (1,)) == (
        ActivationDependency(owner_id="first_if", branch="true_input", frame=(1,)),
    )
    assert state._get_source_activation_dependencies("first_false", (2,)) == (
        ActivationDependency(owner_id="first_if", branch="false_input", frame=(2,)),
    )
    assert state._get_source_activation_dependencies("second_true", (3,)) == (
        ActivationDependency(owner_id="second_if", branch="true_input", frame=(3,)),
    )
    assert state._get_source_activation_dependencies("second_false", (4,)) == (
        ActivationDependency(owner_id="second_if", branch="false_input", frame=(4,)),
    )
    assert state._get_source_activation_dependencies("shared") == ()
    assert state._get_source_activation_dependencies("first_true", (5,)) != state._get_source_activation_dependencies(
        "first_true", (6,)
    )


def test_three_independent_sibling_ifs_admit_owner_local_frame_local_dependencies() -> None:
    state = GraphExecutionState(graph=_sibling_if_graph_with_count(3))

    assert state._can_use_fresh_flat_if_activation()
    for index, name in enumerate(("first", "second", "third"), start=1):
        assert state._get_source_activation_dependencies(f"{name}_true", (index,)) == (
            ActivationDependency(owner_id=f"{name}_if", branch="true_input", frame=(index,)),
        )
        assert state._get_source_activation_dependencies(f"{name}_false", (index + 3,)) == (
            ActivationDependency(owner_id=f"{name}_if", branch="false_input", frame=(index + 3,)),
        )

    assert state._get_source_activation_dependencies("shared") == ()


def test_four_independent_sibling_ifs_admit_owner_local_frame_local_dependencies() -> None:
    state = GraphExecutionState(graph=_sibling_if_graph_with_count(4))

    assert state._can_use_fresh_flat_if_activation()
    for index, name in enumerate(("first", "second", "third", "fourth"), start=1):
        assert state._get_source_activation_dependencies(f"{name}_true", (index,)) == (
            ActivationDependency(owner_id=f"{name}_if", branch="true_input", frame=(index,)),
        )
        assert state._get_source_activation_dependencies(f"{name}_false", (index + 4,)) == (
            ActivationDependency(owner_id=f"{name}_if", branch="false_input", frame=(index + 4,)),
        )

    assert state._get_source_activation_dependencies("shared") == ()


@pytest.mark.parametrize(
    ("graph_factory", "source_node_id"),
    [
        pytest.param(lambda: _sibling_if_graph_with_count(5), "first_true", id="five-sibling-ifs"),
        pytest.param(_indirectly_connected_if_graph, "first_true", id="indirect-chain"),
        pytest.param(_five_nested_if_graph, "inner_true", id="five-nested-ifs"),
        pytest.param(_four_nested_if_fanout_graph, "inner_true", id="four-nested-if-fanout"),
        pytest.param(_four_nested_if_terminal_fanout_graph, "inner_true", id="four-nested-if-terminal-fanout"),
        pytest.param(_three_nested_if_extra_fanout_graph, "inner_true", id="three-nested-if-extra-fanout"),
        pytest.param(_three_nested_if_non_leaf_fanout_graph, "inner_true", id="three-nested-if-non-leaf-fanout"),
        pytest.param(_three_nested_if_inner_fanout_graph, "inner_true", id="three-nested-if-inner-fanout"),
    ],
)
def test_unsupported_if_shapes_use_controller_fallback(
    monkeypatch: pytest.MonkeyPatch,
    graph_factory: Any,
    source_node_id: str,
) -> None:
    expected = (ActivationDependency(owner_id="fallback", branch="true_input", frame=(9,)),)
    calls: list[tuple[str, tuple[int, ...]]] = []

    def get_source_dependencies(
        _controller: Any, actual_source_node_id: str, iteration_path: tuple[int, ...]
    ) -> tuple[ActivationDependency, ...]:
        calls.append((actual_source_node_id, iteration_path))
        return expected

    monkeypatch.setattr(graph_module._IfActivationController, "get_source_dependencies", get_source_dependencies)
    state = GraphExecutionState(graph=graph_factory())

    assert not state._can_use_fresh_flat_if_activation()
    assert state._get_source_activation_dependencies(source_node_id, (9,)) == expected
    assert calls == [(source_node_id, (9,))]


@pytest.mark.parametrize("outer_condition", [False, True])
@pytest.mark.parametrize("middle_condition", [False, True])
@pytest.mark.parametrize("inner_condition", [False, True])
def test_three_nested_ifs_admit_owner_and_frame_dependencies(
    outer_condition: bool, middle_condition: bool, inner_condition: bool
) -> None:
    graph = _three_nested_if_graph()
    graph.get_node("outer_condition").value = outer_condition
    graph.get_node("middle_condition").value = middle_condition
    graph.get_node("inner_condition").value = inner_condition
    state = GraphExecutionState(graph=graph)

    assert state._can_use_fresh_flat_if_activation()
    assert state._get_source_activation_dependencies("inner_true", (1,)) == (
        ActivationDependency(owner_id="inner_if", branch="true_input", frame=(1,)),
        ActivationDependency(owner_id="middle_if", branch="true_input", frame=(1,)),
        ActivationDependency(owner_id="outer_if", branch="true_input", frame=(1,)),
    )
    assert state._get_source_activation_dependencies("inner_false", (2,)) == (
        ActivationDependency(owner_id="inner_if", branch="false_input", frame=(2,)),
        ActivationDependency(owner_id="middle_if", branch="true_input", frame=(2,)),
        ActivationDependency(owner_id="outer_if", branch="true_input", frame=(2,)),
    )
    assert state._get_source_activation_dependencies("middle_false", (3,)) == (
        ActivationDependency(owner_id="middle_if", branch="false_input", frame=(3,)),
        ActivationDependency(owner_id="outer_if", branch="true_input", frame=(3,)),
    )
    assert state._get_source_activation_dependencies("outer_false", (4,)) == (
        ActivationDependency(owner_id="outer_if", branch="false_input", frame=(4,)),
    )


def test_three_nested_ifs_with_one_middle_leaf_fanout_admit_exact_dependencies() -> None:
    state = GraphExecutionState(graph=_three_nested_if_fanout_graph())

    assert state._can_use_fresh_flat_if_activation()
    assert state._get_source_activation_dependencies("inner_true", (1,)) == (
        ActivationDependency(owner_id="inner_if", branch="true_input", frame=(1,)),
        ActivationDependency(owner_id="middle_if", branch="true_input", frame=(1,)),
        ActivationDependency(owner_id="outer_if", branch="true_input", frame=(1,)),
    )
    assert state._get_source_activation_dependencies("middle_false", (2,)) == (
        ActivationDependency(owner_id="middle_if", branch="false_input", frame=(2,)),
        ActivationDependency(owner_id="outer_if", branch="true_input", frame=(2,)),
    )
    assert state._get_source_activation_dependencies("outer_false", (3,)) == (
        ActivationDependency(owner_id="outer_if", branch="false_input", frame=(3,)),
    )
    assert state._get_source_activation_dependencies("middle_side_consumer", (4,)) == (
        ActivationDependency(owner_id="outer_if", branch="true_input", frame=(4,)),
    )


def test_four_nested_ifs_admit_owner_and_frame_dependencies() -> None:
    state = GraphExecutionState(graph=_four_nested_if_graph())

    assert state._can_use_fresh_flat_if_activation()
    assert state._get_source_activation_dependencies("inner_true", (1,)) == (
        ActivationDependency(owner_id="inner_if", branch="true_input", frame=(1,)),
        ActivationDependency(owner_id="middle_if", branch="true_input", frame=(1,)),
        ActivationDependency(owner_id="outer_if", branch="true_input", frame=(1,)),
        ActivationDependency(owner_id="deepest_if", branch="true_input", frame=(1,)),
    )
    assert state._get_source_activation_dependencies("outer_false", (2,)) == (
        ActivationDependency(owner_id="outer_if", branch="false_input", frame=(2,)),
        ActivationDependency(owner_id="deepest_if", branch="true_input", frame=(2,)),
    )
    assert state._get_source_activation_dependencies("deepest_false", (3,)) == (
        ActivationDependency(owner_id="deepest_if", branch="false_input", frame=(3,)),
    )


@pytest.mark.parametrize("middle_branch", ["true_input", "false_input"])
@pytest.mark.parametrize("outer_branch", ["true_input", "false_input"])
def test_three_nested_ifs_with_one_middle_leaf_fanout_preserve_parent_branch_ports(
    middle_branch: str, outer_branch: str
) -> None:
    state = GraphExecutionState(
        graph=_three_nested_if_graph(
            middle_branch=middle_branch,
            outer_branch=outer_branch,
        )
    )
    # This fixture does not include fan-out. Add the exact permitted side leaf.
    state.graph.add_node(AddInvocation(id="middle_side_consumer", b=1))
    state.graph.add_edge(_edge("middle_if", "value", "middle_side_consumer", "a"))

    assert state._can_use_fresh_flat_if_activation()
    assert state._get_source_activation_dependencies("middle_side_consumer", (4,)) == (
        ActivationDependency(owner_id="outer_if", branch=outer_branch, frame=(4,)),
    )


def test_indirectly_connected_ifs_retain_controller_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    state = GraphExecutionState(graph=_indirectly_connected_if_graph())
    calls: list[tuple[str, str]] = []
    original_branch_sources = graph_module._IfActivationController._branch_sources

    def record_branch_analysis(
        controller: graph_module._IfActivationController,
        if_node_id: str,
        branch_field: str,
        source_graph: Any,
    ) -> set[str]:
        calls.append((if_node_id, branch_field))
        return original_branch_sources(controller, if_node_id, branch_field, source_graph)

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", record_branch_analysis)

    assert not state._can_use_fresh_flat_if_activation()
    assert state._get_source_activation_dependencies("first_true", (7,)) == (
        ActivationDependency(owner_id="first_if", branch="true_input", frame=(7,)),
    )
    assert ("first_if", "true_input") in calls
