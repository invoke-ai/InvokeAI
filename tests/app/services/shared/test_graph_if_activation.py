from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.services.shared import graph as graph_module
from invokeai.app.services.shared.execution_engine import ActivationDependency
from invokeai.app.services.shared.graph_if_activation import _IfActivationController
from tests.app.services.shared.import_test_utils import assert_module_imports_without_graph


def _edge(source: str, destination: str, field: str) -> Any:
    return SimpleNamespace(
        source=SimpleNamespace(node_id=source),
        destination=SimpleNamespace(node_id=destination, field=field),
    )


class _Graph:
    def __init__(self, *, condition_input: bool = True, condition: bool = False) -> None:
        self.nodes = {
            "if": IfInvocation(id="if", condition=condition),
            "true": object(),
            "false": object(),
            "outside": object(),
        }
        self._input_edges = {
            ("if", "true_input"): [_edge("true", "if", "true_input")],
            ("if", "false_input"): [_edge("false", "if", "false_input")],
        }
        if condition_input:
            self._input_edges[("if", "condition")] = [_edge("condition", "if", "condition")]
        self._output_edges = {
            "true": self._input_edges[("if", "true_input")],
            "false": self._input_edges[("if", "false_input")],
            "outside": [],
        }

    def _get_input_edges(self, node_id: str, field: str) -> list[Any]:
        return self._input_edges.get((node_id, field), [])

    def _get_output_edges(self, node_id: str) -> list[Any]:
        return self._output_edges.get(node_id, [])


class _State:
    def __init__(self) -> None:
        self.graph = _Graph()

    def _get_source_graph_flat(self) -> object:
        return object()


@pytest.mark.platform_independent
def test_standalone_import_does_not_import_graph() -> None:
    assert_module_imports_without_graph("invokeai.app.services.shared.graph_if_activation")


def test_graph_reexports_controller_class() -> None:
    assert graph_module._IfActivationController is _IfActivationController


def test_graph_nx_patch_seam_and_branch_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _NetworkX:
        def ancestors(self, _graph: object, node_id: str) -> set[str]:
            calls.append(node_id)
            return set()

    monkeypatch.setattr(graph_module, "nx", _NetworkX())
    controller = _IfActivationController(_State())

    assert controller._branch_sources("if", "true_input", object()) == {"true"}
    assert controller._branch_sources("if", "true_input", object()) == {"true"}
    assert calls == ["true"]


def test_controller_gets_source_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NetworkX:
        def ancestors(self, _graph: object, _node_id: str) -> set[str]:
            return set()

    monkeypatch.setattr(graph_module, "nx", _NetworkX())
    state = _State()
    controller = _IfActivationController(state)
    true_dependency = ActivationDependency(owner_id="if", branch="true_input", frame=())
    false_dependency = ActivationDependency(owner_id="if", branch="false_input", frame=())

    assert controller.get_source_dependencies("true") == (true_dependency,)
    assert controller.get_source_dependencies("false") == (false_dependency,)
    assert controller.get_source_dependencies("outside") == ()


@pytest.mark.parametrize(
    ("condition", "selected_source", "unselected_source", "unselected_field"),
    [
        (True, "true", "false", "false_input"),
        (False, "false", "true", "true_input"),
    ],
)
def test_controller_selects_source_dependencies_for_source_level_branch(
    monkeypatch: pytest.MonkeyPatch,
    condition: bool,
    selected_source: str,
    unselected_source: str,
    unselected_field: str,
) -> None:
    class _NetworkX:
        def ancestors(self, _graph: object, _node_id: str) -> set[str]:
            return set()

    monkeypatch.setattr(graph_module, "nx", _NetworkX())
    state = _State()
    state.graph = _Graph(condition_input=False, condition=condition)
    controller = _IfActivationController(state)

    assert controller.get_source_dependencies(selected_source) == ()
    assert controller.get_source_dependencies(unselected_source) == (
        ActivationDependency(owner_id="if", branch=unselected_field, frame=()),
    )
