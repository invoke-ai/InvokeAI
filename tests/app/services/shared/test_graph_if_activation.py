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
    def __init__(self) -> None:
        self.nodes = {
            "if": IfInvocation(id="if"),
            "true": object(),
            "false": object(),
            "outside": object(),
        }
        self._input_edges = {
            ("if", "true_input"): [_edge("true", "if", "true_input")],
            ("if", "false_input"): [_edge("false", "if", "false_input")],
        }
        self._output_edges = {
            "true": self._input_edges[("if", "true_input")],
            "false": self._input_edges[("if", "false_input")],
            "outside": [],
        }

    def _get_input_edges(self, node_id: str, field: str) -> list[Any]:
        return self._input_edges.get((node_id, field), [])

    def _get_output_edges(self, node_id: str) -> list[Any]:
        return self._output_edges.get(node_id, [])


class _Registry:
    def get_source_node_id(self, exec_node_id: str) -> str:
        return exec_node_id

    def get_prepared_ids(self, source_node_id: str) -> list[str]:
        return [source_node_id] if source_node_id == "if" else []


class _State:
    def __init__(self) -> None:
        self.graph = _Graph()
        self.admitted: set[ActivationDependency] = set()
        self.rejected: set[ActivationDependency] = set()

    def _prepared_registry(self) -> _Registry:
        return _Registry()

    def _get_iteration_path(self, _: str) -> tuple[int, ...]:
        return ()

    def _get_source_graph_flat(self) -> object:
        return object()

    def _is_activation_dependency_satisfied(self, dependency: ActivationDependency) -> bool:
        return dependency in self.admitted

    def _is_activation_dependency_rejected(self, dependency: ActivationDependency) -> bool:
        return dependency in self.rejected


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


def test_controller_dependencies_admission_rejection_and_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert not controller.is_source_admitted("true")
    assert not controller.is_source_rejected("true")
    assert not controller.is_source_inactive("true")

    state.admitted.add(true_dependency)
    assert controller.is_source_admitted("true")

    state.rejected.add(true_dependency)
    assert controller.is_source_rejected("true")
    state.rejected.add(ActivationDependency(owner_id="if", branch="true_input", frame=(1,)))
    assert controller.is_source_inactive("true", (1,))
