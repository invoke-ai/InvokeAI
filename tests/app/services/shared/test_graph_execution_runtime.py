import subprocess
import sys
import textwrap

from invokeai.app.services.shared import graph as graph_facade
from invokeai.app.services.shared import graph_execution_runtime


def test_runtime_module_imports_without_graph_cycle() -> None:
    script = textwrap.dedent(
        """
        import sys

        from invokeai.app.services.shared import graph_execution_runtime

        assert graph_execution_runtime._ExecutionRuntime
        assert "invokeai.app.services.shared.graph" not in sys.modules
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_graph_facade_reexports_runtime_aliases() -> None:
    assert graph_facade._ExecutionRuntime is graph_execution_runtime._ExecutionRuntime
    assert graph_facade._DirectIterateCollectFanIn is graph_execution_runtime._DirectIterateCollectFanIn
    assert graph_facade._BodyIterateCollectFanIn is graph_execution_runtime._BodyIterateCollectFanIn


def test_graph_execution_state_runtime_seam_uses_extracted_runtime() -> None:
    state = graph_facade.GraphExecutionState(graph=graph_facade.Graph())

    runtime = state._runtime()

    assert type(runtime) is graph_execution_runtime._ExecutionRuntime
    assert state._runtime() is runtime


def test_graph_execution_state_delegates_runtime_helpers(monkeypatch) -> None:
    state = graph_facade.GraphExecutionState(graph=graph_facade.Graph())
    calls: list[tuple[str, object]] = []

    class StubRuntime:
        def get_iteration_path(self, exec_node_id: str) -> tuple[int, ...]:
            calls.append(("get_iteration_path", exec_node_id))
            return (3,)

        def prepare_inputs(self, node: object) -> None:
            calls.append(("prepare_inputs", node))

    runtime = StubRuntime()
    monkeypatch.setattr(state, "_runtime", lambda: runtime)
    node = object()

    assert state._get_iteration_path("exec-node") == (3,)
    state._prepare_inputs(node)

    assert calls == [("get_iteration_path", "exec-node"), ("prepare_inputs", node)]
