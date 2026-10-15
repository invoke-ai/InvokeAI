import importlib
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("planner_first", [True, False])
def test_planner_import_boundary(planner_first: bool) -> None:
    script = textwrap.dedent(
        """
        import importlib
        import sys
        from importlib.abc import MetaPathFinder

        class BlockGraph(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "invokeai.app.services.shared.graph":
                    raise AssertionError("planner imported graph eagerly")

        blocker = BlockGraph()
        if PLANNER_FIRST:
            sys.meta_path.insert(0, blocker)
            planner = importlib.import_module("invokeai.app.services.shared.graph_iterate_planner")
            assert "invokeai.app.services.shared.graph" not in sys.modules
            sys.meta_path.remove(blocker)
        from invokeai.app.services.shared import graph
        planner = importlib.import_module("invokeai.app.services.shared.graph_iterate_planner")
        state = graph.GraphExecutionState(graph=graph.Graph())
        assert planner._get_direct_iterate_collect_nodes(state) is None
        assert planner._can_use_direct_iterate_collect_planner(state) is False
        """
    ).replace("PLANNER_FIRST", repr(planner_first))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr


def test_planner_dispatch_preserves_state_override_and_outer_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    from invokeai.app.services.shared.graph import Graph, GraphExecutionState
    from invokeai.app.services.shared.graph_runtime_records import _ApplyTransaction

    planner = importlib.import_module("invokeai.app.services.shared.graph_iterate_planner")
    state = GraphExecutionState(graph=Graph())
    transaction = _ApplyTransaction()
    object.__setattr__(state, "_apply_transaction", transaction)
    calls = []
    monkeypatch.setattr(
        state, "_prepare_direct_iterate_collect_unchecked", lambda: calls.append(state._apply_transaction)
    )

    planner._prepare_direct_iterate_collect(state)

    assert calls == [transaction]
    assert state._apply_transaction is transaction


def test_execution_copy_preserves_graph_uuid_seam_and_iteration_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from invokeai.app.services.shared import graph

    planner = importlib.import_module("invokeai.app.services.shared.graph_iterate_planner")
    state = graph.GraphExecutionState(graph=graph.Graph())
    source = graph.IterateInvocation(id="iterator", collection=[1, 2], use_cache=True)
    state.graph.add_node(source)
    monkeypatch.setattr(graph, "uuid_string", lambda: "prepared-iterator")

    copied = planner._create_direct_execution_node_copy(state, "iterator", 1, (3, 1))

    assert copied.id == "prepared-iterator"
    assert copied.index == 1
    assert copied.use_cache is False
    assert source.id == "iterator"
    assert source.use_cache is True
    assert state.execution_graph.get_node(copied.id) is copied
    assert state.prepared_source_mapping[copied.id] == "iterator"
    assert state._prepared_registry().get_iteration_path(copied.id) == (3, 1)


def test_planner_accepts_equivalent_generic_scheduler_instance() -> None:
    from invokeai.app.invocations.math import AddInvocation
    from invokeai.app.services.shared import graph

    planner = importlib.import_module("invokeai.app.services.shared.graph_iterate_planner")
    state = graph.GraphExecutionState(graph=graph.Graph(nodes={"source": AddInvocation(id="source", a=1, b=2)}))
    facade_scheduler = graph._GenericGraphSchedulerAdapter(state)
    planner_scheduler = graph._GenericGraphSchedulerAdapter(state)
    object.__setattr__(state, "_generic_graph_scheduler", facade_scheduler)
    object.__setattr__(state, "_execution_scheduler", planner_scheduler)

    copied = planner._create_direct_execution_node_copy(state, "source")
    planner._initialize_direct_execution_node(state, copied.id, ())

    assert state._is_generic_graph_scheduler(planner_scheduler)
