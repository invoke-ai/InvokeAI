import subprocess
import sys
import textwrap

import pytest

from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.primitives import IntegerOutput
from invokeai.app.services.shared import graph as graph_facade


@pytest.mark.platform_independent
def test_materializer_import_is_leaf_and_keeps_networkx_lazy() -> None:
    script = textwrap.dedent(
        """
        import sys
        from invokeai.app.services.shared import graph_materializer

        assert graph_materializer._ExecutionMaterializer
        assert "invokeai.app.services.shared.graph" not in sys.modules
        from invokeai.app.services.shared import graph
        assert graph._ExecutionMaterializer is graph_materializer._ExecutionMaterializer
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_materializer_facade_alias_and_class_patch_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    from invokeai.app.services.shared import graph_materializer

    assert graph_facade._ExecutionMaterializer is graph_materializer._ExecutionMaterializer
    state = graph_facade.GraphExecutionState(graph=graph_facade.Graph())
    assert type(state._materializer()) is graph_materializer._ExecutionMaterializer
    assert state._materializer() is state._materializer()
    calls = []
    monkeypatch.setattr(graph_facade._ExecutionMaterializer, "prepare", lambda self: calls.append(self))
    state._materializer().prepare()
    assert calls == [state._materializer()]


def test_materializer_preserves_execution_edges_and_source_nodes() -> None:
    from invokeai.app.services.shared import graph_materializer

    source = AddInvocation(id="source", a=2, b=3)
    sink = AddInvocation(id="sink", b=7)
    graph = graph_facade.Graph(nodes={source.id: source, sink.id: sink})
    graph.add_edge(
        graph_facade.Edge(
            source=graph_facade.EdgeConnection(node_id="source", field="value"),
            destination=graph_facade.EdgeConnection(node_id="sink", field="a"),
        )
    )
    state = graph_facade.GraphExecutionState(graph=graph)
    assert isinstance(state._materializer(), graph_materializer._ExecutionMaterializer)
    prepared_source = state.next()
    assert prepared_source is not None
    assert prepared_source.id != source.id
    state.complete(prepared_source.id, IntegerOutput(value=5))
    prepared_sink = state.next()
    assert isinstance(prepared_sink, AddInvocation)
    assert (prepared_sink.a, prepared_sink.b) == (5, 7)
    assert state.execution_graph._get_input_edges(prepared_sink.id) == [
        graph_facade.Edge(
            source=graph_facade.EdgeConnection(node_id=prepared_source.id, field="value"),
            destination=graph_facade.EdgeConnection(node_id=prepared_sink.id, field="a"),
        )
    ]
    assert sink.a == 0
    state.complete(prepared_sink.id, IntegerOutput(value=12))
    assert state.is_complete()


def test_prepare_admits_only_selected_frame_after_unresolved_conditions() -> None:
    from invokeai.app.invocations.logic import IfInvocation
    from invokeai.app.services.shared.graph_validation import IterateInvocation, nx

    nodes = {
        "iterator": IterateInvocation(id="iterator"),
        "condition-a": AddInvocation(id="condition-a"),
        "condition-b": AddInvocation(id="condition-b"),
        "inner-a": IfInvocation(id="inner-a"),
        "inner-b": IfInvocation(id="inner-b"),
        "blocked-a": IfInvocation(id="blocked-a", condition=True),
        "blocked-b": IfInvocation(id="blocked-b", condition=True),
        "selected": IfInvocation(id="selected", condition=True),
    }
    edges = [
        graph_facade.Edge(
            source=graph_facade.EdgeConnection(node_id=source, field=field),
            destination=graph_facade.EdgeConnection(node_id=destination, field=input_field),
        )
        for source, field, destination, input_field in (
            ("iterator", "item", "inner-a", "true_input"),
            ("condition-a", "value", "inner-a", "condition"),
            ("condition-b", "value", "inner-b", "condition"),
            ("inner-a", "value", "blocked-a", "true_input"),
            ("inner-a", "value", "blocked-b", "true_input"),
            ("inner-b", "value", "selected", "true_input"),
        )
    ]
    state = graph_facade.GraphExecutionState(graph=graph_facade.Graph())
    state.graph = graph_facade.Graph(nodes=nodes, edges=edges)
    state.executed.add("iterator")
    graph = state._get_source_graph_flat()
    # Keep condition sources outside this preparation view; their If nodes stay unresolved.
    preparation_graph = nx.subgraph_view(graph, filter_node=lambda node_id: not node_id.startswith("condition-"))
    prepared_id = state._materializer().prepare(preparation_graph)
    assert prepared_id in state.source_prepared_mapping["selected"]
    assert len(state.source_prepared_mapping["selected"]) == 1
    assert "blocked-a" not in state.source_prepared_mapping
    assert "blocked-b" not in state.source_prepared_mapping
    assert "inner-a" not in state.source_prepared_mapping
    assert "inner-b" not in state.source_prepared_mapping
