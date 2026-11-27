import random

import networkx as nx
import pytest

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph, GraphExecutionState
from invokeai.app.services.shared.graph_iterator_ancestry import _IteratorAncestry
from invokeai.app.services.shared.graph_validation import CollectInvocation, IterateInvocation


@pytest.mark.parametrize("seed", range(5))
def test_iterator_ancestry_matches_filtered_networkx_on_generated_dags(seed: int) -> None:
    randomizer = random.Random(seed)
    classes = (AddInvocation, IterateInvocation, ForInvocation, CollectInvocation)
    nodes = {str(i): randomizer.choice(classes)(id=str(i)) for i in range(40)}
    edges = []
    for i in range(40):
        for j in range(i + 1, 40):
            if randomizer.random() < 0.12:
                source = nodes[str(i)]
                field = randomizer.choice(("item", "final_state")) if isinstance(source, ForInvocation) else "value"
                edges.append(
                    Edge(
                        source=EdgeConnection(node_id=str(i), field=field),
                        destination=EdgeConnection(node_id=str(j), field="a"),
                    )
                )
    # These are topology fixtures, not executable invocation workflows.
    state = GraphExecutionState(graph=Graph())
    state.graph = Graph(nodes=nodes, edges=edges)
    source_graph = state._get_source_graph_flat()
    for graph in (source_graph, nx.subgraph_view(source_graph, filter_edge=lambda u, v: (int(u) + int(v)) % 3 != 0)):
        filtered = state._iterator_graph(graph)
        ancestry = _IteratorAncestry(state, graph)
        for node_id in reversed(list(graph)):
            expected = any(
                isinstance(nodes[ancestor], (ForInvocation, IterateInvocation))
                for ancestor in nx.ancestors(filtered, node_id)
            )
            assert ancestry.has_iterator_ancestor(node_id) == expected
            assert ancestry.has_iterator_ancestor(node_id) == expected


def test_iterator_self_collectors_and_final_for_outputs_are_not_ancestors() -> None:
    nodes = {
        "iterate": IterateInvocation(id="iterate"),
        "for": ForInvocation(id="for"),
        "collect": CollectInvocation(id="collect"),
        "after_collect": AddInvocation(id="after_collect"),
        "final": AddInvocation(id="final"),
        "body": AddInvocation(id="body"),
    }
    edges = [
        Edge(
            source=EdgeConnection(node_id=source, field=field),
            destination=EdgeConnection(node_id=destination, field="a"),
        )
        for source, field, destination in (
            ("iterate", "item", "collect"),
            ("collect", "collection", "after_collect"),
            ("for", "final_state", "final"),
            ("for", "item", "body"),
        )
    ]
    state = GraphExecutionState(graph=Graph())
    state.graph = Graph(nodes=nodes, edges=edges)
    ancestry = _IteratorAncestry(state)
    assert {node_id: ancestry.has_iterator_ancestor(node_id) for node_id in nodes} == {
        "iterate": False,
        "for": False,
        "collect": False,
        "after_collect": False,
        "final": False,
        "body": True,
    }


def test_new_context_observes_supplied_graph_mutations() -> None:
    state = GraphExecutionState(graph=Graph())
    state.graph = Graph(nodes={"iterator": IterateInvocation(id="iterator"), "sink": AddInvocation(id="sink")})
    graph = nx.DiGraph()
    graph.add_nodes_from(state.graph.nodes)
    assert not _IteratorAncestry(state, graph).has_iterator_ancestor("sink")
    graph.add_edge("iterator", "sink")
    assert _IteratorAncestry(state, graph).has_iterator_ancestor("sink")
    graph.remove_edge("iterator", "sink")
    assert not _IteratorAncestry(state, graph).has_iterator_ancestor("sink")


def test_supplied_graph_view_keeps_excluded_edges_out_of_collect_ancestry() -> None:
    nodes = {
        "iterator": IterateInvocation(id="iterator"),
        "collect": CollectInvocation(id="collect"),
        "if": IfInvocation(id="if", condition=True),
    }
    edges = [
        Edge(
            source=EdgeConnection(node_id="iterator", field="item"),
            destination=EdgeConnection(node_id="collect", field="collection"),
        ),
        Edge(
            source=EdgeConnection(node_id="collect", field="collection"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        ),
    ]
    state = GraphExecutionState(graph=Graph())
    state.graph = Graph(nodes=nodes, edges=edges)
    source_graph = state._get_source_graph_flat()
    constrained_graph = nx.subgraph_view(
        source_graph, filter_edge=lambda source, destination: (source, destination) != ("iterator", "collect")
    )

    assert not _IteratorAncestry(state, constrained_graph).has_iterator_ancestor("if")
