import pickle
import re

import pytest

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.invocations.collections import CollectionConcatInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.graph_loop_topology import (
    LoopTopology,
    NestedForBody,
    NestedIterateBody,
    NestedIterateChain,
    NestedIterateSequence,
)
from invokeai.app.services.shared.graph_validation import CollectInvocation, InvalidEdgeError, IterateInvocation
from tests.test_nodes import AnyTypeTestInvocation, PolymorphicStringTestInvocation, create_edge, create_loop_linkage

_Connection = tuple[str, str, str, str]


def _graph(
    nodes: list[BaseInvocation], connections: list[_Connection], linkages: tuple[tuple[str, str], ...] = ()
) -> Graph:
    graph = Graph()
    for node in nodes:
        graph.add_node(node)
    for connection in connections:
        graph.add_edge(create_edge(*connection))
    for for_id, return_id in linkages:
        graph.add_edge(create_loop_linkage(for_id, return_id))
    return graph


def _classify(graph: Graph) -> LoopTopology:
    return LoopTopology.classify(graph, graph.nx_graph_flat())


def _for_iterate_collect_graph(
    *, extra_collect_item: bool = False, extra_connections: tuple[_Connection, ...] = ()
) -> Graph:
    nodes: list[BaseInvocation] = [
        ForInvocation(id="for", collection=[["a"]]),
        PolymorphicStringTestInvocation(id="prep"),
        IterateInvocation(id="iterate"),
        AnyTypeTestInvocation(id="body"),
        CollectInvocation(id="collect"),
        ForReturnInvocation(id="return"),
    ]
    connections = [
        ("for", "item", "prep", "value"),
        ("prep", "collection", "iterate", "collection"),
        ("iterate", "item", "body", "value"),
        ("body", "value", "collect", "item"),
        ("collect", "collection", "return", "output"),
    ]
    if extra_collect_item:
        nodes.append(AnyTypeTestInvocation(id="second_body"))
        connections += [("iterate", "item", "second_body", "value"), ("second_body", "value", "collect", "item")]
    return _graph(nodes, [*connections, *extra_connections], (("for", "return"),))


def _assert_rejected(graph: Graph, message: str) -> None:
    with pytest.raises(InvalidEdgeError, match=re.escape(message)):
        graph.validate_self()


def _iterate_sequence_graph(*, body_input: str = "inner") -> Graph:
    nodes: list[BaseInvocation] = [
        CollectionConcatInvocation(id="source", first=[["a"]]),
        IterateInvocation(id="outer"),
        PolymorphicStringTestInvocation(id="prep"),
        IterateInvocation(id="inner"),
        AnyTypeTestInvocation(id="body"),
    ]
    connections = [
        ("source", "collection", "outer", "collection"),
        ("outer", "item", "prep", "value"),
        ("prep", "collection", "inner", "collection"),
        (body_input, "item", "body", "value"),
    ]
    return _graph(nodes, connections)


def test_flat_for_has_a_body_path_but_no_nested_shape() -> None:
    graph = _graph(
        [ForInvocation(id="for", collection=[1]), AnyTypeTestInvocation(id="body"), ForReturnInvocation(id="return")],
        [("for", "item", "body", "value"), ("body", "value", "return", "output")],
        (("for", "return"),),
    )

    topology = _classify(graph)

    assert topology.for_body_paths == {"for": (frozenset({"body", "return"}), "return")}
    assert topology.for_id_by_return_id == {"return": "for"}
    assert topology.for_body_shapes == {}
    assert topology.nested_iterate_sequence is None


def test_for_with_one_internal_iterate_is_a_nested_iterate_body() -> None:
    topology = _classify(_for_iterate_collect_graph())

    assert topology.for_body_shapes == {
        "for": NestedIterateBody(
            body_path_nodes=frozenset({"prep", "iterate", "body", "collect", "return"}),
            return_node_id="return",
            iterate_node_id="iterate",
            collect_node_id="collect",
        )
    }
    assert topology.nested_iterate_chain("for") is None
    assert topology.nested_for_body("for") is None


def test_iterate_body_with_two_collect_items_is_unsupported() -> None:
    topology = _classify(_for_iterate_collect_graph(extra_collect_item=True))

    assert topology.for_body_shapes == {}
    assert "for" in topology.for_body_paths


def test_iterate_body_whose_body_node_feeds_the_return_state_is_rejected() -> None:
    graph = _for_iterate_collect_graph(extra_connections=(("body", "value", "return", "state"),))

    _assert_rejected(graph, "Invalid For node (for): Iterate nodes inside For loop bodies are unsupported")


def test_iterate_body_node_unrelated_to_the_iterate_is_rejected() -> None:
    graph = _graph(
        [
            ForInvocation(id="for", collection=[["a"]]),
            PolymorphicStringTestInvocation(id="prep"),
            IterateInvocation(id="iterate"),
            AnyTypeTestInvocation(id="side"),
            CollectionConcatInvocation(id="mix"),
            CollectInvocation(id="collect"),
            ForReturnInvocation(id="return"),
        ],
        [
            ("for", "item", "prep", "value"),
            ("prep", "collection", "iterate", "collection"),
            ("iterate", "item", "mix", "first"),
            ("for", "item", "side", "value"),
            ("side", "value", "mix", "second"),
            ("mix", "collection", "collect", "item"),
            ("collect", "collection", "return", "output"),
        ],
        (("for", "return"),),
    )

    _assert_rejected(graph, "Invalid For node (for): Iterate nodes inside For loop bodies are unsupported")


def test_for_return_outside_its_for_body_is_rejected() -> None:
    # The ForReturn comes first, so its own check runs before the For's.
    graph = _graph(
        [ForReturnInvocation(id="return"), ForInvocation(id="for", collection=[1]), AnyTypeTestInvocation(id="body")],
        [("for", "item", "body", "value")],
        (("for", "return"),),
    )

    _assert_rejected(graph, "Invalid ForReturn node (return): ForReturn must belong to exactly one matching For")


def test_for_with_two_serial_iterates_is_a_nested_iterate_chain() -> None:
    graph = _graph(
        [
            ForInvocation(id="for", collection=[["a"]]),
            PolymorphicStringTestInvocation(id="prep_1"),
            IterateInvocation(id="iterate_1"),
            PolymorphicStringTestInvocation(id="prep_2"),
            IterateInvocation(id="iterate_2"),
            AnyTypeTestInvocation(id="body"),
            CollectInvocation(id="collect"),
            ForReturnInvocation(id="return"),
        ],
        [
            ("for", "item", "prep_1", "value"),
            ("prep_1", "collection", "iterate_1", "collection"),
            ("iterate_1", "item", "prep_2", "value"),
            ("prep_2", "collection", "iterate_2", "collection"),
            ("iterate_2", "item", "body", "value"),
            ("body", "value", "collect", "item"),
            ("collect", "collection", "return", "output"),
        ],
        (("for", "return"),),
    )

    assert _classify(graph).for_body_shapes == {
        "for": NestedIterateChain(
            body_path_nodes=frozenset({"prep_1", "iterate_1", "prep_2", "iterate_2", "body", "collect", "return"}),
            return_node_id="return",
            iterate_node_ids=("iterate_1", "iterate_2"),
            collect_node_id="collect",
        )
    }


def test_three_level_nested_for_is_classified_recursively() -> None:
    graph = _graph(
        [
            ForInvocation(id="outer", collection=[[["a"]]]),
            ForInvocation(id="middle"),
            ForInvocation(id="inner"),
            AnyTypeTestInvocation(id="body"),
            ForReturnInvocation(id="inner_return"),
            ForReturnInvocation(id="middle_return"),
            ForReturnInvocation(id="outer_return"),
        ],
        [
            ("outer", "item", "middle", "collection"),
            ("middle", "item", "inner", "collection"),
            ("inner", "item", "body", "value"),
            ("body", "value", "inner_return", "output"),
            ("inner", "output_collection", "middle_return", "output"),
            ("middle", "output_collection", "outer_return", "output"),
        ],
        (("outer", "outer_return"), ("middle", "middle_return"), ("inner", "inner_return")),
    )

    topology = _classify(graph)

    assert topology.for_body_shapes == {
        "outer": NestedForBody(
            body_path_nodes=frozenset({"middle", "inner", "body", "inner_return", "middle_return", "outer_return"}),
            outer_return_id="outer_return",
            inner_for_ids=("middle",),
            continuation_nodes=frozenset(),
        ),
        "middle": NestedForBody(
            body_path_nodes=frozenset({"inner", "body", "inner_return", "middle_return"}),
            outer_return_id="middle_return",
            inner_for_ids=("inner",),
            continuation_nodes=frozenset(),
        ),
    }
    assert topology.for_id_by_return_id == {"outer_return": "outer", "middle_return": "middle", "inner_return": "inner"}


def test_sibling_nested_fors_join_through_a_continuation() -> None:
    graph = _graph(
        [
            ForInvocation(id="outer", collection=[["a"]]),
            ForInvocation(id="left"),
            ForInvocation(id="right"),
            AnyTypeTestInvocation(id="left_body"),
            AnyTypeTestInvocation(id="right_body"),
            ForReturnInvocation(id="left_return"),
            ForReturnInvocation(id="right_return"),
            CollectionConcatInvocation(id="join"),
            ForReturnInvocation(id="outer_return"),
        ],
        [
            ("outer", "item", "left", "collection"),
            ("outer", "item", "right", "collection"),
            ("left", "item", "left_body", "value"),
            ("right", "item", "right_body", "value"),
            ("left_body", "value", "left_return", "output"),
            ("right_body", "value", "right_return", "output"),
            ("left", "output_collection", "join", "first"),
            ("right", "output_collection", "join", "second"),
            ("join", "collection", "outer_return", "output"),
        ],
        (("outer", "outer_return"), ("left", "left_return"), ("right", "right_return")),
    )

    shape = _classify(graph).nested_for_body("outer")

    assert shape is not None
    assert shape.inner_for_ids == ("left", "right")
    assert shape.continuation_nodes == {"join"}
    assert shape.outer_return_id == "outer_return"
    assert shape.body_path_nodes == {
        "left",
        "right",
        "left_body",
        "right_body",
        "left_return",
        "right_return",
        "join",
        "outer_return",
    }


def test_nested_for_continuation_fed_from_outside_the_inner_loop_is_rejected() -> None:
    graph = _graph(
        [
            ForInvocation(id="outer", collection=[["a"]]),
            ForInvocation(id="inner"),
            AnyTypeTestInvocation(id="inner_body"),
            ForReturnInvocation(id="inner_return"),
            AnyTypeTestInvocation(id="side"),
            CollectionConcatInvocation(id="join"),
            ForReturnInvocation(id="outer_return"),
        ],
        [
            ("outer", "item", "inner", "collection"),
            ("inner", "item", "inner_body", "value"),
            ("inner_body", "value", "inner_return", "output"),
            ("outer", "item", "side", "value"),
            ("inner", "output_collection", "join", "first"),
            ("side", "value", "join", "second"),
            ("join", "collection", "outer_return", "output"),
        ],
        (("outer", "outer_return"), ("inner", "inner_return")),
    )

    _assert_rejected(
        graph, "Invalid For node (outer): Nested For loops require one linked inner For with a matching ForReturn"
    )


def test_nested_for_around_an_iterate_body_classifies_only_the_inner_loop() -> None:
    graph = _graph(
        [
            ForInvocation(id="outer", collection=[[["a"]]]),
            ForInvocation(id="inner"),
            PolymorphicStringTestInvocation(id="prep"),
            IterateInvocation(id="iterate"),
            AnyTypeTestInvocation(id="body"),
            CollectInvocation(id="collect"),
            ForReturnInvocation(id="inner_return"),
            ForReturnInvocation(id="outer_return"),
        ],
        [
            ("outer", "item", "inner", "collection"),
            ("inner", "item", "prep", "value"),
            ("prep", "collection", "iterate", "collection"),
            ("iterate", "item", "body", "value"),
            ("body", "value", "collect", "item"),
            ("collect", "collection", "inner_return", "output"),
            ("inner", "output_collection", "outer_return", "output"),
        ],
        (("outer", "outer_return"), ("inner", "inner_return")),
    )

    topology = _classify(graph)

    assert set(topology.for_body_shapes) == {"inner"}
    assert topology.nested_iterate_body("inner") is not None
    assert topology.nested_for_body("outer") is None


def test_whole_graph_iterate_chain_is_a_nested_iterate_sequence() -> None:
    assert _classify(_iterate_sequence_graph()).nested_iterate_sequence == NestedIterateSequence(
        body_path_nodes=frozenset({"source", "outer", "prep", "inner", "body"}),
        source_node_id="source",
        iterate_node_ids=("outer", "inner"),
        preparation_node_ids=("prep",),
        body_node_id="body",
    )


def test_iterate_chain_whose_body_skips_the_inner_iterate_is_not_a_sequence() -> None:
    assert _classify(_iterate_sequence_graph(body_input="outer")).nested_iterate_sequence is None


def test_iterate_chain_ending_in_a_collect_is_not_a_sequence() -> None:
    graph = _graph(
        [
            CollectionConcatInvocation(id="source", first=[["a"]]),
            IterateInvocation(id="outer"),
            PolymorphicStringTestInvocation(id="prep"),
            IterateInvocation(id="inner"),
            CollectInvocation(id="body"),
        ],
        [
            ("source", "collection", "outer", "collection"),
            ("outer", "item", "prep", "value"),
            ("prep", "collection", "inner", "collection"),
            ("inner", "item", "body", "item"),
        ],
    )

    assert _classify(graph).nested_iterate_sequence is None


def test_execution_state_reclassifies_after_its_source_graph_changes() -> None:
    graph = _graph(
        [ForInvocation(id="for", collection=[1]), AnyTypeTestInvocation(id="body"), ForReturnInvocation(id="return")],
        [("for", "item", "body", "value"), ("body", "value", "return", "output")],
        (("for", "return"),),
    )
    state = GraphExecutionState(graph=graph)
    first = state._loop_topology()
    assert state._loop_topology() is first

    state.delete_edge(create_edge("body", "value", "return", "output"))
    state.add_node(AnyTypeTestInvocation(id="second_body"))
    state.add_edge(create_edge("body", "value", "second_body", "value"))
    state.add_edge(create_edge("second_body", "value", "return", "output"))

    assert state._loop_topology().for_body_paths["for"] == (
        frozenset({"body", "second_body", "return"}),
        "return",
    )


def test_topology_and_execution_state_survive_pickling() -> None:
    topology = _classify(_for_iterate_collect_graph())

    restored = pickle.loads(pickle.dumps(topology))

    assert restored.for_body_paths == topology.for_body_paths
    assert restored.for_body_shapes == topology.for_body_shapes
    assert restored.for_id_by_return_id == topology.for_id_by_return_id
    assert restored.nested_iterate_sequence == topology.nested_iterate_sequence
    with pytest.raises(TypeError):
        restored.for_body_shapes["other"] = topology.for_body_shapes["for"]  # type: ignore[index]

    state = GraphExecutionState(graph=_for_iterate_collect_graph())
    built = state._loop_topology()
    restored_state = pickle.loads(pickle.dumps(state))
    restored_topology = restored_state.__pydantic_private__["_runtime_state"].loop_topology

    # The topology and the state's flat graph unpickle as one graph, so the restored topology is still the cache entry.
    assert restored_topology.for_body_shapes == built.for_body_shapes
    assert restored_state._loop_topology() is restored_topology
