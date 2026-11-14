from unittest.mock import Mock

import pytest

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.primitives import BooleanCollectionInvocation, BooleanInvocation
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    Edge,
    EdgeConnection,
    Graph,
    GraphExecutionState,
    InvalidEdgeError,
    IterateInvocation,
)
from tests.test_nodes import AnyTypeTestInvocation, ErrorInvocation, PromptTestInvocation, create_loop_linkage


@pytest.mark.parametrize("condition", [False, True])
@pytest.mark.parametrize("connected_condition", [False, True])
def test_if_executes_both_connected_branches_before_resolving_and_preserves_fanout(
    condition: bool, connected_condition: bool
) -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="shared", a=2, b=3))
    graph.add_node(AddInvocation(id="true_branch", b=10))
    graph.add_node(AddInvocation(id="false_branch", b=20))
    graph.add_node(IfInvocation(id="if", condition=condition if not connected_condition else False))
    graph.add_node(AddInvocation(id="consumer"))
    if connected_condition:
        graph.add_node(BooleanInvocation(id="condition", value=condition))
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id="condition", field="value"),
                destination=EdgeConnection(node_id="if", field="condition"),
            )
        )
    for branch_id, branch_port in (("true_branch", "true_input"), ("false_branch", "false_input")):
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id="shared", field="value"),
                destination=EdgeConnection(node_id=branch_id, field="a"),
            )
        )
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=branch_id, field="value"),
                destination=EdgeConnection(node_id="if", field=branch_port),
            )
        )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="true_branch", field="value"),
            destination=EdgeConnection(node_id="consumer", field="a"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="if", field="value"),
            destination=EdgeConnection(node_id="consumer", field="b"),
        )
    )

    state = GraphExecutionState(graph=graph)
    execution_order: list[str] = []
    while (node := state.next()) is not None:
        source_id = state.prepared_source_mapping[node.id]
        execution_order.append(source_id)
        if source_id == "if":
            assert node.true_input == (15 if condition else None)
            assert node.false_input == (25 if not condition else None)
        state.complete(node.id, node.invoke(Mock()))

    if_index = execution_order.index("if")
    assert execution_order.index("true_branch") < if_index
    assert execution_order.index("false_branch") < if_index
    assert execution_order.count("shared") == 1

    result_by_source = {
        source_id: state.results[execution_id]
        for source_id, execution_ids in state.source_prepared_mapping.items()
        for execution_id in execution_ids
    }
    assert result_by_source["if"].value == (15 if condition else 25)
    assert result_by_source["consumer"].value == (30 if condition else 40)
    assert state.is_complete()


@pytest.mark.parametrize("condition", [False, True])
def test_if_resolves_literal_branch_inputs(condition: bool) -> None:
    graph = Graph(nodes={"if": IfInvocation(id="if", condition=condition, true_input=11, false_input=22)})

    state = GraphExecutionState(graph=graph)
    node = state.next()
    assert node is not None
    state.complete(node.id, node.invoke(Mock()))

    assert state.results[node.id].value == (11 if condition else 22)
    assert state.is_complete()


def test_failure_in_unselected_if_branch_fails_execution() -> None:
    graph = Graph()
    graph.add_node(IfInvocation(id="if", condition=False))
    graph.add_node(ErrorInvocation(id="unselected_failure"))
    graph.add_node(AddInvocation(id="selected", a=4, b=5))
    graph.add_node(PromptTestInvocation(id="external_consumer"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="unselected_failure", field="prompt"),
            destination=EdgeConnection(node_id="if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="selected", field="value"),
            destination=EdgeConnection(node_id="if", field="false_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="unselected_failure", field="prompt"),
            destination=EdgeConnection(node_id="external_consumer", field="prompt"),
        )
    )

    state = GraphExecutionState(graph=graph)
    with pytest.raises(Exception, match="supposed to fail"):
        while (node := state.next()) is not None:
            state.complete(node.id, node.invoke(Mock()))


@pytest.mark.parametrize(("condition", "expected"), [(True, 17), (False, 27)])
def test_root_if_executes_only_the_selected_private_branch_and_shared_ancestor_once(
    condition: bool, expected: int
) -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=condition))
    graph.add_node(AddInvocation(id="shared", a=3, b=4))
    graph.add_node(AddInvocation(id="true_branch", b=10))
    graph.add_node(AddInvocation(id="false_branch", b=20))
    graph.add_node(IfInvocation(id="if"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="condition", field="value"),
            destination=EdgeConnection(node_id="if", field="condition"),
        )
    )
    for branch_id, branch_field in (("true_branch", "true_input"), ("false_branch", "false_input")):
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id="shared", field="value"),
                destination=EdgeConnection(node_id=branch_id, field="a"),
            )
        )
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=branch_id, field="value"),
                destination=EdgeConnection(node_id="if", field=branch_field),
            )
        )

    state = GraphExecutionState(graph=graph)
    executed: list[str] = []
    while (node := state.next()) is not None:
        source_id = state.prepared_source_mapping[node.id]
        executed.append(source_id)
        state.complete(node.id, node.invoke(Mock()))

    assert executed.count("shared") == 1
    selected_branch = "true_branch" if condition else "false_branch"
    unselected_branch = "false_branch" if condition else "true_branch"
    assert selected_branch in executed
    assert executed.index("condition") < executed.index(selected_branch)
    assert unselected_branch not in executed
    if_exec_id = next(iter(state.source_prepared_mapping["if"]))
    assert state.results[if_exec_id].value == expected
    assert state.is_complete()


def test_single_iterate_if_collect_executes_only_selected_branch_per_frame_in_order() -> None:
    graph = Graph()
    graph.add_node(BooleanCollectionInvocation(id="conditions", collection=[True, False, True]))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(BooleanInvocation(id="true_branch"))
    graph.add_node(BooleanInvocation(id="false_branch"))
    graph.add_node(CollectInvocation(id="collect"))
    for source, source_field, destination, destination_field in (
        ("conditions", "collection", "iterate", "collection"),
        ("iterate", "item", "if", "condition"),
        ("iterate", "item", "true_branch", "value"),
        ("true_branch", "value", "if", "true_input"),
        ("iterate", "item", "false_branch", "value"),
        ("false_branch", "value", "if", "false_input"),
        ("if", "value", "collect", "item"),
    ):
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source, field=source_field),
                destination=EdgeConnection(node_id=destination, field=destination_field),
            )
        )

    state = GraphExecutionState(graph=graph)
    executed: list[str] = []
    while (node := state.next()) is not None:
        source_id = state.prepared_source_mapping[node.id]
        executed.append(source_id)
        state.complete(node.id, node.invoke(Mock()))

    assert executed.count("conditions") == 1
    assert executed.count("iterate") == 3
    assert executed.count("true_branch") == 2
    assert executed.count("false_branch") == 1
    collect_exec_id = next(iter(state.source_prepared_mapping["collect"]))
    assert state.results[collect_exec_id].collection == [True, False, True]
    assert state.is_complete()


def test_iterate_if_with_second_collect_consumer_remains_eager() -> None:
    graph = Graph()
    graph.add_node(BooleanCollectionInvocation(id="conditions", collection=[True, False, True]))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(BooleanInvocation(id="true_branch"))
    graph.add_node(BooleanInvocation(id="false_branch"))
    graph.add_node(CollectInvocation(id="collect_a"))
    graph.add_node(CollectInvocation(id="collect_b"))
    for source, source_field, destination, destination_field in (
        ("conditions", "collection", "iterate", "collection"),
        ("iterate", "item", "if", "condition"),
        ("iterate", "item", "true_branch", "value"),
        ("true_branch", "value", "if", "true_input"),
        ("iterate", "item", "false_branch", "value"),
        ("false_branch", "value", "if", "false_input"),
        ("if", "value", "collect_a", "item"),
        ("if", "value", "collect_b", "item"),
    ):
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source, field=source_field),
                destination=EdgeConnection(node_id=destination, field=destination_field),
            )
        )

    state = GraphExecutionState(graph=graph)
    executed: list[str] = []
    while (node := state.next()) is not None:
        source_id = state.prepared_source_mapping[node.id]
        executed.append(source_id)
        state.complete(node.id, node.invoke(Mock()))

    assert executed.count("true_branch") == 3
    assert executed.count("false_branch") == 3
    for collect_id in ("collect_a", "collect_b"):
        collect_exec_id = next(iter(state.source_prepared_mapping[collect_id]))
        assert state.results[collect_exec_id].collection == [True, False, True]
    assert state.is_complete()


def test_nested_ifs_remain_eager() -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="outer_condition", value=True))
    graph.add_node(BooleanInvocation(id="inner_condition", value=False))
    graph.add_node(BooleanInvocation(id="inner_true", value=True))
    graph.add_node(BooleanInvocation(id="inner_false", value=False))
    graph.add_node(BooleanInvocation(id="outer_false", value=False))
    graph.add_node(IfInvocation(id="inner_if"))
    graph.add_node(IfInvocation(id="outer_if"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="inner_condition", field="value"),
            destination=EdgeConnection(node_id="inner_if", field="condition"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="inner_true", field="value"),
            destination=EdgeConnection(node_id="inner_if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="inner_false", field="value"),
            destination=EdgeConnection(node_id="inner_if", field="false_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="outer_condition", field="value"),
            destination=EdgeConnection(node_id="outer_if", field="condition"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="inner_if", field="value"),
            destination=EdgeConnection(node_id="outer_if", field="true_input"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="outer_false", field="value"),
            destination=EdgeConnection(node_id="outer_if", field="false_input"),
        )
    )

    state = GraphExecutionState(graph=graph)
    executed: list[str] = []
    while (node := state.next()) is not None:
        executed.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))

    assert {"inner_true", "inner_false", "outer_false"}.issubset(executed)
    assert state.is_complete()


def test_sibling_ifs_remain_eager() -> None:
    graph = Graph()
    for node_id, value in (
        ("condition_a", True),
        ("condition_b", False),
        ("true_a", True),
        ("false_a", False),
        ("true_b", True),
        ("false_b", False),
    ):
        graph.add_node(BooleanInvocation(id=node_id, value=value))
    graph.add_node(IfInvocation(id="if_a"))
    graph.add_node(IfInvocation(id="if_b"))
    graph.add_node(CollectInvocation(id="collect"))
    for condition, if_id, true_id, false_id in (
        ("condition_a", "if_a", "true_a", "false_a"),
        ("condition_b", "if_b", "true_b", "false_b"),
    ):
        for source_id, source_field, destination_id, destination_field in (
            (condition, "value", if_id, "condition"),
            (true_id, "value", if_id, "true_input"),
            (false_id, "value", if_id, "false_input"),
            (if_id, "value", "collect", "item"),
        ):
            graph.add_edge(
                Edge(
                    source=EdgeConnection(node_id=source_id, field=source_field),
                    destination=EdgeConnection(node_id=destination_id, field=destination_field),
                )
            )

    state = GraphExecutionState(graph=graph)
    executed: list[str] = []
    while (node := state.next()) is not None:
        executed.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))

    assert {"true_a", "false_a", "true_b", "false_b"}.issubset(executed)
    collect_exec_id = next(iter(state.source_prepared_mapping["collect"]))
    assert sorted(state.results[collect_exec_id].collection) == [False, True]
    assert state.is_complete()


def test_if_with_for_axis_remains_eager() -> None:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=[True, False]))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AnyTypeTestInvocation(id="true_branch"))
    graph.add_node(AnyTypeTestInvocation(id="false_branch"))
    graph.add_node(ForReturnInvocation(id="return"))
    for source_id, source_field, destination_id, destination_field in (
        ("for", "item", "if", "condition"),
        ("for", "item", "true_branch", "value"),
        ("true_branch", "value", "if", "true_input"),
        ("for", "item", "false_branch", "value"),
        ("false_branch", "value", "if", "false_input"),
        ("if", "value", "return", "output"),
    ):
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source_id, field=source_field),
                destination=EdgeConnection(node_id=destination_id, field=destination_field),
            )
        )
    graph.add_edge(create_loop_linkage("for", "return"))

    state = GraphExecutionState(graph=graph)
    executed: list[str] = []
    while (node := state.next()) is not None:
        executed.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))

    assert executed.count("true_branch") == 2
    assert executed.count("false_branch") == 2
    assert state.is_complete()


def test_iterate_if_with_unscoped_branch_inputs_remains_eager() -> None:
    graph = Graph()
    graph.add_node(BooleanCollectionInvocation(id="conditions", collection=[True, False]))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(BooleanInvocation(id="true_branch", value=True))
    graph.add_node(BooleanInvocation(id="false_branch", value=False))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(CollectInvocation(id="collect"))
    for source_id, source_field, destination_id, destination_field in (
        ("conditions", "collection", "iterate", "collection"),
        ("iterate", "item", "if", "condition"),
        ("true_branch", "value", "if", "true_input"),
        ("false_branch", "value", "if", "false_input"),
        ("if", "value", "collect", "item"),
    ):
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source_id, field=source_field),
                destination=EdgeConnection(node_id=destination_id, field=destination_field),
            )
        )

    state = GraphExecutionState(graph=graph)
    executed: list[str] = []
    while (node := state.next()) is not None:
        executed.append(state.prepared_source_mapping[node.id])
        state.complete(node.id, node.invoke(Mock()))

    assert executed.count("true_branch") == 1
    assert executed.count("false_branch") == 1
    collect_exec_id = next(iter(state.source_prepared_mapping["collect"]))
    assert state.results[collect_exec_id].collection == [True, False]
    assert state.is_complete()


def test_invalid_edge_still_fails_graph_validation() -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(IfInvocation(id="if"))
    with pytest.raises(InvalidEdgeError, match="Field types are incompatible"):
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id="condition", field="not_an_output"),
                destination=EdgeConnection(node_id="if", field="condition"),
            )
        )
