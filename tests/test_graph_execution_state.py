from typing import Optional
from unittest.mock import Mock

import pytest
from pydantic import TypeAdapter

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext
from invokeai.app.invocations.collections import RangeInvocation
from invokeai.app.invocations.logic import IfInvocation, IfInvocationOutput
from invokeai.app.invocations.math import AddInvocation, MultiplyInvocation
from invokeai.app.invocations.primitives import (
    BooleanCollectionInvocation,
    BooleanCollectionOutput,
    BooleanInvocation,
    BooleanOutput,
)
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    Graph,
    GraphExecutionState,
    IterateInvocation,
)

# This import must happen before other invoke imports or test in other files(!!) break
from tests.test_nodes import (
    AnyTypeTestInvocation,
    AnyTypeTestInvocationOutput,
    PromptCollectionTestInvocation,
    PromptTestInvocation,
    TextToImageTestInvocation,
    create_edge,
)


@pytest.fixture
def simple_graph() -> Graph:
    g = Graph()
    g.add_node(PromptTestInvocation(id="1", prompt="Banana sushi"))
    g.add_node(TextToImageTestInvocation(id="2"))
    g.add_edge(create_edge("1", "prompt", "2", "prompt"))
    return g


def invoke_next(g: GraphExecutionState) -> tuple[Optional[BaseInvocation], Optional[BaseInvocationOutput]]:
    n = g.next()
    if n is None:
        return (None, None)

    print(f"invoking {n.id}: {type(n)}")
    o = n.invoke(Mock(InvocationContext))
    g.complete(n.id, o)

    return (n, o)


def execute_all_nodes(g: GraphExecutionState) -> list[str]:
    """Execute the graph to completion and return source node ids in execution order."""

    executed_source_ids: list[str] = []
    while True:
        invocation, _output = invoke_next(g)
        if invocation is None:
            break
        executed_source_ids.append(g.prepared_source_mapping[invocation.id])

    return executed_source_ids


def test_graph_state_executes_in_order(simple_graph: Graph):
    g = GraphExecutionState(graph=simple_graph)

    n1 = invoke_next(g)
    n2 = invoke_next(g)
    n3 = g.next()

    assert g.prepared_source_mapping[n1[0].id] == "1"
    assert g.prepared_source_mapping[n2[0].id] == "2"
    assert n3 is None
    assert g.results[n1[0].id].prompt == n1[0].prompt
    assert n2[0].prompt == n1[0].prompt


def test_graph_is_complete(simple_graph: Graph):
    g = GraphExecutionState(graph=simple_graph)
    _ = invoke_next(g)
    _ = invoke_next(g)
    _ = g.next()

    assert g.is_complete()


def test_graph_is_not_complete(simple_graph: Graph):
    g = GraphExecutionState(graph=simple_graph)
    _ = invoke_next(g)
    _ = g.next()

    assert not g.is_complete()


# TODO: test completion with iterators/subgraphs


def test_graph_state_expands_iterator():
    graph = Graph()
    graph.add_node(RangeInvocation(id="0", start=0, stop=3, step=1))
    graph.add_node(IterateInvocation(id="1"))
    graph.add_node(MultiplyInvocation(id="2", b=10))
    graph.add_node(AddInvocation(id="3", b=1))
    graph.add_edge(create_edge("0", "collection", "1", "collection"))
    graph.add_edge(create_edge("1", "item", "2", "a"))
    graph.add_edge(create_edge("2", "value", "3", "a"))

    g = GraphExecutionState(graph=graph)
    while not g.is_complete():
        invoke_next(g)

    prepared_add_nodes = g.source_prepared_mapping["3"]
    results = {g.results[n].value for n in prepared_add_nodes}
    expected = {1, 11, 21}
    assert results == expected


def test_graph_state_collects():
    graph = Graph()
    test_prompts = ["Banana sushi", "Cat sushi"]
    graph.add_node(PromptCollectionTestInvocation(id="1", collection=list(test_prompts)))
    graph.add_node(IterateInvocation(id="2"))
    graph.add_node(PromptTestInvocation(id="3"))
    graph.add_node(CollectInvocation(id="4"))
    graph.add_edge(create_edge("1", "collection", "2", "collection"))
    graph.add_edge(create_edge("2", "item", "3", "prompt"))
    graph.add_edge(create_edge("3", "prompt", "4", "item"))

    g = GraphExecutionState(graph=graph)
    _ = invoke_next(g)
    _ = invoke_next(g)
    _ = invoke_next(g)
    _ = invoke_next(g)
    _ = invoke_next(g)
    n6 = invoke_next(g)

    assert isinstance(n6[0], CollectInvocation)

    assert sorted(g.results[n6[0].id].collection) == sorted(test_prompts)


def test_graph_state_resumes_partially_executed_session_after_json_round_trip():
    graph = Graph()
    graph.add_node(RangeInvocation(id="c", start=1, stop=5, step=1))
    graph.add_node(IterateInvocation(id="iter"))
    graph.add_node(AddInvocation(id="add", b=1))
    graph.add_node(CollectInvocation(id="collect"))

    graph.add_edge(create_edge("c", "collection", "iter", "collection"))
    graph.add_edge(create_edge("iter", "item", "add", "a"))
    graph.add_edge(create_edge("add", "value", "collect", "item"))

    state = GraphExecutionState(graph=graph)

    for _ in range(4):
        invocation, output = invoke_next(state)
        assert invocation is not None
        assert output is not None

    raw = state.model_dump_json(warnings=False, exclude_none=True)
    resumed = TypeAdapter(GraphExecutionState).validate_json(raw, strict=False)
    registry = resumed._prepared_registry()

    assert all(
        registry.get_iteration_path(exec_node_id) is not None for exec_node_id in resumed.prepared_source_mapping
    )

    executed_source_ids = execute_all_nodes(resumed)

    assert executed_source_ids
    assert "add" in executed_source_ids
    assert "collect" in resumed.source_prepared_mapping

    prepared_collect_id = next(iter(resumed.source_prepared_mapping["collect"]))
    assert resumed.results[prepared_collect_id].collection == [2, 3, 4, 5]


def test_if_graph_state_resumes_resolved_branch_after_json_round_trip():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(PromptTestInvocation(id="true_value", prompt="true branch"))
    graph.add_node(PromptTestInvocation(id="false_value", prompt="false branch"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(PromptTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("true_value", "prompt", "if", "true_input"))
    graph.add_edge(create_edge("false_value", "prompt", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "selected_output", "prompt"))

    state = GraphExecutionState(graph=graph)

    for _ in range(2):
        invocation, output = invoke_next(state)
        assert invocation is not None
        assert output is not None

    raw = state.model_dump_json(warnings=False, exclude_none=True)
    resumed = TypeAdapter(GraphExecutionState).validate_json(raw, strict=False)

    executed_source_ids = execute_all_nodes(resumed)

    prepared_selected_output_id = next(iter(resumed.source_prepared_mapping["selected_output"]))
    assert resumed.results[prepared_selected_output_id].prompt == "true branch"
    assert set(executed_source_ids) == {"if", "selected_output"}
    assert "false_value" not in executed_source_ids


def test_graph_state_prepares_eagerly():
    """Tests that all prepareable nodes are prepared"""
    graph = Graph()

    test_prompts = ["Banana sushi", "Cat sushi"]
    graph.add_node(PromptCollectionTestInvocation(id="prompt_collection", collection=list(test_prompts)))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(PromptTestInvocation(id="prompt_iterated"))
    graph.add_edge(create_edge("prompt_collection", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "prompt_iterated", "prompt"))

    # separated, fully-preparable chain of nodes
    graph.add_node(PromptTestInvocation(id="prompt_chain_1", prompt="Dinosaur sushi"))
    graph.add_node(PromptTestInvocation(id="prompt_chain_2"))
    graph.add_node(PromptTestInvocation(id="prompt_chain_3"))
    graph.add_edge(create_edge("prompt_chain_1", "prompt", "prompt_chain_2", "prompt"))
    graph.add_edge(create_edge("prompt_chain_2", "prompt", "prompt_chain_3", "prompt"))

    g = GraphExecutionState(graph=graph)
    g.next()

    assert "prompt_collection" in g.source_prepared_mapping
    assert "prompt_chain_1" in g.source_prepared_mapping
    assert "prompt_chain_2" in g.source_prepared_mapping
    assert "prompt_chain_3" in g.source_prepared_mapping
    assert "iterate" not in g.source_prepared_mapping
    assert "prompt_iterated" not in g.source_prepared_mapping


def test_graph_executes_depth_first():
    """Tests that the graph executes depth-first, executing a branch as far as possible before moving to the next branch"""

    def assert_topo_order_and_all_executed(state: GraphExecutionState, order: list[str]):
        """
        Validates:
          1) Every materialized exec node executed exactly once.
          2) Execution order respects all exec-graph dependencies (u→v ⇒ u before v).
        """
        # order must be EXEC node ids in run order
        exec_nodes = set(state.execution_graph.nodes.keys())

        # 1) coverage: all exec nodes ran, and no duplicates
        pos = {nid: i for i, nid in enumerate(order)}
        assert set(pos.keys()) == exec_nodes, (
            f"Executed {len(pos)} of {len(exec_nodes)} nodes. Missing: {sorted(exec_nodes - set(pos))[:10]}"
        )
        assert len(pos) == len(order), "Duplicate execution detected"

        # 2) topo order: parents before children
        for e in state.execution_graph.edges:
            u = e.source.node_id
            v = e.destination.node_id
            assert pos[u] < pos[v], f"child {v} ran before parent {u}"

    graph = Graph()

    test_prompts = ["Banana sushi", "Cat sushi"]
    graph.add_node(PromptCollectionTestInvocation(id="prompt_collection", collection=list(test_prompts)))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(PromptTestInvocation(id="prompt_iterated"))
    graph.add_node(PromptTestInvocation(id="prompt_successor"))
    graph.add_edge(create_edge("prompt_collection", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "prompt_iterated", "prompt"))
    graph.add_edge(create_edge("prompt_iterated", "prompt", "prompt_successor", "prompt"))

    g = GraphExecutionState(graph=graph)
    order: list[str] = []

    while True:
        n = g.next()
        if n is None:
            break
        o = n.invoke(Mock(InvocationContext))
        g.complete(n.id, o)
        order.append(n.id)

    assert_topo_order_and_all_executed(g, order)


def test_graph_scheduler_drains_active_class_before_switching():
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt_a", prompt="a"))
    graph.add_node(PromptTestInvocation(id="prompt_b", prompt="b"))
    graph.add_node(TextToImageTestInvocation(id="image"))

    g = GraphExecutionState(graph=graph)
    g.set_ready_order([PromptTestInvocation, TextToImageTestInvocation])

    first = invoke_next(g)[0]
    second = invoke_next(g)[0]
    third = invoke_next(g)[0]

    assert first is not None
    assert g.prepared_source_mapping[first.id] == "prompt_a"
    assert g.prepared_source_mapping[second.id] == "prompt_b"
    assert g.prepared_source_mapping[third.id] == "image"


def test_graph_scheduler_skips_stale_ready_entries():
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt_a", prompt="a"))
    graph.add_node(PromptTestInvocation(id="prompt_b", prompt="b"))

    g = GraphExecutionState(graph=graph)
    g.set_ready_order([PromptTestInvocation])

    first = invoke_next(g)[0]
    assert first is not None

    prompt_queue = g._queue_for(PromptTestInvocation.__name__)
    prompt_queue.appendleft(first.id)

    second = g.next()

    assert second is not None
    assert second.id != first.id
    assert g.prepared_source_mapping[second.id] == "prompt_b"


def test_graph_scheduler_falls_back_to_non_priority_ready_classes():
    graph = Graph()
    graph.add_node(TextToImageTestInvocation(id="image"))

    g = GraphExecutionState(graph=graph)
    g.set_ready_order([PromptTestInvocation])

    next_node = g.next()

    assert next_node is not None
    assert g.prepared_source_mapping[next_node.id] == "image"


# Because this tests deterministic ordering, we run it multiple times
@pytest.mark.parametrize("execution_number", range(5))
def test_graph_iterate_execution_order(execution_number: int):
    """Tests that iterate nodes execution is ordered by the order of the collection"""

    graph = Graph()

    test_prompts = ["Banana sushi", "Cat sushi", "Strawberry Sushi", "Dinosaur Sushi"]
    graph.add_node(PromptCollectionTestInvocation(id="prompt_collection", collection=list(test_prompts)))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(PromptTestInvocation(id="prompt_iterated"))
    graph.add_edge(create_edge("prompt_collection", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "prompt_iterated", "prompt"))

    g = GraphExecutionState(graph=graph)
    _ = invoke_next(g)
    _ = invoke_next(g)
    assert _[1].item == "Banana sushi"
    _ = invoke_next(g)
    assert _[1].item == "Cat sushi"
    _ = invoke_next(g)
    assert _[1].item == "Strawberry Sushi"
    _ = invoke_next(g)
    assert _[1].item == "Dinosaur Sushi"
    _ = invoke_next(g)


# Because this tests deterministic ordering, we run it multiple times
@pytest.mark.parametrize("execution_number", range(5))
def test_graph_nested_iterate_execution_order(execution_number: int):
    """
    Validates best-effort in-order execution for nodes expanded under nested iterators.
    Expected lexicographic order by (outer_index, inner_index), subject to readiness.
    """
    graph = Graph()

    # Outer iterator: [0, 1]
    graph.add_node(RangeInvocation(id="outer_range", start=0, stop=2, step=1))
    graph.add_node(IterateInvocation(id="outer_iter"))

    # Inner iterator is derived from the outer item:
    # start = outer_item * 10
    # stop  = start + 2  => yields 2 items per outer item
    graph.add_node(MultiplyInvocation(id="mul10", b=10))
    graph.add_node(AddInvocation(id="stop_plus2", b=2))
    graph.add_node(RangeInvocation(id="inner_range", start=0, stop=1, step=1))
    graph.add_node(IterateInvocation(id="inner_iter"))

    # Observe inner items (they encode outer via start=outer*10)
    graph.add_node(AddInvocation(id="sum", b=0))

    graph.add_edge(create_edge("outer_range", "collection", "outer_iter", "collection"))
    graph.add_edge(create_edge("outer_iter", "item", "mul10", "a"))
    graph.add_edge(create_edge("mul10", "value", "stop_plus2", "a"))
    graph.add_edge(create_edge("mul10", "value", "inner_range", "start"))
    graph.add_edge(create_edge("stop_plus2", "value", "inner_range", "stop"))
    graph.add_edge(create_edge("inner_range", "collection", "inner_iter", "collection"))
    graph.add_edge(create_edge("inner_iter", "item", "sum", "a"))

    g = GraphExecutionState(graph=graph)
    sum_values: list[int] = []

    while True:
        n, o = invoke_next(g)
        if n is None:
            break
        if g.prepared_source_mapping[n.id] == "sum":
            sum_values.append(o.value)

    assert sum_values == [0, 1, 10, 11]


def test_graph_validate_self_iterator_without_collection_input_raises_invalid_edge_error():
    """Iterator nodes with no collection input should fail validation cleanly.

    This test exposes the bug where validation crashes with IndexError instead of raising InvalidEdgeError.
    """
    from invokeai.app.services.shared.graph import InvalidEdgeError

    graph = Graph()
    graph.add_node(IterateInvocation(id="iterate"))

    with pytest.raises(InvalidEdgeError):
        graph.validate_self()


def test_graph_validate_self_collector_without_item_inputs_raises_invalid_edge_error():
    """Collector nodes with no item inputs should fail validation cleanly.

    This test exposes the bug where validation can crash (e.g. StopIteration) instead of raising InvalidEdgeError.
    """
    from invokeai.app.services.shared.graph import InvalidEdgeError

    graph = Graph()
    graph.add_node(CollectInvocation(id="collect"))

    with pytest.raises(InvalidEdgeError):
        graph.validate_self()


def test_if_invocation_selects_true_input_value():
    invocation = IfInvocation(id="if", condition=True, true_input="true", false_input="false")

    output = invocation.invoke(Mock(InvocationContext))

    assert output.value == "true"


def test_if_invocation_outputs_none_when_selected_input_is_missing():
    invocation = IfInvocation(id="if", condition=False, true_input="true")

    output = invocation.invoke(Mock(InvocationContext))

    assert output.value is None


def test_if_invocation_output_allows_missing_value_on_deserialization():
    output = IfInvocationOutput.model_validate({"type": "if_output"})

    assert output.value is None


def test_if_invocation_output_connects_to_downstream_input():
    graph = Graph()
    graph.add_node(IfInvocation(id="if", condition=True, true_input="connected value", false_input="unused"))
    graph.add_node(PromptTestInvocation(id="prompt"))
    graph.add_edge(create_edge("if", "value", "prompt", "prompt"))

    g = GraphExecutionState(graph=graph)
    while not g.is_complete():
        invoke_next(g)

    prepared_prompt_nodes = g.source_prepared_mapping["prompt"]
    assert len(prepared_prompt_nodes) == 1
    prepared_prompt_node_id = next(iter(prepared_prompt_nodes))
    assert g.results[prepared_prompt_node_id].prompt == "connected value"


@pytest.mark.xfail(strict=True, reason="Legacy eager If-node execution should no longer occur")
def test_if_graph_current_behavior_executes_both_branches_and_shared_ancestors():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(PromptTestInvocation(id="shared", prompt="shared value"))
    graph.add_node(PromptTestInvocation(id="true_mid"))
    graph.add_node(PromptTestInvocation(id="true_leaf"))
    graph.add_node(PromptTestInvocation(id="false_mid"))
    graph.add_node(PromptTestInvocation(id="false_leaf"))
    graph.add_node(PromptTestInvocation(id="side_consumer"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(PromptTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("shared", "prompt", "true_mid", "prompt"))
    graph.add_edge(create_edge("true_mid", "prompt", "true_leaf", "prompt"))
    graph.add_edge(create_edge("true_leaf", "prompt", "if", "true_input"))
    graph.add_edge(create_edge("shared", "prompt", "false_mid", "prompt"))
    graph.add_edge(create_edge("false_mid", "prompt", "false_leaf", "prompt"))
    graph.add_edge(create_edge("false_leaf", "prompt", "if", "false_input"))
    graph.add_edge(create_edge("shared", "prompt", "side_consumer", "prompt"))
    graph.add_edge(create_edge("if", "value", "selected_output", "prompt"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    assert set(executed_source_ids) == {
        "condition",
        "shared",
        "true_mid",
        "true_leaf",
        "false_mid",
        "false_leaf",
        "side_consumer",
        "if",
        "selected_output",
    }
    assert executed_source_ids.count("false_mid") == 1
    assert executed_source_ids.count("false_leaf") == 1

    prepared_selected_output_id = next(iter(g.source_prepared_mapping["selected_output"]))
    assert g.results[prepared_selected_output_id].prompt == "shared value"


@pytest.mark.xfail(strict=True, reason="Legacy eager If-node execution should no longer occur")
def test_if_graph_current_behavior_executes_both_simple_branches():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(PromptTestInvocation(id="true_value", prompt="true branch"))
    graph.add_node(PromptTestInvocation(id="false_value", prompt="false branch"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(PromptTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("true_value", "prompt", "if", "true_input"))
    graph.add_edge(create_edge("false_value", "prompt", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "selected_output", "prompt"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    assert set(executed_source_ids) == {"condition", "true_value", "false_value", "if", "selected_output"}
    prepared_selected_output_id = next(iter(g.source_prepared_mapping["selected_output"]))
    assert g.results[prepared_selected_output_id].prompt == "true branch"


def test_if_graph_optimized_behavior_executes_only_selected_simple_branch():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(PromptTestInvocation(id="true_value", prompt="true branch"))
    graph.add_node(PromptTestInvocation(id="false_value", prompt="false branch"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(PromptTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("true_value", "prompt", "if", "true_input"))
    graph.add_edge(create_edge("false_value", "prompt", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "selected_output", "prompt"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    assert set(executed_source_ids) == {"condition", "true_value", "if", "selected_output"}
    assert "false_value" not in executed_source_ids


def test_if_graph_optimized_behavior_records_skipped_branch_in_execution_history():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(PromptTestInvocation(id="true_value", prompt="true branch"))
    graph.add_node(PromptTestInvocation(id="false_value", prompt="false branch"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(PromptTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("true_value", "prompt", "if", "true_input"))
    graph.add_edge(create_edge("false_value", "prompt", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "selected_output", "prompt"))

    g = GraphExecutionState(graph=graph)
    execute_all_nodes(g)

    assert set(g.executed_history) == {"condition", "true_value", "false_value", "if", "selected_output"}
    assert g.executed_history.count("false_value") == 1


def test_if_graph_optimized_behavior_skips_unselected_branch_but_keeps_shared_ancestors():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(PromptTestInvocation(id="shared", prompt="shared value"))
    graph.add_node(PromptTestInvocation(id="true_mid"))
    graph.add_node(PromptTestInvocation(id="true_leaf"))
    graph.add_node(PromptTestInvocation(id="false_mid"))
    graph.add_node(PromptTestInvocation(id="false_leaf"))
    graph.add_node(PromptTestInvocation(id="side_consumer"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(PromptTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("shared", "prompt", "true_mid", "prompt"))
    graph.add_edge(create_edge("true_mid", "prompt", "true_leaf", "prompt"))
    graph.add_edge(create_edge("true_leaf", "prompt", "if", "true_input"))
    graph.add_edge(create_edge("shared", "prompt", "false_mid", "prompt"))
    graph.add_edge(create_edge("false_mid", "prompt", "false_leaf", "prompt"))
    graph.add_edge(create_edge("false_leaf", "prompt", "if", "false_input"))
    graph.add_edge(create_edge("shared", "prompt", "side_consumer", "prompt"))
    graph.add_edge(create_edge("if", "value", "selected_output", "prompt"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    assert set(executed_source_ids) == {
        "condition",
        "shared",
        "true_mid",
        "true_leaf",
        "side_consumer",
        "if",
        "selected_output",
    }
    assert "false_mid" not in executed_source_ids
    assert "false_leaf" not in executed_source_ids


def test_if_graph_optimized_behavior_skips_distant_unselected_ancestors_only_when_exclusive():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=False))
    graph.add_node(PromptTestInvocation(id="shared_root", prompt="shared value"))
    graph.add_node(PromptTestInvocation(id="true_shared_mid"))
    graph.add_node(PromptTestInvocation(id="true_exclusive_leaf"))
    graph.add_node(PromptTestInvocation(id="false_mid"))
    graph.add_node(PromptTestInvocation(id="false_leaf"))
    graph.add_node(PromptTestInvocation(id="shared_observer"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(PromptTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("shared_root", "prompt", "true_shared_mid", "prompt"))
    graph.add_edge(create_edge("true_shared_mid", "prompt", "true_exclusive_leaf", "prompt"))
    graph.add_edge(create_edge("true_exclusive_leaf", "prompt", "if", "true_input"))
    graph.add_edge(create_edge("shared_root", "prompt", "false_mid", "prompt"))
    graph.add_edge(create_edge("false_mid", "prompt", "false_leaf", "prompt"))
    graph.add_edge(create_edge("false_leaf", "prompt", "if", "false_input"))
    graph.add_edge(create_edge("true_shared_mid", "prompt", "shared_observer", "prompt"))
    graph.add_edge(create_edge("if", "value", "selected_output", "prompt"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    assert set(executed_source_ids) == {
        "condition",
        "shared_root",
        "true_shared_mid",
        "false_mid",
        "false_leaf",
        "shared_observer",
        "if",
        "selected_output",
    }
    assert "true_exclusive_leaf" not in executed_source_ids


def test_if_graph_optimized_behavior_allows_selected_missing_branch_input():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=False))
    graph.add_node(PromptTestInvocation(id="true_value", prompt="true branch"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AnyTypeTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("true_value", "prompt", "if", "true_input"))
    graph.add_edge(create_edge("if", "value", "selected_output", "value"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    prepared_selected_output_id = next(iter(g.source_prepared_mapping["selected_output"]))
    assert g.results[prepared_selected_output_id].value is None
    assert set(executed_source_ids) == {"condition", "if", "selected_output"}
    assert "true_value" not in executed_source_ids


def test_if_graph_optimized_behavior_does_not_cross_defer_independent_ifs():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition_a", value=True))
    graph.add_node(BooleanInvocation(id="condition_b", value=False))
    graph.add_node(PromptTestInvocation(id="true_a", prompt="true a"))
    graph.add_node(PromptTestInvocation(id="false_a", prompt="false a"))
    graph.add_node(PromptTestInvocation(id="true_b", prompt="true b"))
    graph.add_node(PromptTestInvocation(id="false_b", prompt="false b"))
    graph.add_node(IfInvocation(id="if_a"))
    graph.add_node(IfInvocation(id="if_b"))
    graph.add_node(CollectInvocation(id="collect"))

    graph.add_edge(create_edge("condition_a", "value", "if_a", "condition"))
    graph.add_edge(create_edge("true_a", "prompt", "if_a", "true_input"))
    graph.add_edge(create_edge("false_a", "prompt", "if_a", "false_input"))
    graph.add_edge(create_edge("condition_b", "value", "if_b", "condition"))
    graph.add_edge(create_edge("true_b", "prompt", "if_b", "true_input"))
    graph.add_edge(create_edge("false_b", "prompt", "if_b", "false_input"))
    graph.add_edge(create_edge("if_a", "value", "collect", "item"))
    graph.add_edge(create_edge("if_b", "value", "collect", "item"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    prepared_collect_id = next(iter(g.source_prepared_mapping["collect"]))
    assert sorted(g.results[prepared_collect_id].collection) == ["false b", "true a"]
    assert set(executed_source_ids) == {
        "condition_a",
        "condition_b",
        "true_a",
        "false_b",
        "if_a",
        "if_b",
        "collect",
    }
    assert "false_a" not in executed_source_ids
    assert "true_b" not in executed_source_ids


def test_if_graph_optimized_behavior_supports_nested_ifs():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="outer_condition", value=True))
    graph.add_node(BooleanInvocation(id="inner_condition", value=False))
    graph.add_node(PromptTestInvocation(id="outer_false", prompt="outer false"))
    graph.add_node(PromptTestInvocation(id="inner_true", prompt="inner true"))
    graph.add_node(PromptTestInvocation(id="inner_false", prompt="inner false"))
    graph.add_node(IfInvocation(id="inner_if"))
    graph.add_node(IfInvocation(id="outer_if"))
    graph.add_node(PromptTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("inner_condition", "value", "inner_if", "condition"))
    graph.add_edge(create_edge("inner_true", "prompt", "inner_if", "true_input"))
    graph.add_edge(create_edge("inner_false", "prompt", "inner_if", "false_input"))
    graph.add_edge(create_edge("outer_condition", "value", "outer_if", "condition"))
    graph.add_edge(create_edge("inner_if", "value", "outer_if", "true_input"))
    graph.add_edge(create_edge("outer_false", "prompt", "outer_if", "false_input"))
    graph.add_edge(create_edge("outer_if", "value", "selected_output", "prompt"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    prepared_selected_output_id = next(iter(g.source_prepared_mapping["selected_output"]))
    assert g.results[prepared_selected_output_id].prompt == "inner false"
    assert set(executed_source_ids) == {
        "outer_condition",
        "inner_condition",
        "inner_false",
        "inner_if",
        "outer_if",
        "selected_output",
    }
    assert "inner_true" not in executed_source_ids
    assert "outer_false" not in executed_source_ids


def test_if_graph_optimized_behavior_prunes_branches_per_iteration():
    graph = Graph()
    graph.add_node(BooleanCollectionInvocation(id="conditions", collection=[True, False, True]))
    graph.add_node(IterateInvocation(id="condition_iter"))
    graph.add_node(AnyTypeTestInvocation(id="true_branch"))
    graph.add_node(AnyTypeTestInvocation(id="false_branch"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(CollectInvocation(id="collect"))

    graph.add_edge(create_edge("conditions", "collection", "condition_iter", "collection"))
    graph.add_edge(create_edge("condition_iter", "item", "if", "condition"))
    graph.add_edge(create_edge("condition_iter", "item", "true_branch", "value"))
    graph.add_edge(create_edge("true_branch", "value", "if", "true_input"))
    graph.add_edge(create_edge("condition_iter", "item", "false_branch", "value"))
    graph.add_edge(create_edge("false_branch", "value", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "collect", "item"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    prepared_collect_id = next(iter(g.source_prepared_mapping["collect"]))
    assert g.results[prepared_collect_id].collection == [True, False, True]
    assert executed_source_ids.count("condition_iter") == 3
    assert executed_source_ids.count("true_branch") == 2
    assert executed_source_ids.count("false_branch") == 1
    assert executed_source_ids.count("if") == 3


def test_if_graph_optimized_behavior_keeps_shared_live_consumers_per_iteration():
    graph = Graph()
    graph.add_node(BooleanCollectionInvocation(id="conditions", collection=[True, False, False]))
    graph.add_node(IterateInvocation(id="condition_iter"))
    graph.add_node(AnyTypeTestInvocation(id="shared_branch"))
    graph.add_node(AnyTypeTestInvocation(id="true_leaf"))
    graph.add_node(AnyTypeTestInvocation(id="false_branch"))
    graph.add_node(AnyTypeTestInvocation(id="observer"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(CollectInvocation(id="selected_collect"))
    graph.add_node(CollectInvocation(id="observer_collect"))

    graph.add_edge(create_edge("conditions", "collection", "condition_iter", "collection"))
    graph.add_edge(create_edge("condition_iter", "item", "if", "condition"))
    graph.add_edge(create_edge("condition_iter", "item", "shared_branch", "value"))
    graph.add_edge(create_edge("shared_branch", "value", "true_leaf", "value"))
    graph.add_edge(create_edge("true_leaf", "value", "if", "true_input"))
    graph.add_edge(create_edge("condition_iter", "item", "false_branch", "value"))
    graph.add_edge(create_edge("false_branch", "value", "if", "false_input"))
    graph.add_edge(create_edge("shared_branch", "value", "observer", "value"))
    graph.add_edge(create_edge("if", "value", "selected_collect", "item"))
    graph.add_edge(create_edge("observer", "value", "observer_collect", "item"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    prepared_selected_collect_id = next(iter(g.source_prepared_mapping["selected_collect"]))
    assert g.results[prepared_selected_collect_id].collection == [True, False, False]
    prepared_observer_collect_id = next(iter(g.source_prepared_mapping["observer_collect"]))
    assert g.results[prepared_observer_collect_id].collection == [True, False, False]

    assert executed_source_ids.count("condition_iter") == 3
    assert executed_source_ids.count("shared_branch") == 3
    assert executed_source_ids.count("observer") == 3
    assert executed_source_ids.count("true_leaf") == 1
    assert executed_source_ids.count("false_branch") == 2


def test_if_graph_optimized_behavior_handles_selected_true_branch_with_shared_false_input_ancestor():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(AnyTypeTestInvocation(id="shared_item", value="shared"))
    graph.add_node(AnyTypeTestInvocation(id="true_item", value="true"))
    graph.add_node(CollectInvocation(id="shared_collect"))
    graph.add_node(CollectInvocation(id="true_collect"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AnyTypeTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("shared_item", "value", "shared_collect", "item"))
    graph.add_edge(create_edge("shared_collect", "collection", "true_collect", "collection"))
    graph.add_edge(create_edge("true_item", "value", "true_collect", "item"))
    graph.add_edge(create_edge("shared_collect", "collection", "if", "false_input"))
    graph.add_edge(create_edge("true_collect", "collection", "if", "true_input"))
    graph.add_edge(create_edge("if", "value", "selected_output", "value"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    prepared_selected_output_id = next(iter(g.source_prepared_mapping["selected_output"]))
    assert g.results[prepared_selected_output_id].value == ["shared", "true"]
    assert set(executed_source_ids) == {
        "condition",
        "shared_item",
        "true_item",
        "shared_collect",
        "true_collect",
        "if",
        "selected_output",
    }


def test_if_graph_optimized_behavior_handles_selected_false_branch_with_shared_true_input_ancestor():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=False))
    graph.add_node(AnyTypeTestInvocation(id="shared_item", value="shared"))
    graph.add_node(AnyTypeTestInvocation(id="true_item", value="true"))
    graph.add_node(CollectInvocation(id="shared_collect"))
    graph.add_node(CollectInvocation(id="true_collect"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AnyTypeTestInvocation(id="selected_output"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("shared_item", "value", "shared_collect", "item"))
    graph.add_edge(create_edge("shared_collect", "collection", "true_collect", "collection"))
    graph.add_edge(create_edge("true_item", "value", "true_collect", "item"))
    graph.add_edge(create_edge("shared_collect", "collection", "if", "false_input"))
    graph.add_edge(create_edge("true_collect", "collection", "if", "true_input"))
    graph.add_edge(create_edge("if", "value", "selected_output", "value"))

    g = GraphExecutionState(graph=graph)
    executed_source_ids = execute_all_nodes(g)

    prepared_selected_output_id = next(iter(g.source_prepared_mapping["selected_output"]))
    assert g.results[prepared_selected_output_id].value == ["shared"]
    assert set(executed_source_ids) == {
        "condition",
        "shared_item",
        "shared_collect",
        "if",
        "selected_output",
    }
    assert "true_item" not in executed_source_ids
    assert "true_collect" not in executed_source_ids


def test_prepare_if_inputs_raises_when_selected_branch_source_has_no_result():
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=True))
    graph.add_node(PromptTestInvocation(id="true_value", prompt="true branch"))
    graph.add_node(IfInvocation(id="if"))

    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("true_value", "prompt", "if", "true_input"))

    g = GraphExecutionState(graph=graph)

    condition_exec_id = g._create_execution_node("condition", [])[0]
    true_value_exec_id = g._create_execution_node("true_value", [])[0]
    if_exec_id = g._create_execution_node(
        "if",
        [("condition", condition_exec_id), ("true_value", true_value_exec_id)],
    )[0]

    g.executed.add(condition_exec_id)
    g.results[condition_exec_id] = BooleanOutput(value=True)
    g.executed.add(true_value_exec_id)
    g._resolved_if_exec_branches[if_exec_id] = "true_input"

    if_node = g.execution_graph.get_node(if_exec_id)
    with pytest.raises(RuntimeError) as exc_info:
        g._prepare_inputs(if_node)

    message = str(exc_info.value)
    assert if_exec_id in message
    assert true_value_exec_id in message
    assert "iteration_path=()" in message


def test_get_collect_iteration_mappings_ignores_skipped_prepared_exec_nodes():
    graph = Graph()
    graph.add_node(AnyTypeTestInvocation(id="parent", value="value"))

    g = GraphExecutionState(graph=graph)

    skipped_exec_id = g._create_execution_node("parent", [])[0]
    active_exec_id = g._create_execution_node("parent", [])[0]
    g._set_prepared_exec_state(skipped_exec_id, "skipped")

    mappings = g._materializer()._get_collect_iteration_mappings(["parent"])

    assert mappings == [("parent", active_exec_id)]


def test_get_iteration_node_ignores_skipped_prepared_exec_nodes():
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="value", prompt="branch value"))

    g = GraphExecutionState(graph=graph)

    skipped_exec_id = g._create_execution_node("value", [])[0]
    active_exec_id = g._create_execution_node("value", [])[0]
    g._set_prepared_exec_state(skipped_exec_id, "skipped")

    selected_exec_id = g._get_iteration_node("value", graph.nx_graph_flat(), g.execution_graph.nx_graph_flat(), [])

    assert selected_exec_id == active_exec_id


def test_get_iteration_node_returns_single_active_prepared_exec_node():
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="value", prompt="branch value"))

    g = GraphExecutionState(graph=graph)

    active_exec_id = g._create_execution_node("value", [])[0]

    selected_exec_id = g._get_iteration_node("value", graph.nx_graph_flat(), g.execution_graph.nx_graph_flat(), [])

    assert selected_exec_id == active_exec_id


def test_get_iteration_node_returns_none_when_only_skipped_prepared_exec_nodes_exist():
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="value", prompt="branch value"))

    g = GraphExecutionState(graph=graph)

    skipped_exec_id = g._create_execution_node("value", [])[0]
    g._set_prepared_exec_state(skipped_exec_id, "skipped")

    selected_exec_id = g._get_iteration_node("value", graph.nx_graph_flat(), g.execution_graph.nx_graph_flat(), [])

    assert selected_exec_id is None


def test_get_iteration_node_does_not_reuse_wrong_iterator_when_only_other_iteration_is_live():
    graph = Graph()
    graph.add_node(BooleanCollectionInvocation(id="conditions", collection=[True, False]))
    graph.add_node(IterateInvocation(id="condition_iter"))
    graph.add_node(AnyTypeTestInvocation(id="value"))

    graph.add_edge(create_edge("conditions", "collection", "condition_iter", "collection"))
    graph.add_edge(create_edge("condition_iter", "item", "value", "value"))

    g = GraphExecutionState(graph=graph)

    conditions_exec_id = g._create_execution_node("conditions", [])[0]
    g.executed.add(conditions_exec_id)
    g.results[conditions_exec_id] = BooleanCollectionOutput(collection=[True, False])

    iterator_exec_ids = g._create_execution_node("condition_iter", [("conditions", conditions_exec_id)])
    assert len(iterator_exec_ids) == 2
    iterator_exec_ids_by_index = {g.execution_graph.get_node(exec_id).index: exec_id for exec_id in iterator_exec_ids}
    first_iter_exec_id = iterator_exec_ids_by_index[0]
    second_iter_exec_id = iterator_exec_ids_by_index[1]

    value_exec_ids = []
    value_exec_ids.extend(g._create_execution_node("value", [("condition_iter", first_iter_exec_id)]))
    value_exec_ids.extend(g._create_execution_node("value", [("condition_iter", second_iter_exec_id)]))
    assert len(value_exec_ids) == 2

    for exec_id in value_exec_ids:
        if g._get_iteration_path(exec_id) == (1,):
            active_value_exec_id = exec_id
        else:
            skipped_value_exec_id = exec_id

    g._set_prepared_exec_state(skipped_value_exec_id, "skipped")

    selected_exec_id = g._get_iteration_node(
        "value", graph.nx_graph_flat(), g.execution_graph.nx_graph_flat(), [first_iter_exec_id]
    )

    assert selected_exec_id is None
    assert active_value_exec_id != skipped_value_exec_id


def test_mark_exec_node_skipped_does_not_hide_already_executed_results():
    graph = Graph()
    graph.add_node(AnyTypeTestInvocation(id="value", value="value"))

    g = GraphExecutionState(graph=graph)

    exec_id = g._create_execution_node("value", [])[0]
    g.results[exec_id] = AnyTypeTestInvocationOutput(value="value")
    g.executed.add(exec_id)
    g._set_prepared_exec_state(exec_id, "executed")

    g._if_scheduler().mark_exec_node_skipped(exec_id)

    assert g._get_prepared_exec_metadata(exec_id).state == "executed"
    assert g.results[exec_id].value == "value"


def test_mark_exec_node_skipped_is_idempotent_for_skipped_state():
    graph = Graph()
    graph.add_node(AnyTypeTestInvocation(id="value", value="value"))

    g = GraphExecutionState(graph=graph)

    exec_id = g._create_execution_node("value", [])[0]

    g._if_scheduler().mark_exec_node_skipped(exec_id)
    g._if_scheduler().mark_exec_node_skipped(exec_id)

    assert g._get_prepared_exec_metadata(exec_id).state == "skipped"
    assert g.executed_history.count("value") == 1


def test_are_connection_types_compatible_accepts_subclass_to_base():
    """A subclass output should be connectable to a base-class input.

    This test exposes the bug where non-Union targets reject valid subclass connections.
    """
    from invokeai.app.services.shared.graph import are_connection_types_compatible

    class Base:
        pass

    class Child(Base):
        pass

    assert are_connection_types_compatible(Child, Base) is True
