from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Callable
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.collections import CollectionConcatInvocation, RangeInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import (
    ForInvocation,
    ForReturnInvocation,
    LoopState,
    StateSetInvocation,
)
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.primitives import BooleanInvocation, BooleanOutput
from invokeai.app.services.shared import graph as graph_module
from invokeai.app.services.shared.execution_state_migration import (
    CURRENT_EXECUTION_STATE_VERSION,
    UnsupportedExecutionStateVersionError,
    dump_execution_state,
    load_execution_state,
)
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    Edge,
    EdgeConnection,
    Graph,
    GraphExecutionState,
    IterateInvocation,
    _ExecutionScheduler,
    _GenericGraphSchedulerAdapter,
)
from invokeai.app.services.shared.graph_nested_iterate_planner import get_nested_iterate_sequence_depth
from invokeai.app.services.shared.invocation_context import InvocationContextData, build_invocation_context
from tests.test_nodes import (
    AnyTypeTestInvocation,
    ErrorInvocation,
    MarkedAnyTypeTestInvocation,
    PolymorphicStringTestInvocation,
    UnionCollectionTestInvocation,
    create_edge,
    create_loop_linkage,
)

FIXTURE_PATH = Path(__file__).parents[3] / "fixtures" / "execution_engine" / "static_dag_v1.json"


def _load_fixture(name: str = "static_dag_v1.json") -> GraphExecutionState:
    fixture_path = FIXTURE_PATH.with_name(name)
    with fixture_path.open(encoding="utf-8") as fixture:
        return load_execution_state(json.load(fixture))


def _run(
    state: GraphExecutionState,
    *,
    force_compatibility_scheduler: bool = False,
    stop_after: int | None = None,
    fail_source_id: str | None = None,
) -> tuple[list[str], GraphExecutionState]:
    if force_compatibility_scheduler:
        state._execution_scheduler = _ExecutionScheduler(state)

    trace: list[str] = []
    while (node := state.next()) is not None:
        source_id = state.prepared_source_mapping[node.id]
        trace.append(source_id)
        if source_id == fail_source_id:
            state.set_node_error(node.id, "injected failure")
            break
        state.complete(node.id, node.invoke(Mock()))
        if stop_after is not None and len(trace) == stop_after:
            break
    return trace, state


def _restore_compatibility_scheduler(state: GraphExecutionState) -> None:
    state._ready_queues = {}
    state._ready_node_ids = set()
    state._active_class = None
    state._execution_scheduler = _ExecutionScheduler(state)
    state._rehydrate_ready_queues()


def _nested_if_graph() -> Graph:
    """Build a mixed graph whose selected outer branch contains an inner If."""
    graph = Graph()
    graph.add_node(BooleanInvocation(id="outer_condition", value=True))
    graph.add_node(BooleanInvocation(id="inner_condition", value=False))
    graph.add_node(AddInvocation(id="inner_true", a=2, b=2))
    graph.add_node(AddInvocation(id="inner_false", a=3, b=3))
    graph.add_node(AddInvocation(id="outer_false", a=10, b=0))
    graph.add_node(IfInvocation(id="inner_if"))
    graph.add_node(IfInvocation(id="outer_if"))
    graph.add_node(AddInvocation(id="sink", b=1))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source, field=source_field),
                destination=EdgeConnection(node_id=destination, field=destination_field),
            )
        )

    connect("outer_condition", "value", "outer_if", "condition")
    connect("inner_condition", "value", "inner_if", "condition")
    connect("inner_true", "value", "inner_if", "true_input")
    connect("inner_false", "value", "inner_if", "false_input")
    connect("inner_if", "value", "outer_if", "true_input")
    connect("outer_false", "value", "outer_if", "false_input")
    connect("outer_if", "value", "sink", "a")
    return graph


def _nested_if_graph_with_shared_ancestor() -> Graph:
    graph = _nested_if_graph()
    graph.add_node(AddInvocation(id="shared", a=1, b=1))
    graph.add_edge(create_edge("shared", "value", "inner_true", "a"))
    graph.add_edge(create_edge("shared", "value", "inner_false", "a"))
    return graph


def _three_nested_if_graph(
    *,
    outer_condition: bool = True,
    middle_condition: bool = True,
    inner_condition: bool = True,
    middle_branch: str = "true_input",
    outer_branch: str = "true_input",
) -> Graph:
    assert middle_branch in {"true_input", "false_input"}
    assert outer_branch in {"true_input", "false_input"}
    middle_other_branch = "false_input" if middle_branch == "true_input" else "true_input"
    outer_other_branch = "false_input" if outer_branch == "true_input" else "true_input"
    middle_other_id = "middle_false" if middle_branch == "true_input" else "middle_true"
    outer_other_id = "outer_false" if outer_branch == "true_input" else "outer_true"
    graph = Graph()
    graph.add_node(BooleanInvocation(id="outer_condition", value=outer_condition))
    graph.add_node(BooleanInvocation(id="middle_condition", value=middle_condition))
    graph.add_node(BooleanInvocation(id="inner_condition", value=inner_condition))
    graph.add_node(AddInvocation(id="inner_true", a=2, b=2))
    graph.add_node(AddInvocation(id="inner_false", a=3, b=3))
    graph.add_node(AddInvocation(id=middle_other_id, a=10, b=0))
    graph.add_node(AddInvocation(id=outer_other_id, a=20, b=0))
    graph.add_node(IfInvocation(id="inner_if"))
    graph.add_node(IfInvocation(id="middle_if"))
    graph.add_node(IfInvocation(id="outer_if"))
    graph.add_node(AddInvocation(id="sink", b=1))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source, field=source_field),
                destination=EdgeConnection(node_id=destination, field=destination_field),
            )
        )

    connect("inner_condition", "value", "inner_if", "condition")
    connect("inner_true", "value", "inner_if", "true_input")
    connect("inner_false", "value", "inner_if", "false_input")
    connect("middle_condition", "value", "middle_if", "condition")
    connect("inner_if", "value", "middle_if", middle_branch)
    connect(middle_other_id, "value", "middle_if", middle_other_branch)
    connect("outer_condition", "value", "outer_if", "condition")
    connect("middle_if", "value", "outer_if", outer_branch)
    connect(outer_other_id, "value", "outer_if", outer_other_branch)
    connect("outer_if", "value", "sink", "a")
    return graph


def _three_nested_if_graph_with_middle_leaf_fanout(**kwargs: Any) -> Graph:
    graph = _three_nested_if_graph(**kwargs)
    graph.add_node(AddInvocation(id="middle_side_consumer", b=1))
    graph.add_edge(create_edge("middle_if", "value", "middle_side_consumer", "a"))
    return graph


def _three_nested_if_graph_with_middle_leaf_fanout_and_shared_ancestor(**kwargs: Any) -> Graph:
    graph = _three_nested_if_graph_with_middle_leaf_fanout(**kwargs)
    graph.add_node(AddInvocation(id="shared", a=1, b=1))
    graph.add_edge(create_edge("shared", "value", "inner_true", "a"))
    graph.add_edge(create_edge("shared", "value", "inner_false", "a"))
    return graph


def _three_nested_if_graph_with_extra_middle_fanout() -> Graph:
    graph = _three_nested_if_graph_with_middle_leaf_fanout()
    graph.add_node(AddInvocation(id="middle_side_consumer_2", b=2))
    graph.add_edge(create_edge("middle_if", "value", "middle_side_consumer_2", "a"))
    return graph


def _three_nested_if_graph_with_non_leaf_middle_fanout() -> Graph:
    graph = _three_nested_if_graph_with_middle_leaf_fanout()
    graph.add_node(AddInvocation(id="middle_side_tail", b=2))
    graph.add_edge(create_edge("middle_side_consumer", "value", "middle_side_tail", "a"))
    return graph


def _three_nested_if_graph_with_inner_fanout() -> Graph:
    graph = _three_nested_if_graph_with_middle_leaf_fanout()
    graph.add_node(AddInvocation(id="inner_side_consumer", b=2))
    graph.add_edge(create_edge("inner_if", "value", "inner_side_consumer", "a"))
    return graph


def _three_nested_if_graph_with_shared_ancestor(**kwargs: bool) -> Graph:
    graph = _three_nested_if_graph(**kwargs)
    graph.add_node(AddInvocation(id="shared", a=1, b=1))
    graph.add_edge(create_edge("shared", "value", "inner_true", "a"))
    graph.add_edge(create_edge("shared", "value", "inner_false", "a"))
    return graph


def _four_nested_if_graph(
    *,
    root_condition: bool = True,
    outer_condition: bool = True,
    middle_condition: bool = True,
    inner_condition: bool = True,
    middle_branch: str = "true_input",
    outer_branch: str = "true_input",
    root_branch: str = "true_input",
    with_shared_ancestor: bool = False,
) -> Graph:
    branches = (middle_branch, outer_branch, root_branch)
    assert all(branch in {"true_input", "false_input"} for branch in branches)

    graph = Graph()
    graph.add_node(BooleanInvocation(id="root_condition", value=root_condition))
    graph.add_node(BooleanInvocation(id="outer_condition", value=outer_condition))
    graph.add_node(BooleanInvocation(id="middle_condition", value=middle_condition))
    graph.add_node(BooleanInvocation(id="inner_condition", value=inner_condition))
    graph.add_node(AddInvocation(id="inner_true", b=2))
    graph.add_node(AddInvocation(id="inner_false", b=3))
    graph.add_node(
        AddInvocation(
            id="middle_false" if middle_branch == "true_input" else "middle_true",
            a=10,
            b=0,
        )
    )
    graph.add_node(
        AddInvocation(
            id="outer_false" if outer_branch == "true_input" else "outer_true",
            a=20,
            b=0,
        )
    )
    graph.add_node(
        AddInvocation(
            id="root_false" if root_branch == "true_input" else "root_true",
            a=30,
            b=0,
        )
    )
    graph.add_node(IfInvocation(id="inner_if"))
    graph.add_node(IfInvocation(id="middle_if"))
    graph.add_node(IfInvocation(id="outer_if"))
    graph.add_node(IfInvocation(id="root_if"))
    graph.add_node(AddInvocation(id="sink", b=1))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("root_condition", "value", "root_if", "condition")
    connect("outer_condition", "value", "outer_if", "condition")
    connect("middle_condition", "value", "middle_if", "condition")
    connect("inner_condition", "value", "inner_if", "condition")
    connect("inner_true", "value", "inner_if", "true_input")
    connect("inner_false", "value", "inner_if", "false_input")
    connect("inner_if", "value", "middle_if", middle_branch)
    connect(
        "middle_false" if middle_branch == "true_input" else "middle_true",
        "value",
        "middle_if",
        "false_input" if middle_branch == "true_input" else "true_input",
    )
    connect("middle_if", "value", "outer_if", outer_branch)
    connect(
        "outer_false" if outer_branch == "true_input" else "outer_true",
        "value",
        "outer_if",
        "false_input" if outer_branch == "true_input" else "true_input",
    )
    connect("outer_if", "value", "root_if", root_branch)
    connect(
        "root_false" if root_branch == "true_input" else "root_true",
        "value",
        "root_if",
        "false_input" if root_branch == "true_input" else "true_input",
    )
    connect("root_if", "value", "sink", "a")

    if with_shared_ancestor:
        graph.add_node(AddInvocation(id="shared", a=1, b=1))
        connect("shared", "value", "inner_true", "a")
        connect("shared", "value", "inner_false", "a")
    else:
        graph.get_node("inner_true").a = 2
        graph.get_node("inner_false").a = 3
    return graph


def _flat_if_graph(*, condition: bool = True) -> Graph:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=condition))
    graph.add_node(AddInvocation(id="true_branch", a=2, b=3))
    graph.add_node(AddInvocation(id="false_branch", a=10, b=20))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AddInvocation(id="sink", b=1))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source, field=source_field),
                destination=EdgeConnection(node_id=destination, field=destination_field),
            )
        )

    connect("condition", "value", "if", "condition")
    connect("true_branch", "value", "if", "true_input")
    connect("false_branch", "value", "if", "false_input")
    connect("if", "value", "sink", "a")
    return graph


def _noncanonical_flat_if_graph(*, condition: bool = True) -> Graph:
    graph = _flat_if_graph(condition=condition)
    graph.delete_node("true_branch")
    graph.add_node(AddInvocation(id="true_source", a=2, b=3))
    graph.add_node(AddInvocation(id="true_branch", b=4))
    graph.add_edge(create_edge("true_source", "value", "true_branch", "a"))
    graph.add_edge(create_edge("true_branch", "value", "if", "true_input"))
    return graph


def _sibling_if_graph(
    *, first_condition: bool = True, second_condition: bool = True, with_shared_ancestor: bool = False
) -> Graph:
    graph = Graph()
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

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source, field=source_field),
                destination=EdgeConnection(node_id=destination, field=destination_field),
            )
        )

    if with_shared_ancestor:
        graph.add_node(AddInvocation(id="shared", a=1, b=1))
        connect("shared", "value", "first_true", "a")
        connect("shared", "value", "second_true", "a")
    connect("first_condition", "value", "first_if", "condition")
    connect("first_true", "value", "first_if", "true_input")
    connect("first_false", "value", "first_if", "false_input")
    connect("second_condition", "value", "second_if", "condition")
    connect("second_true", "value", "second_if", "true_input")
    connect("second_false", "value", "second_if", "false_input")
    connect("first_if", "value", "first_sink", "a")
    connect("second_if", "value", "second_sink", "a")
    return graph


def _three_sibling_if_graph(
    *,
    first_condition: bool = True,
    second_condition: bool = True,
    third_condition: bool = True,
    with_shared_ancestor: bool = False,
) -> Graph:
    graph = _sibling_if_graph(
        first_condition=first_condition,
        second_condition=second_condition,
        with_shared_ancestor=with_shared_ancestor,
    )
    graph.add_node(BooleanInvocation(id="third_condition", value=third_condition))
    graph.add_node(AddInvocation(id="third_true", b=5))
    graph.add_node(AddInvocation(id="third_false", a=30, b=1))
    graph.add_node(IfInvocation(id="third_if"))
    graph.add_node(AddInvocation(id="third_sink", b=300))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source, field=source_field),
                destination=EdgeConnection(node_id=destination, field=destination_field),
            )
        )

    if with_shared_ancestor:
        connect("shared", "value", "third_true", "a")
    connect("third_condition", "value", "third_if", "condition")
    connect("third_true", "value", "third_if", "true_input")
    connect("third_false", "value", "third_if", "false_input")
    connect("third_if", "value", "third_sink", "a")
    return graph


def _four_sibling_if_graph(
    *,
    first_condition: bool = True,
    second_condition: bool = True,
    third_condition: bool = True,
    fourth_condition: bool = True,
    with_shared_ancestor: bool = False,
) -> Graph:
    graph = _three_sibling_if_graph(
        first_condition=first_condition,
        second_condition=second_condition,
        third_condition=third_condition,
        with_shared_ancestor=with_shared_ancestor,
    )
    graph.add_node(BooleanInvocation(id="fourth_condition", value=fourth_condition))
    graph.add_node(AddInvocation(id="fourth_true", b=6))
    graph.add_node(AddInvocation(id="fourth_false", a=40, b=1))
    graph.add_node(IfInvocation(id="fourth_if"))
    graph.add_node(AddInvocation(id="fourth_sink", b=400))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source, field=source_field),
                destination=EdgeConnection(node_id=destination, field=destination_field),
            )
        )

    if with_shared_ancestor:
        connect("shared", "value", "fourth_true", "a")
    connect("fourth_condition", "value", "fourth_if", "condition")
    connect("fourth_true", "value", "fourth_if", "true_input")
    connect("fourth_false", "value", "fourth_if", "false_input")
    connect("fourth_if", "value", "fourth_sink", "a")
    return graph


def _indirectly_connected_sibling_if_graph() -> Graph:
    graph = _sibling_if_graph()
    graph.delete_node("second_true")
    graph.add_node(AddInvocation(id="bridge", b=5))
    graph.add_edge(create_edge("first_if", "value", "bridge", "a"))
    graph.add_edge(create_edge("bridge", "value", "second_if", "true_input"))
    return graph


def _flat_for_graph(
    *,
    with_after: bool = False,
    collection: list[Any] | None = None,
    input_collection: list[Any] | None = None,
    body_returns_none: bool = False,
) -> Graph:
    graph = Graph()
    collection = [1, 2] if collection is None else collection
    graph.add_node(ForInvocation(id="for", collection=collection))
    if input_collection is not None:
        graph.add_node(AnyTypeTestInvocation(id="collection", value=input_collection))
    graph.add_node(UnionCollectionTestInvocation(id="body") if body_returns_none else AddInvocation(id="body", b=10))
    graph.add_node(ForReturnInvocation(id="return"))
    if with_after:
        graph.add_node(AnyTypeTestInvocation(id="after"))
    if input_collection is not None:
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id="collection", field="value"),
                destination=EdgeConnection(node_id="for", field="collection"),
            )
        )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="body", field="value" if body_returns_none else "a"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="body", field="value"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            type="loop_linkage",
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
        )
    )
    if with_after:
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id="for", field="output_collection"),
                destination=EdgeConnection(node_id="after", field="value"),
            )
        )
    return graph


def _nested_for_graph(*, outer_collection: list[list[str]] | None = None) -> Graph:
    graph = Graph()
    graph.add_node(
        ForInvocation(id="outer_for", collection=[["a", "b"], ["c"]] if outer_collection is None else outer_collection)
    )
    graph.add_node(ForInvocation(id="inner_for"))
    graph.add_node(AnyTypeTestInvocation(id="inner_body"))
    graph.add_node(ForReturnInvocation(id="inner_return"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    graph.add_node(AnyTypeTestInvocation(id="after"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source, field=source_field),
                destination=EdgeConnection(node_id=destination, field=destination_field),
            )
        )

    connect("outer_for", "item", "inner_for", "collection")
    connect("inner_for", "item", "inner_body", "value")
    connect("inner_body", "value", "inner_return", "output")
    connect("inner_for", "output_collection", "outer_return", "output")
    connect("outer_for", "output_collection", "after", "value")
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    graph.add_edge(create_loop_linkage("inner_for", "inner_return"))
    return graph


def _two_sibling_nested_for_fan_in_graph(
    *, outer_collection: list[list[str]] | None = None, outer_output_field: str = "output_collection"
) -> Graph:
    graph = Graph()
    graph.add_node(
        ForInvocation(id="outer_for", collection=[["a", "b"], ["c"]] if outer_collection is None else outer_collection)
    )
    graph.add_node(ForInvocation(id="left_for"))
    graph.add_node(ForInvocation(id="right_for"))
    graph.add_node(MarkedAnyTypeTestInvocation(id="left_body", marker="left:"))
    graph.add_node(MarkedAnyTypeTestInvocation(id="right_body", marker="right:"))
    graph.add_node(ForReturnInvocation(id="left_return"))
    graph.add_node(ForReturnInvocation(id="right_return"))
    graph.add_node(CollectionConcatInvocation(id="join"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    graph.add_node(AnyTypeTestInvocation(id="after"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("outer_for", "item", "left_for", "collection")
    connect("outer_for", "item", "right_for", "collection")
    connect("left_for", "item", "left_body", "value")
    connect("right_for", "item", "right_body", "value")
    connect("left_body", "value", "left_return", "output")
    connect("right_body", "value", "right_return", "output")
    connect("left_for", "output_collection", "join", "first")
    connect("right_for", "output_collection", "join", "second")
    connect("join", "collection", "outer_return", "output")
    connect("outer_for", outer_output_field, "after", "value")
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    graph.add_edge(create_loop_linkage("left_for", "left_return"))
    graph.add_edge(create_loop_linkage("right_for", "right_return"))
    return graph


def _three_level_nested_for_graph(*, outer_collection: list[list[list[str]]] | None = None) -> Graph:
    graph = Graph()
    graph.add_node(
        ForInvocation(
            id="outer_for",
            collection=[[["a", "b"], ["c"]], [["d"]]] if outer_collection is None else outer_collection,
        )
    )
    graph.add_node(ForInvocation(id="middle_for"))
    graph.add_node(ForInvocation(id="inner_for"))
    graph.add_node(AnyTypeTestInvocation(id="inner_body"))
    graph.add_node(ForReturnInvocation(id="inner_return"))
    graph.add_node(ForReturnInvocation(id="middle_return"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    graph.add_node(AnyTypeTestInvocation(id="after"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("outer_for", "item", "middle_for", "collection")
    connect("middle_for", "item", "inner_for", "collection")
    connect("inner_for", "item", "inner_body", "value")
    connect("inner_body", "value", "inner_return", "output")
    connect("inner_for", "output_collection", "middle_return", "output")
    connect("middle_for", "output_collection", "outer_return", "output")
    connect("outer_for", "output_collection", "after", "value")
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    graph.add_edge(create_loop_linkage("middle_for", "middle_return"))
    graph.add_edge(create_loop_linkage("inner_for", "inner_return"))
    return graph


def _four_level_nested_for_graph(*, outer_collection: list[list[list[list[str]]]] | None = None) -> Graph:
    graph = _three_level_nested_for_graph()
    graph.get_node("outer_for").collection = (
        [[[["a", "b"], ["c"]], [["d"]]], [[["e"]]]] if outer_collection is None else outer_collection
    )
    graph.delete_node("inner_body")
    graph.add_node(ForInvocation(id="deep_for"))
    graph.add_node(AnyTypeTestInvocation(id="deep_body"))
    graph.add_node(ForReturnInvocation(id="deep_return"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("inner_for", "item", "deep_for", "collection")
    connect("deep_for", "item", "deep_body", "value")
    connect("deep_body", "value", "deep_return", "output")
    connect("deep_for", "output_collection", "inner_return", "output")
    graph.add_edge(create_loop_linkage("deep_for", "deep_return"))
    return graph


def _sibling_nested_for_graph() -> Graph:
    graph = _three_level_nested_for_graph()
    middle_return_edge = next(
        edge for edge in graph._get_input_edges("middle_return", "output") if edge.source.node_id == "inner_for"
    )
    graph.delete_edge(middle_return_edge)
    graph.add_node(ForInvocation(id="sibling_for"))
    graph.add_node(AnyTypeTestInvocation(id="sibling_body"))
    graph.add_node(ForReturnInvocation(id="sibling_return"))
    graph.add_node(CollectionConcatInvocation(id="middle_join"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("middle_for", "item", "sibling_for", "collection")
    connect("sibling_for", "item", "sibling_body", "value")
    connect("sibling_body", "value", "sibling_return", "output")
    connect("inner_for", "output_collection", "middle_join", "first")
    connect("sibling_for", "output_collection", "middle_join", "second")
    connect("middle_join", "collection", "middle_return", "output")
    graph.add_edge(create_loop_linkage("sibling_for", "sibling_return"))
    return graph


def _input_driven_three_level_nested_for_graph() -> Graph:
    graph = _three_level_nested_for_graph()
    graph.get_node("outer_for").collection = []
    graph.add_node(CollectionConcatInvocation(id="outer_source", first=[[["a", "b"], ["c"]], [["d"]]]))
    graph.add_edge(create_edge("outer_source", "collection", "outer_for", "collection"))
    return graph


def _input_driven_three_level_nested_for_graph_with_collection(
    outer_collection: list[list[list[str]]],
) -> Graph:
    graph = _three_level_nested_for_graph(outer_collection=outer_collection)
    graph.get_node("outer_for").collection = []
    graph.add_node(CollectionConcatInvocation(id="outer_source", first=outer_collection))
    graph.add_edge(create_edge("outer_source", "collection", "outer_for", "collection"))
    return graph


def _input_driven_three_level_nested_for_graph_with_extra_node() -> Graph:
    graph = _input_driven_three_level_nested_for_graph()
    graph.add_node(AnyTypeTestInvocation(id="extra"))
    return graph


def _input_driven_nested_for_graph(*, outer_collection: list[list[str]] | None = None) -> Graph:
    graph = _nested_for_graph(outer_collection=outer_collection)
    graph.get_node("outer_for").collection = []
    graph.add_node(
        CollectionConcatInvocation(
            id="outer_source",
            first=[["a", "b"], ["c"]] if outer_collection is None else outer_collection,
        )
    )
    graph.add_edge(create_edge("outer_source", "collection", "outer_for", "collection"))
    return graph


def _malformed_three_level_nested_for_graph() -> Graph:
    graph = _three_level_nested_for_graph()
    outer_item_edge = next(
        edge for edge in graph._get_input_edges("middle_for", "collection") if edge.source.node_id == "outer_for"
    )
    graph.delete_edge(outer_item_edge)
    graph.add_node(AnyTypeTestInvocation(id="outer_bridge"))
    graph.add_edge(create_edge("outer_for", "item", "outer_bridge", "value"))
    graph.add_edge(create_edge("outer_bridge", "value", "middle_for", "collection"))
    return graph


def _input_driven_four_level_nested_for_graph() -> Graph:
    graph = _four_level_nested_for_graph()
    graph.add_node(CollectionConcatInvocation(id="outer_source", first=[[[["a"]]]]))
    graph.add_edge(create_edge("outer_source", "collection", "outer_for", "collection"))
    return graph


def _malformed_four_level_nested_for_graph() -> Graph:
    graph = _four_level_nested_for_graph()
    outer_item_edge = next(
        edge for edge in graph._get_input_edges("middle_for", "collection") if edge.source.node_id == "outer_for"
    )
    graph.delete_edge(outer_item_edge)
    graph.add_node(AnyTypeTestInvocation(id="outer_bridge"))
    graph.add_edge(create_edge("outer_for", "item", "outer_bridge", "value"))
    graph.add_edge(create_edge("outer_bridge", "value", "middle_for", "collection"))
    return graph


def _five_level_nested_for_graph() -> Graph:
    graph = _four_level_nested_for_graph()
    graph.delete_node("deep_body")
    graph.add_node(ForInvocation(id="deepest_for"))
    graph.add_node(AnyTypeTestInvocation(id="deepest_body"))
    graph.add_node(ForReturnInvocation(id="deepest_return"))
    graph.add_edge(create_edge("deep_for", "item", "deepest_for", "collection"))
    graph.add_edge(create_edge("deepest_for", "item", "deepest_body", "value"))
    graph.add_edge(create_edge("deepest_body", "value", "deepest_return", "output"))
    graph.add_edge(create_edge("deepest_for", "output_collection", "deep_return", "output"))
    graph.add_edge(create_loop_linkage("deepest_for", "deepest_return"))
    return graph


def _four_level_nested_for_without_consumer() -> Graph:
    graph = _four_level_nested_for_graph()
    graph.delete_node("after")
    return graph


def _four_level_nested_for_with_extra_node() -> Graph:
    graph = _four_level_nested_for_graph()
    graph.add_node(AnyTypeTestInvocation(id="extra"))
    return graph


def _nested_for_iterate_collect_graph(*, outer_collection: list[list[str]] | None = None) -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="outer_for", collection=outer_collection or [["a", "b"], ["c"]]))
    graph.add_node(PolymorphicStringTestInvocation(id="nested_collection"))
    graph.add_node(IterateInvocation(id="nested_iterate"))
    graph.add_node(AnyTypeTestInvocation(id="nested_body"))
    graph.add_node(CollectInvocation(id="nested_collect"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    graph.add_node(AnyTypeTestInvocation(id="after"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source, field=source_field),
                destination=EdgeConnection(node_id=destination, field=destination_field),
            )
        )

    connect("outer_for", "item", "nested_collection", "value")
    connect("nested_collection", "collection", "nested_iterate", "collection")
    connect("nested_iterate", "item", "nested_body", "value")
    connect("nested_body", "value", "nested_collect", "item")
    connect("nested_collect", "collection", "outer_return", "output")
    connect("outer_for", "output_collection", "after", "value")
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    return graph


def _input_driven_nested_for_iterate_collect_graph() -> Graph:
    graph = _nested_for_iterate_collect_graph()
    graph.add_node(CollectionConcatInvocation(id="outer_source", first=[["a", "b"], ["c"]]))
    graph.add_edge(create_edge("outer_source", "collection", "outer_for", "collection"))
    return graph


def _serial_two_level_nested_for_iterate_collect_graph(*, outer_collection: list[list[str]] | None = None) -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="outer_for", collection=outer_collection or [["a", "b"], ["c"]]))
    graph.add_node(PolymorphicStringTestInvocation(id="nested_collection_1"))
    graph.add_node(IterateInvocation(id="nested_iterate_1"))
    graph.add_node(PolymorphicStringTestInvocation(id="nested_collection_2"))
    graph.add_node(IterateInvocation(id="nested_iterate_2"))
    graph.add_node(AnyTypeTestInvocation(id="nested_body"))
    graph.add_node(CollectInvocation(id="nested_collect"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    graph.add_node(AnyTypeTestInvocation(id="after"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("outer_for", "item", "nested_collection_1", "value")
    connect("nested_collection_1", "collection", "nested_iterate_1", "collection")
    connect("nested_iterate_1", "item", "nested_collection_2", "value")
    connect("nested_collection_2", "collection", "nested_iterate_2", "collection")
    connect("nested_iterate_2", "item", "nested_body", "value")
    connect("nested_body", "value", "nested_collect", "item")
    connect("nested_collect", "collection", "outer_return", "output")
    connect("outer_for", "output_collection", "after", "value")
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    return graph


def _nested_iterate_chain_graph(*, outer_collection: list[list[str]] | None = None) -> Graph:
    """Existing nested-iterator shape: outer Iterate -> preparation -> inner Iterate -> body."""
    graph = Graph()
    graph.add_node(
        CollectionConcatInvocation(
            id="outer_source", first=[["a", "b"], ["c"]] if outer_collection is None else outer_collection
        )
    )
    graph.add_node(IterateInvocation(id="outer_iterate"))
    graph.add_node(PolymorphicStringTestInvocation(id="inner_collection"))
    graph.add_node(IterateInvocation(id="inner_iterate"))
    graph.add_node(AnyTypeTestInvocation(id="body"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("outer_source", "collection", "outer_iterate", "collection")
    connect("outer_iterate", "item", "inner_collection", "value")
    connect("inner_collection", "collection", "inner_iterate", "collection")
    connect("inner_iterate", "item", "body", "value")
    return graph


def _three_level_nested_iterate_chain_graph(*, outer_collection: list[list[list[str]]] | None = None) -> Graph:
    """Exact three-level serial nested-iterator shape."""
    graph = Graph()
    graph.add_node(
        CollectionConcatInvocation(
            id="outer_source",
            first=[[["a", "b"], ["c"]], [["d"]]] if outer_collection is None else outer_collection,
        )
    )
    graph.add_node(IterateInvocation(id="outer_iterate"))
    graph.add_node(CollectionConcatInvocation(id="middle_collection"))
    graph.add_node(IterateInvocation(id="middle_iterate"))
    graph.add_node(CollectionConcatInvocation(id="inner_collection"))
    graph.add_node(IterateInvocation(id="inner_iterate"))
    graph.add_node(AnyTypeTestInvocation(id="body"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("outer_source", "collection", "outer_iterate", "collection")
    connect("outer_iterate", "item", "middle_collection", "first")
    connect("middle_collection", "collection", "middle_iterate", "collection")
    connect("middle_iterate", "item", "inner_collection", "first")
    connect("inner_collection", "collection", "inner_iterate", "collection")
    connect("inner_iterate", "item", "body", "value")
    return graph


def _four_level_nested_iterate_chain_graph(*, outer_collection: list[list[list[list[str]]]] | None = None) -> Graph:
    """Exact four-level serial nested-iterator shape."""
    graph = Graph()
    graph.add_node(
        CollectionConcatInvocation(
            id="outer_source",
            first=[[[["a", "b"], ["c"]], [["d"]]], [[["e"]]]] if outer_collection is None else outer_collection,
        )
    )
    graph.add_node(IterateInvocation(id="outer_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level1_collection"))
    graph.add_node(IterateInvocation(id="level1_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level2_collection"))
    graph.add_node(IterateInvocation(id="level2_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level3_collection"))
    graph.add_node(IterateInvocation(id="level3_iterate"))
    graph.add_node(AnyTypeTestInvocation(id="body"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("outer_source", "collection", "outer_iterate", "collection")
    connect("outer_iterate", "item", "level1_collection", "first")
    connect("level1_collection", "collection", "level1_iterate", "collection")
    connect("level1_iterate", "item", "level2_collection", "first")
    connect("level2_collection", "collection", "level2_iterate", "collection")
    connect("level2_iterate", "item", "level3_collection", "first")
    connect("level3_collection", "collection", "level3_iterate", "collection")
    connect("level3_iterate", "item", "body", "value")
    return graph


def _five_level_nested_iterate_chain_graph(
    *, outer_collection: list[list[list[list[list[str]]]]] | None = None
) -> Graph:
    """Exact five-level serial nested-iterator shape."""
    graph = Graph()
    graph.add_node(
        CollectionConcatInvocation(
            id="outer_source",
            first=[
                [
                    [[["a", "b"]], [["c"]]],
                    [[["d"]]],
                ],
                [[[["e"]]]],
            ]
            if outer_collection is None
            else outer_collection,
        )
    )
    graph.add_node(IterateInvocation(id="outer_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level1_collection"))
    graph.add_node(IterateInvocation(id="level1_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level2_collection"))
    graph.add_node(IterateInvocation(id="level2_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level3_collection"))
    graph.add_node(IterateInvocation(id="level3_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level4_collection"))
    graph.add_node(IterateInvocation(id="level4_iterate"))
    graph.add_node(AnyTypeTestInvocation(id="body"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("outer_source", "collection", "outer_iterate", "collection")
    connect("outer_iterate", "item", "level1_collection", "first")
    connect("level1_collection", "collection", "level1_iterate", "collection")
    connect("level1_iterate", "item", "level2_collection", "first")
    connect("level2_collection", "collection", "level2_iterate", "collection")
    connect("level2_iterate", "item", "level3_collection", "first")
    connect("level3_collection", "collection", "level3_iterate", "collection")
    connect("level3_iterate", "item", "level4_collection", "first")
    connect("level4_collection", "collection", "level4_iterate", "collection")
    connect("level4_iterate", "item", "body", "value")
    return graph


def _six_level_nested_iterate_chain_graph(
    *, outer_collection: list[list[list[list[list[list[str]]]]]] | None = None
) -> Graph:
    """Exact six-level serial nested-iterator shape."""
    graph = Graph()
    graph.add_node(
        CollectionConcatInvocation(
            id="outer_source",
            first=[
                [
                    [
                        [
                            [
                                [
                                    "a",
                                    "b",
                                ]
                            ],
                            [["c"]],
                        ],
                        [
                            [
                                [
                                    "d",
                                ]
                            ]
                        ],
                    ],
                ],
                [[[[["e"]]]]],
            ]
            if outer_collection is None
            else outer_collection,
        )
    )
    graph.add_node(IterateInvocation(id="outer_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level1_collection"))
    graph.add_node(IterateInvocation(id="level1_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level2_collection"))
    graph.add_node(IterateInvocation(id="level2_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level3_collection"))
    graph.add_node(IterateInvocation(id="level3_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level4_collection"))
    graph.add_node(IterateInvocation(id="level4_iterate"))
    graph.add_node(CollectionConcatInvocation(id="level5_collection"))
    graph.add_node(IterateInvocation(id="level5_iterate"))
    graph.add_node(AnyTypeTestInvocation(id="body"))

    def connect(source: str, source_field: str, destination: str, destination_field: str) -> None:
        graph.add_edge(create_edge(source, source_field, destination, destination_field))

    connect("outer_source", "collection", "outer_iterate", "collection")
    connect("outer_iterate", "item", "level1_collection", "first")
    connect("level1_collection", "collection", "level1_iterate", "collection")
    connect("level1_iterate", "item", "level2_collection", "first")
    connect("level2_collection", "collection", "level2_iterate", "collection")
    connect("level2_iterate", "item", "level3_collection", "first")
    connect("level3_collection", "collection", "level3_iterate", "collection")
    connect("level3_iterate", "item", "level4_collection", "first")
    connect("level4_collection", "collection", "level4_iterate", "collection")
    connect("level4_iterate", "item", "level5_collection", "first")
    connect("level5_collection", "collection", "level5_iterate", "collection")
    connect("level5_iterate", "item", "body", "value")
    return graph


def _serial_nested_iterate_chain_graph_at_depth(iterate_count: int) -> Graph:
    """Build one non-empty serial nested-iterator chain for admission-boundary tests."""
    collection: list[Any] = ["value"]
    for _ in range(iterate_count - 1):
        collection = [collection]

    graph = Graph()
    graph.add_node(CollectionConcatInvocation(id="source", first=collection))
    previous_iterate_id = "source"
    previous_field = "collection"
    for level in range(iterate_count):
        iterate_id = f"iterate_{level}"
        graph.add_node(IterateInvocation(id=iterate_id))
        graph.add_edge(create_edge(previous_iterate_id, previous_field, iterate_id, "collection"))
        if level == iterate_count - 1:
            graph.add_node(AnyTypeTestInvocation(id="body"))
            graph.add_edge(create_edge(iterate_id, "item", "body", "value"))
        else:
            preparation_id = f"preparation_{level}"
            graph.add_node(CollectionConcatInvocation(id=preparation_id))
            graph.add_edge(create_edge(iterate_id, "item", preparation_id, "first"))
            previous_iterate_id = preparation_id
            previous_field = "collection"
    return graph


def _direct_iterate_body_collect_graph(
    *, collection: list[Any] | None = None, with_after: bool = False, downstream_count: int | None = None
) -> Graph:
    graph = Graph()
    graph.add_node(CollectionConcatInvocation(id="source", first=[] if collection is None else collection))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(AnyTypeTestInvocation(id="body"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="source", field="collection"),
            destination=EdgeConnection(node_id="iterate", field="collection"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="iterate", field="item"),
            destination=EdgeConnection(node_id="body", field="value"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="body", field="value"),
            destination=EdgeConnection(node_id="collect", field="item"),
        )
    )
    if downstream_count is None:
        downstream_count = 1 if with_after else 0
    for downstream_index in range(downstream_count):
        downstream_id = "after" if downstream_count == 1 else f"after_{downstream_index + 1}"
        graph.add_node(AnyTypeTestInvocation(id=downstream_id))
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id="collect", field="collection"),
                destination=EdgeConnection(node_id=downstream_id, field="value"),
            )
        )
    return graph


def _input_driven_iterate_body_collect_graph(values: list[Any] | None = None) -> Graph:
    graph = Graph()
    graph.add_node(CollectionConcatInvocation(id="producer", first=[1, 2, 3] if values is None else values))
    graph.add_node(CollectionConcatInvocation(id="source"))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(AnyTypeTestInvocation(id="body"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_node(AnyTypeTestInvocation(id="after"))

    graph.add_edge(create_edge("producer", "collection", "source", "first"))
    graph.add_edge(create_edge("source", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "body", "value"))
    graph.add_edge(create_edge("body", "value", "collect", "item"))
    graph.add_edge(create_edge("collect", "collection", "after", "value"))
    return graph


def _direct_iterate_fan_in_graph(*, left: list[Any], right: list[Any]) -> Graph:
    graph = Graph()
    graph.add_node(CollectionConcatInvocation(id="left_source", first=left))
    graph.add_node(CollectionConcatInvocation(id="right_source", first=right))
    graph.add_node(IterateInvocation(id="left_iterate"))
    graph.add_node(IterateInvocation(id="right_iterate"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="left_source", field="collection"),
            destination=EdgeConnection(node_id="left_iterate", field="collection"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="right_source", field="collection"),
            destination=EdgeConnection(node_id="right_iterate", field="collection"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="left_iterate", field="item"),
            destination=EdgeConnection(node_id="collect", field="item"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="right_iterate", field="item"),
            destination=EdgeConnection(node_id="collect", field="item"),
        )
    )
    return graph


def _body_iterate_fan_in_graph(*, branches: tuple[tuple[str, str, str, list[Any]], ...]) -> Graph:
    graph = Graph()
    graph.add_node(CollectInvocation(id="collect"))
    for source_id, iterate_id, body_id, values in branches:
        graph.add_node(CollectionConcatInvocation(id=source_id, first=values))
        graph.add_node(IterateInvocation(id=iterate_id))
        graph.add_node(AnyTypeTestInvocation(id=body_id))
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source_id, field="collection"),
                destination=EdgeConnection(node_id=iterate_id, field="collection"),
            )
        )
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=iterate_id, field="item"),
                destination=EdgeConnection(node_id=body_id, field="value"),
            )
        )
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=body_id, field="value"),
                destination=EdgeConnection(node_id="collect", field="item"),
            )
        )
    return graph


def _direct_iterate_fan_in_graph_with_branches(
    branches: tuple[tuple[str, str, list[Any]], ...],
) -> Graph:
    graph = Graph()
    graph.add_node(CollectInvocation(id="collect"))
    for source_id, iterator_id, values in branches:
        graph.add_node(CollectionConcatInvocation(id=source_id, first=values))
        graph.add_node(IterateInvocation(id=iterator_id))
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=source_id, field="collection"),
                destination=EdgeConnection(node_id=iterator_id, field="collection"),
            )
        )
        graph.add_edge(
            Edge(
                source=EdgeConnection(node_id=iterator_id, field="item"),
                destination=EdgeConnection(node_id="collect", field="item"),
            )
        )
    return graph


def _direct_iterate_three_fan_in_graph(
    *,
    a: list[Any],
    m: list[Any],
    z: list[Any],
    unsorted: bool = False,
) -> Graph:
    branches = (
        ("a_source", "a_iterate", a),
        ("m_source", "m_iterate", m),
        ("z_source", "z_iterate", z),
    )
    if unsorted:
        branches = (branches[2], branches[0], branches[1])
    return _direct_iterate_fan_in_graph_with_branches(
        branches=branches,
    )


def _flat_for_state_graph(continue_condition: bool) -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=[1, 2, 3], state=LoopState(values={"count": 0})))
    graph.add_node(StateSetInvocation(id="body", key="count"))
    graph.add_node(ForReturnInvocation(id="return", continue_condition=continue_condition))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="state"),
            destination=EdgeConnection(node_id="body", field="state"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="body", field="value"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="for", field="item"),
            destination=EdgeConnection(node_id="return", field="output"),
        )
    )
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="body", field="state"),
            destination=EdgeConnection(node_id="return", field="state"),
        )
    )
    graph.add_edge(
        Edge(
            type="loop_linkage",
            source=EdgeConnection(node_id="for", field="loop_linkage"),
            destination=EdgeConnection(node_id="return", field="loop_linkage"),
        )
    )
    return graph


def _run_graph(
    state: GraphExecutionState,
    *,
    force_compatibility_scheduler: bool = False,
    stop_after: int | None = None,
    fail_source_id: str | None = None,
) -> tuple[list[str], GraphExecutionState]:
    """Run a constructed graph through either scheduler path."""
    if force_compatibility_scheduler:
        state._execution_scheduler = _ExecutionScheduler(state)

    trace: list[str] = []
    while (node := state.next()) is not None:
        source_id = state.prepared_source_mapping[node.id]
        trace.append(source_id)
        if source_id == fail_source_id:
            state.set_node_error(node.id, "injected failure")
            break
        state.complete(node.id, node.invoke(Mock()))
        if stop_after is not None and len(trace) == stop_after:
            break
    return trace, state


def _run_graph_with_effects(
    state: GraphExecutionState,
    *,
    force_compatibility_scheduler: bool = False,
    stop_after_source: str | None = None,
    fail_source_id: str | None = None,
    stop_after: int | None = None,
) -> tuple[list[str], GraphExecutionState]:
    """Run a graph through the same invocation/effect/apply path as the session runner."""
    if force_compatibility_scheduler:
        state._execution_scheduler = _ExecutionScheduler(state)

    services = Mock()
    services.invocation_cache.get.return_value = None
    trace: list[str] = []
    while (node := state.next()) is not None:
        source_id = state.prepared_source_mapping[node.id]
        trace.append(source_id)
        execution_ref = state.get_execution_ref(node.id)
        context = build_invocation_context(
            services=services,
            data=InvocationContextData(
                queue_item=None,  # type: ignore[arg-type]
                invocation=node,
                source_invocation_id=source_id,
                execution_frame=execution_ref.frame.iteration_path,
                execution_state_id=execution_ref.state_id,
                execution_frame_id=execution_ref.frame.frame_id,
                execution_workflow_call_depth=execution_ref.frame.workflow_call_depth,
            ),
            is_canceled=lambda: False,
        )
        try:
            if source_id == fail_source_id:
                with patch.object(
                    type(node),
                    "invoke_internal_with_effects",
                    side_effect=RuntimeError("injected failure"),
                ):
                    run_result = node.invoke_internal_with_effects(context, services)
            else:
                run_result = node.invoke_internal_with_effects(context, services)
        except RuntimeError as exc:
            state.set_node_error(node.id, str(exc))
            break
        state.apply(state.get_execution_ref(node.id, effect_count=len(run_result.effects)), run_result)
        if stop_after_source == source_id:
            break
        if stop_after is not None and len(trace) == stop_after:
            break
    return trace, state


def _run_until_source(
    state: GraphExecutionState,
    source_id_to_stop: str,
    *,
    force_compatibility_scheduler: bool = False,
) -> tuple[list[str], GraphExecutionState]:
    """Run through one source node, modeling a queue cancellation boundary."""
    if force_compatibility_scheduler:
        state._execution_scheduler = _ExecutionScheduler(state)

    trace: list[str] = []
    while (node := state.next()) is not None:
        source_id = state.prepared_source_mapping[node.id]
        trace.append(source_id)
        state.complete(node.id, node.invoke(Mock()))
        if source_id == source_id_to_stop:
            break
    return trace, state


def _run_both(**kwargs: Any) -> tuple[tuple[list[str], GraphExecutionState], tuple[list[str], GraphExecutionState]]:
    base_kwargs = {key: value for key, value in kwargs.items() if key != "force_compatibility_scheduler"}
    generic = _run(_load_fixture(), force_compatibility_scheduler=False, **base_kwargs)
    compatibility = _run(_load_fixture(), force_compatibility_scheduler=True, **base_kwargs)
    return generic, compatibility


def _state_projection(state: GraphExecutionState) -> tuple[Any, ...]:
    """Compare durable behavior without depending on generated execution-node IDs."""
    prepared_sources = tuple(sorted(state.prepared_source_mapping.values()))
    executed_prepared_sources = tuple(
        sorted(
            state.prepared_source_mapping[execution_id]
            for execution_id in state.executed
            if execution_id in state.prepared_source_mapping
        )
    )
    completed_sources = tuple(sorted(source_id for source_id in state.graph.nodes if source_id in state.executed))
    results = tuple(
        sorted(
            (
                state.prepared_source_mapping[execution_id],
                json.dumps(output.model_dump(mode="json"), sort_keys=True),
            )
            for execution_id, output in state.results.items()
            if execution_id in state.prepared_source_mapping
        )
    )
    indegree = tuple(
        sorted(
            (state.prepared_source_mapping[execution_id], degree)
            for execution_id, degree in state.indegree.items()
            if execution_id in state.prepared_source_mapping
        )
    )
    errors = tuple(
        sorted(
            (state.prepared_source_mapping.get(execution_id, execution_id), message)
            for execution_id, message in state.errors.items()
        )
    )
    return (
        prepared_sources,
        executed_prepared_sources,
        completed_sources,
        tuple(state.executed_history),
        results,
        indegree,
        errors,
        state.is_complete(),
    )


def _activation_projection(state: GraphExecutionState) -> tuple[tuple[str, str, Any, tuple[int, ...]], ...]:
    """Normalize activation tokens to source IDs and frame paths."""
    return tuple(
        sorted(
            (
                state.prepared_source_mapping[token.owner_node_id],
                token.port,
                token.value,
                tuple(token.frame.iteration_path),
            )
            for token in state.execution_tokens.values()
            if token.token_kind == "activation"
        )
    )


def _execution_identity_projection(state: GraphExecutionState) -> tuple[Any, ...]:
    def json_projection(value: Any) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(value, sort_keys=True)

    def normalize_effect_identity(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", warnings=False)
        if isinstance(value, dict):
            return {
                key: normalize_effect_identity(item)
                for key, item in value.items()
                if key
                not in {
                    "state_id",
                    "execution_node_id",
                    "frame_id",
                    "reference_id",
                    "template_node_id",
                    "token",
                }
            }
        if isinstance(value, list):
            return [normalize_effect_identity(item) for item in value]
        return value

    references = tuple(
        sorted(
            (
                state.prepared_source_mapping.get(exec_node_id, exec_node_id),
                reference.source_node_id,
                tuple(reference.frame.iteration_path),
                reference.frame.workflow_call_depth,
                reference.effect_count,
            )
            for exec_node_id, reference in state.execution_refs.items()
            if exec_node_id in state.prepared_source_mapping
        )
    )
    tokens = tuple(
        sorted(
            (
                state.prepared_source_mapping.get(token.owner_node_id, token.owner_node_id),
                token.port,
                json_projection(token.value),
                token.token_kind,
                token.sequence,
                tuple(token.frame.iteration_path),
                token.frame.workflow_call_depth,
            )
            for token in state.execution_tokens.values()
        )
    )
    effects = tuple(
        sorted(
            (
                state.prepared_source_mapping.get(
                    next(
                        (
                            execution_id
                            for execution_id, reference in state.execution_refs.items()
                            if reference.reference_id == reference_id
                        ),
                        reference_id,
                    ),
                    reference_id,
                ),
                json.dumps(
                    [normalize_effect_identity(effect) for effect in effect_values],
                    sort_keys=True,
                ),
            )
            for reference_id, effect_values in state.execution_effects.items()
        )
    )
    return references, tokens, effects


def _continuation_projection(state: GraphExecutionState) -> tuple[Any, ...]:
    return tuple(
        sorted(
            (
                state.prepared_source_mapping.get(continuation.owner_id, continuation.owner_id),
                continuation.kind,
                continuation.status,
                tuple(continuation.frame.iteration_path),
                continuation.frame.workflow_call_depth,
                json.dumps(continuation.payload, sort_keys=True),
                json.dumps(continuation.result, sort_keys=True),
                continuation.error,
            )
            for continuation in state._generic_runtime().continuations.values()
        )
    )


def _final_for_output(state: GraphExecutionState) -> Any:
    final_for_id = max(
        (
            exec_node_id
            for exec_node_id, source_node_id in state.prepared_source_mapping.items()
            if source_node_id == "for"
        ),
        key=lambda exec_node_id: state.execution_graph.get_node(exec_node_id).index,
    )
    return state.results[final_for_id]


def _source_output(state: GraphExecutionState, source_id: str) -> Any:
    execution_id = next(
        execution_id
        for execution_id, prepared_source_id in state.prepared_source_mapping.items()
        if prepared_source_id == source_id and execution_id in state.results
    )
    return state.results[execution_id]


def _execution_edge_projection(state: GraphExecutionState) -> tuple[tuple[str, str, str, str, str], ...]:
    return tuple(
        sorted(
            (
                state.prepared_source_mapping[edge.source.node_id],
                edge.source.field,
                state.prepared_source_mapping[edge.destination.node_id],
                edge.destination.field,
                edge.type,
            )
            for edge in state.execution_graph.edges
        )
    )


def _effect_ledger_projection(state: GraphExecutionState) -> tuple[Any, ...]:
    """Compare persisted effects while ignoring generated state/reference IDs."""

    def normalized(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (key, normalized(item))
                    for key, item in value.items()
                    if key
                    not in {
                        "state_id",
                        "execution_node_id",
                        "frame_id",
                        "reference_id",
                        "template_node_id",
                        "token",
                    }
                    and not (key == "execution_ref" and item is None)
                )
            )
        if isinstance(value, list):
            return tuple(normalized(item) for item in value)
        return value

    return tuple(
        sorted(
            (
                (
                    next(
                        (
                            state.prepared_source_mapping.get(exec_node_id, exec_node_id)
                            for exec_node_id, reference in state.execution_refs.items()
                            if reference.reference_id == reference_id
                        ),
                        reference_id,
                    ),
                    normalized(effects),
                )
                for reference_id, effects in state.execution_effects.items()
            ),
            key=repr,
        )
    )


def _execution_token_projection(state: GraphExecutionState) -> tuple[Any, ...]:
    """Project durable activation tokens without generated execution identities."""
    return tuple(
        sorted(
            (
                (
                    state.prepared_source_mapping.get(token.owner_node_id, token.owner_node_id),
                    token.port,
                    _stable_json(token.value),
                    token.token_kind,
                    token.sequence,
                    tuple(token.frame.iteration_path),
                    token.frame.workflow_call_depth,
                )
                for token in state.execution_tokens.values()
                if token.token_kind == "activation"
            ),
            key=repr,
        )
    )


def _stable_json(value: Any) -> str:
    def jsonable(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return jsonable(item.model_dump(mode="json", warnings=False))
        if isinstance(item, dict):
            return {str(key): jsonable(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonable(value) for value in item]
        if isinstance(item, set):
            return sorted((jsonable(value) for value in item), key=repr)
        return item

    return json.dumps(jsonable(value), sort_keys=True, default=str)


def _body_fan_in_graph_projection(state: GraphExecutionState) -> tuple[Any, ...]:
    nodes = tuple(sorted((node_id, _stable_json(node)) for node_id, node in state.execution_graph.nodes.items()))
    edges = tuple(
        sorted(
            (
                edge.source.node_id,
                edge.source.field,
                edge.destination.node_id,
                edge.destination.field,
                edge.type,
            )
            for edge in state.execution_graph.edges
        )
    )
    return nodes, edges


def _body_fan_in_durable_projection(state: GraphExecutionState) -> tuple[Any, ...]:
    """Exact pre/post transaction projection; generated IDs intentionally remain exact."""
    return (
        tuple(sorted(state.prepared_source_mapping.items())),
        tuple(
            sorted(
                (source_id, tuple(sorted(exec_ids))) for source_id, exec_ids in state.source_prepared_mapping.items()
            )
        ),
        tuple(sorted(state.prepared_iteration_paths.items())),
        tuple(sorted(state.finalized_loop_contexts)),
        _body_fan_in_graph_projection(state),
        tuple(sorted(state.indegree.items())),
        tuple(sorted(state.executed)),
        tuple(state.executed_history),
        tuple(sorted((exec_node_id, _stable_json(output)) for exec_node_id, output in state.results.items())),
        tuple(sorted(state.errors.items())),
        tuple(
            sorted((exec_node_id, _stable_json(reference)) for exec_node_id, reference in state.execution_refs.items())
        ),
        tuple(sorted((token_id, _stable_json(token)) for token_id, token in state.execution_tokens.items())),
        tuple(
            sorted((reference_id, _stable_json(effects)) for reference_id, effects in state.execution_effects.items())
        ),
    )


def _body_fan_in_runtime_projection(state: GraphExecutionState) -> tuple[Any, ...]:
    metadata = tuple(
        sorted(
            (
                exec_node_id,
                metadata.source_node_id,
                metadata.iteration_path,
                metadata.state,
            )
            for exec_node_id, metadata in state._prepared_exec_metadata.items()
        )
    )
    ready_queues = tuple(sorted((class_name, tuple(queue)) for class_name, queue in state._ready_queues.items()))
    return (
        _direct_iterate_fan_in_stream_projection(state),
        metadata,
        ready_queues,
        tuple(sorted(state._ready_node_ids)),
        state._active_class,
        tuple(state.ready_order),
    )


def _direct_iterate_fan_in_stream_projection(state: GraphExecutionState) -> tuple[Any, ...]:
    """Project fan-in streams without comparing state-generated UUID prefixes."""

    def value_projection(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return value

    streams = []
    for stream in state._generic_runtime().streams.values():
        assert stream.frame.state_id == state.id
        streams.append(
            (
                stream.stream_id.removeprefix(f"{state.id}:"),
                stream.owner_id,
                stream.frame.frame_id.removeprefix(f"{state.id}:"),
                tuple(stream.frame.iteration_path),
                stream.frame.workflow_call_depth,
                tuple(
                    (event.kind, event.sequence, value_projection(getattr(event, "value", None)))
                    for event in stream.events
                ),
                tuple(value_projection(value) for value in stream.values),
                stream.next_sequence,
                stream.closed,
                stream.end_sequence,
            )
        )
    return tuple(sorted(streams))


def _assert_retained_iterate_streams_match_expected(
    resumed_state: GraphExecutionState, expected_state: GraphExecutionState
) -> None:
    """Rehydration may omit finalized streams that have no consumer after loading."""
    expected_streams = {stream[0]: stream for stream in _direct_iterate_fan_in_stream_projection(expected_state)}
    retained_streams = _direct_iterate_fan_in_stream_projection(resumed_state)
    assert {stream[0] for stream in retained_streams} <= expected_streams.keys()
    assert all(stream == expected_streams[stream[0]] for stream in retained_streams)


def _direct_iterate_fan_in_execution_ref_projection(
    state: GraphExecutionState, *, execution_ids: set[str] | None = None
) -> tuple[Any, ...]:
    """Project fan-in execution references without comparing generated IDs."""

    refs = [
        (
            state.prepared_source_mapping.get(exec_node_id, exec_node_id),
            tuple(reference.frame.iteration_path),
            reference.frame.workflow_call_depth,
        )
        for exec_node_id, reference in state.execution_refs.items()
        if exec_node_id in state.prepared_source_mapping and (execution_ids is None or exec_node_id in execution_ids)
    ]
    return tuple(sorted(refs, key=lambda ref: (ref[0], ref[1], ref[2])))


def _source_edge_projection(graph: Graph) -> tuple[tuple[str, str, str, str, str], ...]:
    return tuple(
        sorted(
            (
                edge.source.node_id,
                edge.source.field,
                edge.destination.node_id,
                edge.destination.field,
                edge.type,
            )
            for edge in graph.edges
        )
    )


def _expected_edge_projection(
    graph: Graph,
    *,
    force_compatibility_scheduler: bool,
    outer_condition: bool,
    inner_condition: bool,
) -> tuple[tuple[str, str, str, str, str], ...]:
    """Project the live execution graph for either scheduler adapter."""
    del force_compatibility_scheduler
    live_edges = {("outer_condition", "value", "outer_if", "condition", "default")}
    if outer_condition:
        live_edges.add(("inner_condition", "value", "inner_if", "condition", "default"))
        live_edges.add(
            (
                "inner_true" if inner_condition else "inner_false",
                "value",
                "inner_if",
                "true_input" if inner_condition else "false_input",
                "default",
            )
        )
        live_edges.add(("inner_if", "value", "outer_if", "true_input", "default"))
    else:
        live_edges.add(("outer_false", "value", "outer_if", "false_input", "default"))
    live_edges.add(("outer_if", "value", "sink", "a", "default"))
    return tuple(sorted(live_edges))


def _normalized_indegree(state: GraphExecutionState) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        sorted(
            (
                source_id,
                tuple(
                    sorted(
                        degree
                        for execution_id, degree in state.indegree.items()
                        if state.prepared_source_mapping.get(execution_id) == source_id
                    )
                ),
            )
            for source_id in state.graph.nodes
        )
    )


def _expected_remaining_input_indegree(
    state: GraphExecutionState,
    *,
    force_compatibility_scheduler: bool,
    outer_condition: bool,
    inner_condition: bool,
) -> dict[str, int]:
    """Calculate remaining indegrees from an independent source-graph oracle."""
    expected_edges = _expected_edge_projection(
        state.graph,
        force_compatibility_scheduler=force_compatibility_scheduler,
        outer_condition=outer_condition,
        inner_condition=inner_condition,
    )
    prepared_by_source = {
        source_id: sorted(
            prepared_ids,
            key=lambda exec_id: (state._get_iteration_path(exec_id), exec_id),
        )
        for source_id, prepared_ids in state.source_prepared_mapping.items()
    }
    return {
        execution_id: sum(
            source_exec_id not in state.executed
            for source_id, _source_field, destination_id, _destination_field, _edge_type in expected_edges
            if destination_id == state.prepared_source_mapping[execution_id]
            for source_exec_id in prepared_by_source.get(source_id, ())
            if state._get_iteration_path(source_exec_id) == state._get_iteration_path(execution_id)
            and not (
                isinstance(state.execution_graph.get_node(execution_id), IfInvocation)
                and isinstance(state.execution_graph.get_node(source_exec_id), IfInvocation)
                and source_exec_id not in state.executed
            )
        )
        for execution_id in state.prepared_source_mapping
    }


def _assert_execution_identity_consistent(state: GraphExecutionState) -> None:
    def value_from_object(value: Any, *names: str) -> Any:
        if isinstance(value, dict):
            for name in names:
                if name in value:
                    return value[name]
            return None
        for name in names:
            candidate = getattr(value, name, None)
            if candidate is not None:
                return candidate
        return None

    for exec_node_id, reference in state.execution_refs.items():
        if exec_node_id not in state.prepared_source_mapping:
            continue
        expected = state._expected_execution_ref(exec_node_id, effect_count=reference.effect_count)
        assert reference.reference_id == expected.reference_id
        assert reference.state_id == expected.state_id
        assert reference.exec_node_id == expected.exec_node_id
        assert reference.source_node_id == expected.source_node_id
        assert reference.frame == expected.frame
    for token_key, token in state.execution_tokens.items():
        expected = state.execution_refs.get(token.owner_node_id) or state._expected_execution_ref(token.owner_node_id)
        assert token_key == token.token_id
        assert token.reference_id == expected.reference_id
        assert token.owner_node_id == expected.exec_node_id
        assert token.frame.state_id == expected.frame.state_id
        assert token.frame.frame_id == expected.frame.frame_id
        assert token.frame.iteration_path == expected.frame.iteration_path
        assert token.frame.workflow_call_depth == expected.frame.workflow_call_depth
        if token.token_kind == "activation":
            owner = state.execution_graph.nodes[token.owner_node_id]
            activation_fields = getattr(type(owner), "execution_activation_fields", frozenset())
            assert token.port in activation_fields
            assert token.token_id == f"{expected.reference_id}:activation:{token.port}"
            assert token.value == token.port
    references_by_id = {reference.reference_id: reference for reference in state.execution_refs.values()}
    for reference_id, effects in state.execution_effects.items():
        reference = references_by_id[reference_id]
        for effect in effects:
            effect_reference = value_from_object(
                effect,
                "execution_ref",
                "execution_reference",
                "owner_ref",
                "owner",
            )
            assert effect_reference is not None
            assert value_from_object(effect_reference, "state_id", "session_id") == reference.state_id
            assert value_from_object(effect_reference, "execution_node_id", "node_id") == reference.exec_node_id
            assert value_from_object(effect_reference, "frame_id") == reference.frame.frame_id
            assert tuple(value_from_object(effect_reference, "frame_path", "iteration_path") or ()) == tuple(
                reference.frame.iteration_path
            )
            assert value_from_object(effect_reference, "workflow_call_depth", "call_depth", "depth") == (
                reference.frame.workflow_call_depth
            )


def _assert_generic_and_compatibility_schedulers(
    generic_state: GraphExecutionState, compatibility_state: GraphExecutionState
) -> None:
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)


def test_static_dag_fresh_execution_has_matching_source_trace() -> None:
    generic, compatibility = _run_both()

    assert generic[0] == compatibility[0] == ["left", "right", "join"]
    assert generic[1].is_complete()
    assert compatibility[1].is_complete()
    assert _state_projection(generic[1]) == _state_projection(compatibility[1])


@pytest.mark.parametrize(
    ("outer_condition", "inner_condition", "expected_sources", "expected_value"),
    [
        (
            True,
            True,
            {"outer_condition", "inner_condition", "inner_true", "inner_if", "outer_if", "sink"},
            5,
        ),
        (
            True,
            False,
            {"outer_condition", "inner_condition", "inner_false", "inner_if", "outer_if", "sink"},
            7,
        ),
        (
            False,
            True,
            {"outer_condition", "outer_false", "outer_if", "sink"},
            11,
        ),
        (
            False,
            False,
            {"outer_condition", "outer_false", "outer_if", "sink"},
            11,
        ),
    ],
)
def test_nested_if_fresh_execution_matches_compatibility_scheduler(
    outer_condition: bool,
    inner_condition: bool,
    expected_sources: set[str],
    expected_value: int,
) -> None:
    def run(force_compatibility_scheduler: bool) -> tuple[list[str], GraphExecutionState]:
        graph = _nested_if_graph()
        graph.get_node("outer_condition").value = outer_condition
        graph.get_node("inner_condition").value = inner_condition
        return _run_graph(
            GraphExecutionState(graph=graph),
            force_compatibility_scheduler=force_compatibility_scheduler,
        )

    generic_trace, generic_state = run(False)
    compatibility_trace, compatibility_state = run(True)

    assert generic_trace == compatibility_trace
    assert {
        source_id
        for exec_node_id, source_id in generic_state.prepared_source_mapping.items()
        if exec_node_id in generic_state.results
    } == expected_sources
    assert {
        source_id
        for exec_node_id, source_id in compatibility_state.prepared_source_mapping.items()
        if exec_node_id in compatibility_state.results
    } == expected_sources
    assert generic_state.results[next(iter(generic_state.source_prepared_mapping["sink"]))].value == expected_value
    assert (
        compatibility_state.results[next(iter(compatibility_state.source_prepared_mapping["sink"]))].value
        == expected_value
    )
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    _assert_generic_and_compatibility_schedulers(generic_state, compatibility_state)
    expected_activations = [
        ("outer_if", "true_input" if outer_condition else "false_input"),
    ]
    if outer_condition:
        expected_activations.append(("inner_if", "true_input" if inner_condition else "false_input"))
    expected_activation_projection = tuple(
        sorted((*activation, activation[1], ()) for activation in expected_activations)
    )
    assert (
        _activation_projection(generic_state)
        == _activation_projection(compatibility_state)
        == expected_activation_projection
    )
    assert _execution_edge_projection(generic_state) == _expected_edge_projection(
        generic_state.graph,
        force_compatibility_scheduler=False,
        outer_condition=outer_condition,
        inner_condition=inner_condition,
    )
    assert _execution_edge_projection(compatibility_state) == _expected_edge_projection(
        compatibility_state.graph,
        force_compatibility_scheduler=True,
        outer_condition=outer_condition,
        inner_condition=inner_condition,
    )
    assert dict(generic_state.indegree) == _expected_remaining_input_indegree(
        generic_state,
        force_compatibility_scheduler=False,
        outer_condition=outer_condition,
        inner_condition=inner_condition,
    )
    assert dict(compatibility_state.indegree) == _expected_remaining_input_indegree(
        compatibility_state,
        force_compatibility_scheduler=True,
        outer_condition=outer_condition,
        inner_condition=inner_condition,
    )
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()


def test_flat_for_fresh_execution_matches_compatibility_scheduler() -> None:
    generic_trace, generic_state = _run_graph_with_effects(GraphExecutionState(graph=_flat_for_graph()))
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph()),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace == ["for", "body", "return", "for", "body", "return"]
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    _assert_generic_and_compatibility_schedulers(generic_state, compatibility_state)
    final_for_id = max(
        (
            exec_node_id
            for exec_node_id, source_node_id in generic_state.prepared_source_mapping.items()
            if source_node_id == "for"
        ),
        key=lambda exec_node_id: generic_state.execution_graph.get_node(exec_node_id).index,
    )
    assert generic_state.results[final_for_id].output_collection == [11, 12]
    assert generic_state._generic_runtime().continuations
    assert all(
        continuation.status == "completed" for continuation in generic_state._generic_runtime().continuations.values()
    )
    assert len(generic_state._generic_runtime().continuations) == 2
    assert {continuation.owner_id for continuation in generic_state._generic_runtime().continuations.values()} == {
        exec_node_id
        for exec_node_id, source_node_id in generic_state.prepared_source_mapping.items()
        if source_node_id == "for"
    }
    expected_continuations = [
        (
            (0,),
            "completed",
            {"index": 0, "total": 2, "state": {"values": {}}},
            {"output": 11, "state": None, "continue_condition": True},
        ),
        (
            (1,),
            "completed",
            {"index": 1, "total": 2, "state": {"values": {}}},
            {"output": 12, "state": None, "continue_condition": True},
        ),
    ]
    expected_effects = [
        ("start", {"index": 0, "total": 2, "state": {"values": {}}}),
        ("complete", {"output": 11, "state": None, "continue_condition": True}),
        ("start", {"index": 1, "total": 2, "state": {"values": {}}}),
        ("complete", {"output": 12, "state": None, "continue_condition": True}),
    ]
    for state in (generic_state, compatibility_state):
        assert [
            (tuple(continuation.frame.iteration_path), continuation.status, continuation.payload, continuation.result)
            for continuation in sorted(
                state._generic_runtime().continuations.values(),
                key=lambda continuation: continuation.frame.iteration_path,
            )
        ] == expected_continuations
        actual_effects = [
            (effect.operation, effect.payload)
            for effects in state.execution_effects.values()
            for effect in effects
            if effect.kind == "continuation"
        ]
        assert sorted(actual_effects, key=lambda item: (item[0], json.dumps(item[1], sort_keys=True))) == sorted(
            expected_effects, key=lambda item: (item[0], json.dumps(item[1], sort_keys=True))
        )
        assert _final_for_output(state).output_collection == [11, 12]
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)


def test_flat_for_generic_path_does_not_use_compatibility_continuation_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_compatibility_bridge(*_: object, **__: object) -> None:
        raise AssertionError("generic For execution used the compatibility continuation bridge")

    monkeypatch.setattr(GraphExecutionState, "_try_schedule_next_for_iteration", fail_compatibility_bridge)

    trace, state = _run_graph_with_effects(GraphExecutionState(graph=_flat_for_graph()))

    assert trace == ["for", "body", "return", "for", "body", "return"]
    assert state.is_complete()
    assert _final_for_output(state).output_collection == [11, 12]


def test_flat_for_generic_path_does_not_use_compatibility_materializer_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_compatibility_helper(*_: object, **__: object) -> None:
        raise AssertionError("generic For execution used a compatibility materializer helper")

    def fail_compatibility_materializer(*_: object, **__: object) -> None:
        raise AssertionError("generic For instantiated compatibility materializer")

    materializer_type = graph_module._ExecutionMaterializer
    monkeypatch.setattr(graph_module, "_ExecutionMaterializer", fail_compatibility_materializer)
    for method_name in ("create_for_iteration", "create_for_body_iteration", "create_nested_for_return"):
        monkeypatch.setattr(materializer_type, method_name, fail_compatibility_helper)

    trace, state = _run_graph_with_effects(GraphExecutionState(graph=_flat_for_graph()))

    assert trace == ["for", "body", "return", "for", "body", "return"]
    assert state.is_complete()
    assert _final_for_output(state).output_collection == [11, 12]


@pytest.mark.parametrize(
    "graph_factory",
    [
        pytest.param(_flat_for_graph, id="flat"),
        pytest.param(_nested_for_graph, id="nested"),
        pytest.param(_input_driven_nested_for_graph, id="input-driven-nested"),
        pytest.param(_two_sibling_nested_for_fan_in_graph, id="two-sibling"),
        pytest.param(_three_level_nested_for_graph, id="three-level"),
        pytest.param(_input_driven_three_level_nested_for_graph, id="input-driven-three-level"),
        pytest.param(_four_level_nested_for_graph, id="four-level"),
        pytest.param(_nested_for_iterate_collect_graph, id="nested-iterate-collect"),
        pytest.param(_input_driven_nested_for_iterate_collect_graph, id="input-driven-nested-iterate-collect"),
        pytest.param(_serial_two_level_nested_for_iterate_collect_graph, id="serial-nested-iterate"),
    ],
)
def test_admitted_generic_for_shapes_do_not_construct_compatibility_materializer(
    monkeypatch: pytest.MonkeyPatch,
    graph_factory: Callable[[], Graph],
) -> None:
    def fail_compatibility_materializer(*_: object, **__: object) -> None:
        raise AssertionError("generic For instantiated compatibility materializer")

    monkeypatch.setattr(graph_module, "_ExecutionMaterializer", fail_compatibility_materializer)

    trace, state = _run_graph_with_effects(GraphExecutionState(graph=graph_factory()))

    assert trace
    assert state.is_complete()


@pytest.mark.parametrize(
    "graph_factory",
    [
        pytest.param(_nested_for_graph, id="nested"),
        pytest.param(_input_driven_nested_for_graph, id="input-driven-nested"),
        pytest.param(_two_sibling_nested_for_fan_in_graph, id="two-sibling"),
        pytest.param(_input_driven_three_level_nested_for_graph, id="input-driven-three-level"),
        pytest.param(_nested_for_iterate_collect_graph, id="nested-iterate-collect"),
    ],
)
def test_admitted_nested_for_checkpoint_does_not_construct_compatibility_materializer(
    monkeypatch: pytest.MonkeyPatch,
    graph_factory: Callable[[], Graph],
) -> None:
    partial_trace, partial_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph_factory()),
        stop_after=4,
    )

    def fail_compatibility_materializer(*_: object, **__: object) -> None:
        raise AssertionError("generic nested For rehydration instantiated compatibility materializer")

    monkeypatch.setattr(graph_module, "_ExecutionMaterializer", fail_compatibility_materializer)
    resumed_trace, resumed_state = _run_graph_with_effects(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace
    assert resumed_trace
    assert resumed_state.is_complete()


def test_legacy_nested_for_snapshot_uses_compatibility_materializer() -> None:
    _partial_trace, partial_state = _run_graph_with_effects(
        GraphExecutionState(graph=_nested_for_graph()),
        stop_after=2,
    )
    snapshot = dump_execution_state(partial_state)
    snapshot.pop("execution_state_version")
    snapshot.pop("execution_effects")
    materializer_type = graph_module._ExecutionMaterializer

    with patch.object(graph_module, "_ExecutionMaterializer", wraps=materializer_type) as materializer:
        restored = load_execution_state(snapshot)
        assert restored._legacy_snapshot_loaded
        _resumed_trace, restored = _run_graph(restored)

    assert materializer.call_count == 1
    assert restored.is_complete()


def test_generic_nested_for_path_does_not_use_compatibility_deferred_body_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_compatibility_helper(*_: object, **__: object) -> None:
        raise AssertionError("generic nested For execution used the compatibility deferred-body helper")

    monkeypatch.setattr(_ExecutionScheduler, "_try_materialize_deferred_nested_for_body", fail_compatibility_helper)

    trace, state = _run_graph_with_effects(GraphExecutionState(graph=_nested_for_graph()))

    assert trace
    assert state.is_complete()


def test_compatibility_for_path_does_not_construct_generic_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_generic_planner(*_: object, **__: object) -> None:
        raise AssertionError("compatibility For instantiated generic planner")

    monkeypatch.setattr(GraphExecutionState, "_for_planner", fail_generic_planner)

    trace, state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph()),
        force_compatibility_scheduler=True,
    )

    assert trace == ["for", "body", "return", "for", "body", "return"]
    assert state.is_complete()


def test_nested_for_generic_and_compatibility_paths_have_matching_completion() -> None:
    def run(force_compatibility: bool) -> tuple[list[str], GraphExecutionState]:
        state = GraphExecutionState(graph=_nested_for_graph())
        if force_compatibility:
            state._execution_scheduler = _ExecutionScheduler(state)
        return _run(state)

    compatibility_trace, compatibility_state = run(force_compatibility=True)
    generic_trace, generic_state = run(force_compatibility=False)

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    after_exec_id = next(
        exec_node_id
        for exec_node_id, source_node_id in generic_state.prepared_source_mapping.items()
        if source_node_id == "after"
    )
    assert generic_state.results[after_exec_id].value == [["a", "b"], ["c"]]
    compatibility_after_exec_id = next(
        exec_node_id
        for exec_node_id, source_node_id in compatibility_state.prepared_source_mapping.items()
        if source_node_id == "after"
    )
    assert compatibility_state.results[compatibility_after_exec_id].value == [["a", "b"], ["c"]]


def test_input_driven_nested_for_uses_generic_scheduler() -> None:
    state = GraphExecutionState(graph=_input_driven_nested_for_graph())

    assert state._can_use_generic_scheduler()

    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_input_driven_nested_for_graph()),
        force_compatibility_scheduler=True,
    )
    generic_trace, generic_state = _run_graph(state)

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _source_output(generic_state, "after").value == [["a", "b"], ["c"]]
    assert generic_trace.count("outer_source") == compatibility_trace.count("outer_source") == 1


def test_input_driven_nested_for_empty_outer_result_remains_compatibility_owned() -> None:
    graph = _input_driven_nested_for_graph(outer_collection=[])
    state = GraphExecutionState(graph=graph)

    assert not state._can_use_generic_scheduler()
    assert isinstance(state._scheduler(), _ExecutionScheduler)


def test_input_driven_nested_for_empty_range_remains_compatibility_owned_and_completes() -> None:
    graph = _input_driven_nested_for_graph()
    graph.delete_node("outer_source")
    graph.add_node(RangeInvocation(id="outer_source", start=0, stop=1, step=-1))
    graph.add_edge(create_edge("outer_source", "collection", "outer_for", "collection"))
    state = GraphExecutionState(graph=graph)

    assert not state._can_use_generic_scheduler()

    trace, state = _run_graph(state, stop_after=1)

    assert trace == ["outer_source"]
    assert _source_output(state, "outer_source").collection == []
    assert isinstance(state._execution_scheduler, _ExecutionScheduler)


def test_input_driven_nested_for_generic_handles_empty_inner_collection() -> None:
    def graph_factory() -> Graph:
        return _input_driven_nested_for_graph(outer_collection=[[], ["c"]])

    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=graph_factory()),
        force_compatibility_scheduler=True,
    )
    generic_trace, generic_state = _run_graph(GraphExecutionState(graph=graph_factory()))

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _source_output(generic_state, "after").value == [[], ["c"]]
    assert _source_output(compatibility_state, "after").value == [[], ["c"]]


@pytest.mark.parametrize("stop_after", [1, 4], ids=["after-producer", "mid-inner"])
def test_input_driven_nested_for_reload_does_not_replay_completed_work(stop_after: int) -> None:
    expected_trace, expected_state = _run_graph(GraphExecutionState(graph=_input_driven_nested_for_graph()))
    partial_trace, partial_state = _run_graph(
        GraphExecutionState(graph=_input_driven_nested_for_graph()),
        stop_after=stop_after,
    )

    resumed_trace, resumed_state = _run_graph(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert "outer_source" not in resumed_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)

    compatibility_partial_trace, compatibility_partial_state = _run_graph(
        GraphExecutionState(graph=_input_driven_nested_for_graph()),
        force_compatibility_scheduler=True,
        stop_after=stop_after,
    )
    compatibility_resumed_trace, compatibility_resumed_state = _run_graph(
        load_execution_state(dump_execution_state(compatibility_partial_state)),
        force_compatibility_scheduler=True,
    )

    assert compatibility_partial_trace + compatibility_resumed_trace == expected_trace
    assert "outer_source" not in compatibility_resumed_trace
    assert compatibility_resumed_state.is_complete()
    assert _state_projection(compatibility_resumed_state) == _state_projection(expected_state)
    assert resumed_trace == compatibility_resumed_trace


def test_input_driven_nested_for_failure_matches_compatibility() -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_input_driven_nested_for_graph()),
        fail_source_id="inner_body",
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_input_driven_nested_for_graph()),
        force_compatibility_scheduler=True,
        fail_source_id="inner_body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert "inner_return" not in generic_trace
    assert "outer_return" not in generic_trace
    assert "after" not in generic_trace
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_two_sibling_nested_for_fan_in_uses_generic_scheduler() -> None:
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_two_sibling_nested_for_fan_in_graph()),
        force_compatibility_scheduler=True,
    )
    generic_trace, generic_state = _run_graph(GraphExecutionState(graph=_two_sibling_nested_for_fan_in_graph()))

    assert generic_trace == compatibility_trace
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _source_output(generic_state, "after").value == [
        ["left:a", "left:b", "right:a", "right:b"],
        ["left:c", "right:c"],
    ]


def test_two_sibling_nested_for_final_state_consumer_uses_compatibility_scheduler() -> None:
    state = GraphExecutionState(graph=_two_sibling_nested_for_fan_in_graph(outer_output_field="final_state"))
    trace, state = _run_graph(state)

    assert trace
    assert isinstance(state._execution_scheduler, _ExecutionScheduler)
    assert state.is_complete()
    assert _source_output(state, "after").value.values == {}


def test_two_sibling_nested_for_fan_in_failure_matches_compatibility() -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_two_sibling_nested_for_fan_in_graph()),
        fail_source_id="left_body",
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_two_sibling_nested_for_fan_in_graph()),
        force_compatibility_scheduler=True,
        fail_source_id="left_body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert "join" not in generic_trace
    assert "outer_return" not in generic_trace
    assert "after" not in generic_trace
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_nested_for_generic_and_compatibility_paths_handle_empty_inner_collection() -> None:
    def run(force_compatibility: bool) -> tuple[list[str], GraphExecutionState]:
        state = GraphExecutionState(graph=_nested_for_graph(outer_collection=[[], ["c"]]))
        if force_compatibility:
            state._execution_scheduler = _ExecutionScheduler(state)
        return _run(state)

    compatibility_trace, compatibility_state = run(force_compatibility=True)
    generic_trace, generic_state = run(force_compatibility=False)

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    # Empty inner iterations use different synthetic execution-node projections, but must preserve trace and outputs.
    after_exec_id = next(
        exec_node_id
        for exec_node_id, source_node_id in generic_state.prepared_source_mapping.items()
        if source_node_id == "after"
    )
    assert generic_state.results[after_exec_id].value == [[], ["c"]]
    compatibility_after_exec_id = next(
        exec_node_id
        for exec_node_id, source_node_id in compatibility_state.prepared_source_mapping.items()
        if source_node_id == "after"
    )
    assert compatibility_state.results[compatibility_after_exec_id].value == [[], ["c"]]


def test_nested_for_generic_path_rehydrates_after_inner_completion() -> None:
    expected_state = GraphExecutionState(graph=_nested_for_graph())
    expected_state._execution_scheduler = _ExecutionScheduler(expected_state)
    expected_trace, expected_state = _run(expected_state)
    assert expected_trace == [
        "outer_for",
        "inner_for",
        "inner_body",
        "inner_return",
        "inner_for",
        "inner_body",
        "inner_return",
        "outer_return",
        "outer_for",
        "inner_for",
        "inner_body",
        "inner_return",
        "outer_return",
        "after",
    ]

    state = GraphExecutionState(graph=_nested_for_graph())
    state._execution_scheduler = _GenericGraphSchedulerAdapter(state)
    partial_trace, partial_state = _run(state, stop_after=4)
    assert partial_trace == ["outer_for", "inner_for", "inner_body", "inner_return"]

    restored = load_execution_state(dump_execution_state(partial_state))
    restored._execution_scheduler = _GenericGraphSchedulerAdapter(restored)
    resumed_trace, resumed_state = _run(restored)

    assert resumed_trace == [
        "inner_for",
        "inner_body",
        "inner_return",
        "outer_return",
        "outer_for",
        "inner_for",
        "inner_body",
        "inner_return",
        "outer_return",
        "after",
    ]
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)

    compatibility_partial = GraphExecutionState(graph=_nested_for_graph())
    compatibility_partial._execution_scheduler = _ExecutionScheduler(compatibility_partial)
    _, compatibility_partial = _run(compatibility_partial, stop_after=4)
    compatibility_restored = load_execution_state(dump_execution_state(compatibility_partial))
    compatibility_restored._execution_scheduler = _ExecutionScheduler(compatibility_restored)
    compatibility_resumed_trace, compatibility_resumed = _run(compatibility_restored)
    assert compatibility_resumed_trace == resumed_trace
    assert _state_projection(compatibility_resumed) == _state_projection(expected_state)


def test_three_level_nested_for_generic_and_compatibility_paths_match() -> None:
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_three_level_nested_for_graph()),
        force_compatibility_scheduler=True,
    )
    generic_trace, generic_state = _run_graph(GraphExecutionState(graph=_three_level_nested_for_graph()))

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _source_output(generic_state, "after").value == [[["a", "b"], ["c"]], [["d"]]]


def test_three_level_nested_for_generic_and_compatibility_paths_match_empty_inner_collection() -> None:
    graph = _three_level_nested_for_graph(outer_collection=[[[], ["c"]], [["d"]]])
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )
    generic_trace, generic_state = _run_graph(
        GraphExecutionState(graph=_three_level_nested_for_graph(outer_collection=[[[], ["c"]], [["d"]]]))
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _source_output(generic_state, "after").value == [[[], ["c"]], [["d"]]]
    assert _source_output(compatibility_state, "after").value == [[[], ["c"]], [["d"]]]


def test_three_level_nested_for_generic_rehydrates_after_inner_completion() -> None:
    expected_trace, expected_state = _run_graph(GraphExecutionState(graph=_three_level_nested_for_graph()))
    partial_trace, partial_state = _run_graph(
        GraphExecutionState(graph=_three_level_nested_for_graph()),
        stop_after=5,
    )

    restored = load_execution_state(dump_execution_state(partial_state))
    resumed_trace, resumed_state = _run_graph(restored)

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_three_level_nested_for_generic_failure_matches_compatibility() -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_three_level_nested_for_graph()),
        fail_source_id="inner_body",
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_three_level_nested_for_graph()),
        force_compatibility_scheduler=True,
        fail_source_id="inner_body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert "inner_return" not in generic_trace
    assert "middle_return" not in generic_trace
    assert "outer_return" not in generic_trace
    assert "after" not in generic_trace
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_input_driven_three_level_nested_for_uses_generic_scheduler() -> None:
    state = GraphExecutionState(graph=_input_driven_three_level_nested_for_graph())

    assert state._can_use_generic_scheduler()

    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_input_driven_three_level_nested_for_graph()),
        force_compatibility_scheduler=True,
    )
    generic_trace, generic_state = _run_graph(state)

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _source_output(generic_state, "after").value == [[["a", "b"], ["c"]], [["d"]]]
    assert generic_trace.count("outer_source") == compatibility_trace.count("outer_source") == 1


def test_input_driven_three_level_nested_for_matches_empty_nested_collections() -> None:
    def graph_factory() -> Graph:
        return _input_driven_three_level_nested_for_graph_with_collection([[[], ["c"]], [[]]])

    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=graph_factory()),
        force_compatibility_scheduler=True,
    )
    generic_trace, generic_state = _run_graph(GraphExecutionState(graph=graph_factory()))

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _source_output(generic_state, "after").value == [[[], ["c"]], [[]]]
    assert _source_output(compatibility_state, "after").value == [[[], ["c"]], [[]]]


@pytest.mark.parametrize("stop_after", [1, 7], ids=["after-producer", "mid-inner"])
def test_input_driven_three_level_nested_for_reload_does_not_replay_completed_work(stop_after: int) -> None:
    def graph_factory() -> Graph:
        return _input_driven_three_level_nested_for_graph()

    expected_trace, expected_state = _run_graph(GraphExecutionState(graph=graph_factory()))
    partial_trace, partial_state = _run_graph(GraphExecutionState(graph=graph_factory()), stop_after=stop_after)

    resumed_trace, resumed_state = _run_graph(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert "outer_source" not in resumed_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_input_driven_three_level_nested_for_failure_matches_compatibility() -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_input_driven_three_level_nested_for_graph()),
        fail_source_id="inner_body",
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_input_driven_three_level_nested_for_graph()),
        force_compatibility_scheduler=True,
        fail_source_id="inner_body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert "inner_return" not in generic_trace
    assert "middle_return" not in generic_trace
    assert "outer_return" not in generic_trace
    assert "after" not in generic_trace
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


@pytest.mark.parametrize(
    "graph_factory",
    [
        pytest.param(_sibling_nested_for_graph, id="sibling"),
        pytest.param(_malformed_three_level_nested_for_graph, id="malformed-link"),
        pytest.param(
            lambda: _input_driven_three_level_nested_for_graph_with_collection([]),
            id="empty-outer-producer",
        ),
        pytest.param(_input_driven_three_level_nested_for_graph_with_extra_node, id="extra-node"),
    ],
)
def test_three_level_nested_for_gate_falls_back_for_unsupported_shapes(graph_factory: Any) -> None:
    state = GraphExecutionState(graph=graph_factory())

    assert not state._can_use_generic_scheduler()


@pytest.mark.parametrize(
    "graph_factory",
    [
        pytest.param(_four_level_nested_for_without_consumer, id="missing-consumer"),
        pytest.param(_four_level_nested_for_with_extra_node, id="extra-disconnected-node"),
        pytest.param(_input_driven_four_level_nested_for_graph, id="input-driven"),
        pytest.param(_malformed_four_level_nested_for_graph, id="malformed-link"),
        pytest.param(_sibling_nested_for_graph, id="sibling"),
        pytest.param(_five_level_nested_for_graph, id="depth-five"),
    ],
)
def test_four_level_nested_for_gate_falls_back_for_non_exact_shapes(graph_factory: Any) -> None:
    state = GraphExecutionState(graph=graph_factory())

    assert not state._can_use_generic_scheduler()


def test_four_level_nested_for_generic_and_compatibility_paths_match() -> None:
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_four_level_nested_for_graph()),
        force_compatibility_scheduler=True,
    )
    generic_trace, generic_state = _run_graph(GraphExecutionState(graph=_four_level_nested_for_graph()))

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _source_output(generic_state, "after").value == [
        [[["a", "b"], ["c"]], [["d"]]],
        [[["e"]]],
    ]


def test_four_level_nested_for_generic_handles_deepest_empty_collection() -> None:
    """Compare user-visible behavior; generic synthetic empty frames have a different durable projection."""
    graph = _four_level_nested_for_graph(outer_collection=[[[[]]]])
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )
    generic_trace, generic_state = _run_graph(
        GraphExecutionState(graph=_four_level_nested_for_graph(outer_collection=[[[[]]]]))
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _source_output(generic_state, "after").value == [[[[]]]]
    assert _source_output(compatibility_state, "after").value == [[[[]]]]


def test_four_level_nested_for_generic_rehydrates_without_replaying_completed_body() -> None:
    expected_trace, expected_state = _run_graph(GraphExecutionState(graph=_four_level_nested_for_graph()))
    partial_trace, partial_state = _run_graph(
        GraphExecutionState(graph=_four_level_nested_for_graph()),
        stop_after=6,
    )

    resumed_trace, resumed_state = _run_graph(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_trace[0] != "outer_for"
    assert resumed_state.is_complete()
    assert isinstance(resumed_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert _state_projection(resumed_state) == _state_projection(expected_state)

    compatibility_partial_trace, compatibility_partial_state = _run_graph(
        GraphExecutionState(graph=_four_level_nested_for_graph()),
        force_compatibility_scheduler=True,
        stop_after=6,
    )
    compatibility_resumed_trace, compatibility_resumed_state = _run_graph(
        load_execution_state(dump_execution_state(compatibility_partial_state)),
        force_compatibility_scheduler=True,
    )

    assert compatibility_partial_trace + compatibility_resumed_trace == expected_trace
    assert _state_projection(compatibility_resumed_state) == _state_projection(expected_state)


def test_four_level_nested_for_generic_failure_matches_compatibility() -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_four_level_nested_for_graph()),
        fail_source_id="deep_body",
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_four_level_nested_for_graph()),
        force_compatibility_scheduler=True,
        fail_source_id="deep_body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert "deep_return" not in generic_trace
    assert "inner_return" not in generic_trace
    assert "middle_return" not in generic_trace
    assert "outer_return" not in generic_trace
    assert "after" not in generic_trace
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_four_level_nested_for_legacy_snapshot_remains_compatibility_owned() -> None:
    _, partial_state = _run_graph(
        GraphExecutionState(graph=_four_level_nested_for_graph()),
        stop_after=2,
    )
    snapshot = dump_execution_state(partial_state)
    snapshot.pop("execution_state_version")
    snapshot.pop("execution_effects")

    restored = load_execution_state(snapshot)

    assert restored._legacy_snapshot_loaded
    assert restored._can_use_generic_scheduler() is False
    assert isinstance(restored._scheduler(), _ExecutionScheduler)


def test_nested_for_iterate_collect_generic_and_compatibility_paths_have_matching_completion() -> None:
    compatibility_state = GraphExecutionState(graph=_nested_for_iterate_collect_graph())
    compatibility_trace, compatibility_state = _run(
        compatibility_state,
        force_compatibility_scheduler=True,
    )
    generic_state = GraphExecutionState(graph=_nested_for_iterate_collect_graph())
    generic_trace, generic_state = _run(generic_state)

    assert (
        generic_trace
        == compatibility_trace
        == [
            "outer_for",
            "nested_collection",
            "nested_iterate",
            "nested_iterate",
            "nested_body",
            "nested_body",
            "nested_collect",
            "outer_return",
            "outer_for",
            "nested_collection",
            "nested_iterate",
            "nested_body",
            "nested_collect",
            "outer_return",
            "after",
        ]
    )
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    for state in (generic_state, compatibility_state):
        after_exec_id = next(
            exec_node_id
            for exec_node_id, source_node_id in state.prepared_source_mapping.items()
            if source_node_id == "after"
        )
        assert state.results[after_exec_id].value == [["a", "b"], ["c"]]


def test_nested_for_iterate_collect_generic_owns_nested_stream_expansion() -> None:
    compatibility_state = GraphExecutionState(graph=_nested_for_iterate_collect_graph())
    compatibility_trace, compatibility_state = _run(
        compatibility_state,
        force_compatibility_scheduler=True,
    )

    generic_state = GraphExecutionState(graph=_nested_for_iterate_collect_graph())
    with patch.object(
        graph_module._ExecutionMaterializer,
        "_create_nested_iterate_body_iteration",
        side_effect=AssertionError("generic nested Iterate/Collect used materializer copy creation"),
    ):
        generic_trace, generic_state = _run(generic_state)

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _direct_iterate_fan_in_stream_projection(generic_state) == _direct_iterate_fan_in_stream_projection(
        compatibility_state
    )


def test_nested_for_iterate_collect_generic_handles_empty_inner_collection() -> None:
    compatibility_state = GraphExecutionState(graph=_nested_for_iterate_collect_graph(outer_collection=[[], ["c"]]))
    compatibility_trace, compatibility_state = _run(
        compatibility_state,
        force_compatibility_scheduler=True,
    )
    generic_state = GraphExecutionState(graph=_nested_for_iterate_collect_graph(outer_collection=[[], ["c"]]))
    generic_trace, generic_state = _run(generic_state)

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    for state in (generic_state, compatibility_state):
        after_exec_id = next(
            exec_node_id
            for exec_node_id, source_node_id in state.prepared_source_mapping.items()
            if source_node_id == "after"
        )
        assert state.results[after_exec_id].value == [[], ["c"]]
    generic_streams = generic_state._generic_runtime().streams
    assert generic_streams[generic_state._iteration_stream_id("nested_iterate", (0,))].closed
    assert generic_streams[generic_state._iteration_stream_id("nested_iterate", (0,))].values == ()
    assert generic_streams[generic_state._iteration_stream_id("nested_iterate", (1,))].values == ("c",)


def test_nested_for_iterate_collect_generic_rehydrates_after_inner_completion() -> None:
    expected_state = GraphExecutionState(graph=_nested_for_iterate_collect_graph())
    expected_state._execution_scheduler = _ExecutionScheduler(expected_state)
    expected_trace, expected_state = _run(expected_state)

    state = GraphExecutionState(graph=_nested_for_iterate_collect_graph())
    state._execution_scheduler = _GenericGraphSchedulerAdapter(state)
    partial_trace, partial_state = _run(state, stop_after=3)
    restored = load_execution_state(dump_execution_state(partial_state))
    resumed_trace, resumed_state = _run(restored)

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert isinstance(resumed_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _direct_iterate_fan_in_stream_projection(resumed_state) == _direct_iterate_fan_in_stream_projection(
        expected_state
    )


def test_nested_for_iterate_collect_generic_failure_does_not_release_outer_return() -> None:
    graph = _nested_for_iterate_collect_graph()
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        fail_source_id="nested_body",
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id="nested_body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert "nested_collect" not in generic_trace
    assert "outer_return" not in generic_trace
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_input_driven_nested_for_iterate_collect_uses_generic_stream_planner() -> None:
    generic_state = GraphExecutionState(graph=_input_driven_nested_for_iterate_collect_graph())
    compatibility_state = GraphExecutionState(graph=_input_driven_nested_for_iterate_collect_graph())

    assert generic_state._can_use_generic_scheduler()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "_create_nested_iterate_body_iteration",
        side_effect=AssertionError("input-driven nested Iterate/Collect used materializer copy creation"),
    ):
        generic_trace, generic_state = _run_graph_with_effects(generic_state)
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        compatibility_state,
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    after_exec_id = next(
        exec_node_id
        for exec_node_id, source_node_id in generic_state.prepared_source_mapping.items()
        if source_node_id == "after"
    )
    assert generic_state.results[after_exec_id].value == [["a", "b"], ["c"]]


def test_input_driven_nested_for_iterate_collect_rehydrates_without_replay() -> None:
    graph = _input_driven_nested_for_iterate_collect_graph()
    expected_trace, expected_state = _run(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run(GraphExecutionState(graph=graph), stop_after=4)

    restored = load_execution_state(dump_execution_state(partial_state))
    resumed_trace, resumed_state = _run(restored)

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_input_driven_nested_for_iterate_collect_failure_matches_compatibility() -> None:
    graph = _input_driven_nested_for_iterate_collect_graph()
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        fail_source_id="nested_body",
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id="nested_body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert "nested_collect" not in generic_trace
    assert "outer_return" not in generic_trace
    assert "after" not in generic_state.source_prepared_mapping
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_serial_two_level_nested_for_iterate_collect_uses_generic_planner() -> None:
    compatibility_state = GraphExecutionState(graph=_serial_two_level_nested_for_iterate_collect_graph())
    compatibility_trace, compatibility_state = _run(
        compatibility_state,
        force_compatibility_scheduler=True,
    )
    state = GraphExecutionState(graph=_serial_two_level_nested_for_iterate_collect_graph())
    assert state._can_use_generic_scheduler()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "_create_nested_iterate_body_iteration",
        side_effect=AssertionError("serial nested Iterate/Collect used materializer copy creation"),
    ):
        trace, state = _run(state)
    assert state.is_complete()
    assert compatibility_state.is_complete()
    assert trace == compatibility_trace


def test_serial_two_level_nested_for_iterate_collect_generic_closes_empty_streams() -> None:
    graph = _serial_two_level_nested_for_iterate_collect_graph(outer_collection=[[], ["c"]])
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )
    state = GraphExecutionState(graph=graph)
    trace, state = _run(state)

    assert state.is_complete()
    assert compatibility_state.is_complete()
    assert trace == compatibility_trace
    assert _state_projection(state) == _state_projection(compatibility_state)
    assert trace[-1] == "after"
    runtime = state._generic_runtime()
    assert runtime.streams[state._iteration_stream_id("nested_iterate_1", (0,))].values == ()
    assert runtime.streams[state._iteration_stream_id("nested_iterate_2", (1, 0))].values == ("c",)
    assert _source_output(state, "after").value == [[], ["c"]]


def test_serial_two_level_nested_for_iterate_collect_generic_rehydrates() -> None:
    graph = _serial_two_level_nested_for_iterate_collect_graph()
    expected_trace, expected_state = _run(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run(GraphExecutionState(graph=graph), stop_after=5)

    resumed_trace, resumed_state = _run(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )
    assert compatibility_trace == expected_trace
    assert _state_projection(compatibility_state) == _state_projection(expected_state)


def test_serial_two_level_nested_for_iterate_collect_generic_failure_stops_outer_return() -> None:
    graph = _serial_two_level_nested_for_iterate_collect_graph()
    trace, state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        fail_source_id="nested_body",
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id="nested_body",
    )

    assert state.has_error()
    assert state.is_complete()
    assert compatibility_state.has_error()
    assert compatibility_state.is_complete()
    assert trace == compatibility_trace
    assert "nested_collect" not in trace
    assert "outer_return" not in trace
    assert "after" not in trace


def test_serial_two_level_nested_for_iterate_collect_retry_isolated() -> None:
    graph = _serial_two_level_nested_for_iterate_collect_graph()
    canceled_trace, canceled_state = _run_until_source(
        GraphExecutionState(graph=graph),
        "nested_collection_2",
    )
    retry_trace, retry_state = _run_graph(GraphExecutionState(graph=graph.model_copy(deep=True)))

    assert canceled_trace[:3] == ["outer_for", "nested_collection_1", "nested_iterate_1"]
    assert not canceled_state.is_complete()
    assert retry_trace[-2:] == ["outer_return", "after"]
    assert retry_state.id != canceled_state.id
    assert _source_output(retry_state, "after").value == [["a", "b"], ["c"]]
    assert retry_state.is_complete()


def test_nested_iterate_chain_matches_compatibility() -> None:
    graph = _nested_iterate_chain_graph()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("nested Iterate chain used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_trace == [
        "outer_source",
        "outer_iterate",
        "outer_iterate",
        "inner_collection",
        "inner_collection",
        "inner_iterate",
        "inner_iterate",
        "inner_iterate",
        "body",
        "body",
        "body",
    ]
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_nested_iterate_chain_closes_empty_inner_frame() -> None:
    graph = _nested_iterate_chain_graph(outer_collection=[[], ["c"]])
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("nested Iterate chain used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_trace == [
        "outer_source",
        "outer_iterate",
        "outer_iterate",
        "inner_collection",
        "inner_collection",
        "inner_iterate",
        "body",
    ]
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert (
        generic_state._generic_runtime().streams[generic_state._iteration_stream_id("inner_iterate", (0,))].values == ()
    )


def test_nested_iterate_chain_completes_when_all_collections_are_empty() -> None:
    graph = _nested_iterate_chain_graph(outer_collection=[[], []])
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("nested Iterate chain used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_trace == [
        "outer_source",
        "outer_iterate",
        "outer_iterate",
        "inner_collection",
        "inner_collection",
    ]
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_nested_iterate_chain_rehydrates_without_replay() -> None:
    graph = _nested_iterate_chain_graph()
    expected_trace, expected_state = _run(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run(GraphExecutionState(graph=graph), stop_after=5)

    resumed_trace, resumed_state = _run(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_nested_iterate_chain_failure_does_not_release_later_frames() -> None:
    graph = _nested_iterate_chain_graph()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("nested Iterate chain used compatibility materializer"),
    ):
        generic_trace, generic_state = _run_graph_with_effects(
            GraphExecutionState(graph=graph),
            fail_source_id="body",
        )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id="body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert "body" in generic_trace
    assert generic_trace.count("body") == 1


def test_three_level_nested_iterate_matches_compatibility() -> None:
    graph = _three_level_nested_iterate_chain_graph()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("three-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_trace == [
        "outer_source",
        "outer_iterate",
        "outer_iterate",
        "middle_collection",
        "middle_collection",
        "middle_iterate",
        "middle_iterate",
        "middle_iterate",
        "inner_collection",
        "inner_collection",
        "inner_collection",
        "inner_iterate",
        "inner_iterate",
        "inner_iterate",
        "inner_iterate",
        "body",
        "body",
        "body",
        "body",
    ]
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _direct_iterate_fan_in_stream_projection(generic_state) == _direct_iterate_fan_in_stream_projection(
        compatibility_state
    )
    assert sorted(
        generic_state._get_iteration_path(exec_id)
        for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
    ) == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]


@pytest.mark.parametrize(
    ("outer_collection", "expected_body_paths"),
    [
        ([], []),
        ([[], [["c"]]], [(1, 0, 0)]),
        ([[[], []]], []),
    ],
    ids=["outer-empty", "mixed-empty", "all-empty"],
)
def test_three_level_nested_iterate_closes_empty_frames(
    outer_collection: list[list[list[str]]], expected_body_paths: list[tuple[int, ...]]
) -> None:
    graph = _three_level_nested_iterate_chain_graph(outer_collection=outer_collection)
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("three-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert (
        sorted(
            generic_state._get_iteration_path(exec_id)
            for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
        )
        == expected_body_paths
    )


def test_three_level_nested_iterate_rehydrates_without_replay() -> None:
    graph = _three_level_nested_iterate_chain_graph()
    expected_trace, expected_state = _run(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run(GraphExecutionState(graph=graph), stop_after=8)

    resumed_trace, resumed_state = _run(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    _assert_retained_iterate_streams_match_expected(resumed_state, expected_state)


def test_three_level_nested_iterate_failure_matches_compatibility() -> None:
    graph = _three_level_nested_iterate_chain_graph()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("three-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run_graph_with_effects(
            GraphExecutionState(graph=graph),
            fail_source_id="body",
        )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id="body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert generic_trace.count("body") == 1
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_four_level_nested_iterate_matches_compatibility() -> None:
    graph = _four_level_nested_iterate_chain_graph()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("four-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_trace == (
        ["outer_source"]
        + ["outer_iterate"] * 2
        + ["level1_collection"] * 2
        + ["level1_iterate"] * 3
        + ["level2_collection"] * 3
        + ["level2_iterate"] * 4
        + ["level3_collection"] * 4
        + ["level3_iterate"] * 5
        + ["body"] * 5
    )
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _direct_iterate_fan_in_stream_projection(generic_state) == _direct_iterate_fan_in_stream_projection(
        compatibility_state
    )
    assert sorted(
        generic_state._get_iteration_path(exec_id)
        for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
    ) == [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]


@pytest.mark.parametrize(
    ("outer_collection", "expected_body_paths"),
    [
        ([], []),
        (
            [
                [],
                [
                    [
                        [
                            "c",
                        ]
                    ]
                ],
            ],
            [(1, 0, 0, 0)],
        ),
        ([[[]]], []),
        ([[[], []]], []),
    ],
    ids=["outer-empty", "mixed-empty", "deepest-empty", "all-empty"],
)
def test_four_level_nested_iterate_closes_empty_frames(
    outer_collection: list[list[list[list[str]]]], expected_body_paths: list[tuple[int, ...]]
) -> None:
    graph = _four_level_nested_iterate_chain_graph(outer_collection=outer_collection)
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("four-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert (
        sorted(
            generic_state._get_iteration_path(exec_id)
            for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
        )
        == expected_body_paths
    )


def test_four_level_nested_iterate_rehydrates_without_replay() -> None:
    graph = _four_level_nested_iterate_chain_graph()
    expected_trace, expected_state = _run(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run(GraphExecutionState(graph=graph), stop_after=12)

    resumed_trace, resumed_state = _run(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    _assert_retained_iterate_streams_match_expected(resumed_state, expected_state)


def test_four_level_nested_iterate_failure_matches_compatibility() -> None:
    graph = _four_level_nested_iterate_chain_graph()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("four-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run_graph_with_effects(
            GraphExecutionState(graph=graph),
            fail_source_id="body",
        )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id="body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert generic_trace.count("body") == 1
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_five_level_nested_iterate_matches_compatibility() -> None:
    graph = _five_level_nested_iterate_chain_graph()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("five-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _direct_iterate_fan_in_stream_projection(generic_state) == _direct_iterate_fan_in_stream_projection(
        compatibility_state
    )
    assert sorted(
        generic_state._get_iteration_path(exec_id)
        for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
    ) == [
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1),
        (0, 0, 1, 0, 0),
        (0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0),
    ]


@pytest.mark.parametrize(
    ("outer_collection", "expected_body_paths"),
    [
        ([], []),
        ([[], [[[["c"]]]]], [(1, 0, 0, 0, 0)]),
        ([[[[]]]], []),
        ([[[[], []]]], []),
    ],
    ids=["outer-empty", "mixed-empty", "deepest-empty", "all-empty"],
)
def test_five_level_nested_iterate_closes_empty_frames(
    outer_collection: list[list[list[list[list[str]]]]], expected_body_paths: list[tuple[int, ...]]
) -> None:
    graph = _five_level_nested_iterate_chain_graph(outer_collection=outer_collection)
    generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert (
        sorted(
            generic_state._get_iteration_path(exec_id)
            for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
        )
        == expected_body_paths
    )


def test_five_level_nested_iterate_rehydrates_without_replay() -> None:
    graph = _five_level_nested_iterate_chain_graph()
    expected_trace, expected_state = _run(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run(GraphExecutionState(graph=graph), stop_after=15)

    resumed_trace, resumed_state = _run(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    _assert_retained_iterate_streams_match_expected(resumed_state, expected_state)


def test_five_level_nested_iterate_failure_matches_compatibility() -> None:
    graph = _five_level_nested_iterate_chain_graph()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("five-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run_graph_with_effects(
            GraphExecutionState(graph=graph),
            fail_source_id="body",
        )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id="body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert generic_trace.count("body") == 1
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_six_level_nested_iterate_matches_compatibility() -> None:
    graph = _six_level_nested_iterate_chain_graph()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("six-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert sorted(
        generic_state._get_iteration_path(exec_id)
        for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
    ) == [
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0, 0),
        (0, 0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 0),
    ]


def _nested_collection(value: Any, depth: int) -> list[Any]:
    collection: Any = [value]
    for _ in range(depth - 1):
        collection = [collection]
    return collection


@pytest.mark.parametrize(
    ("outer_collection", "expected_body_paths"),
    [
        ([], []),
        ([[], _nested_collection("c", 5)], [(1, 0, 0, 0, 0, 0)]),
        ([_nested_collection([], 4)], []),
        ([_nested_collection([], 3)], []),
    ],
    ids=["outer-empty", "mixed-empty", "deepest-empty", "intermediate-empty"],
)
def test_six_level_nested_iterate_closes_empty_frames(
    outer_collection: list[Any], expected_body_paths: list[tuple[int, ...]]
) -> None:
    graph = _six_level_nested_iterate_chain_graph(outer_collection=outer_collection)
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("six-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert (
        sorted(
            generic_state._get_iteration_path(exec_id)
            for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
        )
        == expected_body_paths
    )


def test_six_level_nested_iterate_rehydrates_without_replay() -> None:
    graph = _six_level_nested_iterate_chain_graph()
    expected_trace, expected_state = _run(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run(GraphExecutionState(graph=graph), stop_after=8)

    resumed_trace, resumed_state = _run(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_six_level_nested_iterate_failure_matches_compatibility() -> None:
    graph = _six_level_nested_iterate_chain_graph()
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("six-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run_graph_with_effects(
            GraphExecutionState(graph=graph),
            fail_source_id="body",
        )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id="body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert generic_trace.count("body") == 1


def test_seven_level_nested_iterate_matches_compatibility() -> None:
    graph = _serial_nested_iterate_chain_graph_at_depth(7)
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("seven-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert sorted(
        generic_state._get_iteration_path(exec_id)
        for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
    ) == [(0, 0, 0, 0, 0, 0, 0)]


@pytest.mark.parametrize(
    ("outer_collection", "expected_body_paths"),
    [
        ([], []),
        ([[], _nested_collection("c", 6)], [(1, 0, 0, 0, 0, 0, 0)]),
        ([_nested_collection([], 5)], []),
    ],
    ids=["outer-empty", "mixed-empty", "deepest-empty"],
)
def test_seven_level_nested_iterate_closes_empty_frames(
    outer_collection: list[Any], expected_body_paths: list[tuple[int, ...]]
) -> None:
    graph = _serial_nested_iterate_chain_graph_at_depth(7)
    graph.get_node("source").first = outer_collection
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("seven-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert (
        sorted(
            generic_state._get_iteration_path(exec_id)
            for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
        )
        == expected_body_paths
    )


def test_seven_level_nested_iterate_rehydrates_without_replay() -> None:
    graph = _serial_nested_iterate_chain_graph_at_depth(7)
    expected_trace, expected_state = _run(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run(GraphExecutionState(graph=graph), stop_after=8)

    resumed_trace, resumed_state = _run(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_seven_level_nested_iterate_failure_matches_compatibility() -> None:
    graph = _serial_nested_iterate_chain_graph_at_depth(7)
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("seven-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run_graph_with_effects(
            GraphExecutionState(graph=graph),
            fail_source_id="body",
        )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id="body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert generic_trace.count("body") == 1


def test_eight_level_nested_iterate_matches_compatibility() -> None:
    graph = _serial_nested_iterate_chain_graph_at_depth(8)
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("eight-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert sorted(
        generic_state._get_iteration_path(exec_id)
        for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
    ) == [(0, 0, 0, 0, 0, 0, 0, 0)]


@pytest.mark.parametrize(
    ("outer_collection", "expected_body_paths"),
    [
        ([], []),
        ([[], _nested_collection("c", 7)], [(1, 0, 0, 0, 0, 0, 0, 0)]),
        ([_nested_collection([], 6)], []),
    ],
    ids=["outer-empty", "mixed-empty", "deepest-empty"],
)
def test_eight_level_nested_iterate_closes_empty_frames(
    outer_collection: list[Any], expected_body_paths: list[tuple[int, ...]]
) -> None:
    graph = _serial_nested_iterate_chain_graph_at_depth(8)
    graph.get_node("source").first = outer_collection
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("eight-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert (
        sorted(
            generic_state._get_iteration_path(exec_id)
            for exec_id in generic_state._prepared_registry().get_prepared_ids("body")
        )
        == expected_body_paths
    )


def test_eight_level_nested_iterate_rehydrates_without_replay() -> None:
    graph = _serial_nested_iterate_chain_graph_at_depth(8)
    expected_trace, expected_state = _run(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run(GraphExecutionState(graph=graph), stop_after=8)

    resumed_trace, resumed_state = _run(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_eight_level_nested_iterate_failure_matches_compatibility() -> None:
    graph = _serial_nested_iterate_chain_graph_at_depth(8)
    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("eight-level nested Iterate used compatibility materializer"),
    ):
        generic_trace, generic_state = _run_graph_with_effects(
            GraphExecutionState(graph=graph),
            fail_source_id="body",
        )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id="body",
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() and compatibility_state.has_error()
    assert generic_state.is_complete() and compatibility_state.is_complete()
    assert generic_trace.count("body") == 1


def test_nine_level_nested_iterate_remains_on_compatibility_fallback() -> None:
    state = GraphExecutionState(graph=_serial_nested_iterate_chain_graph_at_depth(9))
    materializer = state._materializer()
    with patch.object(materializer, "prepare", wraps=materializer.prepare) as prepare:
        trace, state = _run(state)

    assert trace.count("body") == 1
    assert state.is_complete()
    assert prepare.call_count > 0


@pytest.mark.parametrize("depth", range(2, 9))
def test_nested_iterate_sequence_depth_matcher_matches_supported_inventory(depth: int) -> None:
    state = GraphExecutionState(graph=_serial_nested_iterate_chain_graph_at_depth(depth))

    assert get_nested_iterate_sequence_depth(state) == depth


@pytest.mark.parametrize("depth", [1, 9])
def test_nested_iterate_sequence_depth_matcher_rejects_unsupported_depth(depth: int) -> None:
    state = GraphExecutionState(graph=_serial_nested_iterate_chain_graph_at_depth(depth))

    assert get_nested_iterate_sequence_depth(state) is None


@pytest.mark.parametrize("values", [[1, None, 1], []], ids=["ordered-duplicates-and-none", "empty"])
def test_input_driven_iterate_collect_uses_generic_stream_ownership(values: list[Any]) -> None:
    compatibility_state = GraphExecutionState(graph=_input_driven_iterate_body_collect_graph(values))
    compatibility_trace, compatibility_state = _run(
        compatibility_state,
        force_compatibility_scheduler=True,
    )

    generic_state = GraphExecutionState(graph=_input_driven_iterate_body_collect_graph(values))
    with patch.object(
        generic_state._materializer(),
        "prepare",
        side_effect=AssertionError("input-driven Iterate/Collect must not use materializer.prepare"),
    ):
        generic_trace, generic_state = _run(generic_state)

    assert generic_trace == compatibility_trace
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _source_output(generic_state, "after").value == values


def test_input_driven_iterate_collect_partial_dump_load_preserves_generic_streams() -> None:
    graph = _input_driven_iterate_body_collect_graph(["first", None, "last"])
    expected_trace, expected_state = _run_graph(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run_graph(GraphExecutionState(graph=graph), stop_after=3)

    with patch.object(
        graph_module._ExecutionMaterializer,
        "prepare",
        side_effect=AssertionError("input-driven checkpoint must not use materializer.prepare"),
    ):
        resumed_trace, resumed_state = _run_graph(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert _source_output(resumed_state, "after").value == ["first", None, "last"]
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_input_driven_iterate_collect_failure_matches_compatibility() -> None:
    graph = _input_driven_iterate_body_collect_graph(["first", "last"])
    generic_trace, generic_state = _run_graph_with_effects(GraphExecutionState(graph=graph), fail_source_id="body")
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph), force_compatibility_scheduler=True, fail_source_id="body"
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() == compatibility_state.has_error()
    assert generic_state.is_complete() == compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert generic_state.next() is None


def test_input_driven_iterate_collect_planner_rolls_back_partial_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = GraphExecutionState(graph=_input_driven_iterate_body_collect_graph(["first", "last"]))
    for _ in range(2):
        node = state.next()
        assert node is not None
        state.complete(node.id, node.invoke(Mock()))

    original_create = GraphExecutionState._create_direct_execution_node_copy

    def fail_at_body(self: GraphExecutionState, source_node_id: str, *args: Any, **kwargs: Any):
        node = original_create(self, source_node_id, *args, **kwargs)
        if source_node_id == "body":
            raise RuntimeError("injected input-driven planner failure")
        return node

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", fail_at_body)
    with pytest.raises(RuntimeError, match="injected input-driven planner failure"):
        state.next()

    assert set(state.source_prepared_mapping) == {"producer", "source"}
    assert len(state.execution_graph.nodes) == 2
    assert len(state.execution_graph.edges) == 1

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", original_create)
    trace, state = _run_graph(state)
    assert trace == ["iterate", "iterate", "body", "body", "collect", "after"]
    assert _source_output(state, "after").value == ["first", "last"]
    assert state.is_complete()


def test_input_driven_iterate_collect_retry_after_input_boundary_isolated() -> None:
    graph = _input_driven_iterate_body_collect_graph(["first", None, "last"])
    canceled_trace, canceled_state = _run_until_source(
        GraphExecutionState(graph=graph),
        "source",
    )
    retry_trace, retry_state = _run_graph(GraphExecutionState(graph=graph.model_copy(deep=True)))

    assert canceled_trace == ["producer", "source"]
    assert not canceled_state.is_complete()
    assert retry_trace == [
        "producer",
        "source",
        "iterate",
        "iterate",
        "iterate",
        "body",
        "body",
        "body",
        "collect",
        "after",
    ]
    assert retry_state.id != canceled_state.id
    assert _source_output(retry_state, "after").value == ["first", None, "last"]
    assert retry_state.is_complete()


def test_direct_iterate_body_collect_fresh_execution_preserves_order_and_none() -> None:
    collection = ["first", None, "last"]
    state = GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection))

    with (
        patch.object(
            graph_module._ExecutionMaterializer,
            "prepare",
            side_effect=AssertionError("canonical Iterate -> body -> Collect must not use materializer.prepare"),
        ) as prepare,
        patch.object(
            graph_module._ExecutionMaterializer,
            "_get_collect_iteration_mapping_groups",
            side_effect=AssertionError("canonical Iterate -> body -> Collect must not group collector inputs"),
        ) as group_collector_inputs,
    ):
        trace, state = _run_graph(state)

    assert trace.count("iterate") == len(collection)
    assert trace.count("body") == len(collection)
    assert trace.count("collect") == 1
    assert _source_output(state, "collect").collection == collection
    assert state.is_complete()
    assert isinstance(state._execution_scheduler, _GenericGraphSchedulerAdapter)
    prepare.assert_not_called()
    group_collector_inputs.assert_not_called()


def test_direct_iterate_body_collect_fresh_construction_and_execution_do_not_instantiate_materializer() -> None:
    collection = ["first", None, "last"]
    with patch.object(
        graph_module,
        "_ExecutionMaterializer",
        side_effect=AssertionError("fresh direct Iterate/Collect must not instantiate materializer"),
    ) as materializer:
        trace, state = _run_graph(GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection)))

    assert trace == ["source", "iterate", "iterate", "iterate", "body", "body", "body", "collect"]
    assert _source_output(state, "collect").collection == collection
    assert state.is_complete()
    materializer.assert_not_called()


def test_direct_iterate_body_collect_checkpoint_rehydration_does_not_instantiate_materializer() -> None:
    collection = ["first", None, "last"]
    partial_trace, partial_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection)), stop_after=2
    )

    with patch.object(
        graph_module,
        "_ExecutionMaterializer",
        side_effect=AssertionError("direct Iterate/Collect checkpoint must not instantiate materializer"),
    ) as materializer:
        resumed_trace, resumed_state = _run_graph(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == [
        "source",
        "iterate",
        "iterate",
        "iterate",
        "body",
        "body",
        "body",
        "collect",
    ]
    assert _source_output(resumed_state, "collect").collection == collection
    assert resumed_state.is_complete()
    materializer.assert_not_called()


def test_legacy_direct_iterate_body_collect_snapshot_uses_compatibility_materializer() -> None:
    collection = ["first", None, "last"]
    partial_trace, partial_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection)), stop_after=1
    )
    snapshot = dump_execution_state(partial_state)
    snapshot.pop("execution_state_version")
    snapshot.pop("execution_effects")
    materializer_type = graph_module._ExecutionMaterializer

    with patch.object(graph_module, "_ExecutionMaterializer", wraps=materializer_type) as materializer:
        restored = load_execution_state(snapshot)
        assert restored._legacy_snapshot_loaded
        assert isinstance(restored._scheduler(), _GenericGraphSchedulerAdapter)
        resumed_trace, restored = _run_graph(restored)

    assert materializer.call_count == 1
    assert partial_trace + resumed_trace == [
        "source",
        "iterate",
        "iterate",
        "iterate",
        "body",
        "body",
        "body",
        "collect",
    ]
    assert _source_output(restored, "collect").collection == collection
    assert restored.is_complete()


def test_direct_iterate_fan_in_fresh_execution_owns_two_stream_expansion() -> None:
    state = GraphExecutionState(graph=_direct_iterate_fan_in_graph(left=[None, "left-last"], right=["right-first"]))

    with (
        patch.object(
            graph_module._ExecutionMaterializer,
            "prepare",
            side_effect=AssertionError("fresh direct fan-in must not use materializer.prepare"),
        ) as prepare,
        patch.object(
            graph_module._ExecutionMaterializer,
            "_get_collect_iteration_mapping_groups",
            side_effect=AssertionError("fresh direct fan-in must not group collector inputs"),
        ) as group_collector_inputs,
    ):
        trace, state = _run_graph(state)

    assert trace.count("collect") == 1
    assert trace == ["left_source", "right_source", "left_iterate", "right_iterate", "left_iterate", "collect"]
    assert _source_output(state, "collect").collection == [None, "left-last", "right-first"]
    streams = {
        stream.owner_id: stream
        for stream in state._generic_runtime().streams.values()
        if stream.owner_id in {"left_iterate", "right_iterate"}
    }
    assert {owner_id: (stream.values, stream.closed) for owner_id, stream in streams.items()} == {
        "left_iterate": ((None, "left-last"), True),
        "right_iterate": (("right-first",), True),
    }
    assert state.is_complete()
    prepare.assert_not_called()
    group_collector_inputs.assert_not_called()


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([], ["right", None], ["right", None]),
        (["left", None], [], ["left", None]),
        ([], [], []),
    ],
    ids=["left-empty", "right-empty", "both-empty"],
)
def test_direct_iterate_fan_in_fresh_execution_closes_empty_streams(
    left: list[Any], right: list[Any], expected: list[Any]
) -> None:
    trace, state = _run_graph(GraphExecutionState(graph=_direct_iterate_fan_in_graph(left=left, right=right)))

    assert trace.count("collect") == 1
    assert _source_output(state, "collect").collection == expected
    streams = {
        stream.owner_id: stream
        for stream in state._generic_runtime().streams.values()
        if stream.owner_id in {"left_iterate", "right_iterate"}
    }
    assert set(streams) == {"left_iterate", "right_iterate"}
    assert all(stream.closed for stream in streams.values())
    assert state.is_complete()


def test_direct_iterate_fan_in_gates_collect_until_both_streams_close_and_rehydrates() -> None:
    graph = _direct_iterate_fan_in_graph(left=["left-0", "left-1"], right=["right-0", "right-1"])
    expected_trace, expected_state = _run_graph_with_effects(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run_graph_with_effects(GraphExecutionState(graph=graph), stop_after=3)

    assert partial_trace == ["left_source", "right_source", "left_iterate"]
    assert "collect" not in partial_trace
    collect_exec_ids = partial_state.source_prepared_mapping["collect"]
    assert len(collect_exec_ids) == 1
    collect_exec_id = next(iter(collect_exec_ids))
    assert collect_exec_id not in partial_state.executed
    assert not partial_state._generic_runtime().streams[partial_state._iteration_stream_id("left_iterate", ())].closed

    restored = load_execution_state(dump_execution_state(partial_state))
    assert _direct_iterate_fan_in_stream_projection(restored) == _direct_iterate_fan_in_stream_projection(partial_state)
    assert _direct_iterate_fan_in_execution_ref_projection(
        restored, execution_ids=set(partial_state.execution_refs)
    ) == _direct_iterate_fan_in_execution_ref_projection(partial_state)
    assert _effect_ledger_projection(restored) == _effect_ledger_projection(partial_state)
    resumed_trace, resumed_state = _run_graph_with_effects(restored)

    assert partial_trace + resumed_trace == expected_trace
    assert _source_output(resumed_state, "collect").collection == ["left-0", "left-1", "right-0", "right-1"]
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _direct_iterate_fan_in_stream_projection(resumed_state) == _direct_iterate_fan_in_stream_projection(
        expected_state
    )
    assert _direct_iterate_fan_in_execution_ref_projection(
        resumed_state
    ) == _direct_iterate_fan_in_execution_ref_projection(expected_state)
    assert _effect_ledger_projection(resumed_state) == _effect_ledger_projection(expected_state)
    assert collect_exec_id in resumed_state.executed
    assert resumed_state.is_complete()


def _assert_direct_iterate_fan_in_matches_forced_compatibility_scheduler(left: list[Any], right: list[Any]) -> None:
    generic_trace, generic_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_fan_in_graph(left=left, right=right))
    )
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_fan_in_graph(left=left, right=right)),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert (
        _source_output(generic_state, "collect").collection == _source_output(compatibility_state, "collect").collection
    )
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()


@pytest.mark.parametrize(
    ("left", "right"),
    [(["left", None], ["right"]), ([], ["right", None]), ([], [])],
    ids=["ordered-none", "mixed-empty", "both-empty"],
)
def test_direct_iterate_fan_in_parity_cases(left: list[Any], right: list[Any]) -> None:
    _assert_direct_iterate_fan_in_matches_forced_compatibility_scheduler(left, right)


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_direct_iterate_fan_in_partial_dump_load_parity(
    force_compatibility_scheduler: bool,
) -> None:
    graph = _direct_iterate_fan_in_graph(left=["left-0", "left-1"], right=["right-0", "right-1"])
    expected_trace, expected_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph), force_compatibility_scheduler=force_compatibility_scheduler
    )
    partial_trace, partial_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=force_compatibility_scheduler,
        stop_after=3,
    )

    restored = load_execution_state(dump_execution_state(partial_state))
    assert _direct_iterate_fan_in_stream_projection(restored) == _direct_iterate_fan_in_stream_projection(partial_state)
    assert _direct_iterate_fan_in_execution_ref_projection(
        restored, execution_ids=set(partial_state.execution_refs)
    ) == _direct_iterate_fan_in_execution_ref_projection(partial_state)
    assert _effect_ledger_projection(restored) == _effect_ledger_projection(partial_state)
    if force_compatibility_scheduler:
        _restore_compatibility_scheduler(restored)
    resumed_trace, resumed_state = _run_graph_with_effects(
        restored, force_compatibility_scheduler=force_compatibility_scheduler
    )

    assert partial_trace + resumed_trace == expected_trace
    assert _source_output(resumed_state, "collect").collection == ["left-0", "left-1", "right-0", "right-1"]
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _direct_iterate_fan_in_stream_projection(resumed_state) == _direct_iterate_fan_in_stream_projection(
        expected_state
    )
    assert _direct_iterate_fan_in_execution_ref_projection(
        resumed_state
    ) == _direct_iterate_fan_in_execution_ref_projection(expected_state)
    assert _effect_ledger_projection(resumed_state) == _effect_ledger_projection(expected_state)
    assert resumed_state.is_complete()


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_direct_iterate_fan_in_failure_does_not_execute_collect(
    force_compatibility_scheduler: bool,
) -> None:
    trace, state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_fan_in_graph(left=["left"], right=["right"])),
        force_compatibility_scheduler=force_compatibility_scheduler,
        fail_source_id="right_iterate",
    )

    assert trace[-1] == "right_iterate"
    assert "collect" not in trace
    assert state.has_error()
    assert state.is_complete()
    restored = load_execution_state(dump_execution_state(state))
    assert restored.has_error()
    assert restored.is_complete()


@pytest.mark.parametrize(
    ("fail_source_id", "expected_trace"),
    [
        ("right_source", ["left_source", "right_source"]),
        ("right_iterate", ["left_source", "right_source", "left_iterate", "right_iterate"]),
    ],
    ids=["source-failure", "iterator-failure"],
)
def test_direct_iterate_fan_in_apply_failure_matches_compatibility_scheduler(
    fail_source_id: str, expected_trace: list[str]
) -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_direct_iterate_fan_in_graph(left=["left"], right=["right"])),
        fail_source_id=fail_source_id,
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_direct_iterate_fan_in_graph(left=["left"], right=["right"])),
        force_compatibility_scheduler=True,
        fail_source_id=fail_source_id,
    )

    assert generic_trace == compatibility_trace == expected_trace
    for state in (generic_state, compatibility_state):
        assert state.next() is None
        assert {
            state.prepared_source_mapping.get(node_id, node_id): message for node_id, message in state.errors.items()
        } == {fail_source_id: "injected failure"}
        assert "collect" not in {
            state.prepared_source_mapping.get(execution_id, execution_id) for execution_id in state.executed
        }
        assert state.is_complete()
        restored = load_execution_state(dump_execution_state(state))
        assert _direct_iterate_fan_in_stream_projection(restored) == _direct_iterate_fan_in_stream_projection(state)
        assert _effect_ledger_projection(restored) == _effect_ledger_projection(state)
        assert restored.next() is None
        assert restored.is_complete()

    assert _effect_ledger_projection(generic_state) == _effect_ledger_projection(compatibility_state)
    assert _direct_iterate_fan_in_stream_projection(generic_state) == _direct_iterate_fan_in_stream_projection(
        compatibility_state
    )


def test_direct_iterate_fan_in_rolls_back_partial_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    state = GraphExecutionState(graph=_direct_iterate_fan_in_graph(left=["left"], right=["right"]))
    _partial_trace, state = _run_graph(state, stop_after=2)

    original_create = GraphExecutionState._create_direct_execution_node_copy

    def fail_at_right_iterate(self: GraphExecutionState, source_node_id: str, *args: Any, **kwargs: Any):
        if source_node_id == "right_iterate":
            raise RuntimeError("injected fan-in planner failure")
        return original_create(self, source_node_id, *args, **kwargs)

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", fail_at_right_iterate)
    with pytest.raises(RuntimeError, match="injected fan-in planner failure"):
        state.next()

    assert set(state.source_prepared_mapping) == {"left_source", "right_source"}
    assert len(state.execution_graph.nodes) == 2
    assert not state.execution_graph.edges
    assert not state._generic_runtime().streams

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", original_create)
    trace, state = _run_graph(state)

    assert trace == ["left_iterate", "right_iterate", "collect"]
    assert _source_output(state, "collect").collection == ["left", "right"]
    assert state.is_complete()


def test_direct_iterate_three_fan_in_fresh_execution_owns_sorted_stream_expansion() -> None:
    graph = _direct_iterate_three_fan_in_graph(a=["dup", None, "dup"], m=["middle"], z=[None, "dup"], unsorted=True)
    assert tuple(graph.nodes) == (
        "collect",
        "z_source",
        "z_iterate",
        "a_source",
        "a_iterate",
        "m_source",
        "m_iterate",
    )
    state = GraphExecutionState(graph=graph)
    fan_in = state._get_direct_iterate_collect_nodes()
    assert isinstance(fan_in, graph_module._DirectIterateCollectFanIn)
    assert fan_in.branches == (
        ("a_source", "a_iterate"),
        ("m_source", "m_iterate"),
        ("z_source", "z_iterate"),
    )

    with (
        patch.object(
            graph_module._ExecutionMaterializer,
            "prepare",
            side_effect=AssertionError("fresh three-branch fan-in must not use materializer.prepare"),
        ) as prepare,
        patch.object(
            graph_module._ExecutionMaterializer,
            "_get_collect_iteration_mapping_groups",
            side_effect=AssertionError("fresh three-branch fan-in must not group collector inputs"),
        ) as group_collector_inputs,
    ):
        trace, state = _run_graph(state)

    assert trace == [
        "a_source",
        "m_source",
        "z_source",
        "a_iterate",
        "m_iterate",
        "z_iterate",
        "a_iterate",
        "z_iterate",
        "a_iterate",
        "collect",
    ]
    assert _source_output(state, "collect").collection == ["dup", None, "dup", "middle", None, "dup"]
    streams = {
        stream.owner_id: stream
        for stream in state._generic_runtime().streams.values()
        if stream.owner_id in {"a_iterate", "m_iterate", "z_iterate"}
    }
    assert {owner_id: (stream.values, stream.closed) for owner_id, stream in streams.items()} == {
        "a_iterate": (("dup", None, "dup"), True),
        "m_iterate": (("middle",), True),
        "z_iterate": ((None, "dup"), True),
    }
    assert state.is_complete()
    prepare.assert_not_called()
    group_collector_inputs.assert_not_called()


@pytest.mark.parametrize(
    ("a", "m", "z"),
    [
        ([], [], []),
        (["a"], [], []),
        ([], ["m"], []),
        ([], [], ["z"]),
        (["a"], ["m"], []),
        (["a"], [], ["z"]),
        ([], ["m"], ["z"]),
    ],
    ids=["all-empty", "a-only", "m-only", "z-only", "a-m", "a-z", "m-z"],
)
def test_direct_iterate_three_fan_in_closes_every_empty_stream(a: list[Any], m: list[Any], z: list[Any]) -> None:
    trace, state = _run_graph(GraphExecutionState(graph=_direct_iterate_three_fan_in_graph(a=a, m=m, z=z)))

    assert trace.count("collect") == 1
    assert _source_output(state, "collect").collection == a + m + z
    streams = {
        stream.owner_id: stream
        for stream in state._generic_runtime().streams.values()
        if stream.owner_id in {"a_iterate", "m_iterate", "z_iterate"}
    }
    assert set(streams) == {"a_iterate", "m_iterate", "z_iterate"}
    assert all(stream.closed for stream in streams.values())
    assert state.is_complete()


def test_direct_iterate_three_fan_in_partial_dump_load_resumes_without_duplicate_work() -> None:
    graph = _direct_iterate_three_fan_in_graph(a=["a-0", "a-1"], m=["m-0", "m-1"], z=["z-0", "z-1"])
    expected_trace, expected_state = _run_graph_with_effects(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run_graph_with_effects(GraphExecutionState(graph=graph), stop_after=5)

    assert partial_trace == ["a_source", "m_source", "z_source", "a_iterate", "m_iterate"]
    assert "collect" not in partial_trace
    collect_exec_ids = partial_state.source_prepared_mapping["collect"]
    assert len(collect_exec_ids) == 1
    collect_exec_id = next(iter(collect_exec_ids))
    assert collect_exec_id not in partial_state.executed

    restored = load_execution_state(dump_execution_state(partial_state))
    assert _direct_iterate_fan_in_stream_projection(restored) == _direct_iterate_fan_in_stream_projection(partial_state)
    assert _direct_iterate_fan_in_execution_ref_projection(
        restored, execution_ids=set(partial_state.execution_refs)
    ) == _direct_iterate_fan_in_execution_ref_projection(partial_state)
    assert _effect_ledger_projection(restored) == _effect_ledger_projection(partial_state)
    resumed_trace, resumed_state = _run_graph_with_effects(restored)

    assert partial_trace + resumed_trace == expected_trace
    assert _source_output(resumed_state, "collect").collection == ["a-0", "a-1", "m-0", "m-1", "z-0", "z-1"]
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _direct_iterate_fan_in_stream_projection(resumed_state) == _direct_iterate_fan_in_stream_projection(
        expected_state
    )
    assert _direct_iterate_fan_in_execution_ref_projection(
        resumed_state
    ) == _direct_iterate_fan_in_execution_ref_projection(expected_state)
    assert _effect_ledger_projection(resumed_state) == _effect_ledger_projection(expected_state)
    assert collect_exec_id in resumed_state.executed
    assert resumed_state.is_complete()


@pytest.mark.parametrize(
    ("a", "m", "z"),
    [
        (["a", None, "a"], ["m"], ["z", None]),
        ([], ["m", "m"], []),
        ([], [], []),
    ],
    ids=["ordered-none-duplicates", "mixed-empty-duplicates", "all-empty"],
)
def test_direct_iterate_three_fan_in_matches_forced_compatibility_scheduler(
    a: list[Any], m: list[Any], z: list[Any]
) -> None:
    generic_trace, generic_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_three_fan_in_graph(a=a, m=m, z=z))
    )
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_three_fan_in_graph(a=a, m=m, z=z)),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert (
        _source_output(generic_state, "collect").collection == _source_output(compatibility_state, "collect").collection
    )
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()


@pytest.mark.parametrize(
    ("fail_source_id", "expected_trace"),
    [
        ("m_source", ["a_source", "m_source"]),
        (
            "m_iterate",
            ["a_source", "m_source", "z_source", "a_iterate", "m_iterate"],
        ),
    ],
    ids=["source-failure", "iterator-failure"],
)
def test_direct_iterate_three_fan_in_failure_matches_compatibility_and_blocks_collect(
    fail_source_id: str, expected_trace: list[str]
) -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_direct_iterate_three_fan_in_graph(a=["a"], m=["m"], z=["z"])),
        fail_source_id=fail_source_id,
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_direct_iterate_three_fan_in_graph(a=["a"], m=["m"], z=["z"])),
        force_compatibility_scheduler=True,
        fail_source_id=fail_source_id,
    )

    assert generic_trace == compatibility_trace == expected_trace
    for state in (generic_state, compatibility_state):
        assert state.next() is None
        assert state.has_error()
        assert "collect" not in {
            state.prepared_source_mapping.get(execution_id, execution_id) for execution_id in state.executed
        }
        assert state.is_complete()
        restored = load_execution_state(dump_execution_state(state))
        assert _direct_iterate_fan_in_stream_projection(restored) == _direct_iterate_fan_in_stream_projection(state)
        assert _effect_ledger_projection(restored) == _effect_ledger_projection(state)
        assert restored.next() is None
        assert restored.is_complete()

    assert _effect_ledger_projection(generic_state) == _effect_ledger_projection(compatibility_state)
    assert _direct_iterate_fan_in_stream_projection(generic_state) == _direct_iterate_fan_in_stream_projection(
        compatibility_state
    )


def test_direct_iterate_three_fan_in_rolls_back_partial_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    state = GraphExecutionState(graph=_direct_iterate_three_fan_in_graph(a=["a"], m=["m"], z=["z"]))
    _partial_trace, state = _run_graph(state, stop_after=3)

    original_create = GraphExecutionState._create_direct_execution_node_copy

    def fail_at_m_iterate(self: GraphExecutionState, source_node_id: str, *args: Any, **kwargs: Any):
        if source_node_id == "m_iterate":
            raise RuntimeError("injected three-branch fan-in planner failure")
        return original_create(self, source_node_id, *args, **kwargs)

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", fail_at_m_iterate)
    with pytest.raises(RuntimeError, match="injected three-branch fan-in planner failure"):
        state.next()

    assert set(state.source_prepared_mapping) == {"a_source", "m_source", "z_source"}
    assert len(state.execution_graph.nodes) == 3
    assert not state.execution_graph.edges
    assert not state._generic_runtime().streams

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", original_create)
    trace, state = _run_graph(state)

    assert trace == ["a_iterate", "m_iterate", "z_iterate", "collect"]
    assert _source_output(state, "collect").collection == ["a", "m", "z"]
    assert state.is_complete()


def test_direct_iterate_four_fan_in_stays_on_compatibility_fallback() -> None:
    graph = _direct_iterate_fan_in_graph_with_branches(
        branches=(
            ("a_source", "a_iterate", ["a"]),
            ("m_source", "m_iterate", ["m"]),
            ("z_source", "z_iterate", ["z"]),
            ("q_source", "q_iterate", ["q"]),
        )
    )
    state = GraphExecutionState(graph=graph)
    assert state._get_direct_iterate_collect_nodes() is None

    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=graph), force_compatibility_scheduler=True
    )

    assert compatibility_trace
    assert sorted(_source_output(compatibility_state, "collect").collection) == ["a", "m", "q", "z"]
    assert compatibility_state.is_complete()


def test_body_iterate_fan_in_fresh_execution_owns_exact_two_branch_shape() -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("z_source", "z_iterate", "z_body", ["z", None, "z"]),
            ("a_source", "a_iterate", "a_body", ["a", None, "a"]),
        )
    )
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 6
    state = GraphExecutionState(graph=graph)
    fan_in = state._get_direct_iterate_collect_nodes()
    assert isinstance(fan_in, graph_module._BodyIterateCollectFanIn)
    assert fan_in.branches == (
        ("a_source", "a_iterate", "a_body"),
        ("z_source", "z_iterate", "z_body"),
    )

    with (
        patch.object(
            graph_module._ExecutionMaterializer,
            "prepare",
            side_effect=AssertionError("body fan-in must not use materializer.prepare"),
        ) as prepare,
        patch.object(
            graph_module._ExecutionMaterializer,
            "_get_collect_iteration_mapping_groups",
            side_effect=AssertionError("body fan-in must not group collector inputs"),
        ) as group_collector_inputs,
    ):
        trace, state = _run_graph(state)

    assert trace.count("a_source") == trace.count("z_source") == 1
    assert trace.count("a_iterate") == 3
    assert trace.count("z_iterate") == 3
    assert trace.count("a_body") == 3
    assert trace.count("z_body") == 3
    assert trace.count("collect") == 1
    assert _source_output(state, "collect").collection == ["a", None, "a", "z", None, "z"]
    assert state.is_complete()
    assert isinstance(state._execution_scheduler, _GenericGraphSchedulerAdapter)
    prepare.assert_not_called()
    group_collector_inputs.assert_not_called()


def test_body_iterate_fan_in_orders_by_source_id_not_body_id() -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("z_source", "z_iterate", "a_body", ["z-0", "z-1"]),
            ("a_source", "a_iterate", "z_body", ["a-0", "a-1"]),
        )
    )
    state = GraphExecutionState(graph=graph)
    fan_in = state._get_direct_iterate_collect_nodes()
    assert isinstance(fan_in, graph_module._BodyIterateCollectFanIn)
    assert fan_in.branches == (
        ("a_source", "a_iterate", "z_body"),
        ("z_source", "z_iterate", "a_body"),
    )

    trace, state = _run_graph_with_effects(state)

    assert trace[-1] == "collect"
    assert _source_output(state, "collect").collection == ["a-0", "a-1", "z-0", "z-1"]
    assert state.is_complete()


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([], ["right", None], ["right", None]),
        (["left", None], [], ["left", None]),
        ([], [], []),
    ],
    ids=["left-empty", "right-empty", "both-empty"],
)
def test_body_iterate_fan_in_closes_empty_branches(left: list[Any], right: list[Any], expected: list[Any]) -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("right_source", "right_iterate", "right_body", right),
            ("left_source", "left_iterate", "left_body", left),
        )
    )
    trace, state = _run_graph(GraphExecutionState(graph=graph))

    assert trace.count("collect") == 1
    assert _source_output(state, "collect").collection == expected
    streams = {
        stream.owner_id: stream
        for stream in state._generic_runtime().streams.values()
        if stream.owner_id in {"left_iterate", "right_iterate"}
    }
    assert set(streams) == {"left_iterate", "right_iterate"}
    assert all(stream.closed for stream in streams.values())
    assert state.is_complete()


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_body_iterate_fan_in_partial_dump_load_preserves_streams_and_effects(
    force_compatibility_scheduler: bool,
) -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("right_source", "right_iterate", "right_body", ["right-0", "right-1"]),
            ("left_source", "left_iterate", "left_body", ["left-0", "left-1"]),
        )
    )
    expected_trace, expected_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph), force_compatibility_scheduler=force_compatibility_scheduler
    )
    partial_trace, partial_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=force_compatibility_scheduler,
        stop_after=3,
    )

    assert "collect" not in partial_trace
    if not force_compatibility_scheduler:
        assert (
            not partial_state._generic_runtime().streams[partial_state._iteration_stream_id("left_iterate", ())].closed
        )
    restored = load_execution_state(dump_execution_state(partial_state))
    assert _direct_iterate_fan_in_stream_projection(restored) == _direct_iterate_fan_in_stream_projection(partial_state)
    assert _direct_iterate_fan_in_execution_ref_projection(
        restored, execution_ids=set(partial_state.execution_refs)
    ) == _direct_iterate_fan_in_execution_ref_projection(partial_state)
    assert _effect_ledger_projection(restored) == _effect_ledger_projection(partial_state)
    if force_compatibility_scheduler:
        _restore_compatibility_scheduler(restored)
    resumed_trace, resumed_state = _run_graph_with_effects(
        restored, force_compatibility_scheduler=force_compatibility_scheduler
    )

    assert partial_trace + resumed_trace == expected_trace
    assert _source_output(resumed_state, "collect").collection == [
        "left-0",
        "left-1",
        "right-0",
        "right-1",
    ]
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _direct_iterate_fan_in_stream_projection(resumed_state) == _direct_iterate_fan_in_stream_projection(
        expected_state
    )
    assert _direct_iterate_fan_in_execution_ref_projection(
        resumed_state
    ) == _direct_iterate_fan_in_execution_ref_projection(expected_state)
    assert _effect_ledger_projection(resumed_state) == _effect_ledger_projection(expected_state)
    assert resumed_state.is_complete()


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_body_iterate_fan_in_source_prepared_dump_load_resumes(
    force_compatibility_scheduler: bool,
) -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("right_source", "right_iterate", "right_body", ["right-0", "right-1"]),
            ("left_source", "left_iterate", "left_body", ["left-0", "left-1"]),
        )
    )
    expected_trace, expected_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph), force_compatibility_scheduler=force_compatibility_scheduler
    )
    partial_trace, partial_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=force_compatibility_scheduler,
        stop_after=2,
    )

    assert partial_trace == expected_trace[:2]
    assert set(partial_trace) == {"left_source", "right_source"}
    assert set(partial_state.source_prepared_mapping) == {"left_source", "right_source"}
    assert "collect" not in partial_state.source_prepared_mapping
    assert not partial_state._generic_runtime().streams
    checkpoint = dump_execution_state(partial_state)
    restored = load_execution_state(checkpoint)
    assert _state_projection(restored) == _state_projection(partial_state)
    assert _direct_iterate_fan_in_stream_projection(restored) == _direct_iterate_fan_in_stream_projection(partial_state)
    assert _direct_iterate_fan_in_execution_ref_projection(
        restored, execution_ids=set(partial_state.execution_refs)
    ) == _direct_iterate_fan_in_execution_ref_projection(partial_state)
    assert _execution_token_projection(restored) == _execution_token_projection(partial_state)
    assert _effect_ledger_projection(restored) == _effect_ledger_projection(partial_state)
    if force_compatibility_scheduler:
        _restore_compatibility_scheduler(restored)

    resumed_trace, resumed_state = _run_graph_with_effects(
        restored, force_compatibility_scheduler=force_compatibility_scheduler
    )

    assert partial_trace + resumed_trace == expected_trace
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _direct_iterate_fan_in_stream_projection(resumed_state) == _direct_iterate_fan_in_stream_projection(
        expected_state
    )
    assert _direct_iterate_fan_in_execution_ref_projection(
        resumed_state
    ) == _direct_iterate_fan_in_execution_ref_projection(expected_state)
    assert _execution_token_projection(resumed_state) == _execution_token_projection(expected_state)
    assert _effect_ledger_projection(resumed_state) == _effect_ledger_projection(expected_state)
    assert resumed_state.is_complete()


def test_body_iterate_fan_in_partial_dump_load_after_one_stream_closes() -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("right_source", "right_iterate", "right_body", ["right-0", "right-1"]),
            ("left_source", "left_iterate", "left_body", ["left-0"]),
        )
    )
    expected_trace, expected_state = _run_graph_with_effects(GraphExecutionState(graph=graph))
    partial_trace, partial_state = _run_graph_with_effects(GraphExecutionState(graph=graph), stop_after=3)

    assert partial_trace == ["left_source", "right_source", "left_iterate"]
    streams = partial_state._generic_runtime().streams
    assert streams[partial_state._iteration_stream_id("left_iterate", ())].closed
    right_stream = streams.get(partial_state._iteration_stream_id("right_iterate", ()))
    assert right_stream is None or not right_stream.closed
    assert "collect" not in partial_trace

    restored = load_execution_state(dump_execution_state(partial_state))
    assert _direct_iterate_fan_in_stream_projection(restored) == _direct_iterate_fan_in_stream_projection(partial_state)
    assert _effect_ledger_projection(restored) == _effect_ledger_projection(partial_state)
    resumed_trace, resumed_state = _run_graph_with_effects(restored)

    assert partial_trace + resumed_trace == expected_trace
    assert _source_output(resumed_state, "collect").collection == ["left-0", "right-0", "right-1"]
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _direct_iterate_fan_in_stream_projection(resumed_state) == _direct_iterate_fan_in_stream_projection(
        expected_state
    )
    assert _effect_ledger_projection(resumed_state) == _effect_ledger_projection(expected_state)
    assert resumed_state.is_complete()


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (["left", None, "left"], ["right", None], ["left", None, "left", "right", None]),
        ([], ["right", None], ["right", None]),
        ([], [], []),
    ],
    ids=["ordered-none-duplicates", "mixed-empty", "all-empty"],
)
def test_body_iterate_fan_in_matches_forced_compatibility_scheduler(
    left: list[Any], right: list[Any], expected: list[Any]
) -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("left_source", "left_iterate", "left_body", left),
            ("right_source", "right_iterate", "right_body", right),
        )
    )
    generic_trace, generic_state = _run_graph_with_effects(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph), force_compatibility_scheduler=True
    )

    assert generic_trace == compatibility_trace
    assert _source_output(generic_state, "collect").collection == expected
    assert (
        _source_output(generic_state, "collect").collection == _source_output(compatibility_state, "collect").collection
    )
    if left or right:
        assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _direct_iterate_fan_in_stream_projection(generic_state) == _direct_iterate_fan_in_stream_projection(
        compatibility_state
    )
    assert _direct_iterate_fan_in_execution_ref_projection(
        generic_state
    ) == _direct_iterate_fan_in_execution_ref_projection(compatibility_state)
    assert _execution_token_projection(generic_state) == _execution_token_projection(compatibility_state)
    assert _effect_ledger_projection(generic_state) == _effect_ledger_projection(compatibility_state)
    assert _execution_edge_projection(generic_state) == _execution_edge_projection(compatibility_state)
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
@pytest.mark.parametrize("fail_source_id", ["right_source", "right_iterate", "right_body"])
def test_body_iterate_fan_in_failure_blocks_collect_and_persists(
    force_compatibility_scheduler: bool, fail_source_id: str
) -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("left_source", "left_iterate", "left_body", ["left"]),
            ("right_source", "right_iterate", "right_body", ["right"]),
        )
    )
    trace, state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=force_compatibility_scheduler,
        fail_source_id=fail_source_id,
    )

    assert trace[-1] == fail_source_id
    assert "collect" not in trace
    assert state.has_error()
    assert state.is_complete()
    assert state.next() is None
    assert tuple(
        sorted(
            (state.prepared_source_mapping.get(execution_id, execution_id), message)
            for execution_id, message in state.errors.items()
        )
    ) == ((fail_source_id, "injected failure"),)
    restored = load_execution_state(dump_execution_state(state))
    assert _state_projection(restored) == _state_projection(state)
    assert _execution_edge_projection(restored) == _execution_edge_projection(state)
    assert _direct_iterate_fan_in_stream_projection(restored) == _direct_iterate_fan_in_stream_projection(state)
    assert _direct_iterate_fan_in_execution_ref_projection(
        restored, execution_ids=set(state.execution_refs)
    ) == _direct_iterate_fan_in_execution_ref_projection(state)
    assert _execution_token_projection(restored) == _execution_token_projection(state)
    assert _effect_ledger_projection(restored) == _effect_ledger_projection(state)
    assert restored.has_error()
    assert tuple(
        sorted(
            (restored.prepared_source_mapping.get(execution_id, execution_id), message)
            for execution_id, message in restored.errors.items()
        )
    ) == ((fail_source_id, "injected failure"),)
    assert restored.next() is None
    assert restored.is_complete()


@pytest.mark.parametrize("fail_source_id", ["right_source", "right_iterate", "right_body"])
def test_body_iterate_fan_in_failure_matches_forced_compatibility_scheduler(fail_source_id: str) -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("left_source", "left_iterate", "left_body", ["left"]),
            ("right_source", "right_iterate", "right_body", ["right"]),
        )
    )
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph), fail_source_id=fail_source_id
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=True,
        fail_source_id=fail_source_id,
    )

    assert generic_trace == compatibility_trace
    assert generic_state.has_error() == compatibility_state.has_error()
    assert generic_state.is_complete() == compatibility_state.is_complete()
    assert {
        generic_state.prepared_source_mapping.get(execution_id, execution_id): message
        for execution_id, message in generic_state.errors.items()
    } == {
        compatibility_state.prepared_source_mapping.get(execution_id, execution_id): message
        for execution_id, message in compatibility_state.errors.items()
    }
    assert _direct_iterate_fan_in_stream_projection(generic_state) == _direct_iterate_fan_in_stream_projection(
        compatibility_state
    )
    assert _direct_iterate_fan_in_execution_ref_projection(
        generic_state
    ) == _direct_iterate_fan_in_execution_ref_projection(compatibility_state)
    assert _execution_token_projection(generic_state) == _execution_token_projection(compatibility_state)
    assert _effect_ledger_projection(generic_state) == _effect_ledger_projection(compatibility_state)
    for state in (generic_state, compatibility_state):
        assert state.has_error()
        assert "collect" not in {
            state.prepared_source_mapping.get(execution_id, execution_id) for execution_id in state.executed
        }
        assert state.is_complete()


def test_body_iterate_fan_in_rolls_back_mid_expansion_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("left_source", "left_iterate", "left_body", ["left"]),
            ("right_source", "right_iterate", "right_body", ["right"]),
        )
    )
    state = GraphExecutionState(graph=graph)
    _partial_trace, state = _run_graph_with_effects(state, stop_after=2)
    before_failure_durable = _body_fan_in_durable_projection(state)
    before_failure_runtime = _body_fan_in_runtime_projection(state)

    original_create = GraphExecutionState._create_direct_execution_node_copy

    def fail_at_right_body(self: GraphExecutionState, source_node_id: str, *args: Any, **kwargs: Any):
        if source_node_id == "right_body":
            raise RuntimeError("injected body fan-in planner failure")
        return original_create(self, source_node_id, *args, **kwargs)

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", fail_at_right_body)
    with pytest.raises(RuntimeError, match="injected body fan-in planner failure"):
        state.next()

    assert _body_fan_in_durable_projection(state) == before_failure_durable
    assert _body_fan_in_runtime_projection(state) == before_failure_runtime

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", original_create)
    trace, state = _run_graph_with_effects(state)
    expected_trace, expected_state = _run_graph_with_effects(
        GraphExecutionState(
            graph=_body_iterate_fan_in_graph(
                branches=(
                    ("left_source", "left_iterate", "left_body", ["left"]),
                    ("right_source", "right_iterate", "right_body", ["right"]),
                )
            )
        )
    )

    assert _partial_trace + trace == expected_trace
    assert trace == ["left_iterate", "right_iterate", "left_body", "right_body", "collect"]
    assert _source_output(state, "collect").collection == ["left", "right"]
    assert _state_projection(state) == _state_projection(expected_state)
    assert _direct_iterate_fan_in_stream_projection(state) == _direct_iterate_fan_in_stream_projection(expected_state)
    assert _direct_iterate_fan_in_execution_ref_projection(state) == _direct_iterate_fan_in_execution_ref_projection(
        expected_state
    )
    assert _execution_token_projection(state) == _execution_token_projection(expected_state)
    assert _effect_ledger_projection(state) == _effect_ledger_projection(expected_state)
    assert state.is_complete()


def test_body_iterate_fan_in_unsupported_topology_uses_compatibility_fallback() -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("m_source", "m_iterate", "m_body", ["m"]),
            ("z_source", "z_iterate", "z_body", ["z"]),
            ("a_source", "a_iterate", "a_body", ["a"]),
        )
    )
    assert GraphExecutionState(graph=graph)._get_direct_iterate_collect_nodes() is None

    generic_trace, generic_state = _run_graph_with_effects(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph), force_compatibility_scheduler=True
    )

    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter) and isinstance(
        compatibility_state._execution_scheduler, _ExecutionScheduler
    )
    assert generic_trace == compatibility_trace
    generic_collection = _source_output(generic_state, "collect").collection
    compatibility_collection = _source_output(compatibility_state, "collect").collection
    # Unsupported body-mediated fan-in stays on the legacy materializer. Its item ordering
    # is not part of this bounded planner's contract, but values (including duplicates) must
    # be retained; only the admitted two-branch planner promises lexical source ordering.
    assert sorted(generic_collection) == sorted(compatibility_collection) == ["a", "m", "z"]
    assert _execution_edge_projection(generic_state) == _execution_edge_projection(compatibility_state)
    assert _direct_iterate_fan_in_stream_projection(generic_state) == _direct_iterate_fan_in_stream_projection(
        compatibility_state
    )
    assert _direct_iterate_fan_in_execution_ref_projection(
        generic_state
    ) == _direct_iterate_fan_in_execution_ref_projection(compatibility_state)
    assert _effect_ledger_projection(generic_state) == _effect_ledger_projection(compatibility_state)
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()


def test_body_iterate_fan_in_downstream_topology_stays_on_compatibility_fallback() -> None:
    graph = _body_iterate_fan_in_graph(
        branches=(
            ("right_source", "right_iterate", "right_body", ["right"]),
            ("left_source", "left_iterate", "left_body", ["left"]),
        )
    )
    graph.add_node(AnyTypeTestInvocation(id="after"))
    graph.add_edge(
        Edge(
            source=EdgeConnection(node_id="collect", field="collection"),
            destination=EdgeConnection(node_id="after", field="value"),
        )
    )

    assert GraphExecutionState(graph=graph)._get_direct_iterate_collect_nodes() is None

    generic_trace, generic_state = _run_graph_with_effects(GraphExecutionState(graph=graph))
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph), force_compatibility_scheduler=True
    )

    assert generic_trace == compatibility_trace
    assert generic_trace[-2:] == ["collect", "after"]
    generic_collection = _source_output(generic_state, "collect").collection
    compatibility_collection = _source_output(compatibility_state, "collect").collection
    assert sorted(generic_collection) == sorted(compatibility_collection) == ["left", "right"]
    assert _source_output(generic_state, "after").value == generic_collection
    assert _source_output(compatibility_state, "after").value == compatibility_collection
    assert _execution_edge_projection(generic_state) == _execution_edge_projection(compatibility_state)
    assert _direct_iterate_fan_in_stream_projection(generic_state) == _direct_iterate_fan_in_stream_projection(
        compatibility_state
    )
    assert _direct_iterate_fan_in_execution_ref_projection(
        generic_state
    ) == _direct_iterate_fan_in_execution_ref_projection(compatibility_state)
    assert _effect_ledger_projection(generic_state) == _effect_ledger_projection(compatibility_state)
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()


def test_direct_iterate_body_collect_fresh_execution_handles_empty_input() -> None:
    state = GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=[]))
    with (
        patch.object(
            graph_module._ExecutionMaterializer,
            "prepare",
            side_effect=AssertionError("canonical empty Iterate -> body -> Collect must not use materializer.prepare"),
        ) as prepare,
        patch.object(
            graph_module._ExecutionMaterializer,
            "_get_collect_iteration_mapping_groups",
            side_effect=AssertionError("canonical empty Iterate -> body -> Collect must not group collector inputs"),
        ) as group_collector_inputs,
    ):
        trace, state = _run_graph(state)

    assert "iterate" not in trace
    assert "body" not in trace
    assert trace[-1] == "collect"
    assert _source_output(state, "collect").collection == []
    assert state.is_complete()
    prepare.assert_not_called()
    group_collector_inputs.assert_not_called()


def test_direct_iterate_body_collect_planner_rolls_back_partial_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=["first", "last"]))
    source_node = state.next()
    assert source_node is not None
    state.complete(source_node.id, source_node.invoke(Mock()))

    original_create = GraphExecutionState._create_direct_execution_node_copy
    failed = False

    def fail_after_body(self: GraphExecutionState, source_node_id: str, *args: Any, **kwargs: Any):
        nonlocal failed
        node = original_create(self, source_node_id, *args, **kwargs)
        if source_node_id == "body" and not failed:
            failed = True
            raise RuntimeError("injected direct planner failure")
        return node

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", fail_after_body)
    with pytest.raises(RuntimeError, match="injected direct planner failure"):
        state.next()

    assert set(state.source_prepared_mapping) == {"source"}
    assert len(state.execution_graph.nodes) == 1
    assert not state.execution_graph.edges

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", original_create)
    trace, state = _run(state)

    assert trace == ["iterate", "iterate", "body", "body", "collect"]
    assert _source_output(state, "collect").collection == ["first", "last"]
    assert state.is_complete()


def test_direct_iterate_body_collect_downstream_planner_rolls_back_partial_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=["first", "last"], with_after=True))
    source_node = state.next()
    assert source_node is not None
    state.complete(source_node.id, source_node.invoke(Mock()))

    original_create = GraphExecutionState._create_direct_execution_node_copy

    def fail_at_downstream(self: GraphExecutionState, source_node_id: str, *args: Any, **kwargs: Any):
        if source_node_id == "after":
            raise RuntimeError("injected downstream planner failure")
        return original_create(self, source_node_id, *args, **kwargs)

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", fail_at_downstream)
    with pytest.raises(RuntimeError, match="injected downstream planner failure"):
        state.next()

    assert set(state.source_prepared_mapping) == {"source"}
    assert len(state.execution_graph.nodes) == 1
    assert not state.execution_graph.edges

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", original_create)
    trace, state = _run_graph(state)

    assert trace == ["iterate", "iterate", "body", "body", "collect", "after"]
    assert _source_output(state, "after").value == ["first", "last"]
    assert state.is_complete()


@pytest.mark.parametrize(
    "collection",
    [
        ["first", None, "last"],
        [],
    ],
    ids=["ordered-values-including-none", "empty"],
)
def test_direct_iterate_body_collect_matches_forced_compatibility_scheduler(collection: list[Any]) -> None:
    generic_trace, generic_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection))
    )
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection)),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert _source_output(generic_state, "collect").collection == collection
    assert _source_output(compatibility_state, "collect").collection == collection
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_direct_iterate_body_collect_partial_dump_load_resume_matches_fresh_execution(
    force_compatibility_scheduler: bool,
) -> None:
    collection = ["first", None, "last"]
    expected_trace, expected_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection)),
        force_compatibility_scheduler=force_compatibility_scheduler,
    )
    partial_trace, partial_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection)),
        force_compatibility_scheduler=force_compatibility_scheduler,
        stop_after=2,
    )

    restored = load_execution_state(dump_execution_state(partial_state))
    if force_compatibility_scheduler:
        _restore_compatibility_scheduler(restored)
    resumed_trace, resumed_state = _run_graph(
        restored,
        force_compatibility_scheduler=force_compatibility_scheduler,
    )

    assert partial_trace + resumed_trace == expected_trace
    assert _source_output(resumed_state, "collect").collection == collection
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    if force_compatibility_scheduler:
        assert isinstance(resumed_state._execution_scheduler, _ExecutionScheduler)
    else:
        assert isinstance(resumed_state._execution_scheduler, _GenericGraphSchedulerAdapter)


def test_direct_iterate_body_collect_empty_checkpoint_rehydrates_without_materializer() -> None:
    _partial_trace, partial_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=[])), stop_after=1
    )
    with (
        patch.object(
            graph_module._ExecutionMaterializer,
            "prepare",
            side_effect=AssertionError("empty canonical checkpoint must not use materializer.prepare"),
        ) as prepare,
        patch.object(
            graph_module._ExecutionMaterializer,
            "_get_collect_iteration_mapping_groups",
            side_effect=AssertionError("empty canonical checkpoint must not group collector inputs"),
        ) as group_collector_inputs,
    ):
        resumed_trace, resumed_state = _run_graph(load_execution_state(dump_execution_state(partial_state)))

    assert resumed_trace == ["collect"]
    assert _source_output(resumed_state, "collect").collection == []
    assert resumed_state.is_complete()
    prepare.assert_not_called()
    group_collector_inputs.assert_not_called()


def test_direct_iterate_body_collect_failure_rehydrates_pending_state_without_replanning() -> None:
    trace, failed_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=["first", "last"])),
        fail_source_id="body",
    )
    assert trace[0] == "source"
    assert trace.count("iterate") == 2
    assert trace[-1] == "body"
    assert failed_state.has_error()

    restored = load_execution_state(dump_execution_state(failed_state))
    assert restored.has_error()
    assert restored.source_prepared_mapping.get("collect")
    assert restored.is_complete()


def test_direct_iterate_body_collect_with_downstream_consumer_uses_private_planner() -> None:
    with (
        patch.object(
            graph_module._ExecutionMaterializer,
            "prepare",
            side_effect=AssertionError("direct Iterate -> body -> Collect -> after must not use materializer.prepare"),
        ) as prepare,
        patch.object(
            graph_module._ExecutionMaterializer,
            "_get_collect_iteration_mapping_groups",
            side_effect=AssertionError("direct downstream Collect must not group collector inputs"),
        ) as group_collector_inputs,
    ):
        trace, state = _run_graph(
            GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=["value"], with_after=True))
        )

    assert trace[-1] == "after"
    assert _source_output(state, "after").value == ["value"]
    assert state.is_complete()
    prepare.assert_not_called()
    group_collector_inputs.assert_not_called()


def test_direct_iterate_body_collect_two_downstream_consumers_admits_exact_shape() -> None:
    state = GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=["value"], downstream_count=2))

    assert len(state.graph.nodes) == 6
    assert len(state.graph.edges) == 5
    assert state._get_direct_iterate_collect_nodes() == (
        "source",
        "iterate",
        "body",
        "collect",
        ("after_1", "after_2"),
    )
    assert state._can_use_direct_iterate_collect_planner()


def test_direct_iterate_body_collect_three_downstream_consumers_uses_private_planner() -> None:
    state = GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=["value"], downstream_count=3))
    assert state._get_direct_iterate_collect_nodes() == (
        "source",
        "iterate",
        "body",
        "collect",
        ("after_1", "after_2", "after_3"),
    )
    assert state._can_use_direct_iterate_collect_planner()

    with (
        patch.object(
            graph_module._ExecutionMaterializer,
            "prepare",
            side_effect=AssertionError("three direct downstream consumers must not use materializer.prepare"),
        ) as prepare,
        patch.object(
            graph_module._ExecutionMaterializer,
            "_get_collect_iteration_mapping_groups",
            side_effect=AssertionError("three direct downstream consumers must not group collector inputs"),
        ) as group_collector_inputs,
    ):
        trace, state = _run_graph(state)

    assert trace == ["source", "iterate", "body", "collect", "after_1", "after_2", "after_3"]
    for downstream_id in ("after_1", "after_2", "after_3"):
        assert _source_output(state, downstream_id).value == ["value"]
    assert state.is_complete()
    prepare.assert_not_called()
    group_collector_inputs.assert_not_called()


@pytest.mark.parametrize("collection", [["first", None, "last"], []], ids=["ordered-none", "empty"])
def test_direct_iterate_body_collect_three_downstream_consumers_matches_compatibility(
    collection: list[Any],
) -> None:
    generic_trace, generic_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection, downstream_count=3))
    )
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection, downstream_count=3)),
        force_compatibility_scheduler=True,
    )

    expected_trace = ["source"] + ["iterate"] * len(collection) + ["body"] * len(collection)
    expected_trace += ["collect", "after_1", "after_2", "after_3"]
    assert generic_trace == expected_trace
    assert generic_trace[:-3] == compatibility_trace[:-3]
    assert sorted(generic_trace[-3:]) == sorted(compatibility_trace[-3:]) == ["after_1", "after_2", "after_3"]
    for state in (generic_state, compatibility_state):
        assert _source_output(state, "collect").collection == collection
        for downstream_id in ("after_1", "after_2", "after_3"):
            assert _source_output(state, downstream_id).value == collection
        assert state.is_complete()


def test_direct_iterate_body_collect_two_downstream_consumers_uses_private_planner() -> None:
    with (
        patch.object(
            graph_module._ExecutionMaterializer,
            "prepare",
            side_effect=AssertionError("two direct downstream consumers must not use materializer.prepare"),
        ) as prepare,
        patch.object(
            graph_module._ExecutionMaterializer,
            "_get_collect_iteration_mapping_groups",
            side_effect=AssertionError("two direct downstream consumers must not group collector inputs"),
        ) as group_collector_inputs,
    ):
        trace, state = _run_graph(
            GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=["value"], downstream_count=2))
        )

    assert trace == ["source", "iterate", "body", "collect", "after_1", "after_2"]
    assert _source_output(state, "after_1").value == ["value"]
    assert _source_output(state, "after_2").value == ["value"]
    assert state.is_complete()
    prepare.assert_not_called()
    group_collector_inputs.assert_not_called()


@pytest.mark.parametrize("collection", [["first", None, "last"], []], ids=["ordered-none", "empty"])
def test_direct_iterate_body_collect_two_downstream_consumers_matches_compatibility(
    collection: list[Any],
) -> None:
    generic_trace, generic_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection, downstream_count=2))
    )
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection, downstream_count=2)),
        force_compatibility_scheduler=True,
    )

    expected_trace = ["source"] + ["iterate"] * len(collection) + ["body"] * len(collection)
    expected_trace += ["collect", "after_1", "after_2"]
    assert generic_trace == expected_trace
    assert generic_trace[:-2] == compatibility_trace[:-2]
    assert sorted(generic_trace[-2:]) == sorted(compatibility_trace[-2:]) == ["after_1", "after_2"]
    for state in (generic_state, compatibility_state):
        assert _source_output(state, "collect").collection == collection
        assert _source_output(state, "after_1").value == collection
        assert _source_output(state, "after_2").value == collection
        assert state.is_complete()


def test_direct_iterate_body_collect_two_downstream_consumers_rolls_back_partial_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = GraphExecutionState(
        graph=_direct_iterate_body_collect_graph(collection=["first", "last"], downstream_count=2)
    )
    source_node = state.next()
    assert source_node is not None
    state.complete(source_node.id, source_node.invoke(Mock()))

    original_create = GraphExecutionState._create_direct_execution_node_copy

    def fail_at_second_downstream(self: GraphExecutionState, source_node_id: str, *args: Any, **kwargs: Any):
        if source_node_id == "after_2":
            raise RuntimeError("injected second downstream planner failure")
        return original_create(self, source_node_id, *args, **kwargs)

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", fail_at_second_downstream)
    with pytest.raises(RuntimeError, match="injected second downstream planner failure"):
        state.next()

    assert set(state.source_prepared_mapping) == {"source"}
    assert len(state.execution_graph.nodes) == 1
    assert not state.execution_graph.edges

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", original_create)
    trace, state = _run_graph(state)

    assert trace == ["iterate", "iterate", "body", "body", "collect", "after_1", "after_2"]
    assert _source_output(state, "after_1").value == ["first", "last"]
    assert _source_output(state, "after_2").value == ["first", "last"]
    assert state.is_complete()


def test_direct_iterate_body_collect_three_downstream_consumers_rolls_back_partial_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = GraphExecutionState(
        graph=_direct_iterate_body_collect_graph(collection=["first", "last"], downstream_count=3)
    )
    source_node = state.next()
    assert source_node is not None
    state.complete(source_node.id, source_node.invoke(Mock()))

    original_create = GraphExecutionState._create_direct_execution_node_copy

    def fail_at_third_downstream(self: GraphExecutionState, source_node_id: str, *args: Any, **kwargs: Any):
        if source_node_id == "after_3":
            raise RuntimeError("injected third downstream planner failure")
        return original_create(self, source_node_id, *args, **kwargs)

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", fail_at_third_downstream)
    with pytest.raises(RuntimeError, match="injected third downstream planner failure"):
        state.next()

    assert set(state.source_prepared_mapping) == {"source"}
    assert len(state.execution_graph.nodes) == 1
    assert not state.execution_graph.edges

    monkeypatch.setattr(GraphExecutionState, "_create_direct_execution_node_copy", original_create)
    trace, state = _run_graph(state)

    assert trace == ["iterate", "iterate", "body", "body", "collect", "after_1", "after_2", "after_3"]
    for downstream_id in ("after_1", "after_2", "after_3"):
        assert _source_output(state, downstream_id).value == ["first", "last"]
    assert state.is_complete()


def test_direct_iterate_body_collect_two_downstream_consumers_checkpoint_rehydrates() -> None:
    collection = ["first", None, "last"]
    expected_trace, expected_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection, downstream_count=2))
    )
    partial_trace, partial_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection, downstream_count=2)),
        stop_after=3,
    )

    resumed_trace, resumed_state = _run_graph(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert _source_output(resumed_state, "collect").collection == collection
    assert _source_output(resumed_state, "after_1").value == collection
    assert _source_output(resumed_state, "after_2").value == collection
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_direct_iterate_body_collect_three_downstream_consumers_checkpoint_rehydrates() -> None:
    collection = ["first", None, "last"]
    expected_trace, expected_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection, downstream_count=3))
    )
    partial_trace, partial_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection, downstream_count=3)),
        stop_after=3,
    )

    resumed_trace, resumed_state = _run_graph(load_execution_state(dump_execution_state(partial_state)))

    assert partial_trace + resumed_trace == expected_trace
    assert _source_output(resumed_state, "collect").collection == collection
    for downstream_id in ("after_1", "after_2", "after_3"):
        assert _source_output(resumed_state, downstream_id).value == collection
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_direct_iterate_body_collect_with_downstream_consumer_matches_forced_compatibility_scheduler() -> None:
    generic_trace, generic_state = _run_graph(
        GraphExecutionState(
            graph=_direct_iterate_body_collect_graph(collection=["first", None, "last"], with_after=True)
        )
    )
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(
            graph=_direct_iterate_body_collect_graph(collection=["first", None, "last"], with_after=True)
        ),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace
    assert generic_trace[-2:] == ["collect", "after"]
    assert _source_output(generic_state, "after").value == ["first", None, "last"]
    assert _source_output(compatibility_state, "after").value == ["first", None, "last"]
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_direct_iterate_body_collect_downstream_checkpoint_after_none_item_rehydrates() -> None:
    collection = ["first", None, "last"]
    expected_trace, expected_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection, with_after=True))
    )
    partial_trace, partial_state = _run_graph(
        GraphExecutionState(graph=_direct_iterate_body_collect_graph(collection=collection, with_after=True)),
        stop_after=3,
    )

    assert partial_trace == ["source", "iterate", "iterate"]
    iterate_exec_id = next(
        exec_node_id
        for exec_node_id, source_node_id in partial_state.prepared_source_mapping.items()
        if source_node_id == "iterate"
        and exec_node_id in partial_state.results
        and partial_state.results[exec_node_id].index == 1  # type: ignore[union-attr]
    )
    snapshot = dump_execution_state(partial_state)
    assert snapshot["results"][iterate_exec_id]["item"] is None

    resumed_trace, resumed_state = _run_graph(load_execution_state(snapshot))

    assert partial_trace + resumed_trace == expected_trace
    assert _source_output(resumed_state, "collect").collection == collection
    assert _source_output(resumed_state, "after").value == collection
    assert resumed_state.is_complete()
    assert _state_projection(resumed_state) == _state_projection(expected_state)


def test_direct_flat_for_completion_persists_continuations_and_result_outputs() -> None:
    trace, state = _run(GraphExecutionState(graph=_flat_for_graph()))

    assert trace == ["for", "body", "return", "for", "body", "return"]
    snapshot = dump_execution_state(state)
    restored = load_execution_state(snapshot)
    final_for_id = max(
        state.source_prepared_mapping["for"],
        key=lambda exec_node_id: state.execution_graph.get_node(exec_node_id).index,
    )
    final_output = state.results[final_for_id]

    assert snapshot["execution_effects"]
    assert final_output.output_collection == [11, 12]
    assert snapshot["execution_refs"] == {}
    assert restored.execution_tokens == {}
    assert restored.execution_refs
    assert restored.results[final_for_id].output_collection == [11, 12]
    assert restored.results[final_for_id].final_state == final_output.final_state
    assert sorted(
        (effect.operation, effect.continuation_kind)
        for effects in state.execution_effects.values()
        for effect in effects
        if effect.kind == "continuation"
    ) == [("complete", "for"), ("complete", "for"), ("start", "for"), ("start", "for")]
    assert restored.is_complete()


def test_legacy_flat_for_snapshot_uses_compatibility_scheduler() -> None:
    _trace, partial_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph()),
        stop_after=1,
    )
    snapshot = dump_execution_state(partial_state)
    snapshot.pop("execution_state_version")
    snapshot.pop("execution_effects")

    restored = load_execution_state(snapshot)
    migrated = load_execution_state(dump_execution_state(restored))

    assert isinstance(restored._scheduler(), _ExecutionScheduler)
    remaining_trace, restored = _run(restored)
    migrated_trace, migrated = _run(migrated)
    assert remaining_trace == ["body", "return", "for", "body", "return"]
    assert migrated_trace == remaining_trace
    assert restored.is_complete()
    assert migrated.is_complete()
    assert _final_for_output(restored).output_collection == [11, 12]
    assert _final_for_output(migrated).output_collection == [11, 12]


def test_terminal_legacy_flat_for_snapshot_can_be_resaved_and_reloaded() -> None:
    _trace, state = _run_graph_with_effects(GraphExecutionState(graph=_flat_for_graph()))
    snapshot = dump_execution_state(state)
    snapshot.pop("execution_state_version")
    snapshot.pop("execution_effects")

    restored = load_execution_state(snapshot)
    migrated = load_execution_state(dump_execution_state(restored))

    assert restored.execution_effects == {}
    assert migrated.is_complete()
    assert _final_for_output(migrated).output_collection == [11, 12]


def test_failed_direct_completion_does_not_persist_a_reference() -> None:
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    state = GraphExecutionState(graph=graph)
    node = state.next()
    assert isinstance(node, AddInvocation)
    before = dump_execution_state(state)

    with pytest.raises(TypeError, match="does not belong to execution node"):
        state.complete(node.id, BooleanOutput(value=True))

    assert dump_execution_state(state) == before


def test_flat_for_fresh_execution_releases_after_loop_consumer() -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph(with_after=True)),
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph(with_after=True)),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace == ["for", "body", "return", "for", "body", "return", "after"]
    after_id = next(
        exec_node_id
        for exec_node_id, source_node_id in generic_state.prepared_source_mapping.items()
        if source_node_id == "after"
    )
    assert generic_state.results[after_id].value == [11, 12]
    assert compatibility_state.results[
        next(
            exec_node_id
            for exec_node_id, source_node_id in compatibility_state.prepared_source_mapping.items()
            if source_node_id == "after"
        )
    ].value == [11, 12]
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    _assert_generic_and_compatibility_schedulers(generic_state, compatibility_state)


def test_flat_for_none_output_matches_compatibility_scheduler() -> None:
    graph = _flat_for_graph(collection=[None, None], body_returns_none=True)
    generic_trace, generic_state = _run_graph_with_effects(GraphExecutionState(graph=graph.model_copy(deep=True)))
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=graph.model_copy(deep=True)),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace == ["for", "body", "return", "for", "body", "return"]
    assert _final_for_output(generic_state).output_collection == [None, None]
    assert _final_for_output(compatibility_state).output_collection == [None, None]
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _execution_identity_projection(generic_state) == _execution_identity_projection(compatibility_state)
    _assert_generic_and_compatibility_schedulers(generic_state, compatibility_state)


def test_flat_for_apply_path_matches_compatibility_scheduler() -> None:
    generic_trace, generic_state = _run_graph_with_effects(GraphExecutionState(graph=_flat_for_graph(with_after=True)))
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph(with_after=True)),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace == ["for", "body", "return", "for", "body", "return", "after"]
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    assert _effect_ledger_projection(generic_state) == _effect_ledger_projection(compatibility_state)
    _assert_generic_and_compatibility_schedulers(generic_state, compatibility_state)
    for state in (generic_state, compatibility_state):
        after_id = next(
            execution_id for execution_id, source_id in state.prepared_source_mapping.items() if source_id == "after"
        )
        assert state.results[after_id].value == [11, 12]
        assert _final_for_output(state).output_collection == [11, 12]

        continuations = sorted(
            state._generic_runtime().continuations.values(), key=lambda continuation: continuation.frame.iteration_path
        )
        assert [
            (
                state.prepared_source_mapping[continuation.owner_id],
                continuation.kind,
                continuation.status,
                tuple(continuation.frame.iteration_path),
                continuation.frame.state_id,
                continuation.frame.workflow_call_depth,
            )
            for continuation in continuations
        ] == [
            ("for", "for", "completed", (0,), state.id, 0),
            ("for", "for", "completed", (1,), state.id, 0),
        ]
        continuation_effects = sorted(
            (
                state.prepared_source_mapping[effect.execution_ref.node_id],
                effect.operation,
                effect.continuation_kind,
                tuple(effect.execution_ref.frame),
            )
            for effects in state.execution_effects.values()
            for effect in effects
            if effect.kind == "continuation"
        )
        assert continuation_effects == [
            ("for", "start", "for", (0,)),
            ("for", "start", "for", (1,)),
            ("return", "complete", "for", (0,)),
            ("return", "complete", "for", (1,)),
        ]


def test_flat_for_apply_partial_rehydration_preserves_continuation_runtime() -> None:
    graph = _flat_for_graph(with_after=True, collection=[None, None], body_returns_none=True)
    resumed_projections: list[tuple[Any, ...]] = []

    for force_compatibility_scheduler in (False, True):
        expected_trace, expected_state = _run_graph_with_effects(
            GraphExecutionState(graph=graph.model_copy(deep=True)),
            force_compatibility_scheduler=force_compatibility_scheduler,
        )
        partial_trace, partial_state = _run_graph_with_effects(
            GraphExecutionState(graph=graph.model_copy(deep=True)),
            force_compatibility_scheduler=force_compatibility_scheduler,
            stop_after_source="return",
        )
        restored = load_execution_state(dump_execution_state(partial_state))
        if force_compatibility_scheduler:
            _restore_compatibility_scheduler(restored)

        assert _effect_ledger_projection(restored) == _effect_ledger_projection(partial_state)
        # Rehydration reconstructs references for all prepared nodes, while the partial in-memory
        # state only has references for nodes reached so far. Compare the durable token/effect projection.
        assert _execution_token_projection(restored) == _execution_token_projection(partial_state)
        _assert_execution_identity_consistent(restored)
        assert any(
            (effect.get("kind") if isinstance(effect, dict) else effect.kind) == "continuation"
            and (effect.get("operation") if isinstance(effect, dict) else effect.operation) == "complete"
            and (effect.get("payload") if isinstance(effect, dict) else effect.payload)
            == {"output": None, "state": None, "continue_condition": True}
            for effects in restored.execution_effects.values()
            for effect in effects
        )

        remaining_trace, resumed_state = _run_graph_with_effects(
            restored,
            force_compatibility_scheduler=force_compatibility_scheduler,
        )

        assert partial_trace + remaining_trace == expected_trace
        assert _state_projection(resumed_state) == _state_projection(expected_state)
        assert _continuation_projection(resumed_state) == _continuation_projection(expected_state)
        assert _effect_ledger_projection(resumed_state) == _effect_ledger_projection(expected_state)
        assert _execution_token_projection(resumed_state) == _execution_token_projection(expected_state)
        assert _effect_ledger_projection(resumed_state) == _effect_ledger_projection(expected_state)
        assert _final_for_output(resumed_state).output_collection == [None, None]
        _assert_execution_identity_consistent(resumed_state)
        if force_compatibility_scheduler:
            assert isinstance(resumed_state._execution_scheduler, _ExecutionScheduler)
        else:
            assert isinstance(resumed_state._execution_scheduler, _GenericGraphSchedulerAdapter)
        resumed_projections.append(
            (
                _state_projection(resumed_state),
                _continuation_projection(resumed_state),
                _effect_ledger_projection(resumed_state),
                _execution_identity_projection(resumed_state),
            )
        )

    assert resumed_projections[0] == resumed_projections[1]


@pytest.mark.parametrize(
    ("fail_source_id", "expected_trace"),
    [
        ("body", ["for", "body"]),
        ("return", ["for", "body", "return"]),
    ],
)
def test_flat_for_failure_matches_compatibility_scheduler(fail_source_id: str, expected_trace: list[str]) -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph()),
        fail_source_id=fail_source_id,
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph()),
        force_compatibility_scheduler=True,
        fail_source_id=fail_source_id,
    )

    assert generic_trace == compatibility_trace == expected_trace
    for state in (generic_state, compatibility_state):
        assert state.next() is None
        assert {
            state.prepared_source_mapping.get(node_id, node_id): message for node_id, message in state.errors.items()
        } == {fail_source_id: "injected failure"}
        assert state.is_complete()
        assert sum(len(effects) for effects in state.execution_effects.values()) == 1
        assert all(item.status == "running" for item in state._generic_runtime().continuations.values())
        _assert_execution_identity_consistent(state)
        restored = load_execution_state(dump_execution_state(state))
        if isinstance(state._execution_scheduler, _ExecutionScheduler):
            _restore_compatibility_scheduler(restored)
        assert _effect_ledger_projection(restored) == _effect_ledger_projection(state)
        assert _continuation_projection(restored) == _continuation_projection(state)
        _assert_execution_identity_consistent(restored)
        assert {
            state.prepared_source_mapping.get(node_id, node_id): message for node_id, message in restored.errors.items()
        } == {fail_source_id: "injected failure"}
        assert restored.next() is None
        assert restored.is_complete()
        if isinstance(state._execution_scheduler, _ExecutionScheduler):
            assert isinstance(restored._execution_scheduler, _ExecutionScheduler)
        else:
            assert restored._execution_scheduler is None
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


@pytest.mark.parametrize("stop_after_source", ["for", "body"])
def test_flat_for_partial_rehydration_matches_compatibility_scheduler(stop_after_source: str) -> None:
    graph = _flat_for_graph()
    expected_trace, expected_state = _run_graph_with_effects(GraphExecutionState(graph=graph.model_copy(deep=True)))
    resumed_projections: list[tuple[Any, ...]] = []

    for force_compatibility_scheduler in (False, True):
        partial_trace, partial_state = _run_graph_with_effects(
            GraphExecutionState(graph=graph.model_copy(deep=True)),
            stop_after_source=stop_after_source,
            force_compatibility_scheduler=force_compatibility_scheduler,
        )
        snapshot = dump_execution_state(partial_state)
        restored = load_execution_state(snapshot)
        if force_compatibility_scheduler:
            _restore_compatibility_scheduler(restored)

        remaining_trace, resumed_state = _run_graph_with_effects(
            restored,
            force_compatibility_scheduler=force_compatibility_scheduler,
        )

        assert partial_trace + remaining_trace == expected_trace
        assert _state_projection(resumed_state) == _state_projection(expected_state)
        # Rehydration reconstructs durable execution references for all prepared nodes; a fresh
        # in-memory run does not retain those references after terminal cleanup. Compare durable
        # token/effect state and continuation records instead.
        assert _execution_token_projection(resumed_state) == _execution_token_projection(expected_state)
        assert _continuation_projection(resumed_state) == _continuation_projection(expected_state)
        assert _effect_ledger_projection(resumed_state) == _effect_ledger_projection(expected_state)
        _assert_execution_identity_consistent(resumed_state)
        if force_compatibility_scheduler:
            assert isinstance(resumed_state._execution_scheduler, _ExecutionScheduler)
        else:
            assert isinstance(resumed_state._execution_scheduler, _GenericGraphSchedulerAdapter)
        resumed_projections.append(
            (
                _state_projection(resumed_state),
                _continuation_projection(resumed_state),
                _effect_ledger_projection(resumed_state),
            )
        )

    assert resumed_projections[0] == resumed_projections[1]


def test_flat_for_frame_and_continuation_identity_matches_compatibility_scheduler() -> None:
    generic_trace, generic_state = _run_graph_with_effects(GraphExecutionState(graph=_flat_for_graph()), stop_after=4)
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph()),
        force_compatibility_scheduler=True,
        stop_after=4,
    )

    assert generic_trace == compatibility_trace == ["for", "body", "return", "for"]
    assert _execution_identity_projection(generic_state) == _execution_identity_projection(compatibility_state)
    assert _continuation_projection(generic_state) == _continuation_projection(compatibility_state)
    _assert_generic_and_compatibility_schedulers(generic_state, compatibility_state)
    for state in (generic_state, compatibility_state):
        continuations = sorted(
            state._generic_runtime().continuations.values(), key=lambda item: item.frame.iteration_path
        )
        assert len(continuations) == 2
        assert [tuple(item.frame.iteration_path) for item in continuations] == [(0,), (1,)]
        assert {state.prepared_source_mapping[item.owner_id] for item in continuations} == {"for"}
        assert {item.frame.state_id for item in continuations} == {state.id}
        assert {item.frame.workflow_call_depth for item in continuations} == {0}
        frame_ids = [item.frame.frame_id for item in continuations]
        assert all(frame_ids)
        assert len(set(frame_ids)) == len(frame_ids)
        assert all(
            item.frame.model_dump(mode="json")
            == state._expected_execution_ref(item.owner_id).frame.model_dump(mode="json")
            for item in continuations
        )
        assert [item.status for item in continuations] == ["completed", "running"]
        _assert_execution_identity_consistent(state)

    generic_frames = {item.frame.frame_id for item in generic_state._generic_runtime().continuations.values()}
    compatibility_frames = {
        item.frame.frame_id for item in compatibility_state._generic_runtime().continuations.values()
    }
    assert generic_frames.isdisjoint(compatibility_frames)

    other_trace, other_state = _run_graph_with_effects(GraphExecutionState(graph=_flat_for_graph()), stop_after=4)
    assert other_trace == generic_trace
    other_frames = {item.frame.frame_id for item in other_state._generic_runtime().continuations.values()}
    generic_frames = {item.frame.frame_id for item in generic_state._generic_runtime().continuations.values()}
    assert other_frames.isdisjoint(generic_frames)


def test_empty_literal_flat_for_uses_generic_scheduler() -> None:
    state = GraphExecutionState(graph=_flat_for_graph(collection=[]))

    assert state._can_use_generic_scheduler()
    assert isinstance(state._scheduler(), _GenericGraphSchedulerAdapter)


def test_empty_literal_flat_for_matches_compatibility_scheduler() -> None:
    generic_trace, generic_state = _run_graph_with_effects(GraphExecutionState(graph=_flat_for_graph(collection=[])))
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph(collection=[])),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace == []
    assert _final_for_output(generic_state).output_collection == []
    assert _final_for_output(compatibility_state).output_collection == []
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)


def test_input_driven_empty_flat_for_uses_generic_scheduler_and_exact_terminal_semantics() -> None:
    state = GraphExecutionState(graph=_flat_for_graph(input_collection=[]))

    assert state._can_use_generic_scheduler()
    trace, state = _run_graph_with_effects(state)

    assert trace == ["collection"]
    assert isinstance(state._execution_scheduler, _GenericGraphSchedulerAdapter)
    final_output = _final_for_output(state)
    assert final_output.item is None
    assert final_output.index == -1
    assert final_output.total == 0
    assert final_output.state == LoopState()
    assert final_output.output_collection == []
    assert final_output.final_state == LoopState()
    assert "body" not in state.source_prepared_mapping
    assert "return" not in state.source_prepared_mapping
    assert state.is_complete()
    assert list(state._generic_runtime().continuations.values()) == []
    _assert_execution_identity_consistent(state)


def test_input_driven_empty_flat_for_matches_compatibility_scheduler() -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph(input_collection=[]))
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph(input_collection=[])),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace == ["collection"]
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    _assert_execution_identity_consistent(generic_state)
    _assert_execution_identity_consistent(compatibility_state)


def test_input_driven_empty_flat_for_checkpoint_after_producer_does_not_replay() -> None:
    expected_trace, expected_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph(input_collection=[]))
    )
    partial_trace, partial_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph(input_collection=[])),
        stop_after_source="collection",
    )

    restored = load_execution_state(dump_execution_state(partial_state))
    resumed_trace, restored = _run_graph_with_effects(restored)
    terminal_reload = load_execution_state(dump_execution_state(restored))

    assert expected_trace == partial_trace == ["collection"]
    assert resumed_trace == []
    assert isinstance(restored._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert _state_projection(restored) == _state_projection(expected_state)
    assert terminal_reload.is_complete()
    assert _state_projection(terminal_reload) == _state_projection(restored)
    assert _final_for_output(terminal_reload).output_collection == []
    _assert_execution_identity_consistent(restored)
    _assert_execution_identity_consistent(terminal_reload)


def test_input_driven_empty_flat_for_with_downstream_remains_compatibility_owned() -> None:
    state = GraphExecutionState(graph=_flat_for_graph(input_collection=[], with_after=True))

    assert state._can_use_generic_scheduler() is False
    assert isinstance(state._scheduler(), _ExecutionScheduler)


def test_input_driven_empty_flat_for_with_index_body_edge_remains_compatibility_owned() -> None:
    graph = _flat_for_graph(input_collection=[])
    body_edge = next(edge for edge in graph.edges if edge.destination.node_id == "body")
    graph.delete_edge(body_edge)
    graph.add_edge(create_edge("for", "index", "body", body_edge.destination.field))

    state = GraphExecutionState(graph=graph)

    assert state._can_use_generic_scheduler() is False


def test_input_driven_empty_flat_for_with_wrong_return_edge_remains_compatibility_owned() -> None:
    graph = _flat_for_graph(input_collection=[])
    graph.delete_node("body")
    graph.add_node(StateSetInvocation(id="body"))
    graph.add_edge(create_edge("for", "item", "body", "value"))
    graph.add_edge(create_edge("body", "state", "return", "state"))

    state = GraphExecutionState(graph=graph)

    assert state._can_use_generic_scheduler() is False


def test_input_driven_flat_for_uses_generic_scheduler() -> None:
    state = GraphExecutionState(graph=_flat_for_graph(input_collection=[3, 7]))

    assert state._can_use_generic_scheduler()
    assert isinstance(state._scheduler(), _GenericGraphSchedulerAdapter)


def test_input_driven_flat_for_matches_compatibility_scheduler() -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph(input_collection=[3, 7]))
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_graph(input_collection=[3, 7])),
        force_compatibility_scheduler=True,
    )

    assert (
        generic_trace
        == compatibility_trace
        == [
            "collection",
            "for",
            "body",
            "return",
            "for",
            "body",
            "return",
        ]
    )
    assert _final_for_output(generic_state).output_collection == [13, 17]
    assert _final_for_output(compatibility_state).output_collection == [13, 17]
    assert generic_state.is_complete()
    assert compatibility_state.is_complete()
    assert isinstance(generic_state._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert isinstance(compatibility_state._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)

    restored_generic = load_execution_state(dump_execution_state(generic_state))
    restored_compatibility = load_execution_state(dump_execution_state(compatibility_state))
    _restore_compatibility_scheduler(restored_compatibility)
    assert isinstance(restored_generic._scheduler(), _GenericGraphSchedulerAdapter)
    assert isinstance(restored_compatibility._execution_scheduler, _ExecutionScheduler)
    assert _state_projection(restored_generic) == _state_projection(generic_state)
    assert _state_projection(restored_compatibility) == _state_projection(compatibility_state)
    _assert_execution_identity_consistent(restored_generic)
    _assert_execution_identity_consistent(restored_compatibility)


@pytest.mark.parametrize(
    ("continue_condition", "expected_trace", "expected_collection", "expected_state"),
    [
        (True, ["for", "body", "return"] * 3, [1, 2, 3], {"count": 3}),
        (False, ["for", "body", "return"], [1], {"count": 1}),
    ],
)
def test_flat_for_fresh_execution_matches_state_and_break_semantics(
    continue_condition: bool,
    expected_trace: list[str],
    expected_collection: list[int],
    expected_state: dict[str, int],
) -> None:
    generic_trace, generic_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_state_graph(continue_condition)),
    )
    compatibility_trace, compatibility_state = _run_graph_with_effects(
        GraphExecutionState(graph=_flat_for_state_graph(continue_condition)),
        force_compatibility_scheduler=True,
    )

    assert generic_trace == compatibility_trace == expected_trace
    final_for_id = max(
        (
            exec_node_id
            for exec_node_id, source_node_id in generic_state.prepared_source_mapping.items()
            if source_node_id == "for"
        ),
        key=lambda exec_node_id: generic_state.execution_graph.get_node(exec_node_id).index,
    )
    final_for_output = generic_state.results[final_for_id]
    assert final_for_output.output_collection == expected_collection
    assert final_for_output.final_state == LoopState(values=expected_state)
    assert _state_projection(generic_state) == _state_projection(compatibility_state)
    _assert_generic_and_compatibility_schedulers(generic_state, compatibility_state)


def test_nested_if_checkpoint_restore_matches_compatibility_scheduler() -> None:
    graph = _nested_if_graph()
    expected_trace, expected_state = _run_graph(GraphExecutionState(graph=graph.model_copy(deep=True)))

    for force_compatibility_scheduler in (False, True):
        checkpoint_trace, checkpoint_state = _run_graph(
            GraphExecutionState(graph=graph.model_copy(deep=True)),
            force_compatibility_scheduler=force_compatibility_scheduler,
            stop_after=4,
        )
        restored = load_execution_state(dump_execution_state(checkpoint_state))
        if force_compatibility_scheduler:
            _restore_compatibility_scheduler(restored)
        remaining_trace, restored_state = _run_graph(
            restored,
            force_compatibility_scheduler=force_compatibility_scheduler,
        )

        assert checkpoint_trace + remaining_trace == expected_trace
        assert _state_projection(restored_state) == _state_projection(expected_state)
        assert _activation_projection(restored_state) == _activation_projection(expected_state)
        assert restored_state.is_complete()


def test_nested_if_failure_round_trip_matches_compatibility_scheduler() -> None:
    expected_trace = ["outer_condition", "inner_condition", "inner_false"]
    expected_errors = {"inner_false": "injected failure"}

    generic_trace, generic_state = _run_graph(
        GraphExecutionState(graph=_nested_if_graph()),
        fail_source_id="inner_false",
    )
    compatibility_trace, compatibility_state = _run_graph(
        GraphExecutionState(graph=_nested_if_graph()),
        force_compatibility_scheduler=True,
        fail_source_id="inner_false",
    )

    assert generic_trace == compatibility_trace == expected_trace
    assert generic_state.next() is None
    assert compatibility_state.next() is None
    for state in (generic_state, compatibility_state):
        assert {
            state.prepared_source_mapping.get(execution_id, execution_id): message
            for execution_id, message in state.errors.items()
        } == expected_errors
        result_sources = {state.prepared_source_mapping[execution_id] for execution_id in state.results}
        assert "inner_if" not in result_sources
        assert "outer_if" not in result_sources
        assert "sink" not in result_sources
    assert _state_projection(generic_state) == _state_projection(compatibility_state)

    restored_generic = load_execution_state(dump_execution_state(generic_state))
    restored_compatibility = load_execution_state(dump_execution_state(compatibility_state))
    _restore_compatibility_scheduler(restored_compatibility)
    assert restored_generic.next() is None
    assert restored_compatibility.next() is None
    assert _state_projection(restored_generic) == _state_projection(restored_compatibility)


def test_static_dag_checkpoint_restore_has_matching_remaining_trace() -> None:
    expected_trace, expected_state = _run(_load_fixture())

    for force_compatibility_scheduler in (False, True):
        checkpoint_trace, checkpoint_state = _run(
            _load_fixture(),
            force_compatibility_scheduler=force_compatibility_scheduler,
            stop_after=1,
        )
        restored = load_execution_state(dump_execution_state(checkpoint_state))
        remaining_trace, restored_state = _run(
            restored,
            force_compatibility_scheduler=force_compatibility_scheduler,
        )

        assert checkpoint_trace + remaining_trace == expected_trace
        assert restored_state.is_complete()
        assert _state_projection(restored_state) == _state_projection(expected_state)


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_inflight_checkpoint_replays_claimed_work_after_rehydrate(force_compatibility_scheduler: bool) -> None:
    state = _load_fixture()
    if force_compatibility_scheduler:
        object.__setattr__(state, "_execution_scheduler", _ExecutionScheduler(state))

    claimed = state.next()
    assert claimed is not None
    snapshot = dump_execution_state(state)
    restored = load_execution_state(snapshot)
    if force_compatibility_scheduler:
        object.__setattr__(restored, "_execution_scheduler", _ExecutionScheduler(restored))

    replayed = restored.next()
    assert replayed is not None
    assert state.prepared_source_mapping[claimed.id] == "left"
    assert restored.prepared_source_mapping[replayed.id] == "left"
    assert restored.executed == set()


def test_injected_failure_stops_both_schedulers_without_further_scheduling() -> None:
    for trace, state in _run_both(fail_source_id="right"):
        assert trace == ["left", "right"]
        assert {
            state.prepared_source_mapping.get(node_id, node_id): message for node_id, message in state.errors.items()
        } == {"right": "injected failure"}
        assert state.next() is None
        assert "join" not in state.results


def test_injected_failure_round_trip_preserves_both_scheduler_terminal_state() -> None:
    generic, compatibility = _run_both(fail_source_id="right")
    restored_generic = load_execution_state(dump_execution_state(generic[1]))
    restored_compatibility = load_execution_state(dump_execution_state(compatibility[1]))
    _restore_compatibility_scheduler(restored_compatibility)

    assert restored_generic.next() is None
    assert restored_compatibility.next() is None
    assert restored_generic.is_complete()
    assert restored_compatibility.is_complete()
    assert _state_projection(restored_generic) == _state_projection(restored_compatibility)


def test_real_invocation_failure_round_trip_preserves_both_scheduler_terminal_state() -> None:
    expected_error = "This invocation is supposed to fail"
    terminal_states: list[tuple[str, bool, bool]] = []

    for force_compatibility_scheduler in (False, True):
        graph = Graph()
        graph.add_node(ErrorInvocation(id="error"))
        state = GraphExecutionState(graph=graph)
        if force_compatibility_scheduler:
            state._execution_scheduler = _ExecutionScheduler(state)

        node = state.next()
        assert node is not None
        with pytest.raises(Exception) as invocation_error:
            node.invoke(Mock())
        assert str(invocation_error.value) == expected_error
        state.set_node_error(node.id, str(invocation_error.value))

        assert state.next() is None
        assert state.is_complete()
        assert {
            state.prepared_source_mapping.get(node_id, node_id): message for node_id, message in state.errors.items()
        } == {"error": expected_error}

        restored = load_execution_state(dump_execution_state(state))
        if force_compatibility_scheduler:
            _restore_compatibility_scheduler(restored)

        assert restored.errors == state.errors
        assert restored.next() is None
        assert restored.is_complete()
        terminal_states.append((str(next(iter(restored.errors.values()))), True, restored.is_complete()))

    assert terminal_states == [(expected_error, True, True), (expected_error, True, True)]


@pytest.mark.parametrize("fixture_name", ["static_dag_partial_v1.json", "static_dag_failed_v1.json"])
def test_durable_snapshot_corpus_round_trips_without_losing_terminal_state(fixture_name: str) -> None:
    state = _load_fixture(fixture_name)
    restored = load_execution_state(dump_execution_state(state))

    assert restored.id == state.id
    assert restored.executed == state.executed
    assert restored.executed_history == state.executed_history
    assert restored.errors == state.errors
    assert restored.indegree == state.indegree

    if state.errors:
        for force_compatibility_scheduler in (False, True):
            candidate = load_execution_state(dump_execution_state(state))
            if force_compatibility_scheduler:
                _restore_compatibility_scheduler(candidate)
            assert candidate.next() is None
            assert candidate.is_complete()
        generic = load_execution_state(dump_execution_state(state))
        compatibility = load_execution_state(dump_execution_state(state))
        _restore_compatibility_scheduler(compatibility)
        assert _state_projection(generic) == _state_projection(compatibility)
    else:
        generic_trace, generic_state = _run(restored)
        compatibility_trace, compatibility_state = _run(
            load_execution_state(dump_execution_state(state)),
            force_compatibility_scheduler=True,
        )
        assert generic_trace == compatibility_trace == ["right", "join"]
        assert generic_state.is_complete()
        assert compatibility_state.is_complete()


def test_fixture_is_versioned_and_future_versions_are_explicitly_rejected() -> None:
    with FIXTURE_PATH.open(encoding="utf-8") as fixture:
        snapshot = json.load(fixture)

    assert snapshot["execution_state_version"] == 1
    restored = load_execution_state(snapshot)
    assert restored.id == "fixture-state"
    assert dump_execution_state(restored)["execution_state_version"] == CURRENT_EXECUTION_STATE_VERSION

    future_snapshot = dict(snapshot)
    future_snapshot["execution_state_version"] = CURRENT_EXECUTION_STATE_VERSION + 1
    with pytest.raises(UnsupportedExecutionStateVersionError, match="newer than supported"):
        load_execution_state(future_snapshot)


def test_generic_legacy_shaped_if_does_not_prune_or_skip_during_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _nested_if_graph()
    state = GraphExecutionState(graph=graph)
    deleted_edges: list[Edge] = []

    def record_deleted_edge(self: GraphExecutionState, edge: Edge) -> None:
        deleted_edges.append(edge)

    monkeypatch.setattr(GraphExecutionState, "_tx_delete_execution_edge", record_deleted_edge)

    trace, state = _run_graph(state)

    assert trace == ["outer_condition", "inner_condition", "inner_false", "inner_if", "outer_if", "sink"]
    sink_id = next(iter(state.source_prepared_mapping["sink"]))
    assert state.results[sink_id].value == 7
    assert state.is_complete()
    assert deleted_edges == []


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_if_readiness_preserves_both_schedulers(
    force_compatibility_scheduler: bool,
) -> None:
    graph = _nested_if_graph()

    trace, state = _run_graph(
        GraphExecutionState(graph=graph),
        force_compatibility_scheduler=force_compatibility_scheduler,
    )

    assert trace == ["outer_condition", "inner_condition", "inner_false", "inner_if", "outer_if", "sink"]
    assert state.is_complete()


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_flat_if_admission_is_demand_driven_and_token_authoritative(
    force_compatibility_scheduler: bool,
) -> None:
    trace, state = _run_graph(
        GraphExecutionState(graph=_flat_if_graph()),
        force_compatibility_scheduler=force_compatibility_scheduler,
    )

    assert trace == ["condition", "true_branch", "if", "sink"]
    assert set(state.source_prepared_mapping) == {"condition", "true_branch", "if", "sink"}
    assert "false_branch" not in state.source_prepared_mapping
    assert state.executed_history == trace
    assert all(
        state._get_prepared_exec_metadata(exec_node_id).state != "skipped"
        for exec_node_id in state.prepared_source_mapping
    )
    activation_tokens = [token for token in state.execution_tokens.values() if token.token_kind == "activation"]
    assert len(activation_tokens) == 1
    assert activation_tokens[0].port == "true_input"
    sink_id = next(iter(state.source_prepared_mapping["sink"]))
    assert state.results[sink_id].value == 6
    assert state.is_complete()


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_fresh_flat_if_records_dependencies_without_author_graph_branch_analysis(
    force_compatibility_scheduler: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh flat If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)

    trace, state = _run_graph(
        GraphExecutionState(graph=_flat_if_graph()),
        force_compatibility_scheduler=force_compatibility_scheduler,
    )

    assert trace == ["condition", "true_branch", "if", "sink"]
    true_branch_id = next(
        execution_id for execution_id, source_id in state.prepared_source_mapping.items() if source_id == "true_branch"
    )
    assert [
        (dependency.owner_id, dependency.branch, dependency.frame)
        for dependency in state._if_activation_dependencies_by_exec[true_branch_id]
    ] == [("if", "true_input", ())]
    assert state.is_complete()


@pytest.mark.parametrize(
    ("first_condition", "second_condition"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_fresh_sibling_ifs_preserve_owner_local_selection_and_scheduler_parity(
    first_condition: bool,
    second_condition: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh sibling If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    runs: list[tuple[list[str], GraphExecutionState]] = []

    for force_compatibility_scheduler in (False, True):
        state = GraphExecutionState(
            graph=_sibling_if_graph(first_condition=first_condition, second_condition=second_condition)
        )
        assert state._can_use_fresh_flat_if_activation()
        trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)
        runs.append((trace, state))

        expected_sources = {
            "first_condition",
            "second_condition",
            "first_if",
            "second_if",
            "first_sink",
            "second_sink",
        }
        expected_sources.add("first_true" if first_condition else "first_false")
        expected_sources.add("second_true" if second_condition else "second_false")
        assert set(state.source_prepared_mapping) == expected_sources
        assert ("first_true" in trace) is first_condition
        assert ("first_false" in trace) is not first_condition
        assert ("second_true" in trace) is second_condition
        assert ("second_false" in trace) is not second_condition
        first_sink_id = next(iter(state.source_prepared_mapping["first_sink"]))
        second_sink_id = next(iter(state.source_prepared_mapping["second_sink"]))
        assert state.results[first_sink_id].value == (103 if first_condition else 111)
        assert state.results[second_sink_id].value == (204 if second_condition else 221)

        selected_first = "first_true" if first_condition else "first_false"
        selected_second = "second_true" if second_condition else "second_false"
        dependencies_by_source = {
            source_id: {
                (dependency.owner_id, dependency.branch, dependency.frame)
                for execution_id, source_id_value in state.prepared_source_mapping.items()
                if source_id_value == source_id
                for dependency in state._if_activation_dependencies_by_exec.get(execution_id, ())
            }
            for source_id in (selected_first, selected_second)
        }
        assert dependencies_by_source == {
            selected_first: {("first_if", "true_input" if first_condition else "false_input", ())},
            selected_second: {("second_if", "true_input" if second_condition else "false_input", ())},
        }
        assert state.is_complete()

    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
@pytest.mark.parametrize(
    ("first_condition", "second_condition"),
    [(True, True), (True, False), (False, True), (False, False)],
)
@pytest.mark.parametrize("stop_after", range(1, 8))
def test_fresh_sibling_ifs_checkpoint_resume_matches_fresh_execution(
    force_compatibility_scheduler: bool,
    first_condition: bool,
    second_condition: bool,
    stop_after: int,
) -> None:
    graph = _sibling_if_graph(first_condition=first_condition, second_condition=second_condition)
    expected_trace, expected_state = _run_graph(
        GraphExecutionState(graph=graph.model_copy(deep=True)),
        force_compatibility_scheduler=force_compatibility_scheduler,
    )
    checkpoint_trace, checkpoint_state = _run_graph(
        GraphExecutionState(graph=graph.model_copy(deep=True)),
        force_compatibility_scheduler=force_compatibility_scheduler,
        stop_after=stop_after,
    )

    restored = load_execution_state(dump_execution_state(checkpoint_state))
    if force_compatibility_scheduler:
        _restore_compatibility_scheduler(restored)
    resumed_trace, resumed_state = _run_graph(
        restored,
        force_compatibility_scheduler=force_compatibility_scheduler,
    )

    assert checkpoint_trace + resumed_trace == expected_trace
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _activation_projection(resumed_state) == _activation_projection(expected_state)
    assert resumed_state.is_complete()


@pytest.mark.parametrize(
    ("first_condition", "second_condition"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_fresh_sibling_ifs_shared_ancestor_executes_once(
    first_condition: bool,
    second_condition: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh sibling If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    trace, state = _run_graph(
        GraphExecutionState(
            graph=_sibling_if_graph(
                first_condition=first_condition,
                second_condition=second_condition,
                with_shared_ancestor=True,
            )
        )
    )

    assert trace.count("shared") == 1
    assert state.is_complete()


@pytest.mark.parametrize(
    ("first_condition", "second_condition", "third_condition"),
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_fresh_three_sibling_ifs_select_only_owner_local_branches_and_match_compatibility(
    first_condition: bool,
    second_condition: bool,
    third_condition: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh three-sibling If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    runs: list[tuple[list[str], GraphExecutionState]] = []

    for force_compatibility_scheduler in (False, True):
        state = GraphExecutionState(
            graph=_three_sibling_if_graph(
                first_condition=first_condition,
                second_condition=second_condition,
                third_condition=third_condition,
            )
        )
        assert state._can_use_fresh_flat_if_activation()
        trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)
        runs.append((trace, state))

        selected = {
            "first_true" if first_condition else "first_false",
            "second_true" if second_condition else "second_false",
            "third_true" if third_condition else "third_false",
        }
        expected_sources = {
            "first_condition",
            "second_condition",
            "third_condition",
            "first_if",
            "second_if",
            "third_if",
            "first_sink",
            "second_sink",
            "third_sink",
            *selected,
        }
        unselected = {
            "first_false" if first_condition else "first_true",
            "second_false" if second_condition else "second_true",
            "third_false" if third_condition else "third_true",
        }
        assert set(state.source_prepared_mapping) == expected_sources
        assert set(trace) == expected_sources
        assert not unselected.intersection(trace)

        dependencies_by_source = {
            source_id: {
                (dependency.owner_id, dependency.branch, dependency.frame)
                for execution_id, source_id_value in state.prepared_source_mapping.items()
                if source_id_value == source_id
                for dependency in state._if_activation_dependencies_by_exec.get(execution_id, ())
            }
            for source_id in selected
        }
        assert dependencies_by_source == {
            "first_true" if first_condition else "first_false": {
                ("first_if", "true_input" if first_condition else "false_input", ())
            },
            "second_true" if second_condition else "second_false": {
                ("second_if", "true_input" if second_condition else "false_input", ())
            },
            "third_true" if third_condition else "third_false": {
                ("third_if", "true_input" if third_condition else "false_input", ())
            },
        }
        first_sink_id = next(iter(state.source_prepared_mapping["first_sink"]))
        second_sink_id = next(iter(state.source_prepared_mapping["second_sink"]))
        third_sink_id = next(iter(state.source_prepared_mapping["third_sink"]))
        assert state.results[first_sink_id].value == (103 if first_condition else 111)
        assert state.results[second_sink_id].value == (204 if second_condition else 221)
        assert state.results[third_sink_id].value == (305 if third_condition else 331)
        assert state.is_complete()

    _assert_generic_and_compatibility_schedulers(runs[0][1], runs[1][1])
    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])
    assert _activation_projection(runs[0][1]) == _activation_projection(runs[1][1])


def test_fresh_three_sibling_ifs_shared_ancestor_executes_once_without_cross_owner_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh three-sibling If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    runs: list[tuple[list[str], GraphExecutionState]] = []

    for force_compatibility_scheduler in (False, True):
        state = GraphExecutionState(graph=_three_sibling_if_graph(with_shared_ancestor=True))
        assert state._can_use_fresh_flat_if_activation()
        trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)
        runs.append((trace, state))

        assert trace.count("shared") == 1
        assert state.results[next(iter(state.source_prepared_mapping["first_sink"]))].value == 105
        assert state.results[next(iter(state.source_prepared_mapping["second_sink"]))].value == 206
        assert state.results[next(iter(state.source_prepared_mapping["third_sink"]))].value == 307
        assert state.is_complete()

        for source_id, if_id, branch in (
            ("first_true", "first_if", "true_input"),
            ("second_true", "second_if", "true_input"),
            ("third_true", "third_if", "true_input"),
        ):
            execution_id = next(
                execution_id
                for execution_id, prepared_source_id in state.prepared_source_mapping.items()
                if prepared_source_id == source_id
            )
            assert {
                (dependency.owner_id, dependency.branch, dependency.frame)
                for dependency in state._if_activation_dependencies_by_exec[execution_id]
            } == {(if_id, branch, ())}

    _assert_generic_and_compatibility_schedulers(runs[0][1], runs[1][1])
    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])
    assert _activation_projection(runs[0][1]) == _activation_projection(runs[1][1])


@pytest.mark.parametrize("first_condition", [False, True])
@pytest.mark.parametrize("second_condition", [False, True])
@pytest.mark.parametrize("third_condition", [False, True])
@pytest.mark.parametrize("fourth_condition", [False, True])
def test_fresh_four_sibling_ifs_match_compatibility_for_all_polarities(
    first_condition: bool,
    second_condition: bool,
    third_condition: bool,
    fourth_condition: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh four-sibling If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    graph = _four_sibling_if_graph(
        first_condition=first_condition,
        second_condition=second_condition,
        third_condition=third_condition,
        fourth_condition=fourth_condition,
    )
    runs: list[tuple[list[str], GraphExecutionState]] = []
    for force_compatibility_scheduler in (False, True):
        state = GraphExecutionState(graph=graph.model_copy(deep=True))
        assert state._can_use_fresh_flat_if_activation()
        trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)
        runs.append((trace, state))
        selected = {
            "first_true" if first_condition else "first_false",
            "second_true" if second_condition else "second_false",
            "third_true" if third_condition else "third_false",
            "fourth_true" if fourth_condition else "fourth_false",
        }
        unselected = {
            "first_false" if first_condition else "first_true",
            "second_false" if second_condition else "second_true",
            "third_false" if third_condition else "third_true",
            "fourth_false" if fourth_condition else "fourth_true",
        }
        assert state.is_complete()
        assert selected.issubset(trace)
        assert not unselected.intersection(trace)
        assert not unselected.intersection(state.prepared_source_mapping.values())

    _assert_generic_and_compatibility_schedulers(runs[0][1], runs[1][1])
    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])
    assert _activation_projection(runs[0][1]) == _activation_projection(runs[1][1])


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
@pytest.mark.parametrize("stop_after", [1, 4, 5, 8, 9, 12])
def test_fresh_four_sibling_ifs_checkpoint_resume_matches_fresh_execution(
    force_compatibility_scheduler: bool,
    stop_after: int,
) -> None:
    graph = _four_sibling_if_graph(
        first_condition=True,
        second_condition=False,
        third_condition=True,
        fourth_condition=False,
    )
    expected_trace, expected_state = _run_graph(
        GraphExecutionState(graph=graph.model_copy(deep=True)),
        force_compatibility_scheduler=force_compatibility_scheduler,
    )
    checkpoint_trace, checkpoint_state = _run_graph(
        GraphExecutionState(graph=graph.model_copy(deep=True)),
        force_compatibility_scheduler=force_compatibility_scheduler,
        stop_after=stop_after,
    )

    restored = load_execution_state(dump_execution_state(checkpoint_state))
    if force_compatibility_scheduler:
        _restore_compatibility_scheduler(restored)
    resumed_trace, resumed_state = _run_graph(
        restored,
        force_compatibility_scheduler=force_compatibility_scheduler,
    )

    assert checkpoint_trace + resumed_trace == expected_trace
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _activation_projection(resumed_state) == _activation_projection(expected_state)
    assert resumed_state.is_complete()


def test_fresh_four_sibling_ifs_shared_ancestor_executes_once_without_cross_owner_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh four-sibling If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    graph = _four_sibling_if_graph(with_shared_ancestor=True)
    runs: list[tuple[list[str], GraphExecutionState]] = []
    for force_compatibility_scheduler in (False, True):
        state = GraphExecutionState(graph=graph.model_copy(deep=True))
        assert state._can_use_fresh_flat_if_activation()
        trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)
        runs.append((trace, state))
        assert trace.count("shared") == 1
        assert state.is_complete()

    _assert_generic_and_compatibility_schedulers(runs[0][1], runs[1][1])
    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])
    assert _activation_projection(runs[0][1]) == _activation_projection(runs[1][1])


@pytest.mark.parametrize(
    "graph_factory",
    [
        pytest.param(_flat_if_graph, id="flat"),
        pytest.param(_noncanonical_flat_if_graph, id="noncanonical-flat"),
        pytest.param(_nested_if_graph, id="nested"),
        pytest.param(_three_nested_if_graph, id="three-nested"),
        pytest.param(_three_nested_if_graph_with_middle_leaf_fanout, id="middle-leaf-fanout"),
        pytest.param(_four_sibling_if_graph, id="four-siblings"),
    ],
)
@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_supported_fresh_if_shapes_do_not_use_controller_or_skip_projection(
    graph_factory: Any,
    force_compatibility_scheduler: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh supported shapes use graph-state dependencies in both scheduler routes."""

    def fail_controller(*_: object, **__: object) -> Any:
        raise AssertionError("fresh supported If shape used compatibility controller")

    def fail_generic_retirement(*_: object, **__: object) -> None:
        raise AssertionError("fresh supported If shape used generic skip projection")

    monkeypatch.setattr(GraphExecutionState, "_if_activation_controller", fail_controller)
    monkeypatch.setattr(_GenericGraphSchedulerAdapter, "_retire_unselected_node", fail_generic_retirement)

    state = GraphExecutionState(graph=graph_factory())
    assert state._can_use_fresh_flat_if_activation()
    trace, state = _run(state, force_compatibility_scheduler=force_compatibility_scheduler)

    assert trace
    assert state.is_complete()
    assert all(
        state._get_prepared_exec_metadata(exec_node_id).state != "skipped"
        for exec_node_id in state.prepared_source_mapping
    )


def test_fresh_mixed_iterate_if_collect_uses_selected_branches_in_both_scheduler_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = Graph()
    graph.add_node(CollectionConcatInvocation(id="source", first=[True, False, True, False]))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AnyTypeTestInvocation(id="true_branch"))
    graph.add_node(AnyTypeTestInvocation(id="false_branch"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(create_edge("source", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "if", "condition"))
    graph.add_edge(create_edge("iterate", "item", "true_branch", "value"))
    graph.add_edge(create_edge("iterate", "item", "false_branch", "value"))
    graph.add_edge(create_edge("true_branch", "value", "if", "true_input"))
    graph.add_edge(create_edge("false_branch", "value", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "collect", "item"))

    def fail_controller(*_: object, **__: object) -> Any:
        raise AssertionError("fresh mixed If used compatibility controller")

    def fail_generic_retirement(*_: object, **__: object) -> None:
        raise AssertionError("fresh mixed If used generic skip projection")

    monkeypatch.setattr(GraphExecutionState, "_if_activation_controller", fail_controller)
    monkeypatch.setattr(_GenericGraphSchedulerAdapter, "_retire_unselected_node", fail_generic_retirement)

    runs: list[tuple[list[str], GraphExecutionState]] = []
    for force_compatibility_scheduler in (False, True):
        state = GraphExecutionState(graph=graph.model_copy(deep=True))
        assert state._can_use_fresh_mixed_if_iterate_collect()
        trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)
        runs.append((trace, state))

        assert trace[0] == "source"
        assert trace[-1] == "collect"
        assert trace.count("iterate") == 4
        assert trace.count("true_branch") == 2
        assert trace.count("false_branch") == 2
        assert trace.count("if") == 4
        assert state.executed_history[-1] == "collect"
        assert set(state.executed_history) == {"source", "iterate", "true_branch", "false_branch", "if", "collect"}
        assert list(state.prepared_source_mapping.values()).count("true_branch") == 2
        assert list(state.prepared_source_mapping.values()).count("false_branch") == 2
        assert state.results[next(iter(state.source_prepared_mapping["collect"]))].collection == [
            True,
            False,
            True,
            False,
        ]
        activation_tokens = sorted(
            (token for token in state.execution_tokens.values() if token.token_kind == "activation"),
            key=lambda token: token.frame.iteration_path,
        )
        assert [token.port for token in activation_tokens] == [
            "true_input",
            "false_input",
            "true_input",
            "false_input",
        ]
        assert all(
            state._get_prepared_exec_metadata(exec_node_id).state != "skipped"
            for exec_node_id in state.prepared_source_mapping
        )
        assert state.is_complete()

    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])


@pytest.mark.parametrize(("condition", "branch"), [(True, "true:"), (False, "false:")])
def test_constant_if_inside_iterate_routes_each_loop_item(condition: bool, branch: str) -> None:
    graph = Graph()
    graph.add_node(CollectionConcatInvocation(id="source", first=[1, 2]))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(IfInvocation(id="if", condition=condition))
    graph.add_node(MarkedAnyTypeTestInvocation(id="true_branch", marker="true:"))
    graph.add_node(MarkedAnyTypeTestInvocation(id="false_branch", marker="false:"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(create_edge("source", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "true_branch", "value"))
    graph.add_edge(create_edge("iterate", "item", "false_branch", "value"))
    graph.add_edge(create_edge("true_branch", "value", "if", "true_input"))
    graph.add_edge(create_edge("false_branch", "value", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "collect", "item"))

    trace, state = _run_graph(GraphExecutionState(graph=graph))

    selected_source = "true_branch" if condition else "false_branch"
    unselected_source = "false_branch" if condition else "true_branch"
    assert trace[0] == "source"
    assert trace[-1] == "collect"
    assert trace.count("iterate") == 2
    assert trace.count(selected_source) == 2
    assert trace.count("if") == 2
    assert unselected_source not in trace
    collect_id = next(iter(state.source_prepared_mapping["collect"]))
    assert state.results[collect_id].collection == [f"{branch}1", f"{branch}2"]
    assert state.is_complete()
    assert state.next() is None
    assert not state._ready_node_ids
    assert all(not queue for queue in state._ready_queues.values())


@pytest.mark.parametrize(("condition", "branch"), [(True, "true:"), (False, "false:")])
def test_constant_if_inside_for_routes_each_loop_item(condition: bool, branch: str) -> None:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=[1, 2]))
    graph.add_node(IfInvocation(id="if", condition=condition))
    graph.add_node(MarkedAnyTypeTestInvocation(id="true_branch", marker="true:"))
    graph.add_node(MarkedAnyTypeTestInvocation(id="false_branch", marker="false:"))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(create_edge("for", "item", "true_branch", "value"))
    graph.add_edge(create_edge("for", "item", "false_branch", "value"))
    graph.add_edge(create_edge("true_branch", "value", "if", "true_input"))
    graph.add_edge(create_edge("false_branch", "value", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "return", "output"))
    graph.add_edge(create_loop_linkage("for", "return"))

    trace, state = _run_graph(GraphExecutionState(graph=graph))

    selected_source = "true_branch" if condition else "false_branch"
    unselected_source = "false_branch" if condition else "true_branch"
    assert trace.count("for") == 2
    assert trace.count(selected_source) == 2
    assert trace.count("if") == 2
    assert trace.count("return") == 2
    assert unselected_source not in trace
    assert sorted(state._get_iteration_path(exec_node_id) for exec_node_id in state.source_prepared_mapping["if"]) == [
        (0,),
        (1,),
    ]
    final_for_id = max(
        state.source_prepared_mapping["for"],
        key=lambda exec_node_id: state.execution_graph.get_node(exec_node_id).index,
    )
    assert state.results[final_for_id].output_collection == [f"{branch}1", f"{branch}2"]
    assert state.is_complete()
    assert state.next() is None
    assert not state._ready_node_ids
    assert all(not queue for queue in state._ready_queues.values())


@pytest.mark.parametrize(("condition", "branch"), [(True, "true:"), (False, "false:")])
def test_external_condition_if_inside_iterate_routes_each_loop_item(condition: bool, branch: str) -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=condition))
    graph.add_node(CollectionConcatInvocation(id="source", first=[1, 2]))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(MarkedAnyTypeTestInvocation(id="true_branch", marker="true:"))
    graph.add_node(MarkedAnyTypeTestInvocation(id="false_branch", marker="false:"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("source", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "true_branch", "value"))
    graph.add_edge(create_edge("iterate", "item", "false_branch", "value"))
    graph.add_edge(create_edge("true_branch", "value", "if", "true_input"))
    graph.add_edge(create_edge("false_branch", "value", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "collect", "item"))

    trace, state = _run_graph(GraphExecutionState(graph=graph))

    selected_source = "true_branch" if condition else "false_branch"
    unselected_source = "false_branch" if condition else "true_branch"
    assert trace.count("condition") == 1
    assert trace.count("source") == 1
    assert trace.count("iterate") == 2
    assert trace.count(selected_source) == 2
    assert trace.count("if") == 2
    assert trace.count("collect") == 1
    assert unselected_source not in trace
    assert sorted(state._get_iteration_path(exec_node_id) for exec_node_id in state.source_prepared_mapping["if"]) == [
        (0,),
        (1,),
    ]
    collect_id = next(iter(state.source_prepared_mapping["collect"]))
    assert state.results[collect_id].collection == [f"{branch}1", f"{branch}2"]
    assert state.is_complete()
    assert state.next() is None
    assert not state._pending_if_exec_nodes
    assert not state._ready_node_ids
    assert all(not queue for queue in state._ready_queues.values())


@pytest.mark.parametrize(("condition", "branch"), [(True, "true:"), (False, "false:")])
def test_external_condition_if_inside_for_routes_each_loop_item(condition: bool, branch: str) -> None:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=condition))
    graph.add_node(ForInvocation(id="for", collection=[1, 2]))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(MarkedAnyTypeTestInvocation(id="true_branch", marker="true:"))
    graph.add_node(MarkedAnyTypeTestInvocation(id="false_branch", marker="false:"))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("for", "item", "true_branch", "value"))
    graph.add_edge(create_edge("for", "item", "false_branch", "value"))
    graph.add_edge(create_edge("true_branch", "value", "if", "true_input"))
    graph.add_edge(create_edge("false_branch", "value", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "return", "output"))
    graph.add_edge(create_loop_linkage("for", "return"))

    trace, state = _run_graph(GraphExecutionState(graph=graph))

    selected_source = "true_branch" if condition else "false_branch"
    unselected_source = "false_branch" if condition else "true_branch"
    assert trace.count("condition") == 1
    assert trace.count("for") == 2
    assert trace.count(selected_source) == 2
    assert trace.count("if") == 2
    assert trace.count("return") == 2
    assert unselected_source not in trace
    assert sorted(state._get_iteration_path(exec_node_id) for exec_node_id in state.source_prepared_mapping["if"]) == [
        (0,),
        (1,),
    ]
    final_for_id = max(
        state.source_prepared_mapping["for"],
        key=lambda exec_node_id: state.execution_graph.get_node(exec_node_id).index,
    )
    assert state.results[final_for_id].output_collection == [f"{branch}1", f"{branch}2"]
    assert state.is_complete()
    assert state.next() is None
    assert not state._pending_if_exec_nodes
    assert not state._ready_node_ids
    assert all(not queue for queue in state._ready_queues.values())


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
@pytest.mark.parametrize(
    ("first_condition", "second_condition", "third_condition"),
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
@pytest.mark.parametrize("stop_after", range(1, 12))
def test_fresh_three_sibling_ifs_checkpoint_resume_matches_fresh_execution(
    force_compatibility_scheduler: bool,
    first_condition: bool,
    second_condition: bool,
    third_condition: bool,
    stop_after: int,
) -> None:
    graph = _three_sibling_if_graph(
        first_condition=first_condition,
        second_condition=second_condition,
        third_condition=third_condition,
    )
    expected_trace, expected_state = _run_graph(
        GraphExecutionState(graph=graph.model_copy(deep=True)),
        force_compatibility_scheduler=force_compatibility_scheduler,
    )
    checkpoint_trace, checkpoint_state = _run_graph(
        GraphExecutionState(graph=graph.model_copy(deep=True)),
        force_compatibility_scheduler=force_compatibility_scheduler,
        stop_after=stop_after,
    )

    restored = load_execution_state(dump_execution_state(checkpoint_state))
    if force_compatibility_scheduler:
        _restore_compatibility_scheduler(restored)
    resumed_trace, resumed_state = _run_graph(
        restored,
        force_compatibility_scheduler=force_compatibility_scheduler,
    )

    assert checkpoint_trace + resumed_trace == expected_trace
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _activation_projection(resumed_state) == _activation_projection(expected_state)
    assert resumed_state.is_complete()


def test_sibling_if_unsupported_saved_workflow_keeps_controller_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _sibling_if_graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call", workflow_id="saved-workflow"))
    state = GraphExecutionState(graph=graph)
    branch_analysis_calls: list[tuple[str, str]] = []
    original_branch_sources = graph_module._IfActivationController._branch_sources

    def record_branch_analysis(
        controller: graph_module._IfActivationController,
        if_node_id: str,
        branch_field: str,
        source_graph: Any,
    ) -> set[str]:
        branch_analysis_calls.append((if_node_id, branch_field))
        return original_branch_sources(controller, if_node_id, branch_field, source_graph)

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", record_branch_analysis)

    assert not state._can_use_fresh_flat_if_activation()
    dependencies = state._get_source_activation_dependencies("first_true", (7,))
    assert {dependency.owner_id for dependency in dependencies} == {"first_if"}
    assert branch_analysis_calls


def test_indirectly_connected_sibling_ifs_match_controller_fallback() -> None:
    expected_trace, expected_state = _run_graph(
        GraphExecutionState(graph=_indirectly_connected_sibling_if_graph()),
        force_compatibility_scheduler=True,
    )
    state = GraphExecutionState(graph=_indirectly_connected_sibling_if_graph())
    assert not state._can_use_fresh_flat_if_activation()
    assert not state._can_use_generic_scheduler()
    trace, state = _run_graph(state)

    assert trace == expected_trace
    assert _state_projection(state) == _state_projection(expected_state)
    assert state.is_complete()


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
@pytest.mark.parametrize(
    ("outer_condition", "inner_condition", "expected_trace"),
    [
        (
            True,
            True,
            ["outer_condition", "inner_condition", "inner_true", "inner_if", "outer_if", "sink"],
        ),
        (
            True,
            False,
            ["outer_condition", "inner_condition", "inner_false", "inner_if", "outer_if", "sink"],
        ),
        (False, True, ["outer_condition", "outer_false", "outer_if", "sink"]),
        (False, False, ["outer_condition", "outer_false", "outer_if", "sink"]),
    ],
)
def test_fresh_nested_if_records_graph_state_dependencies_without_controller_analysis(
    force_compatibility_scheduler: bool,
    outer_condition: bool,
    inner_condition: bool,
    expected_trace: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh nested If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)

    graph = _nested_if_graph()
    graph.get_node("outer_condition").value = outer_condition
    graph.get_node("inner_condition").value = inner_condition
    state = GraphExecutionState(graph=graph)
    dependencies_by_source = {
        source_id: {
            (dependency.owner_id, dependency.branch, dependency.frame)
            for dependency in state._get_source_activation_dependencies(source_id)
        }
        for source_id in {"inner_condition", "inner_true", "inner_false", "inner_if"}
    }

    trace, state = _run_graph(
        state,
        force_compatibility_scheduler=force_compatibility_scheduler,
    )

    assert trace == expected_trace
    assert dependencies_by_source == {
        "inner_condition": {("outer_if", "true_input", ())},
        "inner_true": {("outer_if", "true_input", ()), ("inner_if", "true_input", ())},
        "inner_false": {("outer_if", "true_input", ()), ("inner_if", "false_input", ())},
        "inner_if": {("outer_if", "true_input", ())},
    }
    assert state.is_complete()


@pytest.mark.parametrize(
    ("outer_condition", "middle_condition", "inner_condition", "selected", "expected_value", "expected_trace"),
    [
        (
            True,
            True,
            True,
            "inner_true",
            5,
            [
                "outer_condition",
                "middle_condition",
                "inner_condition",
                "inner_true",
                "inner_if",
                "middle_if",
                "outer_if",
                "sink",
            ],
        ),
        (
            True,
            True,
            False,
            "inner_false",
            7,
            [
                "outer_condition",
                "middle_condition",
                "inner_condition",
                "inner_false",
                "inner_if",
                "middle_if",
                "outer_if",
                "sink",
            ],
        ),
        (
            True,
            False,
            True,
            "middle_false",
            11,
            ["outer_condition", "middle_condition", "middle_false", "middle_if", "outer_if", "sink"],
        ),
        (
            True,
            False,
            False,
            "middle_false",
            11,
            ["outer_condition", "middle_condition", "middle_false", "middle_if", "outer_if", "sink"],
        ),
        (False, True, True, "outer_false", 21, ["outer_condition", "outer_false", "outer_if", "sink"]),
        (False, True, False, "outer_false", 21, ["outer_condition", "outer_false", "outer_if", "sink"]),
        (False, False, True, "outer_false", 21, ["outer_condition", "outer_false", "outer_if", "sink"]),
        (False, False, False, "outer_false", 21, ["outer_condition", "outer_false", "outer_if", "sink"]),
    ],
)
def test_fresh_three_nested_ifs_select_only_branches_and_match_compatibility(
    outer_condition: bool,
    middle_condition: bool,
    inner_condition: bool,
    selected: str,
    expected_value: int,
    expected_trace: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh three-nested If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    runs: list[tuple[list[str], GraphExecutionState]] = []
    for force_compatibility_scheduler in (False, True):
        state = GraphExecutionState(
            graph=_three_nested_if_graph(
                outer_condition=outer_condition,
                middle_condition=middle_condition,
                inner_condition=inner_condition,
            )
        )
        assert state._can_use_fresh_flat_if_activation()
        trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)
        runs.append((trace, state))

        assert trace == expected_trace
        assert set(state.source_prepared_mapping) == set(expected_trace)
        assert selected in trace
        sink_id = next(iter(state.source_prepared_mapping["sink"]))
        assert state.results[sink_id].value == expected_value
        assert state.is_complete()

    _assert_generic_and_compatibility_schedulers(runs[0][1], runs[1][1])
    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])
    assert _activation_projection(runs[0][1]) == _activation_projection(runs[1][1])


@pytest.mark.parametrize("middle_branch", ["true_input", "false_input"])
@pytest.mark.parametrize("outer_branch", ["true_input", "false_input"])
def test_fresh_three_nested_ifs_support_direct_false_parent_branch_edges(
    middle_branch: str,
    outer_branch: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh three-nested If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    graph = _three_nested_if_graph(
        outer_condition=outer_branch == "true_input",
        middle_condition=middle_branch == "true_input",
        middle_branch=middle_branch,
        outer_branch=outer_branch,
    )

    runs: list[tuple[list[str], GraphExecutionState]] = []
    for force_compatibility_scheduler in (False, True):
        trace, state = _run_graph(
            GraphExecutionState(graph=graph.model_copy(deep=True)),
            force_compatibility_scheduler=force_compatibility_scheduler,
        )
        runs.append((trace, state))

        assert trace == [
            "outer_condition",
            "middle_condition",
            "inner_condition",
            "inner_true",
            "inner_if",
            "middle_if",
            "outer_if",
            "sink",
        ]
        assert set(state.source_prepared_mapping) == set(trace)
        assert state.results[next(iter(state.source_prepared_mapping["sink"]))].value == 5
        assert state.is_complete()

    _assert_generic_and_compatibility_schedulers(runs[0][1], runs[1][1])
    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
@pytest.mark.parametrize("stop_after", range(1, 10))
def test_fresh_three_nested_ifs_shared_ancestor_isolated_and_checkpoint_resumes(
    force_compatibility_scheduler: bool,
    stop_after: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh three-nested If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    graph = _three_nested_if_graph_with_shared_ancestor(
        outer_condition=True, middle_condition=True, inner_condition=False
    )
    expected_trace, expected_state = _run_graph(GraphExecutionState(graph=graph.model_copy(deep=True)))
    checkpoint_trace, checkpoint_state = _run_graph(
        GraphExecutionState(graph=graph.model_copy(deep=True)),
        force_compatibility_scheduler=force_compatibility_scheduler,
        stop_after=stop_after,
    )
    restored = load_execution_state(dump_execution_state(checkpoint_state))
    if force_compatibility_scheduler:
        _restore_compatibility_scheduler(restored)
    resumed_trace, resumed_state = _run_graph(restored, force_compatibility_scheduler=force_compatibility_scheduler)

    combined_trace = checkpoint_trace + resumed_trace
    assert combined_trace == expected_trace
    assert resumed_state.executed_history == expected_state.executed_history
    assert resumed_state.results[next(iter(resumed_state.source_prepared_mapping["sink"]))].value == 6
    assert resumed_state.is_complete()
    assert resumed_state.executed_history.count("shared") == 1
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _activation_projection(resumed_state) == _activation_projection(expected_state)


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
@pytest.mark.parametrize(
    ("outer_condition", "middle_condition", "inner_condition", "expected_trace", "expected_value"),
    [
        (
            True,
            True,
            True,
            [
                "outer_condition",
                "middle_condition",
                "inner_condition",
                "inner_true",
                "inner_if",
                "middle_if",
                "middle_side_consumer",
                "outer_if",
                "sink",
            ],
            5,
        ),
        (
            True,
            True,
            False,
            [
                "outer_condition",
                "middle_condition",
                "inner_condition",
                "inner_false",
                "inner_if",
                "middle_if",
                "middle_side_consumer",
                "outer_if",
                "sink",
            ],
            7,
        ),
        (
            True,
            False,
            True,
            [
                "outer_condition",
                "middle_condition",
                "middle_false",
                "middle_if",
                "middle_side_consumer",
                "outer_if",
                "sink",
            ],
            11,
        ),
        (
            True,
            False,
            False,
            [
                "outer_condition",
                "middle_condition",
                "middle_false",
                "middle_if",
                "middle_side_consumer",
                "outer_if",
                "sink",
            ],
            11,
        ),
        (False, True, True, ["outer_condition", "outer_false", "outer_if", "sink"], 21),
        (False, True, False, ["outer_condition", "outer_false", "outer_if", "sink"], 21),
        (False, False, True, ["outer_condition", "outer_false", "outer_if", "sink"], 21),
        (False, False, False, ["outer_condition", "outer_false", "outer_if", "sink"], 21),
    ],
)
def test_fresh_three_nested_middle_leaf_fanout_matches_compatibility(
    force_compatibility_scheduler: bool,
    outer_condition: bool,
    middle_condition: bool,
    inner_condition: bool,
    expected_trace: list[str],
    expected_value: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh middle fanout If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    graph = _three_nested_if_graph_with_middle_leaf_fanout(
        outer_condition=outer_condition,
        middle_condition=middle_condition,
        inner_condition=inner_condition,
    )
    state = GraphExecutionState(graph=graph)

    assert state._can_use_fresh_flat_if_activation()
    trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)

    assert trace == expected_trace
    assert set(state.source_prepared_mapping) == set(expected_trace)
    assert state.results[next(iter(state.source_prepared_mapping["sink"]))].value == expected_value
    if outer_condition:
        side_consumer_id = next(iter(state.source_prepared_mapping["middle_side_consumer"]))
        expected_side_value = expected_value
        assert state.results[side_consumer_id].value == expected_side_value
    else:
        assert "middle_side_consumer" not in state.source_prepared_mapping
    assert state.is_complete()


@pytest.mark.parametrize("middle_branch", ["true_input", "false_input"])
@pytest.mark.parametrize("outer_branch", ["true_input", "false_input"])
def test_fresh_three_nested_middle_leaf_fanout_supports_direct_parent_branch_ports(
    middle_branch: str,
    outer_branch: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh middle fanout If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    graph = _three_nested_if_graph_with_middle_leaf_fanout(
        outer_condition=outer_branch == "true_input",
        middle_condition=middle_branch == "true_input",
        inner_condition=True,
        middle_branch=middle_branch,
        outer_branch=outer_branch,
    )
    expected_trace = [
        "outer_condition",
        "middle_condition",
        "inner_condition",
        "inner_true",
        "inner_if",
        "middle_if",
        "middle_side_consumer",
        "outer_if",
        "sink",
    ]
    runs: list[tuple[list[str], GraphExecutionState]] = []
    for force_compatibility_scheduler in (False, True):
        trace, state = _run_graph(
            GraphExecutionState(graph=graph.model_copy(deep=True)),
            force_compatibility_scheduler=force_compatibility_scheduler,
        )
        runs.append((trace, state))
        assert trace == expected_trace
        assert state.is_complete()

    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
@pytest.mark.parametrize("stop_after", range(1, 11))
def test_fresh_three_nested_middle_leaf_fanout_shared_checkpoint_resumes(
    force_compatibility_scheduler: bool,
    stop_after: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh middle fanout If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    graph = _three_nested_if_graph_with_middle_leaf_fanout_and_shared_ancestor(
        outer_condition=True,
        middle_condition=True,
        inner_condition=False,
    )
    expected_trace, expected_state = _run_graph(GraphExecutionState(graph=graph.model_copy(deep=True)))
    checkpoint_trace, checkpoint_state = _run_graph(
        GraphExecutionState(graph=graph.model_copy(deep=True)),
        force_compatibility_scheduler=force_compatibility_scheduler,
        stop_after=stop_after,
    )
    restored = load_execution_state(dump_execution_state(checkpoint_state))
    if force_compatibility_scheduler:
        _restore_compatibility_scheduler(restored)
    resumed_trace, resumed_state = _run_graph(restored, force_compatibility_scheduler=force_compatibility_scheduler)

    assert checkpoint_trace + resumed_trace == expected_trace
    assert resumed_state.executed_history == expected_state.executed_history
    assert resumed_state.executed_history.count("shared") == 1
    assert resumed_state.executed_history.count("middle_side_consumer") == 1
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _activation_projection(resumed_state) == _activation_projection(expected_state)
    assert resumed_state.is_complete()


@pytest.mark.parametrize(
    "graph_factory",
    [
        pytest.param(_three_nested_if_graph_with_extra_middle_fanout, id="extra-middle-fanout"),
        pytest.param(_three_nested_if_graph_with_non_leaf_middle_fanout, id="non-leaf-middle-fanout"),
        pytest.param(_three_nested_if_graph_with_inner_fanout, id="inner-fanout"),
    ],
)
def test_fresh_three_nested_invalid_fanout_shapes_use_controller_fallback(
    graph_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch_analysis_calls: list[tuple[str, str]] = []
    original_branch_sources = graph_module._IfActivationController._branch_sources

    def record_branch_analysis(
        controller: graph_module._IfActivationController,
        if_node_id: str,
        branch_field: str,
        source_graph: Any,
    ) -> set[str]:
        branch_analysis_calls.append((if_node_id, branch_field))
        return original_branch_sources(controller, if_node_id, branch_field, source_graph)

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", record_branch_analysis)
    state = GraphExecutionState(graph=graph_factory())

    assert not state._can_use_fresh_flat_if_activation()
    state._get_source_activation_dependencies("inner_true")
    assert branch_analysis_calls


@pytest.mark.parametrize(
    ("outer_condition", "inner_condition", "expected_value"),
    [(True, True, 5), (True, False, 6), (False, True, 11), (False, False, 11)],
)
def test_fresh_nested_if_preserves_shared_ancestor_isolation(
    outer_condition: bool,
    inner_condition: bool,
    expected_value: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh nested If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    graph = _nested_if_graph_with_shared_ancestor()
    graph.get_node("outer_condition").value = outer_condition
    graph.get_node("inner_condition").value = inner_condition

    trace, state = _run_graph(GraphExecutionState(graph=graph))

    assert trace.count("shared") == int(outer_condition)
    assert ("inner_true" in trace) is (outer_condition and inner_condition)
    assert ("inner_false" in trace) is (outer_condition and not inner_condition)
    assert trace.count("outer_false") == int(not outer_condition)
    sink_id = next(iter(state.source_prepared_mapping["sink"]))
    assert state.results[sink_id].value == expected_value
    assert state.is_complete()


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_fresh_noncanonical_if_compiles_branch_membership_without_controller_analysis(
    force_compatibility_scheduler: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh noncanonical If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)

    trace, state = _run_graph(
        GraphExecutionState(graph=_noncanonical_flat_if_graph()),
        force_compatibility_scheduler=force_compatibility_scheduler,
    )

    assert trace == ["condition", "true_source", "true_branch", "if", "sink"]
    dependencies_by_source = {
        source_id: {
            (dependency.owner_id, dependency.branch, dependency.frame)
            for execution_id, source_id in state.prepared_source_mapping.items()
            for dependency in state._if_activation_dependencies_by_exec.get(execution_id, ())
        }
        for source_id in {"true_source", "true_branch"}
    }
    assert dependencies_by_source == {
        "true_source": {("if", "true_input", ())},
        "true_branch": {("if", "true_input", ())},
    }
    assert state.is_complete()


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_fresh_if_dependency_cache_refreshes_after_branch_source_graph_edit(
    force_compatibility_scheduler: bool,
) -> None:
    state = GraphExecutionState(graph=_flat_if_graph(condition=False))

    assert state._get_source_activation_dependencies("true_branch") == (
        graph_module.ActivationDependency(owner_id="if", branch="true_input", frame=()),
    )

    state.add_node(AddInvocation(id="true_consumer", b=1))
    state.add_edge(create_edge("true_branch", "value", "true_consumer", "a"))

    assert state._get_source_activation_dependencies("true_branch") == ()
    trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)

    assert trace == ["true_branch", "true_consumer", "condition", "false_branch", "if", "sink"]
    assert state.results[next(iter(state.source_prepared_mapping["true_consumer"]))].value == 6
    assert state.results[next(iter(state.source_prepared_mapping["sink"]))].value == 31
    assert state.is_complete()


def test_fresh_flat_if_with_saved_workflow_uses_controller_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _flat_if_graph()
    graph.delete_node("sink")
    graph.add_node(CallSavedWorkflowInvocation(id="call", workflow_id="saved-workflow"))
    graph.add_edge(create_edge("if", "value", "call", "saved_workflow_input::input::value"))
    state = GraphExecutionState(graph=graph)
    branch_analysis_calls: list[tuple[str, str]] = []
    original_branch_sources = graph_module._IfActivationController._branch_sources

    def record_branch_analysis(
        controller: graph_module._IfActivationController,
        if_node_id: str,
        branch_field: str,
        source_graph: Any,
    ) -> set[str]:
        branch_analysis_calls.append((if_node_id, branch_field))
        return original_branch_sources(controller, if_node_id, branch_field, source_graph)

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", record_branch_analysis)

    assert not state._can_use_fresh_flat_if_activation()
    assert state._get_source_activation_dependencies("true_branch") == (
        graph_module.ActivationDependency(owner_id="if", branch="true_input", frame=()),
    )
    assert branch_analysis_calls


@pytest.mark.parametrize("unsupported_shape", ["extra_if", "mixed_iterate", "inner_fanout"])
def test_nested_if_unsupported_shapes_use_controller_fallback(
    unsupported_shape: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _nested_if_graph()
    if unsupported_shape == "extra_if":
        graph.add_node(IfInvocation(id="extra_if"))
    elif unsupported_shape == "mixed_iterate":
        graph.add_node(RangeInvocation(id="extra_range", start=0, stop=1, step=1))
        graph.add_node(IterateInvocation(id="extra_iterate"))
        graph.add_edge(create_edge("extra_range", "collection", "extra_iterate", "collection"))
    else:
        graph.add_node(AddInvocation(id="inner_side_consumer", b=1))
        graph.add_edge(create_edge("inner_if", "value", "inner_side_consumer", "a"))

    state = GraphExecutionState(graph=graph)
    branch_analysis_calls: list[tuple[str, str]] = []
    original_branch_sources = graph_module._IfActivationController._branch_sources

    def record_branch_analysis(
        controller: graph_module._IfActivationController,
        if_node_id: str,
        branch_field: str,
        source_graph: Any,
    ) -> set[str]:
        branch_analysis_calls.append((if_node_id, branch_field))
        return original_branch_sources(controller, if_node_id, branch_field, source_graph)

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", record_branch_analysis)

    assert not state._can_use_fresh_flat_if_activation()
    dependencies = state._get_source_activation_dependencies("inner_true")
    assert {dependency.owner_id for dependency in dependencies} >= {"inner_if"}
    assert branch_analysis_calls


@pytest.mark.parametrize(
    ("outer_condition", "inner_condition", "expected_trace", "expected_history", "expected_value"),
    [
        (
            True,
            True,
            ["outer_condition", "inner_condition", "inner_true", "inner_if", "outer_if", "sink"],
            [
                "outer_condition",
                "outer_false",
                "inner_condition",
                "inner_false",
                "inner_true",
                "inner_if",
                "outer_if",
                "sink",
            ],
            5,
        ),
        (
            True,
            False,
            ["outer_condition", "inner_condition", "inner_false", "inner_if", "outer_if", "sink"],
            [
                "outer_condition",
                "outer_false",
                "inner_condition",
                "inner_true",
                "inner_false",
                "inner_if",
                "outer_if",
                "sink",
            ],
            7,
        ),
        (
            False,
            True,
            ["outer_condition", "outer_false", "outer_if", "sink"],
            [
                "outer_condition",
                "inner_condition",
                "inner_true",
                "inner_false",
                "inner_if",
                "outer_false",
                "outer_if",
                "sink",
            ],
            11,
        ),
        (
            False,
            False,
            ["outer_condition", "outer_false", "outer_if", "sink"],
            [
                "outer_condition",
                "inner_condition",
                "inner_true",
                "inner_false",
                "inner_if",
                "outer_false",
                "outer_if",
                "sink",
            ],
            11,
        ),
    ],
)
def test_fresh_generic_nested_if_preserves_state_without_legacy_branch_projection(
    outer_condition: bool,
    inner_condition: bool,
    expected_trace: list[str],
    expected_history: list[str],
    expected_value: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _nested_if_graph()
    graph.get_node("outer_condition").value = outer_condition
    graph.get_node("inner_condition").value = inner_condition

    def fail_edge_deletion(*_: object, **__: object) -> None:
        raise AssertionError("fresh generic If execution deleted an execution edge")

    generic_retired_nodes: list[str] = []
    original_discard = _GenericGraphSchedulerAdapter._retire_unselected_node

    def record_generic_discard(adapter: _GenericGraphSchedulerAdapter, exec_node_id: str) -> None:
        generic_retired_nodes.append(exec_node_id)
        original_discard(adapter, exec_node_id)

    monkeypatch.setattr(_GenericGraphSchedulerAdapter, "_retire_unselected_node", record_generic_discard)
    monkeypatch.setattr(GraphExecutionState, "_tx_delete_execution_edge", fail_edge_deletion)

    trace, state = _run_graph(GraphExecutionState(graph=graph))

    assert trace == expected_trace
    assert state.executed_history == expected_trace
    assert set(state.indegree) == set(state.prepared_source_mapping)
    assert set(state.indegree.values()) == {0}
    assert state.is_complete()
    sink_id = next(iter(state.source_prepared_mapping["sink"]))
    assert state.results[sink_id].value == expected_value
    assert _execution_edge_projection(state) == _expected_edge_projection(
        graph,
        force_compatibility_scheduler=False,
        outer_condition=outer_condition,
        inner_condition=inner_condition,
    )
    assert generic_retired_nodes == []


@pytest.mark.parametrize("stop_after_source", ["outer_if", "inner_false"])
def test_if_partial_state_round_trip_rebuilds_fresh_runtime_and_matches_both_scheduler_paths(
    stop_after_source: str,
) -> None:
    """A partial If state round-trips and a fresh runtime matches both scheduler paths."""
    expected_trace, expected_state = _run_graph(GraphExecutionState(graph=_nested_if_graph()))
    partial_projections: list[tuple[Any, ...]] = []
    projections: list[tuple[list[str], GraphExecutionState]] = []

    for force_compatibility_scheduler in (False, True):
        partial_trace, canceled_state = _run_until_source(
            GraphExecutionState(graph=_nested_if_graph()),
            stop_after_source,
            force_compatibility_scheduler=force_compatibility_scheduler,
        )
        expected_partial_trace = ["outer_condition", "inner_condition"]
        if stop_after_source == "inner_false":
            expected_partial_trace.append("inner_false")
        else:
            expected_partial_trace.extend(["inner_false", "inner_if", "outer_if"])
        assert partial_trace == expected_partial_trace
        assert "inner_true" not in {
            canceled_state.prepared_source_mapping[execution_id] for execution_id in canceled_state.results
        }
        assert "outer_false" not in {
            canceled_state.prepared_source_mapping[execution_id] for execution_id in canceled_state.results
        }
        assert not canceled_state.is_complete()
        assert dict(canceled_state.indegree) == _expected_remaining_input_indegree(
            canceled_state,
            force_compatibility_scheduler=force_compatibility_scheduler,
            outer_condition=True,
            inner_condition=False,
        )
        assert _activation_projection(canceled_state) == (
            ("inner_if", "false_input", "false_input", ()),
            ("outer_if", "true_input", "true_input", ()),
        )
        partial_snapshot = dump_execution_state(canceled_state)
        restored_canceled = load_execution_state(partial_snapshot)
        restored_expected = load_execution_state(partial_snapshot)
        if force_compatibility_scheduler:
            _restore_compatibility_scheduler(restored_canceled)
        assert _state_projection(restored_canceled) == _state_projection(canceled_state)
        assert _activation_projection(restored_canceled) == _activation_projection(canceled_state)
        assert dump_execution_state(restored_canceled)["execution_tokens"] == partial_snapshot["execution_tokens"]
        assert dump_execution_state(restored_canceled)["execution_effects"] == partial_snapshot["execution_effects"]
        assert _execution_identity_projection(restored_canceled) == _execution_identity_projection(restored_expected)
        _assert_execution_identity_consistent(restored_canceled)
        assert dict(restored_canceled.indegree) == _expected_remaining_input_indegree(
            restored_canceled,
            force_compatibility_scheduler=force_compatibility_scheduler,
            outer_condition=True,
            inner_condition=False,
        )
        partial_projections.append(
            (_state_projection(restored_canceled), _execution_identity_projection(restored_canceled))
        )

        remaining_trace, resumed_state = _run_graph(
            restored_canceled,
            force_compatibility_scheduler=force_compatibility_scheduler,
        )
        assert partial_trace + remaining_trace == expected_trace
        assert resumed_state.executed_history == expected_state.executed_history
        assert _state_projection(resumed_state) == _state_projection(expected_state)
        assert resumed_state.results[next(iter(resumed_state.source_prepared_mapping["sink"]))].value == 7
        assert resumed_state.is_complete()
        assert dict(resumed_state.indegree) == _expected_remaining_input_indegree(
            resumed_state,
            force_compatibility_scheduler=force_compatibility_scheduler,
            outer_condition=True,
            inner_condition=False,
        )

        retried_state = GraphExecutionState(graph=canceled_state.graph.model_copy(deep=True))
        assert retried_state.id != canceled_state.id
        assert retried_state.results == {}
        assert retried_state.execution_refs == {}
        assert retried_state.execution_tokens == {}

        retried_trace, retried_state = _run_graph(
            retried_state,
            force_compatibility_scheduler=force_compatibility_scheduler,
        )
        projections.append((retried_trace, retried_state))

    assert partial_projections[0] == partial_projections[1]
    assert (
        projections[0][0]
        == projections[1][0]
        == [
            "outer_condition",
            "inner_condition",
            "inner_false",
            "inner_if",
            "outer_if",
            "sink",
        ]
    )
    assert _state_projection(projections[0][1]) == _state_projection(projections[1][1])
    assert _activation_projection(projections[0][1]) == _activation_projection(projections[1][1])
    assert _execution_identity_projection(projections[0][1]) == _execution_identity_projection(projections[1][1])
    _assert_execution_identity_consistent(projections[0][1])
    _assert_execution_identity_consistent(projections[1][1])
    assert projections[0][1].results[next(iter(projections[0][1].source_prepared_mapping["sink"]))].value == 7


@pytest.mark.parametrize(
    "tampered_field",
    [
        "token_id",
        "owner_node_id",
        "reference_id",
        "frame_id",
        "state_id",
        "iteration_path",
        "workflow_call_depth",
        "blank_owner_node_id",
        "blank_frame_id",
        "blank_state_id",
        "mapping_key",
        "both_ids",
    ],
)
def test_rehydrated_if_rejects_tampered_activation_identity(tampered_field: str) -> None:
    """A persisted activation token with stale identity or frame data is rejected."""
    _, partial_state = _run_until_source(GraphExecutionState(graph=_nested_if_graph()), "outer_if")
    snapshot = dump_execution_state(partial_state)
    token_id, token = next(
        (token_id, token)
        for token_id, token in snapshot["execution_tokens"].items()
        if token["token_kind"] == "activation"
    )
    source_if_id = partial_state.prepared_source_mapping[token["owner_node_id"]]

    valid_restored = load_execution_state(snapshot)
    valid_plan = valid_restored._scheduler()._scheduler.plan
    dependency = next(
        dependency
        for plan_node in valid_plan.nodes.values()
        for dependency in plan_node.activation_dependencies
        if dependency.owner_id == source_if_id
    )
    assert valid_restored._is_activation_dependency_satisfied(dependency)
    restored_token = valid_restored.execution_tokens[token_id]
    expected_ref = valid_restored.execution_refs[restored_token.owner_node_id]
    assert restored_token.token_id == token_id == f"{expected_ref.reference_id}:activation:{restored_token.port}"
    assert restored_token.reference_id == expected_ref.reference_id
    assert restored_token.owner_node_id == expected_ref.exec_node_id
    assert restored_token.frame == expected_ref.frame

    stale_snapshot = json.loads(json.dumps(snapshot))
    stale_token = stale_snapshot["execution_tokens"][token_id]
    if tampered_field == "token_id":
        stale_token["token_id"] = "stale-token"
    elif tampered_field == "owner_node_id":
        stale_token["owner_node_id"] = "stale-owner"
    elif tampered_field == "reference_id":
        stale_token["reference_id"] = "stale-reference"
    elif tampered_field == "frame_id":
        stale_token["frame"]["frame_id"] = "stale-frame"
    elif tampered_field == "state_id":
        stale_token["frame"]["state_id"] = "stale-state"
    elif tampered_field == "iteration_path":
        stale_token["frame"]["iteration_path"] = [1]
    elif tampered_field == "workflow_call_depth":
        stale_token["frame"]["workflow_call_depth"] = 1
    elif tampered_field == "blank_owner_node_id":
        stale_token["owner_node_id"] = ""
    elif tampered_field == "blank_frame_id":
        stale_token["frame"]["frame_id"] = ""
    elif tampered_field == "blank_state_id":
        stale_token["frame"]["state_id"] = ""
    elif tampered_field == "mapping_key":
        stale_snapshot["execution_tokens"]["stale-key"] = stale_snapshot["execution_tokens"].pop(token_id)
    else:
        stale_snapshot["execution_tokens"]["stale-key"] = stale_snapshot["execution_tokens"].pop(token_id)
        stale_snapshot["execution_tokens"]["stale-key"]["token_id"] = "stale-key"
    with pytest.raises(ValidationError, match="Activation token|Execution token"):
        load_execution_state(stale_snapshot)


def test_rehydrated_if_rejects_activation_token_with_ghost_owner_and_reference() -> None:
    _, partial_state = _run_until_source(GraphExecutionState(graph=_nested_if_graph()), "outer_if")
    snapshot = dump_execution_state(partial_state)
    token_id, token = next(
        (token_id, token)
        for token_id, token in snapshot["execution_tokens"].items()
        if token["token_kind"] == "activation"
    )
    expected_ref = partial_state._expected_execution_ref(token["owner_node_id"]).model_dump(mode="json")
    ghost_ref = {**expected_ref, "exec_node_id": "ghost", "reference_id": f"{snapshot['id']}:ghost"}
    snapshot["execution_refs"]["ghost"] = ghost_ref
    ghost_token_id = f"{ghost_ref['reference_id']}:activation:{token['port']}"
    ghost_token = {
        **token,
        "token_id": ghost_token_id,
        "reference_id": ghost_ref["reference_id"],
        "owner_node_id": "ghost",
    }
    snapshot["execution_tokens"].pop(token_id)
    snapshot["execution_tokens"][ghost_token_id] = ghost_token

    with pytest.raises(ValidationError, match="Activation token|Execution token"):
        load_execution_state(snapshot)


def test_rehydrated_if_rejects_activation_token_on_invocation_without_declared_field() -> None:
    _, partial_state = _run_until_source(GraphExecutionState(graph=_nested_if_graph()), "outer_if")
    snapshot = dump_execution_state(partial_state)
    ordinary_exec_id = next(
        execution_id
        for execution_id, source_id in partial_state.prepared_source_mapping.items()
        if source_id == "inner_false"
    )
    reference = partial_state._expected_execution_ref(ordinary_exec_id).model_dump(mode="json")
    token_id = f"{reference['reference_id']}:activation:value"
    snapshot["execution_tokens"][token_id] = {
        "token_id": token_id,
        "reference_id": reference["reference_id"],
        "owner_node_id": ordinary_exec_id,
        "port": "value",
        "frame": reference["frame"],
        "value": "value",
        "token_kind": "activation",
    }

    with pytest.raises(ValidationError, match="Activation token|Execution token"):
        load_execution_state(snapshot)


def test_rehydrated_activation_token_allows_unknown_frame_metadata() -> None:
    _, partial_state = _run_until_source(GraphExecutionState(graph=_nested_if_graph()), "outer_if")
    snapshot = dump_execution_state(partial_state)
    token_id, token = next(
        (token_id, token)
        for token_id, token in snapshot["execution_tokens"].items()
        if token["token_kind"] == "activation"
    )
    snapshot["execution_tokens"][token_id]["frame"]["future_frame_metadata"] = "preserved"

    restored = load_execution_state(snapshot)
    assert restored.execution_tokens[token_id].frame.model_extra["future_frame_metadata"] == "preserved"


def _expected_four_nested_if_run(conditions: tuple[bool, bool, bool, bool]) -> tuple[list[str], int]:
    root_condition, outer_condition, middle_condition, inner_condition = conditions
    trace = ["root_condition"]
    if not root_condition:
        return trace + ["root_false", "root_if", "sink"], 31

    trace.append("outer_condition")
    if not outer_condition:
        return trace + ["outer_false", "outer_if", "root_if", "sink"], 21

    trace.append("middle_condition")
    if not middle_condition:
        return trace + ["middle_false", "middle_if", "outer_if", "root_if", "sink"], 11

    trace.extend(["inner_condition", "inner_true" if inner_condition else "inner_false"])
    return trace + ["inner_if", "middle_if", "outer_if", "root_if", "sink"], 5 if inner_condition else 7


@pytest.mark.parametrize("conditions", list(product((True, False), repeat=4)))
def test_fresh_four_nested_ifs_select_only_branches_and_match_compatibility(
    conditions: tuple[bool, bool, bool, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh four-nested If used controller-owned branch analysis")

    def fail_controller_access(*_: object, **__: object) -> Any:
        raise AssertionError("fresh four-nested If instantiated the compatibility controller")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    monkeypatch.setattr(GraphExecutionState, "_if_activation_controller", fail_controller_access)
    expected_trace, expected_value = _expected_four_nested_if_run(conditions)
    runs: list[tuple[list[str], GraphExecutionState]] = []
    for force_compatibility_scheduler in (False, True):
        state = GraphExecutionState(
            graph=_four_nested_if_graph(
                root_condition=conditions[0],
                outer_condition=conditions[1],
                middle_condition=conditions[2],
                inner_condition=conditions[3],
            )
        )
        assert state._can_use_fresh_flat_if_activation()
        trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)
        runs.append((trace, state))

        assert trace == expected_trace
        assert set(state.source_prepared_mapping) == set(expected_trace)
        assert all(
            state._get_prepared_exec_metadata(exec_node_id).state != "skipped"
            for exec_node_id in state.prepared_source_mapping
        )
        assert state.results[next(iter(state.source_prepared_mapping["sink"]))].value == expected_value
        assert state.is_complete()

    _assert_generic_and_compatibility_schedulers(runs[0][1], runs[1][1])
    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])
    assert _activation_projection(runs[0][1]) == _activation_projection(runs[1][1])
    assert _execution_token_projection(runs[0][1]) == _execution_token_projection(runs[1][1])


@pytest.mark.parametrize("middle_branch", ["true_input", "false_input"])
@pytest.mark.parametrize("outer_branch", ["true_input", "false_input"])
@pytest.mark.parametrize("root_branch", ["true_input", "false_input"])
def test_fresh_four_nested_ifs_support_all_direct_parent_branch_ports(
    middle_branch: str,
    outer_branch: str,
    root_branch: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh four-nested If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    graph = _four_nested_if_graph(
        middle_branch=middle_branch,
        outer_branch=outer_branch,
        root_branch=root_branch,
        middle_condition=middle_branch == "true_input",
        outer_condition=outer_branch == "true_input",
        root_condition=root_branch == "true_input",
    )
    expected_trace = [
        "root_condition",
        "outer_condition",
        "middle_condition",
        "inner_condition",
        "inner_true",
        "inner_if",
        "middle_if",
        "outer_if",
        "root_if",
        "sink",
    ]
    runs: list[tuple[list[str], GraphExecutionState]] = []
    for force_compatibility_scheduler in (False, True):
        state = GraphExecutionState(graph=graph.model_copy(deep=True))
        assert state._can_use_fresh_flat_if_activation()
        trace, state = _run_graph(state, force_compatibility_scheduler=force_compatibility_scheduler)
        runs.append((trace, state))

        assert trace == expected_trace
        assert state.is_complete()
        assert {
            (dependency.owner_id, dependency.branch, dependency.frame)
            for dependency in state._get_source_activation_dependencies("inner_true")
        } == {
            ("inner_if", "true_input", ()),
            ("middle_if", middle_branch, ()),
            ("outer_if", outer_branch, ()),
            ("root_if", root_branch, ()),
        }

    _assert_generic_and_compatibility_schedulers(runs[0][1], runs[1][1])
    assert runs[0][0] == runs[1][0]
    assert _state_projection(runs[0][1]) == _state_projection(runs[1][1])
    assert _activation_projection(runs[0][1]) == _activation_projection(runs[1][1])


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
@pytest.mark.parametrize("stop_after", range(1, 12))
def test_fresh_four_nested_ifs_shared_ancestor_checkpoint_resume_matches_both_schedulers(
    force_compatibility_scheduler: bool,
    stop_after: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_branch_analysis(*_: object, **__: object) -> set[str]:
        raise AssertionError("fresh four-nested If used controller-owned branch analysis")

    monkeypatch.setattr(graph_module._IfActivationController, "_branch_sources", fail_branch_analysis)
    graph = _four_nested_if_graph(
        root_condition=True,
        outer_condition=True,
        middle_condition=True,
        inner_condition=False,
        with_shared_ancestor=True,
    )
    expected_trace = [
        "root_condition",
        "outer_condition",
        "middle_condition",
        "inner_condition",
        "shared",
        "inner_false",
        "inner_if",
        "middle_if",
        "outer_if",
        "root_if",
        "sink",
    ]
    expected_trace_run, expected_state = _run_graph(GraphExecutionState(graph=graph.model_copy(deep=True)))
    assert expected_trace_run == expected_trace
    checkpoint_trace, checkpoint_state = _run_graph(
        GraphExecutionState(graph=graph.model_copy(deep=True)),
        force_compatibility_scheduler=force_compatibility_scheduler,
        stop_after=stop_after,
    )
    restored = load_execution_state(dump_execution_state(checkpoint_state))
    if force_compatibility_scheduler:
        _restore_compatibility_scheduler(restored)
    resumed_trace, resumed_state = _run_graph(restored, force_compatibility_scheduler=force_compatibility_scheduler)

    assert checkpoint_trace + resumed_trace == expected_trace
    assert resumed_state.executed_history == expected_state.executed_history
    assert resumed_state.executed_history.count("shared") == 1
    assert resumed_state.results[next(iter(resumed_state.source_prepared_mapping["sink"]))].value == 6
    assert _state_projection(resumed_state) == _state_projection(expected_state)
    assert _activation_projection(resumed_state) == _activation_projection(expected_state)
    assert _execution_token_projection(resumed_state) == _execution_token_projection(expected_state)
    assert resumed_state.is_complete()
