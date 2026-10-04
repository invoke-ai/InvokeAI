import copy
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic.json_schema import models_json_schema

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvalidVersionError,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField, OutputField, OutputScope
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.primitives import (
    ColorInvocation,
    FloatCollectionInvocation,
    FloatInvocation,
    IntegerInvocation,
    StringCollectionInvocation,
    StringInvocation,
)
from invokeai.app.invocations.upscale import ESRGANInvocation
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    CollectInvocationOutput,
    Edge,
    EdgeConnection,
    Graph,
    GraphExecutionState,
    InvalidEdgeError,
    IterateInvocation,
    NodeAlreadyInGraphError,
    NodeNotFoundError,
    are_connections_compatible,
    get_output_field_scope,
)
from tests.test_nodes import (
    AnyTypeTestInvocation,
    ImageToImageTestInvocation,
    ListPassThroughInvocation,
    PolymorphicStringTestInvocation,
    PromptCollectionTestInvocation,
    PromptTestInvocation,
    PromptTestInvocationOutput,
    TextToImageTestInvocation,
    UnionCollectionTestInvocation,
    get_single_output_from_session,
    run_session_with_mock_context,
)


# Helpers
def create_edge(from_id: str, from_field: str, to_id: str, to_field: str) -> Edge:
    return Edge(
        source=EdgeConnection(node_id=from_id, field=from_field),
        destination=EdgeConnection(node_id=to_id, field=to_field),
    )


def create_loop_linkage(from_id: str, to_id: str) -> Edge:
    return Edge(
        type="loop_linkage",
        source=EdgeConnection(node_id=from_id, field="loop_linkage"),
        destination=EdgeConnection(node_id=to_id, field="loop_linkage"),
    )


@invocation_output("test_scoped_output")
class ScopedTestInvocationOutput(BaseInvocationOutput):
    iteration_value: str = OutputField(output_scope=OutputScope.Iteration)
    final_value: str = OutputField(output_scope=OutputScope.Final)
    ordinary_value: str = OutputField()


@invocation("test_scoped", version="1.0.0")
class ScopedTestInvocation(BaseInvocation):
    def invoke(self) -> ScopedTestInvocationOutput:
        return ScopedTestInvocationOutput(iteration_value="iteration", final_value="final", ordinary_value="ordinary")


@invocation_output("test_two_any_graph_output")
class TwoAnyGraphTestInvocationOutput(BaseInvocationOutput):
    value: Any = OutputField()


@invocation("test_two_any_graph", version="1.0.0")
class TwoAnyGraphTestInvocation(BaseInvocation):
    first: Any = InputField(default=None)
    second: Any = InputField(default=None)

    def invoke(self) -> TwoAnyGraphTestInvocationOutput:
        return TwoAnyGraphTestInvocationOutput(value=(self.first, self.second))


# Tests
def test_get_output_field_scope_reads_scoped_output_metadata():
    node = ScopedTestInvocation(id="1")

    assert get_output_field_scope(node, "iteration_value") == OutputScope.Iteration
    assert get_output_field_scope(node, "final_value") == OutputScope.Final
    assert get_output_field_scope(node, "ordinary_value") is None
    assert get_output_field_scope(node, "missing_value") is None


def test_connections_are_compatible():
    from_node = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    from_field = "image"
    to_node = ESRGANInvocation(id="2")
    to_field = "image"

    result = are_connections_compatible(from_node, from_field, to_node, to_field)

    assert result is True


def test_graph_validates_direct_for_boundary_pair():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body_return = ForReturnInvocation(id="return")

    g.add_node(loop)
    g.add_node(body_return)
    g.add_edge(create_edge(loop.id, "item", body_return.id, "output"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    g.validate_self()


def test_graph_validates_for_boundary_pair_with_loop_linkage():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body_return = ForReturnInvocation(id="return")

    g.add_node(loop)
    g.add_node(body_return)
    g.add_edge(create_edge(loop.id, "item", body_return.id, "output"))
    g.add_edge(
        Edge(
            type="loop_linkage",
            source=EdgeConnection(node_id=loop.id, field="loop_linkage"),
            destination=EdgeConnection(node_id=body_return.id, field="loop_linkage"),
        )
    )

    g.validate_self()


def test_graph_round_trips_for_loop_linkage():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body_return = ForReturnInvocation(id="return")

    g.add_node(loop)
    g.add_node(body_return)
    g.add_edge(create_edge(loop.id, "item", body_return.id, "output"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    restored = Graph.model_validate_json(g.model_dump_json())

    assert restored.edges[-1].type == "loop_linkage"
    assert restored.edges[-1].source.node_id == loop.id
    assert restored.edges[-1].destination.node_id == body_return.id


def test_graph_rejects_invalid_loop_linkage_endpoints():
    g = Graph()
    source = ForInvocation(id="source", collection=["a"])
    body_return = ForReturnInvocation(id="return")

    g.add_node(source)
    g.add_node(body_return)
    g.edges.append(
        Edge(
            type="loop_linkage",
            source=EdgeConnection(node_id=source.id, field="item"),
            destination=EdgeConnection(node_id=body_return.id, field="loop_linkage"),
        )
    )
    with pytest.raises(InvalidEdgeError, match="Invalid loop linkage"):
        g.validate_self()


def test_graph_rejects_default_edges_using_loop_linkage_fields():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a"])
    body_return = ForReturnInvocation(id="return")

    g.add_node(loop)
    g.add_node(body_return)
    g.edges.extend(
        [
            create_loop_linkage(loop.id, body_return.id),
            create_edge(loop.id, "item", body_return.id, "loop_linkage"),
        ]
    )

    with pytest.raises(InvalidEdgeError, match="must use a loop_linkage edge"):
        g.validate_self()


def test_graph_rejects_duplicate_loop_linkage():
    g = Graph()
    first_loop = ForInvocation(id="first", collection=["a"])
    second_loop = ForInvocation(id="second", collection=["b"])
    body_return = ForReturnInvocation(id="return")

    g.add_node(first_loop)
    g.add_node(second_loop)
    g.add_node(body_return)
    g.edges.extend(
        [
            create_loop_linkage(first_loop.id, body_return.id),
            create_loop_linkage(second_loop.id, body_return.id),
        ]
    )
    with pytest.raises(InvalidEdgeError, match="exactly one loop linkage"):
        g.validate_self()


def test_graph_rejects_edge_to_for_scheduler_index():
    g = Graph()
    index_source = IntegerInvocation(id="index_source", value=99)
    loop = ForInvocation(id="for", collection=["a", "b"])
    body_return = ForReturnInvocation(id="return")

    g.add_node(index_source)
    g.add_node(loop)
    g.add_node(body_return)
    g.edges.extend(
        [
            create_edge(index_source.id, "value", loop.id, "index"),
            create_edge(loop.id, "item", body_return.id, "output"),
            create_loop_linkage(loop.id, body_return.id),
        ]
    )

    with pytest.raises(InvalidEdgeError, match="direct input"):
        g.validate_self()


@pytest.mark.parametrize(
    ("node_type", "destination_field", "expected_message"),
    [
        ("for", "collection", "For loop may have only one collection input edge"),
        ("for", "state", "For loop may have only one state input edge"),
        ("for_return", "output", "ForReturn may have only one input edge per field"),
        ("for_return", "state", "ForReturn may have only one input edge per field"),
        ("for_return", "continue_condition", "ForReturn may have only one input edge per field"),
    ],
)
def test_graph_rejects_duplicate_loop_boundary_inputs(node_type, destination_field, expected_message):
    g = Graph()
    g.add_node(AnyTypeTestInvocation(id="first"))
    g.add_node(AnyTypeTestInvocation(id="second"))
    g.add_node(ForInvocation(id="for", collection=[1]))
    g.add_node(ForReturnInvocation(id="return"))
    g.edges.append(create_loop_linkage("for", "return"))
    if node_type == "for_return":
        g.edges.append(create_edge("for", "item", "return", "output"))
        source_ids = ["first", "second"]
        if destination_field == "state":
            g.edges.append(create_edge("for", "state", "return", "state"))
            source_ids = ["first", "second"]
    else:
        source_ids = ["first", "second"]
    g.edges.extend(
        create_edge(source_id, "value", "for" if node_type == "for" else "return", destination_field)
        for source_id in source_ids
    )

    with pytest.raises(InvalidEdgeError, match=expected_message):
        g.validate_self()


def test_graph_validates_nested_for_boundary_pair():
    g = Graph()
    g.add_node(ForInvocation(id="outer", collection=[["a"]]))
    g.add_node(AnyTypeTestInvocation(id="inner_collection"))
    g.add_node(ForInvocation(id="inner"))
    g.add_node(AnyTypeTestInvocation(id="inner_body"))
    g.add_node(ForReturnInvocation(id="inner_return"))
    g.add_node(ForReturnInvocation(id="outer_return"))
    g.add_edge(create_edge("outer", "item", "inner_collection", "value"))
    g.add_edge(create_edge("inner_collection", "value", "inner", "collection"))
    g.add_edge(create_edge("inner", "item", "inner_body", "value"))
    g.add_edge(create_edge("inner_body", "value", "inner_return", "output"))
    g.add_edge(create_edge("inner", "output_collection", "outer_return", "output"))
    g.add_edge(create_loop_linkage("inner", "inner_return"))
    g.add_edge(create_loop_linkage("outer", "outer_return"))

    g.validate_self()


def test_for_body_path_resolution_uses_loop_linkage_for_ambiguous_reachable_returns():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a"])
    matching_return = ForReturnInvocation(id="matching-return")
    other_return = ForReturnInvocation(id="other-return")

    g.add_node(loop)
    g.add_node(matching_return)
    g.add_node(other_return)
    g.add_edge(create_edge(loop.id, "item", matching_return.id, "output"))
    g.add_edge(create_edge(loop.id, "item", other_return.id, "output"))
    g.add_edge(create_loop_linkage(loop.id, matching_return.id))

    body_path_to_return = g._get_for_body_path_to_return(loop.id, g.nx_graph_flat())

    assert body_path_to_return is not None
    assert body_path_to_return[1] == matching_return.id


def test_for_body_path_resolution_rejects_missing_loop_linkage():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a"])
    first_return = ForReturnInvocation(id="first-return")
    second_return = ForReturnInvocation(id="second-return")

    g.add_node(loop)
    g.add_node(first_return)
    g.add_node(second_return)
    g.add_edge(create_edge(loop.id, "item", first_return.id, "output"))
    g.add_edge(create_edge(loop.id, "item", second_return.id, "output"))

    assert g._get_for_body_path_to_return(loop.id, g.nx_graph_flat()) is None


def test_graph_rejects_missing_for_loop_linkage():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body_return = ForReturnInvocation(id="return")

    g.add_node(loop)
    g.add_node(body_return)
    g.add_edge(create_edge(loop.id, "item", body_return.id, "output"))

    with pytest.raises(InvalidEdgeError, match="exactly one loop linkage"):
        g.validate_self()


def test_graph_rejects_missing_for_return_loop_linkage():
    g = Graph()
    body_return = ForReturnInvocation(id="return")

    g.add_node(body_return)

    with pytest.raises(InvalidEdgeError, match="exactly one loop linkage"):
        g.validate_self()


def test_graph_validates_indirect_for_body():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body = PromptTestInvocation(id="body")
    body_return = ForReturnInvocation(id="return")

    g.add_node(loop)
    g.add_node(body)
    g.add_node(body_return)
    g.add_edge(create_edge(body.id, "prompt", body_return.id, "output"))
    g.add_edge(create_edge(loop.id, "item", body.id, "prompt"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    g.validate_self()


def test_graph_validates_for_body_inputs_from_outside_body_boundary():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    external = PromptTestInvocation(id="external", prompt="outside")
    body = TextToImageTestInvocation(id="body")
    body_return = ForReturnInvocation(id="return")

    g.add_node(loop)
    g.add_node(external)
    g.add_node(body)
    g.add_node(body_return)
    g.add_edge(create_edge(loop.id, "item", body.id, "prompt"))
    g.add_edge(create_edge(external.id, "prompt", body.id, "prompt2"))
    g.add_edge(create_edge(body.id, "image", body_return.id, "output"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    g.validate_self()


def test_graph_rejects_for_body_inputs_from_external_iterator_scope():
    g = Graph()
    external_values = StringCollectionInvocation(id="external_values", collection=["external-a", "external-b"])
    external_iterate = IterateInvocation(id="external_iterate")
    external_adapter = PromptTestInvocation(id="external_adapter")
    loop = ForInvocation(id="for", collection=["loop-a", "loop-b"])
    body = TextToImageTestInvocation(id="body")
    body_return = ForReturnInvocation(id="return")

    g.add_node(external_values)
    g.add_node(external_iterate)
    g.add_node(external_adapter)
    g.add_node(loop)
    g.add_node(body)
    g.add_node(body_return)
    g.add_edge(create_edge(external_values.id, "collection", external_iterate.id, "collection"))
    g.add_edge(create_edge(external_iterate.id, "item", external_adapter.id, "prompt"))
    g.add_edge(create_edge(external_adapter.id, "prompt", body.id, "prompt2"))
    g.add_edge(create_edge(loop.id, "item", body.id, "prompt"))
    g.add_edge(create_edge(body.id, "image", body_return.id, "output"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    with pytest.raises(InvalidEdgeError, match="iterator-derived external inputs"):
        g.validate_self()


def test_graph_rejects_for_without_matching_return():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body = PromptTestInvocation(id="body")

    g.add_node(loop)
    g.add_node(body)
    g.add_edge(create_edge(loop.id, "item", body.id, "prompt"))

    with pytest.raises(InvalidEdgeError, match="exactly one loop linkage"):
        g.validate_self()


def test_graph_rejects_nested_for_until_linkage_exists():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    nested_loop = ForInvocation(id="nested_for", collection=["c", "d"])
    body_return = ForReturnInvocation(id="return")

    g.add_node(loop)
    g.add_node(nested_loop)
    g.add_node(body_return)
    g.add_edge(create_edge(nested_loop.id, "item", body_return.id, "output"))
    g.add_edge(create_edge(loop.id, "item", nested_loop.id, "collection"))

    with pytest.raises(InvalidEdgeError, match="exactly one loop linkage"):
        g.validate_self()


def test_graph_validates_deeper_nested_for_loops_with_one_child_per_boundary():
    g = Graph()
    outer = ForInvocation(id="outer", collection=[[]])
    outer_collection = AnyTypeTestInvocation(id="outer_collection")
    inner = ForInvocation(id="inner")
    inner_collection = AnyTypeTestInvocation(id="inner_collection")
    leaf = ForInvocation(id="leaf")
    leaf_body = AnyTypeTestInvocation(id="leaf_body")
    leaf_return = ForReturnInvocation(id="leaf_return")
    inner_return = ForReturnInvocation(id="inner_return")
    outer_return = ForReturnInvocation(id="outer_return")

    for node in (
        outer,
        outer_collection,
        inner,
        inner_collection,
        leaf,
        leaf_body,
        leaf_return,
        inner_return,
        outer_return,
    ):
        g.add_node(node)
    g.add_edge(create_edge("outer", "item", "outer_collection", "value"))
    g.add_edge(create_edge("outer_collection", "value", "inner", "collection"))
    g.add_edge(create_edge("inner", "item", "inner_collection", "value"))
    g.add_edge(create_edge("inner_collection", "value", "leaf", "collection"))
    g.add_edge(create_edge("leaf", "item", "leaf_body", "value"))
    g.add_edge(create_edge("leaf_body", "value", "leaf_return", "output"))
    g.add_edge(create_edge("leaf", "output_collection", "inner_return", "output"))
    g.add_edge(create_edge("inner", "output_collection", "outer_return", "output"))
    g.add_edge(create_loop_linkage("leaf", "leaf_return"))
    g.add_edge(create_loop_linkage("inner", "inner_return"))
    g.add_edge(create_loop_linkage("outer", "outer_return"))

    g.validate_self()


def test_graph_validates_nested_for_with_shared_outer_continuation_path():
    g = Graph()
    g.add_node(ForInvocation(id="outer", collection=[[]]))
    g.add_node(AnyTypeTestInvocation(id="inner_collection"))
    g.add_node(ForInvocation(id="inner"))
    g.add_node(AnyTypeTestInvocation(id="inner_body"))
    g.add_node(ForReturnInvocation(id="inner_return"))
    g.add_node(AnyTypeTestInvocation(id="continuation"))
    g.add_node(AnyTypeTestInvocation(id="continuation_tail"))
    g.add_node(ForReturnInvocation(id="outer_return"))
    g.add_edge(create_edge("outer", "item", "inner_collection", "value"))
    g.add_edge(create_edge("inner_collection", "value", "inner", "collection"))
    g.add_edge(create_edge("inner", "item", "inner_body", "value"))
    g.add_edge(create_edge("inner_body", "value", "inner_return", "output"))
    g.add_edge(create_edge("inner", "output_collection", "continuation", "value"))
    g.add_edge(create_edge("continuation", "value", "continuation_tail", "value"))
    g.add_edge(create_edge("continuation_tail", "value", "outer_return", "output"))
    g.add_edge(create_edge("continuation_tail", "value", "outer_return", "continue_condition"))
    g.add_edge(create_loop_linkage("inner", "inner_return"))
    g.add_edge(create_loop_linkage("outer", "outer_return"))

    g.validate_self()


def test_graph_validates_nested_for_return_continue_condition():
    g = Graph()
    g.add_node(ForInvocation(id="outer", collection=[[]]))
    g.add_node(AnyTypeTestInvocation(id="inner_collection"))
    g.add_node(ForInvocation(id="inner"))
    g.add_node(AnyTypeTestInvocation(id="inner_body"))
    g.add_node(AnyTypeTestInvocation(id="inner_condition"))
    g.add_node(ForReturnInvocation(id="inner_return"))
    g.add_node(ForReturnInvocation(id="outer_return"))
    g.add_edge(create_edge("outer", "item", "inner_collection", "value"))
    g.add_edge(create_edge("inner_collection", "value", "inner", "collection"))
    g.add_edge(create_edge("inner", "item", "inner_body", "value"))
    g.add_edge(create_edge("inner", "item", "inner_condition", "value"))
    g.add_edge(create_edge("inner_body", "value", "inner_return", "output"))
    g.add_edge(create_edge("inner_condition", "value", "inner_return", "continue_condition"))
    g.add_edge(create_edge("inner", "output_collection", "outer_return", "output"))
    g.add_edge(create_loop_linkage("inner", "inner_return"))
    g.add_edge(create_loop_linkage("outer", "outer_return"))

    g.validate_self()


def test_graph_rejects_nested_for_return_continue_condition_from_external_scope():
    g = Graph()
    g.add_node(ForInvocation(id="outer", collection=[[]]))
    g.add_node(AnyTypeTestInvocation(id="inner_collection"))
    g.add_node(ForInvocation(id="inner"))
    g.add_node(AnyTypeTestInvocation(id="inner_body"))
    g.add_node(ForReturnInvocation(id="inner_return"))
    g.add_node(ForReturnInvocation(id="outer_return"))
    g.add_node(AnyTypeTestInvocation(id="external_condition"))
    g.add_edge(create_edge("outer", "item", "inner_collection", "value"))
    g.add_edge(create_edge("inner_collection", "value", "inner", "collection"))
    g.add_edge(create_edge("inner", "item", "inner_body", "value"))
    g.add_edge(create_edge("inner_body", "value", "inner_return", "output"))
    g.add_edge(create_edge("inner", "output_collection", "outer_return", "output"))
    g.add_edge(create_edge("external_condition", "value", "outer_return", "continue_condition"))
    g.add_edge(create_loop_linkage("inner", "inner_return"))
    g.add_edge(create_loop_linkage("outer", "outer_return"))

    with pytest.raises(InvalidEdgeError, match="Nested For loops"):
        g.validate_self()


def test_graph_rejects_nested_for_return_state_from_outer_scope():
    g = Graph()
    g.add_node(ForInvocation(id="outer", collection=[[]]))
    g.add_node(AnyTypeTestInvocation(id="inner_collection"))
    g.add_node(ForInvocation(id="inner"))
    g.add_node(AnyTypeTestInvocation(id="inner_body"))
    g.add_node(ForReturnInvocation(id="inner_return"))
    g.add_node(ForReturnInvocation(id="outer_return"))
    g.add_edge(create_edge("outer", "item", "inner_collection", "value"))
    g.add_edge(create_edge("inner_collection", "value", "inner", "collection"))
    g.add_edge(create_edge("inner", "item", "inner_body", "value"))
    g.add_edge(create_edge("inner_body", "value", "inner_return", "output"))
    g.add_edge(create_edge("outer", "state", "inner_return", "state"))
    g.add_edge(create_edge("inner", "output_collection", "outer_return", "output"))
    g.add_edge(create_loop_linkage("inner", "inner_return"))
    g.add_edge(create_loop_linkage("outer", "outer_return"))

    with pytest.raises(InvalidEdgeError, match="Nested For loops"):
        g.validate_self()


def test_graph_rejects_nested_for_continuation_branch_without_outer_return():
    g = Graph()
    g.add_node(ForInvocation(id="outer", collection=[[]]))
    g.add_node(AnyTypeTestInvocation(id="inner_collection"))
    g.add_node(ForInvocation(id="inner"))
    g.add_node(AnyTypeTestInvocation(id="inner_body"))
    g.add_node(ForReturnInvocation(id="inner_return"))
    g.add_node(AnyTypeTestInvocation(id="continuation"))
    g.add_node(AnyTypeTestInvocation(id="dead_branch"))
    g.add_node(ForReturnInvocation(id="outer_return"))
    g.add_edge(create_edge("outer", "item", "inner_collection", "value"))
    g.add_edge(create_edge("inner_collection", "value", "inner", "collection"))
    g.add_edge(create_edge("inner", "item", "inner_body", "value"))
    g.add_edge(create_edge("inner_body", "value", "inner_return", "output"))
    g.add_edge(create_edge("inner", "output_collection", "continuation", "value"))
    g.add_edge(create_edge("continuation", "value", "outer_return", "output"))
    g.add_edge(create_edge("continuation", "value", "dead_branch", "value"))
    g.add_edge(create_loop_linkage("inner", "inner_return"))
    g.add_edge(create_loop_linkage("outer", "outer_return"))

    with pytest.raises(InvalidEdgeError, match="Nested For loops"):
        g.validate_self()


def test_graph_validates_independent_nested_for_children_with_explicit_fan_in():
    g = Graph()
    g.add_node(ForInvocation(id="outer", collection=[[]]))
    g.add_node(ForInvocation(id="first"))
    g.add_node(ForInvocation(id="second"))
    g.add_node(AnyTypeTestInvocation(id="first_body"))
    g.add_node(AnyTypeTestInvocation(id="second_body"))
    g.add_node(ForReturnInvocation(id="first_return"))
    g.add_node(ForReturnInvocation(id="second_return"))
    g.add_node(TwoAnyGraphTestInvocation(id="fan_in"))
    g.add_node(ForReturnInvocation(id="outer_return"))
    g.add_edge(create_edge("outer", "item", "first", "collection"))
    g.add_edge(create_edge("outer", "item", "second", "collection"))
    g.add_edge(create_edge("first", "item", "first_body", "value"))
    g.add_edge(create_edge("first_body", "value", "first_return", "output"))
    g.add_edge(create_edge("second", "item", "second_body", "value"))
    g.add_edge(create_edge("second_body", "value", "second_return", "output"))
    g.add_edge(create_edge("first", "output_collection", "fan_in", "first"))
    g.add_edge(create_edge("second", "output_collection", "fan_in", "second"))
    g.add_edge(create_edge("fan_in", "value", "outer_return", "output"))
    g.add_edge(create_loop_linkage("first", "first_return"))
    g.add_edge(create_loop_linkage("second", "second_return"))
    g.add_edge(create_loop_linkage("outer", "outer_return"))

    g.validate_self()


def test_graph_rejects_multiple_direct_nested_for_children():
    g = Graph()
    g.add_node(ForInvocation(id="outer", collection=[[]]))
    g.add_node(ForInvocation(id="first"))
    g.add_node(ForInvocation(id="second"))
    g.add_node(ForReturnInvocation(id="first_return"))
    g.add_node(ForReturnInvocation(id="second_return"))
    g.add_node(ForReturnInvocation(id="outer_return"))
    g.add_edge(create_edge("outer", "item", "first", "collection"))
    g.add_edge(create_edge("outer", "item", "second", "collection"))
    g.add_edge(create_edge("first", "item", "first_return", "output"))
    g.add_edge(create_edge("second", "item", "second_return", "output"))
    g.add_edge(create_edge("first", "output_collection", "outer_return", "output"))
    g.add_edge(create_loop_linkage("first", "first_return"))
    g.add_edge(create_loop_linkage("second", "second_return"))
    g.add_edge(create_loop_linkage("outer", "outer_return"))

    with pytest.raises(InvalidEdgeError, match="Nested For loops"):
        g.validate_self()


def test_graph_rejects_mixed_nested_for_and_iterate_body():
    g = Graph()
    outer = ForInvocation(id="outer", collection=[[]])
    inner_collection = AnyTypeTestInvocation(id="inner_collection")
    inner = ForInvocation(id="inner")
    iterate_collection = PolymorphicStringTestInvocation(id="iterate_collection")
    iterate = IterateInvocation(id="iterate")
    body = AnyTypeTestInvocation(id="body")
    collect = CollectInvocation(id="collect")
    inner_return = ForReturnInvocation(id="inner_return")
    outer_return = ForReturnInvocation(id="outer_return")

    for node in (
        outer,
        inner_collection,
        inner,
        iterate_collection,
        iterate,
        body,
        collect,
        inner_return,
        outer_return,
    ):
        g.add_node(node)
    g.add_edge(create_edge("outer", "item", "inner_collection", "value"))
    g.add_edge(create_edge("inner_collection", "value", "inner", "collection"))
    g.add_edge(create_edge("inner", "item", "iterate_collection", "value"))
    g.add_edge(create_edge("iterate_collection", "collection", "iterate", "collection"))
    g.add_edge(create_edge("iterate", "item", "body", "value"))
    g.add_edge(create_edge("body", "value", "collect", "item"))
    g.add_edge(create_edge("collect", "collection", "inner_return", "output"))
    g.add_edge(create_edge("inner", "output_collection", "outer_return", "output"))
    g.add_edge(create_loop_linkage("inner", "inner_return"))
    g.add_edge(create_loop_linkage("outer", "outer_return"))

    with pytest.raises(InvalidEdgeError, match="Nested For loops"):
        g.validate_self()


def test_graph_rejects_for_return_shared_by_two_loops():
    first = ForInvocation(id="first", collection=["a"])
    second = ForInvocation(id="second", collection=["b"])
    body_return = ForReturnInvocation(id="return")

    g = Graph(
        nodes={first.id: first, second.id: second, body_return.id: body_return},
        edges=[
            create_edge("first", "item", "return", "output"),
            create_edge("second", "item", "return", "output"),
            create_loop_linkage("first", "return"),
            create_loop_linkage("second", "return"),
        ],
    )

    with pytest.raises(InvalidEdgeError, match="exactly one loop linkage"):
        g.validate_self()


def test_graph_rejects_iterate_inside_for_body():
    g = Graph()
    loop = ForInvocation(id="for", collection=[["a", "b"]])
    collection_adapter = PolymorphicStringTestInvocation(id="collection_adapter")
    nested_iterate = IterateInvocation(id="nested_iterate")
    body_return = ForReturnInvocation(id="return")

    g.add_node(loop)
    g.add_node(collection_adapter)
    g.add_node(nested_iterate)
    g.add_node(body_return)
    g.add_edge(create_edge(loop.id, "item", collection_adapter.id, "value"))
    g.add_edge(create_edge(collection_adapter.id, "collection", nested_iterate.id, "collection"))
    g.add_edge(create_edge(nested_iterate.id, "item", body_return.id, "output"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    with pytest.raises(InvalidEdgeError, match="Iterate nodes inside For loop bodies"):
        g.validate_self()


def test_graph_rejects_iterate_collect_for_return_condition_without_scalar_aggregation():
    g = Graph()
    loop = ForInvocation(id="for", collection=[["a", "b"], ["c", "d"]])
    collection_adapter = PolymorphicStringTestInvocation(id="collection_adapter")
    nested_iterate = IterateInvocation(id="nested_iterate")
    body = AnyTypeTestInvocation(id="body")
    condition = AnyTypeTestInvocation(id="condition")
    collect = CollectInvocation(id="collect")
    body_return = ForReturnInvocation(id="return")

    for node in (loop, collection_adapter, nested_iterate, body, condition, collect, body_return):
        g.add_node(node)
    g.add_edge(create_edge(loop.id, "item", collection_adapter.id, "value"))
    g.add_edge(create_edge(collection_adapter.id, "collection", nested_iterate.id, "collection"))
    g.add_edge(create_edge(nested_iterate.id, "item", body.id, "value"))
    g.add_edge(create_edge(nested_iterate.id, "item", condition.id, "value"))
    g.add_edge(create_edge(body.id, "value", collect.id, "item"))
    g.add_edge(create_edge(collect.id, "collection", body_return.id, "output"))
    g.add_edge(create_edge(condition.id, "value", body_return.id, "continue_condition"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    with pytest.raises(InvalidEdgeError, match="Iterate nodes inside For loop bodies"):
        g.validate_self()


def test_graph_rejects_for_body_edges_that_escape_to_after_loop_nodes():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body = PromptTestInvocation(id="body")
    body_return = ForReturnInvocation(id="return")
    after = AnyTypeTestInvocation(id="after")

    g.add_node(loop)
    g.add_node(body)
    g.add_node(body_return)
    g.add_node(after)
    g.add_edge(create_edge(body.id, "prompt", body_return.id, "output"))
    g.add_edge(create_edge(loop.id, "item", body.id, "prompt"))
    g.add_edge(create_edge(body.id, "prompt", after.id, "value"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    with pytest.raises(InvalidEdgeError, match="escape"):
        g.validate_self()


def test_graph_rejects_for_iteration_branch_that_does_not_reach_return():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body_return = ForReturnInvocation(id="return")
    after = AnyTypeTestInvocation(id="after")

    g.add_node(loop)
    g.add_node(body_return)
    g.add_node(after)
    g.add_edge(create_edge(loop.id, "item", body_return.id, "output"))
    g.add_edge(create_edge(loop.id, "index", after.id, "value"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    with pytest.raises(InvalidEdgeError, match="terminate"):
        g.validate_self()


def test_graph_rejects_for_return_outputs_to_after_loop_nodes():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body_return = ForReturnInvocation(id="return")
    after = AnyTypeTestInvocation(id="after")

    g.add_node(loop)
    g.add_node(body_return)
    g.add_node(after)
    g.add_edge(create_edge(loop.id, "item", body_return.id, "output"))
    g.add_edge(create_edge(body_return.id, "output", after.id, "value"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    with pytest.raises(InvalidEdgeError, match="terminate"):
        g.validate_self()


def test_graph_rejects_final_scoped_for_output_into_body():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body_return = ForReturnInvocation(id="return")

    g.add_node(loop)
    g.add_node(body_return)
    g.add_edge(create_edge(loop.id, "item", body_return.id, "output"))
    g.add_edge(create_edge(loop.id, "final_state", body_return.id, "state"))
    g.add_edge(create_loop_linkage(loop.id, body_return.id))

    with pytest.raises(InvalidEdgeError, match="final-scoped"):
        g.validate_self()


def test_graph_rejects_final_scoped_for_output_through_branch_to_return():
    g = Graph()
    loop = ForInvocation(id="for", collection=["a", "b"])
    body = AnyTypeTestInvocation(id="body")
    body_return = ForReturnInvocation(id="return")
    downstream = AnyTypeTestInvocation(id="downstream")
    downstream_tail = AnyTypeTestInvocation(id="downstream_tail")

    for node in (loop, body, body_return, downstream, downstream_tail):
        g.add_node(node)
    g.edges.extend(
        [
            create_edge(loop.id, "item", body.id, "value"),
            create_edge(body.id, "value", body_return.id, "output"),
            create_edge(loop.id, "final_state", downstream.id, "value"),
            create_edge(downstream.id, "value", downstream_tail.id, "value"),
            create_edge(downstream_tail.id, "value", body_return.id, "state"),
            create_loop_linkage(loop.id, body_return.id),
        ]
    )

    with pytest.raises(InvalidEdgeError, match="final-scoped"):
        g.validate_self()


def test_graph_rejects_orphan_for_return():
    g = Graph()
    body_return = ForReturnInvocation(id="return")

    g.add_node(body_return)

    with pytest.raises(InvalidEdgeError, match="exactly one loop linkage"):
        g.validate_self()


def test_connections_are_incompatible():
    from_node = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    from_field = "image"
    to_node = ESRGANInvocation(id="2")
    to_field = "strength"

    result = are_connections_compatible(from_node, from_field, to_node, to_field)

    assert result is False


def test_connections_incompatible_with_invalid_fields():
    from_node = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    from_field = "invalid_field"
    to_node = ESRGANInvocation(id="2")
    to_field = "image"

    # From field is invalid
    result = are_connections_compatible(from_node, from_field, to_node, to_field)
    assert result is False

    # To field is invalid
    from_field = "image"
    to_field = "invalid_field"

    result = are_connections_compatible(from_node, from_field, to_node, to_field)
    assert result is False


def test_graph_can_add_node():
    g = Graph()
    n = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    g.add_node(n)

    assert n.id in g.nodes


def test_graph_fails_to_add_node_with_duplicate_id():
    g = Graph()
    n = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    g.add_node(n)
    n2 = TextToImageTestInvocation(id="1", prompt="Banana sushi the second")

    with pytest.raises(NodeAlreadyInGraphError):
        g.add_node(n2)


def test_graph_updates_node():
    g = Graph()
    n = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    g.add_node(n)
    n2 = TextToImageTestInvocation(id="2", prompt="Banana sushi the second")
    g.add_node(n2)

    nu = TextToImageTestInvocation(id="1", prompt="Banana sushi updated")

    g.update_node("1", nu)

    assert g.nodes["1"].prompt == "Banana sushi updated"


def test_graph_fails_to_update_node_if_type_changes():
    g = Graph()
    n = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    g.add_node(n)
    n2 = ESRGANInvocation(id="2")
    g.add_node(n2)

    nu = ESRGANInvocation(id="1")

    with pytest.raises(TypeError):
        g.update_node("1", nu)


def test_graph_allows_non_conflicting_id_change():
    g = Graph()
    n = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    g.add_node(n)
    n2 = ESRGANInvocation(id="2")
    g.add_node(n2)
    e1 = create_edge(n.id, "image", n2.id, "image")
    g.add_edge(e1)

    nu = TextToImageTestInvocation(id="3", prompt="Banana sushi")
    g.update_node("1", nu)

    with pytest.raises(NodeNotFoundError):
        g.get_node("1")

    assert g.get_node("3").prompt == "Banana sushi"

    assert len(g.edges) == 1
    assert (
        Edge(source=EdgeConnection(node_id="3", field="image"), destination=EdgeConnection(node_id="2", field="image"))
        in g.edges
    )


def test_graph_fails_to_update_node_id_if_conflict():
    g = Graph()
    n = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    g.add_node(n)
    n2 = TextToImageTestInvocation(id="2", prompt="Banana sushi the second")
    g.add_node(n2)

    nu = TextToImageTestInvocation(id="2", prompt="Banana sushi")
    with pytest.raises(NodeAlreadyInGraphError):
        g.update_node("1", nu)


def test_graph_adds_edge():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = ESRGANInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e = create_edge(n1.id, "image", n2.id, "image")

    g.add_edge(e)

    assert e in g.edges


def test_graph_fails_to_add_edge_with_cycle():
    g = Graph()
    n1 = ESRGANInvocation(id="1")
    g.add_node(n1)
    e = create_edge(n1.id, "image", n1.id, "image")
    with pytest.raises(InvalidEdgeError):
        g.add_edge(e)


def test_graph_fails_to_add_edge_with_long_cycle():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = ESRGANInvocation(id="2")
    n3 = ESRGANInvocation(id="3")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    e1 = create_edge(n1.id, "image", n2.id, "image")
    e2 = create_edge(n2.id, "image", n3.id, "image")
    e3 = create_edge(n3.id, "image", n2.id, "image")
    g.add_edge(e1)
    g.add_edge(e2)
    with pytest.raises(InvalidEdgeError):
        g.add_edge(e3)


def test_graph_fails_to_add_edge_with_missing_node_id():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = ESRGANInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e1 = create_edge("1", "image", "3", "image")
    e2 = create_edge("3", "image", "1", "image")
    with pytest.raises(InvalidEdgeError):
        g.add_edge(e1)
    with pytest.raises(InvalidEdgeError):
        g.add_edge(e2)


def test_graph_fails_to_add_edge_when_destination_exists():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = ESRGANInvocation(id="2")
    n3 = ESRGANInvocation(id="3")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    e1 = create_edge(n1.id, "image", n2.id, "image")
    e2 = create_edge(n1.id, "image", n3.id, "image")
    e3 = create_edge(n2.id, "image", n3.id, "image")
    g.add_edge(e1)
    g.add_edge(e2)
    with pytest.raises(InvalidEdgeError):
        g.add_edge(e3)


def test_graph_fails_to_add_edge_with_mismatched_types():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = ESRGANInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e1 = create_edge("1", "image", "2", "strength")
    with pytest.raises(InvalidEdgeError):
        g.add_edge(e1)


def test_graph_connects_collector():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = TextToImageTestInvocation(id="2", prompt="Banana sushi 2")
    n3 = CollectInvocation(id="3")
    n4 = ListPassThroughInvocation(id="4")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)

    e1 = create_edge("1", "image", "3", "item")
    e2 = create_edge("2", "image", "3", "item")
    e3 = create_edge("3", "collection", "4", "collection")
    g.add_edge(e1)
    g.add_edge(e2)
    g.add_edge(e3)


def test_graph_collector_accepts_if_output_when_both_branches_have_matching_type():
    graph = Graph()
    true_value = StringInvocation(id="true", value="true")
    false_value = StringInvocation(id="false", value="false")
    if_node = IfInvocation(id="if")
    collector = CollectInvocation(id="collect")
    consumer = PromptCollectionTestInvocation(id="consumer", collection=[])
    for node in (true_value, false_value, if_node, collector, consumer):
        graph.add_node(node)

    graph.add_edge(create_edge(true_value.id, "value", if_node.id, "true_input"))
    graph.add_edge(create_edge(false_value.id, "value", if_node.id, "false_input"))
    graph.add_edge(create_edge(if_node.id, "value", collector.id, "item"))
    graph.add_edge(create_edge(collector.id, "collection", consumer.id, "collection"))


def test_graph_collectors_accept_if_output_with_and_without_a_string_collection_seed():
    graph = Graph()
    true_value = StringInvocation(id="true", value="true")
    false_value = StringInvocation(id="false", value="false")
    if_node = IfInvocation(id="if")
    unseeded_collector = CollectInvocation(id="unseeded_collect")
    seeded_collector = CollectInvocation(id="seeded_collect")
    seed = StringCollectionInvocation(id="seed", collection=["existing"])
    unseeded_consumer = PromptCollectionTestInvocation(id="unseeded_consumer", collection=[])
    seeded_consumer = PromptCollectionTestInvocation(id="seeded_consumer", collection=[])
    for node in (
        true_value,
        false_value,
        if_node,
        unseeded_collector,
        seeded_collector,
        seed,
        unseeded_consumer,
        seeded_consumer,
    ):
        graph.add_node(node)

    graph.add_edge(create_edge(true_value.id, "value", if_node.id, "true_input"))
    graph.add_edge(create_edge(false_value.id, "value", if_node.id, "false_input"))
    graph.add_edge(create_edge(if_node.id, "value", unseeded_collector.id, "item"))
    graph.add_edge(create_edge(if_node.id, "value", seeded_collector.id, "item"))
    graph.add_edge(create_edge(seed.id, "collection", seeded_collector.id, "collection"))
    graph.add_edge(create_edge(unseeded_collector.id, "collection", unseeded_consumer.id, "collection"))
    graph.add_edge(create_edge(seeded_collector.id, "collection", seeded_consumer.id, "collection"))


def test_graph_rejects_if_output_when_its_branches_do_not_match_target_type():
    graph = Graph()
    true_value = StringInvocation(id="true", value="true")
    false_value = StringInvocation(id="false", value="false")
    if_node = IfInvocation(id="if")
    consumer = ImageToImageTestInvocation(id="consumer")
    for node in (true_value, false_value, if_node, consumer):
        graph.add_node(node)

    graph.add_edge(create_edge(true_value.id, "value", if_node.id, "true_input"))
    graph.add_edge(create_edge(false_value.id, "value", if_node.id, "false_input"))

    with pytest.raises(InvalidEdgeError, match="Field types are incompatible"):
        graph.add_edge(create_edge(if_node.id, "value", consumer.id, "image"))


def test_graph_rejects_if_branch_that_invalidates_existing_output_edge():
    graph = Graph()
    true_value = StringInvocation(id="true", value="true")
    false_value = IntegerInvocation(id="false", value=1)
    if_node = IfInvocation(id="if")
    consumer = ImageToImageTestInvocation(id="consumer")
    for node in (true_value, false_value, if_node, consumer):
        graph.add_node(node)

    graph.add_edge(create_edge(true_value.id, "value", if_node.id, "true_input"))
    graph.add_edge(create_edge(if_node.id, "value", consumer.id, "image"))

    false_branch_edge = create_edge(false_value.id, "value", if_node.id, "false_input")
    with pytest.raises(InvalidEdgeError):
        graph.add_edge(false_branch_edge)

    assert false_branch_edge not in graph.edges
    assert graph.is_valid()


def test_graph_rejects_if_branch_that_makes_collector_items_incompatible():
    graph = Graph()
    true_value = StringInvocation(id="true", value="true")
    false_value = IntegerInvocation(id="false", value=1)
    if_node = IfInvocation(id="if")
    collector = CollectInvocation(id="collect")
    seed = StringCollectionInvocation(id="seed", collection=["existing"])
    consumer = PromptCollectionTestInvocation(id="consumer", collection=[])
    for node in (true_value, false_value, if_node, collector, seed, consumer):
        graph.add_node(node)

    graph.add_edge(create_edge(true_value.id, "value", if_node.id, "true_input"))
    graph.add_edge(create_edge(if_node.id, "value", collector.id, "item"))
    graph.add_edge(create_edge(seed.id, "collection", collector.id, "collection"))
    graph.add_edge(create_edge(collector.id, "collection", consumer.id, "collection"))

    false_branch_edge = create_edge(false_value.id, "value", if_node.id, "false_input")
    with pytest.raises(InvalidEdgeError, match="Collector output type does not match collector input type"):
        graph.add_edge(false_branch_edge)

    assert false_branch_edge not in graph.edges
    assert graph.is_valid()


def test_graph_rejects_mixed_if_branch_types_for_collector_items():
    graph = Graph()
    string_value = StringInvocation(id="string", value="text")
    integer_value = IntegerInvocation(id="integer", value=1)
    if_node = IfInvocation(id="if")
    collector = CollectInvocation(id="collect")
    for node in (string_value, integer_value, if_node, collector):
        graph.add_node(node)

    graph.add_edge(create_edge(string_value.id, "value", if_node.id, "true_input"))
    graph.add_edge(create_edge(integer_value.id, "value", if_node.id, "false_input"))

    with pytest.raises(InvalidEdgeError, match="Collector input collection items must be of a single type"):
        graph.add_edge(create_edge(if_node.id, "value", collector.id, "item"))


@pytest.mark.parametrize("condition", [True, False])
def test_graph_accepts_and_runs_mixed_numeric_if_branches_through_iterator(condition: bool):
    graph = Graph()
    integer_value = IntegerInvocation(id="integer", value=2)
    float_value = FloatInvocation(id="float", value=3.5)
    if_node = IfInvocation(id="if", condition=condition)
    collector = CollectInvocation(id="collect")
    iterator = IterateInvocation(id="iterate")
    float_consumer = FloatInvocation(id="float_consumer")
    for node in (integer_value, float_value, if_node, collector, iterator, float_consumer):
        graph.add_node(node)

    graph.add_edge(create_edge(integer_value.id, "value", if_node.id, "true_input"))
    graph.add_edge(create_edge(float_value.id, "value", if_node.id, "false_input"))
    graph.add_edge(create_edge(if_node.id, "value", collector.id, "item"))
    graph.add_edge(create_edge(collector.id, "collection", iterator.id, "collection"))
    graph.add_edge(create_edge(iterator.id, "item", float_consumer.id, "value"))

    session = GraphExecutionState(graph=graph)
    run_session_with_mock_context(session)
    output = get_single_output_from_session(session, float_consumer.id)

    assert output.value == (2.0 if condition else 3.5)
    assert isinstance(output.value, float)


@pytest.mark.parametrize("through_if", [False, True], ids=["direct collector", "If branches"])
def test_graph_rejects_integer_iterator_output_for_mixed_numeric_collector(through_if: bool):
    integer_value = IntegerInvocation(id="integer", value=2)
    float_value = FloatInvocation(id="float", value=3.5)
    collector = CollectInvocation(id="collect")
    iterator = IterateInvocation(id="iterate")
    integer_consumer = IntegerInvocation(id="integer_consumer")
    nodes = [integer_value, float_value, collector, iterator, integer_consumer]
    graph = Graph(nodes={node.id: node for node in nodes})

    if through_if:
        if_node = IfInvocation(id="if")
        graph.add_node(if_node)
        graph.add_edge(create_edge(integer_value.id, "value", if_node.id, "true_input"))
        graph.add_edge(create_edge(float_value.id, "value", if_node.id, "false_input"))
        graph.add_edge(create_edge(if_node.id, "value", collector.id, "item"))
    else:
        graph.add_edge(create_edge(integer_value.id, "value", collector.id, "item"))
        graph.add_edge(create_edge(float_value.id, "value", collector.id, "item"))

    graph.add_edge(create_edge(collector.id, "collection", iterator.id, "collection"))
    invalid_output_edge = create_edge(iterator.id, "item", integer_consumer.id, "value")
    with pytest.raises(InvalidEdgeError, match="Iterator collection type must match all iterator output types"):
        graph.add_edge(invalid_output_edge)

    assert invalid_output_edge not in graph.edges


def test_graph_rejects_iterator_consumers_incompatible_with_if_collector_branches():
    graph = Graph()
    string_value = StringInvocation(id="string", value="text")
    image_value = ImageToImageTestInvocation(id="image")
    string_collector = CollectInvocation(id="string_collector")
    image_collector = CollectInvocation(id="image_collector")
    if_node = IfInvocation(id="if")
    iterator = IterateInvocation(id="iterator")
    string_sink = StringInvocation(id="string_sink")
    for node in (string_value, image_value, string_collector, image_collector, if_node, iterator, string_sink):
        graph.add_node(node)

    graph.add_edge(create_edge(string_value.id, "value", string_collector.id, "item"))
    graph.add_edge(create_edge(image_value.id, "image", image_collector.id, "item"))
    graph.add_edge(create_edge(string_collector.id, "collection", if_node.id, "true_input"))
    graph.add_edge(create_edge(image_collector.id, "collection", if_node.id, "false_input"))
    graph.add_edge(create_edge(if_node.id, "value", iterator.id, "collection"))

    with pytest.raises(InvalidEdgeError, match="Iterator output type does not match iterator input type"):
        graph.add_edge(create_edge(iterator.id, "item", string_sink.id, "value"))


@pytest.mark.parametrize("collector_count", [1, 3])
def test_graph_rejects_if_branch_that_invalidates_iterator_through_collectors(collector_count: int):
    true_value = StringInvocation(id="true", value="true")
    false_value = StringInvocation(id="false", value="false")
    if_node = IfInvocation(id="if")
    collectors = [CollectInvocation(id=f"collect_{index}") for index in range(collector_count)]
    iterator = IterateInvocation(id="iterator")
    image_sink = ImageToImageTestInvocation(id="image_sink")
    string_sink = StringInvocation(id="string_sink")
    nodes = [true_value, false_value, if_node, *collectors, iterator, image_sink, string_sink]
    image_edge = create_edge(iterator.id, "item", image_sink.id, "image")
    graph = Graph(
        nodes={node.id: node for node in nodes},
        edges=[
            create_edge(true_value.id, "value", if_node.id, "true_input"),
            create_edge(if_node.id, "value", collectors[0].id, "item"),
            *[
                create_edge(source.id, "collection", target.id, "collection")
                for source, target in zip(collectors, collectors[1:], strict=False)
            ],
            create_edge(collectors[-1].id, "collection", iterator.id, "collection"),
            image_edge,
        ],
    )
    graph.validate_self()
    original_edges = list(graph.edges)
    branch_edge = create_edge(false_value.id, "value", if_node.id, "false_input")

    with pytest.raises(InvalidEdgeError, match="[Ii]terator.*type"):
        graph.add_edge(branch_edge)

    assert graph.edges == original_edges
    graph.validate_self()

    graph.delete_edge(image_edge)
    graph.add_edge(create_edge(iterator.id, "item", string_sink.id, "value"))
    graph.add_edge(branch_edge)
    graph.validate_self()


def test_graph_if_branch_revalidates_only_affected_iterators_once(monkeypatch: pytest.MonkeyPatch):
    nodes = [
        StringInvocation(id="true", value="true"),
        StringInvocation(id="false", value="false"),
        IfInvocation(id="if"),
        CollectInvocation(id="first_collect"),
        CollectInvocation(id="second_collect"),
        IterateInvocation(id="affected_iterator"),
        StringInvocation(id="sink"),
        *[IterateInvocation(id=f"unrelated_{index}") for index in range(32)],
    ]
    graph = Graph(
        nodes={node.id: node for node in nodes},
        edges=[
            create_edge("true", "value", "if", "true_input"),
            create_edge("if", "value", "first_collect", "item"),
            create_edge("if", "value", "second_collect", "item"),
            create_edge("first_collect", "collection", "second_collect", "collection"),
            create_edge("second_collect", "collection", "affected_iterator", "collection"),
            create_edge("affected_iterator", "item", "sink", "value"),
        ],
    )
    validated_iterators: list[str] = []
    original_validate_iterator = Graph._is_iterator_connection_valid

    def count_iterator_validations(graph: Graph, node_id: str, *args: Any, **kwargs: Any):
        validated_iterators.append(node_id)
        return original_validate_iterator(graph, node_id, *args, **kwargs)

    monkeypatch.setattr(Graph, "_is_iterator_connection_valid", count_iterator_validations)
    graph.add_edge(create_edge("false", "value", "if", "false_input"))

    assert validated_iterators == ["affected_iterator"]


def test_graph_collector_validation_avoids_ordinary_source_allocations(monkeypatch: pytest.MonkeyPatch):
    item_sources = [StringInvocation(id=f"string_{index}", value="text") for index in range(16)]
    seed = StringCollectionInvocation(id="seed", collection=["seed"])
    collector = CollectInvocation(id="collect")
    consumer = PromptCollectionTestInvocation(id="consumer", collection=[])
    nodes = [*item_sources, seed, collector, consumer]
    edges = [create_edge(source.id, "value", collector.id, "item") for source in item_sources]
    edges.extend(
        [
            create_edge(seed.id, "collection", collector.id, "collection"),
            create_edge(collector.id, "collection", consumer.id, "collection"),
        ]
    )
    graph = Graph(nodes={node.id: node for node in nodes}, edges=edges)
    original_connection_init = EdgeConnection.__init__
    connection_allocations = 0

    def count_connection_allocations(connection: EdgeConnection, **kwargs: Any):
        nonlocal connection_allocations
        connection_allocations += 1
        original_connection_init(connection, **kwargs)

    monkeypatch.setattr(EdgeConnection, "__init__", count_connection_allocations)

    graph.validate_self()

    assert graph.edges == edges
    assert connection_allocations == 0


def test_graph_validation_bounds_shared_if_branch_validation_work(monkeypatch: pytest.MonkeyPatch):
    from invokeai.app.services.shared import graph as graph_facade

    true_value = StringInvocation(id="true", value="true")
    false_value = StringInvocation(id="false", value="false")
    nodes = {true_value.id: true_value, false_value.id: false_value}
    edges = []

    previous_if_id = "if_0"
    nodes[previous_if_id] = IfInvocation(id=previous_if_id)
    edges.extend(
        [
            create_edge(true_value.id, "value", previous_if_id, "true_input"),
            create_edge(false_value.id, "value", previous_if_id, "false_input"),
        ]
    )

    for index in range(1, 12):
        if_node_id = f"if_{index}"
        nodes[if_node_id] = IfInvocation(id=if_node_id)
        edges.extend(
            [
                create_edge(previous_if_id, "value", if_node_id, "true_input"),
                create_edge(previous_if_id, "value", if_node_id, "false_input"),
            ]
        )
        previous_if_id = if_node_id

    graph = Graph(nodes=nodes, edges=edges)
    compatibility_calls = 0
    branch_input_lookups = 0
    original_are_connections_compatible = graph_facade.are_connections_compatible
    original_get_input_edges = Graph._get_input_edges

    def count_compatibility_calls(*args, **kwargs):
        nonlocal compatibility_calls
        compatibility_calls += 1
        return original_are_connections_compatible(*args, **kwargs)

    def count_branch_input_lookups(graph: Graph, node_id: str, field: str | None = None, **kwargs: Any):
        nonlocal branch_input_lookups
        if field in ("true_input", "false_input"):
            branch_input_lookups += 1
        return original_get_input_edges(graph, node_id, field, **kwargs)

    monkeypatch.setattr(graph_facade, "are_connections_compatible", count_compatibility_calls)
    monkeypatch.setattr(Graph, "_get_input_edges", count_branch_input_lookups)
    graph.validate_self()

    if_node_count = sum(isinstance(node, IfInvocation) for node in nodes.values())
    assert branch_input_lookups <= len(edges) * if_node_count * 2
    assert compatibility_calls <= len(edges) * 2


def test_graph_rejects_collector_output_edge_before_input_edge():
    graph = Graph()
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_node(ListPassThroughInvocation(id="consumer"))

    with pytest.raises(InvalidEdgeError, match="Collector must have at least one item or collection input edge"):
        graph.add_edge(create_edge("collect", "collection", "consumer", "collection"))


# TODO: test that derived types mixed with base types are compatible


def test_graph_collector_invalid_with_varying_input_types():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = PromptTestInvocation(id="2", prompt="banana sushi 2")
    n3 = CollectInvocation(id="3")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)

    e1 = create_edge("1", "image", "3", "item")
    e2 = create_edge("2", "prompt", "3", "item")
    g.add_edge(e1)

    with pytest.raises(InvalidEdgeError):
        g.add_edge(e2)


def test_graph_collector_invalid_with_varying_input_output():
    g = Graph()
    n1 = PromptTestInvocation(id="1", prompt="Banana sushi")
    n2 = PromptTestInvocation(id="2", prompt="Banana sushi 2")
    n3 = CollectInvocation(id="3")
    n4 = ListPassThroughInvocation(id="4")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)

    e1 = create_edge("1", "prompt", "3", "item")
    e2 = create_edge("2", "prompt", "3", "item")
    e3 = create_edge("3", "collection", "4", "collection")
    g.add_edge(e1)
    g.add_edge(e2)

    with pytest.raises(InvalidEdgeError):
        g.add_edge(e3)


def test_graph_collector_invalid_with_non_list_output():
    g = Graph()
    n1 = PromptTestInvocation(id="1", prompt="Banana sushi")
    n2 = PromptTestInvocation(id="2", prompt="Banana sushi 2")
    n3 = CollectInvocation(id="3")
    n4 = PromptTestInvocation(id="4")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)

    e1 = create_edge("1", "prompt", "3", "item")
    e2 = create_edge("2", "prompt", "3", "item")
    e3 = create_edge("3", "collection", "4", "prompt")
    g.add_edge(e1)
    g.add_edge(e2)

    with pytest.raises(InvalidEdgeError):
        g.add_edge(e3)


def test_graph_collector_can_chain_collection_input():
    g = Graph()
    n1 = PromptCollectionTestInvocation(id="1", collection=["Banana", "Sushi"])
    n2 = PromptTestInvocation(id="2", prompt="Ramen")
    n3 = CollectInvocation(id="3")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)

    g.add_edge(create_edge("1", "collection", "3", "collection"))
    g.add_edge(create_edge("2", "prompt", "3", "item"))

    session = GraphExecutionState(graph=g)
    run_session_with_mock_context(session)
    output = get_single_output_from_session(session, n3.id)

    assert isinstance(output, CollectInvocationOutput)
    assert output.collection == ["Banana", "Sushi", "Ramen"]


def test_graph_collector_chain_rejects_mismatched_item_type():
    g = Graph()
    n1 = PromptCollectionTestInvocation(id="1", collection=["Banana", "Sushi"])
    n2 = IntegerInvocation(id="2", value=7)
    n3 = CollectInvocation(id="3")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)

    g.add_edge(create_edge("1", "collection", "3", "collection"))
    with pytest.raises(InvalidEdgeError):
        g.add_edge(create_edge("2", "value", "3", "item"))


def test_graph_iterator_accepts_collector_chained_collection_input():
    g = Graph()
    n1 = PromptTestInvocation(id="1", prompt="Banana")
    n2 = CollectInvocation(id="2")
    n3 = CollectInvocation(id="3")
    n4 = IterateInvocation(id="4")
    n5 = PromptTestInvocation(id="5")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)
    g.add_node(n5)

    g.add_edge(create_edge("1", "prompt", "2", "item"))
    g.add_edge(create_edge("2", "collection", "3", "collection"))
    g.add_edge(create_edge("3", "collection", "4", "collection"))
    g.add_edge(create_edge("4", "item", "5", "prompt"))

    session = GraphExecutionState(graph=g)
    run_session_with_mock_context(session)

    output = get_single_output_from_session(session, n5.id)
    assert isinstance(output, PromptTestInvocationOutput)
    assert output.prompt == "Banana"


def test_graph_collector_chain_rejects_upstream_mismatch_added_late():
    g = Graph()
    n1 = CollectInvocation(id="1")
    n2 = CollectInvocation(id="2")
    n3 = PromptTestInvocation(id="3", prompt="typed-as-string")
    n4 = ColorInvocation(id="4")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)

    # Connect chain first while n1 is still untyped.
    g.add_edge(create_edge("1", "collection", "2", "collection"))
    # Constrain downstream collector to strings.
    g.add_edge(create_edge("3", "prompt", "2", "item"))
    # Now adding an incompatible type to the upstream collector must fail.
    with pytest.raises(InvalidEdgeError):
        g.add_edge(create_edge("4", "color", "1", "item"))


def test_graph_collector_rejects_mismatched_item_with_union_collection_input():
    g = Graph()
    n1 = UnionCollectionTestInvocation(id="1")
    n2 = CollectInvocation(id="2")
    n3 = ColorInvocation(id="3")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)

    g.add_edge(create_edge("1", "value", "2", "collection"))
    with pytest.raises(InvalidEdgeError):
        g.add_edge(create_edge("3", "color", "2", "item"))


def test_graph_connects_iterator():
    g = Graph()
    n1 = ListPassThroughInvocation(id="1")
    n2 = IterateInvocation(id="2")
    n3 = ImageToImageTestInvocation(id="3", prompt="Banana sushi")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)

    e1 = create_edge("1", "collection", "2", "collection")
    e2 = create_edge("2", "item", "3", "image")
    g.add_edge(e1)
    g.add_edge(e2)


# TODO: TEST INVALID ITERATOR SCENARIOS


def test_graph_iterator_invalid_if_multiple_inputs():
    g = Graph()
    n1 = ListPassThroughInvocation(id="1")
    n2 = IterateInvocation(id="2")
    n3 = ImageToImageTestInvocation(id="3", prompt="Banana sushi")
    n4 = ListPassThroughInvocation(id="4")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)

    e1 = create_edge("1", "collection", "2", "collection")
    e2 = create_edge("2", "item", "3", "image")
    e3 = create_edge("4", "collection", "2", "collection")
    g.add_edge(e1)
    g.add_edge(e2)

    with pytest.raises(InvalidEdgeError):
        g.add_edge(e3)


def test_graph_iterator_invalid_if_input_not_list():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = IterateInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)

    e1 = create_edge("1", "collection", "2", "collection")

    with pytest.raises(InvalidEdgeError):
        g.add_edge(e1)


def test_graph_iterator_invalid_if_output_and_input_types_different():
    g = Graph()
    n1 = ListPassThroughInvocation(id="1")
    n2 = IterateInvocation(id="2")
    n3 = PromptTestInvocation(id="3", prompt="Banana sushi")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)

    e1 = create_edge("1", "collection", "2", "collection")
    e2 = create_edge("2", "item", "3", "prompt")
    g.add_edge(e1)

    with pytest.raises(InvalidEdgeError):
        g.add_edge(e2)


def test_graph_validates():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = ESRGANInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e1 = create_edge("1", "image", "2", "image")
    g.add_edge(e1)

    assert g.is_valid() is True


def test_graph_invalid_if_edges_reference_missing_nodes():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    g.nodes[n1.id] = n1
    e1 = create_edge("1", "image", "2", "image")
    g.edges.append(e1)

    assert g.is_valid() is False


def test_graph_invalid_if_has_cycle():
    g = Graph()
    n1 = ESRGANInvocation(id="1")
    n2 = ESRGANInvocation(id="2")
    g.nodes[n1.id] = n1
    g.nodes[n2.id] = n2
    e1 = create_edge("1", "image", "2", "image")
    e2 = create_edge("2", "image", "1", "image")
    g.edges.append(e1)
    g.edges.append(e2)

    assert g.is_valid() is False


def test_graph_invalid_with_invalid_connection():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = ESRGANInvocation(id="2")
    g.nodes[n1.id] = n1
    g.nodes[n2.id] = n2
    e1 = create_edge("1", "image", "2", "strength")
    g.edges.append(e1)

    assert g.is_valid() is False


def test_graph_edge_indexes_follow_direct_edge_list_mutation():
    graph = Graph()
    range_node = PromptCollectionTestInvocation(id="range", collection=["one"])
    iterate_node = IterateInvocation(id="iterate")
    graph.add_node(range_node)
    graph.add_node(iterate_node)
    graph.add_edge(create_edge("range", "collection", "iterate", "collection"))

    assert len(graph._get_input_edges("iterate")) == 1

    graph.edges.clear()

    assert graph._get_input_edges("iterate") == []
    with pytest.raises(InvalidEdgeError):
        graph.validate_self()


def test_graph_edge_indexes_follow_edge_list_replacement():
    edge = create_edge("source", "value", "destination", "value")
    graph = Graph(edges=[edge])
    assert graph._get_output_edges("source") == [edge]

    graph.edges = []

    assert graph._get_output_edges("source") == []


def test_graph_edge_indexes_follow_partially_failed_edge_list_mutation():
    first_edge = create_edge("source", "value", "first", "value")
    second_edge = create_edge("source", "value", "second", "value")
    graph = Graph(edges=[first_edge])
    graph._get_output_edges("source")

    def failing_edges():
        yield second_edge
        raise RuntimeError("failed while extending edges")

    with pytest.raises(RuntimeError):
        graph.edges.extend(failing_edges())

    assert graph._get_output_edges("source") == [first_edge, second_edge]


def test_graph_copy_rebinds_edge_list_invalidation():
    edge = create_edge("source", "value", "destination", "value")
    original = Graph(edges=[edge])
    copied = original.model_copy()
    copied._get_output_edges("source")

    copied.edges.clear()

    assert copied._get_output_edges("source") == []
    assert original._get_output_edges("source") == [edge]


@pytest.mark.parametrize("copy_graph", [copy.copy, copy.deepcopy])
def test_graph_python_copy_rebinds_edge_list_invalidation(copy_graph):
    edge = create_edge("source", "value", "destination", "value")
    original = Graph(edges=[edge])
    copied = copy_graph(original)
    copied._get_output_edges("source")

    copied.edges.clear()

    assert copied._get_output_edges("source") == []
    assert original._get_output_edges("source") == [edge]


def test_graph_pickle_rebinds_edge_list_invalidation():
    edge = create_edge("source", "value", "destination", "value")
    graph = Graph(edges=[edge])

    restored = pickle.loads(pickle.dumps(graph))
    restored._get_output_edges("source")
    restored.edges.clear()

    assert restored._get_output_edges("source") == []


def test_graph_edges_cannot_be_mutated_after_indexing():
    edge = create_edge("source", "value", "destination", "value")
    graph = Graph(edges=[edge])
    graph._get_output_edges("source")

    with pytest.raises(ValidationError):
        edge.source.node_id = "different-source"

    assert graph._get_output_edges("source") == [edge]


def test_graph_equality_ignores_adjacency_cache_state():
    left = Graph(id="same")
    right = Graph(id="same")

    left._get_input_edges("missing")

    assert left == right


def test_graph_gets_networkx_graph():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = ESRGANInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e = create_edge(n1.id, "image", n2.id, "image")
    g.add_edge(e)

    nxg = g.nx_graph()

    assert "1" in nxg.nodes
    assert "2" in nxg.nodes
    assert ("1", "2") in nxg.edges


# TODO: Graph serializes and deserializes
def test_graph_can_serialize():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = ESRGANInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e = create_edge(n1.id, "image", n2.id, "image")
    g.add_edge(e)

    # Not throwing on this line is sufficient
    _ = g.model_dump_json()


def test_graph_can_deserialize():
    g = Graph()
    n1 = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    n2 = ImageToImageTestInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e = create_edge(n1.id, "image", n2.id, "image")
    g.add_edge(e)

    json = g.model_dump_json()
    GraphValidator = TypeAdapter(Graph)
    g2 = GraphValidator.validate_json(json)

    assert g2 is not None
    assert g2.nodes["1"] is not None
    assert g2.nodes["2"] is not None
    assert len(g2.edges) == 1
    assert g2.edges[0].source.node_id == "1"
    assert g2.edges[0].source.field == "image"
    assert g2.edges[0].destination.node_id == "2"
    assert g2.edges[0].destination.field == "image"


def test_invocation_decorator():
    invocation_type = "test_invocation_decorator"
    title = "Test Invocation"
    tags = ["first", "second", "third"]
    category = "category"
    version = "1.2.3"

    @invocation(invocation_type, title=title, tags=tags, category=category, version=version)
    class TestInvocation(BaseInvocation):
        def invoke(self) -> PromptTestInvocationOutput:
            pass

    schema = TestInvocation.model_json_schema()

    assert schema.get("title") == title
    assert schema.get("tags") == tags
    assert schema.get("category") == category
    assert schema.get("version") == version
    assert TestInvocation(id="1").type == invocation_type  # type: ignore (type is dynamically added)


def test_invocation_version_must_be_semver():
    valid_version = "1.0.0"
    invalid_version = "not_semver"

    @invocation("test_invocation_version_valid", version=valid_version)
    class ValidVersionInvocation(BaseInvocation):
        def invoke(self) -> PromptTestInvocationOutput:
            pass

    with pytest.raises(InvalidVersionError):

        @invocation("test_invocation_version_invalid", version=invalid_version)
        class InvalidVersionInvocation(BaseInvocation):
            def invoke(self):
                pass


def test_invocation_output_decorator():
    output_type = "test_output"

    @invocation_output(output_type)
    class TestOutput(BaseInvocationOutput):
        pass

    assert TestOutput().type == output_type  # type: ignore (type is dynamically added)


def test_floats_accept_ints():
    g = Graph()
    n1 = IntegerInvocation(id="1", value=1)
    n2 = FloatInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e = create_edge(n1.id, "value", n2.id, "value")

    # Not throwing on this line is sufficient
    g.add_edge(e)


def test_ints_do_not_accept_floats():
    g = Graph()
    n1 = FloatInvocation(id="1", value=1.0)
    n2 = IntegerInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e = create_edge(n1.id, "value", n2.id, "value")

    with pytest.raises(InvalidEdgeError):
        g.add_edge(e)


def test_polymorphic_accepts_single():
    g = Graph()
    n1 = StringInvocation(id="1", value="banana")
    n2 = PolymorphicStringTestInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e1 = create_edge(n1.id, "value", n2.id, "value")
    # Not throwing on this line is sufficient
    g.add_edge(e1)


def test_polymorphic_accepts_collection_of_same_base_type():
    g = Graph()
    n1 = PromptCollectionTestInvocation(id="1", collection=["banana", "sundae"])
    n2 = PolymorphicStringTestInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e1 = create_edge(n1.id, "collection", n2.id, "value")
    # Not throwing on this line is sufficient
    g.add_edge(e1)


def test_polymorphic_does_not_accept_collection_of_different_base_type():
    g = Graph()
    n1 = FloatCollectionInvocation(id="1", collection=[1.0, 2.0, 3.0])
    n2 = PolymorphicStringTestInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e1 = create_edge(n1.id, "collection", n2.id, "value")
    with pytest.raises(InvalidEdgeError):
        g.add_edge(e1)


def test_polymorphic_does_not_accept_generic_collection():
    g = Graph()
    n1 = IntegerInvocation(id="1", value=1)
    n2 = IntegerInvocation(id="2", value=2)
    n3 = CollectInvocation(id="3")
    n4 = PolymorphicStringTestInvocation(id="4")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)
    e1 = create_edge(n1.id, "value", n3.id, "item")
    e2 = create_edge(n2.id, "value", n3.id, "item")
    e3 = create_edge(n3.id, "collection", n4.id, "value")
    g.add_edge(e1)
    g.add_edge(e2)
    with pytest.raises(InvalidEdgeError):
        g.add_edge(e3)


def test_any_accepts_integer():
    g = Graph()
    n1 = IntegerInvocation(id="1", value=1)
    n2 = AnyTypeTestInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e = create_edge(n1.id, "value", n2.id, "value")
    # Not throwing on this line is sufficient
    g.add_edge(e)


def test_any_accepts_string():
    g = Graph()
    n1 = StringInvocation(id="1", value="banana sundae")
    n2 = AnyTypeTestInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e = create_edge(n1.id, "value", n2.id, "value")
    # Not throwing on this line is sufficient
    g.add_edge(e)


def test_any_accepts_generic_collection():
    g = Graph()
    n1 = IntegerInvocation(id="1", value=1)
    n2 = IntegerInvocation(id="2", value=2)
    n3 = CollectInvocation(id="3")
    n4 = AnyTypeTestInvocation(id="4")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)
    e1 = create_edge(n1.id, "value", n3.id, "item")
    e2 = create_edge(n2.id, "value", n3.id, "item")
    e3 = create_edge(n3.id, "collection", n4.id, "value")
    g.add_edge(e1)
    g.add_edge(e2)
    # Not throwing on this line is sufficient
    g.add_edge(e3)


def test_any_accepts_prompt_collection():
    g = Graph()
    n1 = PromptCollectionTestInvocation(id="1", collection=["banana", "sundae"])
    n2 = AnyTypeTestInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e = create_edge(n1.id, "collection", n2.id, "value")
    # Not throwing on this line is sufficient
    g.add_edge(e)


def test_any_accepts_any():
    g = Graph()
    n1 = AnyTypeTestInvocation(id="1")
    n2 = AnyTypeTestInvocation(id="2")
    g.add_node(n1)
    g.add_node(n2)
    e = create_edge(n1.id, "value", n2.id, "value")
    # Not throwing on this line is sufficient
    g.add_edge(e)


def test_iterate_accepts_collection():
    g = Graph()
    n1 = IntegerInvocation(id="1", value=1)
    n2 = IntegerInvocation(id="2", value=2)
    n3 = CollectInvocation(id="3")
    n4 = IterateInvocation(id="4")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)
    e1 = create_edge(n1.id, "value", n3.id, "item")
    e2 = create_edge(n2.id, "value", n3.id, "item")
    e3 = create_edge(n3.id, "collection", n4.id, "collection")
    g.add_edge(e1)
    g.add_edge(e2)
    g.add_edge(e3)


def test_iterate_accepts_collection_from_any_only_collector():
    g = Graph()
    n1 = AnyTypeTestInvocation(id="1")
    n2 = CollectInvocation(id="2")
    n3 = IterateInvocation(id="3")
    n4 = AnyTypeTestInvocation(id="4")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)
    e1 = create_edge(n1.id, "value", n2.id, "item")
    e2 = create_edge(n2.id, "collection", n3.id, "collection")
    e3 = create_edge(n3.id, "item", n4.id, "value")
    g.add_edge(e1)
    g.add_edge(e2)
    g.add_edge(e3)


def test_iterate_validates_collection_inputs_against_iterator_outputs():
    g = Graph()
    n1 = IntegerInvocation(id="1", value=1)
    n2 = IntegerInvocation(id="2", value=2)
    n3 = CollectInvocation(id="3")
    n4 = IterateInvocation(id="4")
    n5 = AddInvocation(id="5")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_node(n4)
    g.add_node(n5)
    e1 = create_edge(n1.id, "value", n3.id, "item")
    e2 = create_edge(n2.id, "value", n3.id, "item")
    e3 = create_edge(n3.id, "collection", n4.id, "collection")
    e4 = create_edge(n4.id, "item", n5.id, "a")
    g.add_edge(e1)
    g.add_edge(e2)
    g.add_edge(e3)
    # Not throwing on this line indicates the collector's input types validated successfully against the iterator's output types
    g.add_edge(e4)
    with pytest.raises(InvalidEdgeError, match="Iterator collection type must match all iterator output types"):
        # Connect iterator to a node with a different type than the collector inputs which is not allowed
        n6 = ColorInvocation(id="6")
        g.add_node(n6)
        e5 = create_edge(n4.id, "item", n6.id, "color")
        g.add_edge(e5)


def test_graph_can_generate_schema():
    # Not throwing on this line is sufficient
    # NOTE: if this test fails, it's PROBABLY because a new invocation type is breaking schema generation
    models_json_schema([(Graph, "serialization")])


def test_nodes_must_implement_invoke_method():
    with pytest.raises(ValueError, match='must implement the "invoke" method'):

        @invocation("test_no_invoke_method", version="1.0.0")
        class NoInvokeMethodInvocation(BaseInvocation):
            pass


def test_nodes_must_return_invocation_output():
    with pytest.raises(ValueError, match="must have a return annotation of a subclass of BaseInvocationOutput"):

        @invocation("test_no_output", version="1.0.0")
        class NoOutputInvocation(BaseInvocation):
            def invoke(self) -> str:
                return "foo"


def test_nodes_must_return_invocation_output_under_optimized_python():
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            textwrap.dedent(
                """
                from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation

                try:
                    @invocation("test_no_output_optimized", version="1.0.0")
                    class NoOutputInvocation(BaseInvocation):
                        def invoke(self) -> str:
                            return "foo"
                except ValueError:
                    pass
                else:
                    raise SystemExit("invalid invocation return annotation was accepted under python -O")
                """
            ),
        ],
        capture_output=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_collector_different_incomers():
    """Tests an edge case where a collector has incoming edges from invocations with differently-named output fields."""
    g = Graph()
    # This node has a str type output field named "prompt"
    n1 = PromptTestInvocation(id="1", prompt="Banana")
    # This node has a str type output field named "value"
    n2 = StringInvocation(id="2", value="Sushi")
    n3 = CollectInvocation(id="3")
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    e1 = create_edge(n1.id, "prompt", n3.id, "item")
    e2 = create_edge(n2.id, "value", n3.id, "item")
    g.add_edge(e1)
    g.add_edge(e2)
    session = GraphExecutionState(graph=g)
    # The bug resulted in an error like this when calling session.next():
    #   Field types are incompatible (a0f9797b-1179-4200-81ae-6ef981660163.prompt -> ccc6af96-2a65-4bbe-a02f-4189bb4770ac.item)
    run_session_with_mock_context(session)
    output = get_single_output_from_session(session, n3.id)
    assert isinstance(output, CollectInvocationOutput)
    assert set(output.collection) == {"Banana", "Sushi"}  # Both inputs should be collected, no order guarantee


def test_iterator_collector_iterator_chain():
    """Test basic Iterator -> Collector -> Iterator chain execution."""
    g = Graph()
    # Start with a collection of strings
    n1 = PromptCollectionTestInvocation(id="1", collection=["apple", "banana", "cherry"])
    # First iterator breaks down the collection
    n2 = IterateInvocation(id="2")
    # Process each item (pass-through for simplicity)
    n3 = PromptTestInvocation(id="3")
    # Collector reassembles the processed items
    n4 = CollectInvocation(id="4")
    # Second iterator breaks down the collected items again
    n5 = IterateInvocation(id="5")
    # Process each item again
    n6 = PromptTestInvocation(id="6")
    # Final collector
    n7 = CollectInvocation(id="7")

    for node in [n1, n2, n3, n4, n5, n6, n7]:
        g.add_node(node)

    # Chain the nodes together
    g.add_edge(create_edge(n1.id, "collection", n2.id, "collection"))
    g.add_edge(create_edge(n2.id, "item", n3.id, "prompt"))
    g.add_edge(create_edge(n3.id, "prompt", n4.id, "item"))
    g.add_edge(create_edge(n4.id, "collection", n5.id, "collection"))
    g.add_edge(create_edge(n5.id, "item", n6.id, "prompt"))
    g.add_edge(create_edge(n6.id, "prompt", n7.id, "item"))

    # Execute the graph
    session = GraphExecutionState(graph=g)
    run_session_with_mock_context(session)

    # Verify the final output contains all original items
    output = get_single_output_from_session(session, n7.id)
    assert isinstance(output, CollectInvocationOutput)
    assert set(output.collection) == {"apple", "banana", "cherry"}


def test_parallel_iterator_collector_iterator_chains():
    """Test two parallel Iterator -> Collector -> Iterator chains."""
    g = Graph()

    # First chain
    n1 = PromptCollectionTestInvocation(id="1", collection=["a", "b"])
    n2 = IterateInvocation(id="2")
    n3 = PromptTestInvocation(id="3")
    n4 = CollectInvocation(id="4")
    n5 = IterateInvocation(id="5")
    n6 = PromptTestInvocation(id="6")
    n7 = CollectInvocation(id="7")

    # Second chain
    n8 = PromptCollectionTestInvocation(id="8", collection=["x", "y", "z"])
    n9 = IterateInvocation(id="9")
    n10 = PromptTestInvocation(id="10")
    n11 = CollectInvocation(id="11")
    n12 = IterateInvocation(id="12")
    n13 = PromptTestInvocation(id="13")
    n14 = CollectInvocation(id="14")

    for node in [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14]:
        g.add_node(node)

    # First chain edges
    g.add_edge(create_edge(n1.id, "collection", n2.id, "collection"))
    g.add_edge(create_edge(n2.id, "item", n3.id, "prompt"))
    g.add_edge(create_edge(n3.id, "prompt", n4.id, "item"))
    g.add_edge(create_edge(n4.id, "collection", n5.id, "collection"))
    g.add_edge(create_edge(n5.id, "item", n6.id, "prompt"))
    g.add_edge(create_edge(n6.id, "prompt", n7.id, "item"))

    # Second chain edges
    g.add_edge(create_edge(n8.id, "collection", n9.id, "collection"))
    g.add_edge(create_edge(n9.id, "item", n10.id, "prompt"))
    g.add_edge(create_edge(n10.id, "prompt", n11.id, "item"))
    g.add_edge(create_edge(n11.id, "collection", n12.id, "collection"))
    g.add_edge(create_edge(n12.id, "item", n13.id, "prompt"))
    g.add_edge(create_edge(n13.id, "prompt", n14.id, "item"))

    # Execute the graph
    session = GraphExecutionState(graph=g)
    run_session_with_mock_context(session)

    # Verify both chains executed correctly
    output1 = get_single_output_from_session(session, n7.id)
    output2 = get_single_output_from_session(session, n14.id)

    assert isinstance(output1, CollectInvocationOutput)
    assert isinstance(output2, CollectInvocationOutput)
    assert set(output1.collection) == {"a", "b"}
    assert set(output2.collection) == {"x", "y", "z"}


def test_iterator_collector_iterator_chain_with_cross_dependency():
    """Test Iterator -> Collector -> Iterator chain where the second iterator depends on both chains."""
    g = Graph()

    # First chain: process strings
    n1 = PromptCollectionTestInvocation(id="1", collection=["hello", "world"])
    n2 = IterateInvocation(id="2")
    n3 = PromptTestInvocation(id="3")
    n4 = CollectInvocation(id="4")

    # Second chain: process the collected results
    n5 = IterateInvocation(id="5")
    n6 = PromptTestInvocation(id="6")

    # Additional input that gets collected with the iterator results
    n7 = PromptTestInvocation(id="7", prompt="extra")

    # Collector that receives from both the iterator and the additional input
    n8 = CollectInvocation(id="8")

    for node in [n1, n2, n3, n4, n5, n6, n7, n8]:
        g.add_node(node)

    # First chain
    g.add_edge(create_edge(n1.id, "collection", n2.id, "collection"))
    g.add_edge(create_edge(n2.id, "item", n3.id, "prompt"))
    g.add_edge(create_edge(n3.id, "prompt", n4.id, "item"))

    # Second chain
    g.add_edge(create_edge(n4.id, "collection", n5.id, "collection"))
    g.add_edge(create_edge(n5.id, "item", n6.id, "prompt"))

    # Cross-dependency: collector receives from both iterator and regular node
    g.add_edge(create_edge(n6.id, "prompt", n8.id, "item"))
    g.add_edge(create_edge(n7.id, "prompt", n8.id, "item"))

    # Execute the graph
    session = GraphExecutionState(graph=g)
    run_session_with_mock_context(session)

    # Verify the final output contains items from both sources
    output = get_single_output_from_session(session, n8.id)
    assert isinstance(output, CollectInvocationOutput)
    # Should contain the processed items from the iterator plus the extra item
    assert set(output.collection) == {"hello", "world", "extra"}


def test_iterator_collector_iterator_chain_with_empty_collection():
    """Test Iterator -> Collector -> Iterator chain with empty input collection."""
    g = Graph()

    # Start with empty collection
    n1 = PromptCollectionTestInvocation(id="1", collection=[])
    n2 = IterateInvocation(id="2")
    n3 = PromptTestInvocation(id="3")
    n4 = CollectInvocation(id="4")
    n5 = IterateInvocation(id="5")
    n6 = PromptTestInvocation(id="6")
    n7 = CollectInvocation(id="7")

    for node in [n1, n2, n3, n4, n5, n6, n7]:
        g.add_node(node)

    # Chain the nodes
    g.add_edge(create_edge(n1.id, "collection", n2.id, "collection"))
    g.add_edge(create_edge(n2.id, "item", n3.id, "prompt"))
    g.add_edge(create_edge(n3.id, "prompt", n4.id, "item"))
    g.add_edge(create_edge(n4.id, "collection", n5.id, "collection"))
    g.add_edge(create_edge(n5.id, "item", n6.id, "prompt"))
    g.add_edge(create_edge(n6.id, "prompt", n7.id, "item"))

    # Execute the graph
    session = GraphExecutionState(graph=g)
    run_session_with_mock_context(session)

    first_output = get_single_output_from_session(session, n4.id)
    final_output = get_single_output_from_session(session, n7.id)
    assert isinstance(first_output, CollectInvocationOutput)
    assert isinstance(final_output, CollectInvocationOutput)
    assert first_output.collection == []
    assert final_output.collection == []
    assert session.is_complete()
