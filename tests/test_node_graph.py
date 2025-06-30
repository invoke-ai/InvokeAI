import pytest
from pydantic import TypeAdapter
from pydantic.json_schema import models_json_schema

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvalidVersionError,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.primitives import (
    ColorInvocation,
    FloatCollectionInvocation,
    FloatInvocation,
    IntegerInvocation,
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
    get_single_output_from_session,
    run_session_with_mock_context,
)


# Helpers
def create_edge(from_id: str, from_field: str, to_id: str, to_field: str) -> Edge:
    return Edge(
        source=EdgeConnection(node_id=from_id, field=from_field),
        destination=EdgeConnection(node_id=to_id, field=to_field),
    )


# Tests
def test_connections_are_compatible():
    from_node = TextToImageTestInvocation(id="1", prompt="Banana sushi")
    from_field = "image"
    to_node = ESRGANInvocation(id="2")
    to_field = "image"

    result = are_connections_compatible(from_node, from_field, to_node, to_field)

    assert result is True


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
    assert output.collection == ["Banana", "Sushi"]  # Both inputs should be collected


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

    # With empty collection, iterators don't create execution nodes, so collectors don't execute
    # Verify that the final collector was never prepared (which is correct behavior)
    assert n7.id not in session.source_prepared_mapping

    # Verify only the source collection node executed
    assert n1.id in session.source_prepared_mapping
    assert len(session.source_prepared_mapping[n1.id]) == 1
