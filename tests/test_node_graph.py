import pytest
from pydantic import TypeAdapter

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvalidVersionError,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.primitives import (
    FloatCollectionInvocation,
    FloatInvocation,
    IntegerInvocation,
    StringInvocation,
)
from invokeai.app.invocations.upscale import ESRGANInvocation
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    Edge,
    EdgeConnection,
    Graph,
    InvalidEdgeError,
    IterateInvocation,
    NodeAlreadyInGraphError,
    NodeNotFoundError,
    are_connections_compatible,
)

from .test_nodes import (
    AnyTypeTestInvocation,
    ImageToImageTestInvocation,
    ListPassThroughInvocation,
    PolymorphicStringTestInvocation,
    PromptCollectionTestInvocation,
    PromptTestInvocation,
    TextToImageTestInvocation,
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
        def invoke(self):
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
        def invoke(self):
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
    """We need to update the validation for Collect -> Iterate to traverse to the Iterate
    node's output and compare that against the item type of the Collect node's collection. Until
    then, Collect nodes may not output into Iterate nodes."""
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
    # Once we fix the validation logic as described, this should should not raise an error
    with pytest.raises(InvalidEdgeError, match="Cannot connect collector to iterator"):
        g.add_edge(e3)


def test_graph_can_generate_schema():
    # Not throwing on this line is sufficient
    # NOTE: if this test fails, it's PROBABLY because a new invocation type is breaking schema generation
    _ = Graph.model_json_schema()
