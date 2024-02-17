import pytest
from pydantic import TypeAdapter, ValidationError

from invokeai.app.services.session_queue.session_queue_common import (
    Batch,
    BatchDataCollection,
    BatchDatum,
    NodeFieldValue,
    calc_session_count,
    create_session_nfv_tuples,
    prepare_values_to_insert,
)
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.aa_nodes.test_nodes import PromptTestInvocation


@pytest.fixture
def batch_data_collection() -> BatchDataCollection:
    return [
        [
            # zipped
            BatchDatum(node_path="1", field_name="prompt", items=["Banana sushi", "Grape sushi"]),
            BatchDatum(node_path="2", field_name="prompt", items=["Strawberry sushi", "Blueberry sushi"]),
        ],
        [
            BatchDatum(node_path="3", field_name="prompt", items=["Orange sushi", "Apple sushi"]),
        ],
    ]


@pytest.fixture
def batch_graph() -> Graph:
    g = Graph()
    g.add_node(PromptTestInvocation(id="1", prompt="Chevy"))
    g.add_node(PromptTestInvocation(id="2", prompt="Toyota"))
    g.add_node(PromptTestInvocation(id="3", prompt="Subaru"))
    g.add_node(PromptTestInvocation(id="4", prompt="Nissan"))
    return g


# def test_populate_graph_with_subgraph():
#     g1 = Graph()
#     g1.add_node(PromptTestInvocation(id="1", prompt="Banana sushi"))
#     g1.add_node(PromptTestInvocation(id="2", prompt="Banana sushi"))
#     n1 = PromptTestInvocation(id="1", prompt="Banana snake")
#     subgraph = Graph()
#     subgraph.add_node(n1)
#     g1.add_node(GraphInvocation(id="3", graph=subgraph))

#     nfvs = [
#         NodeFieldValue(node_path="1", field_name="prompt", value="Strawberry sushi"),
#         NodeFieldValue(node_path="2", field_name="prompt", value="Strawberry sunday"),
#         NodeFieldValue(node_path="3.1", field_name="prompt", value="Strawberry snake"),
#     ]

#     g2 = populate_graph(g1, nfvs)

#     # do not mutate g1
#     assert g1 is not g2
#     assert g2.get_node("1").prompt == "Strawberry sushi"
#     assert g2.get_node("2").prompt == "Strawberry sunday"
#     assert g2.get_node("3.1").prompt == "Strawberry snake"


def test_create_sessions_from_batch_with_runs(batch_data_collection, batch_graph):
    b = Batch(graph=batch_graph, data=batch_data_collection, runs=2)
    t = list(create_session_nfv_tuples(batch=b, maximum=1000))
    # 2 list[BatchDatum] * length 2 * 2 runs = 8
    assert len(t) == 8

    assert t[0][0].graph.get_node("1").prompt == "Banana sushi"
    assert t[0][0].graph.get_node("2").prompt == "Strawberry sushi"
    assert t[0][0].graph.get_node("3").prompt == "Orange sushi"
    assert t[0][0].graph.get_node("4").prompt == "Nissan"

    assert t[1][0].graph.get_node("1").prompt == "Banana sushi"
    assert t[1][0].graph.get_node("2").prompt == "Strawberry sushi"
    assert t[1][0].graph.get_node("3").prompt == "Apple sushi"
    assert t[1][0].graph.get_node("4").prompt == "Nissan"

    assert t[2][0].graph.get_node("1").prompt == "Grape sushi"
    assert t[2][0].graph.get_node("2").prompt == "Blueberry sushi"
    assert t[2][0].graph.get_node("3").prompt == "Orange sushi"
    assert t[2][0].graph.get_node("4").prompt == "Nissan"

    assert t[3][0].graph.get_node("1").prompt == "Grape sushi"
    assert t[3][0].graph.get_node("2").prompt == "Blueberry sushi"
    assert t[3][0].graph.get_node("3").prompt == "Apple sushi"
    assert t[3][0].graph.get_node("4").prompt == "Nissan"

    # repeat for second run
    assert t[4][0].graph.get_node("1").prompt == "Banana sushi"
    assert t[4][0].graph.get_node("2").prompt == "Strawberry sushi"
    assert t[4][0].graph.get_node("3").prompt == "Orange sushi"
    assert t[4][0].graph.get_node("4").prompt == "Nissan"

    assert t[5][0].graph.get_node("1").prompt == "Banana sushi"
    assert t[5][0].graph.get_node("2").prompt == "Strawberry sushi"
    assert t[5][0].graph.get_node("3").prompt == "Apple sushi"
    assert t[5][0].graph.get_node("4").prompt == "Nissan"

    assert t[6][0].graph.get_node("1").prompt == "Grape sushi"
    assert t[6][0].graph.get_node("2").prompt == "Blueberry sushi"
    assert t[6][0].graph.get_node("3").prompt == "Orange sushi"
    assert t[6][0].graph.get_node("4").prompt == "Nissan"

    assert t[7][0].graph.get_node("1").prompt == "Grape sushi"
    assert t[7][0].graph.get_node("2").prompt == "Blueberry sushi"
    assert t[7][0].graph.get_node("3").prompt == "Apple sushi"
    assert t[7][0].graph.get_node("4").prompt == "Nissan"


def test_create_sessions_from_batch_without_runs(batch_data_collection, batch_graph):
    b = Batch(graph=batch_graph, data=batch_data_collection)
    t = list(create_session_nfv_tuples(batch=b, maximum=1000))
    # 2 list[BatchDatum] * length 2 * 1 runs = 8
    assert len(t) == 4


def test_create_sessions_from_batch_without_batch(batch_graph):
    b = Batch(graph=batch_graph, runs=2)
    t = list(create_session_nfv_tuples(batch=b, maximum=1000))
    # 2 runs
    assert len(t) == 2


def test_create_sessions_from_batch_without_batch_or_runs(batch_graph):
    b = Batch(graph=batch_graph)
    t = list(create_session_nfv_tuples(batch=b, maximum=1000))
    # 1 run
    assert len(t) == 1


def test_create_sessions_from_batch_with_runs_and_max(batch_data_collection, batch_graph):
    b = Batch(graph=batch_graph, data=batch_data_collection, runs=2)
    t = list(create_session_nfv_tuples(batch=b, maximum=5))
    # 2 list[BatchDatum] * length 2 * 2 runs = 8, but max is 5
    assert len(t) == 5


def test_calc_session_count(batch_data_collection, batch_graph):
    b = Batch(graph=batch_graph, data=batch_data_collection, runs=2)
    # 2 list[BatchDatum] * length 2 * 2 runs = 8
    assert calc_session_count(batch=b) == 8


def test_prepare_values_to_insert(batch_data_collection, batch_graph):
    b = Batch(graph=batch_graph, data=batch_data_collection, runs=2)
    values = prepare_values_to_insert(queue_id="default", batch=b, priority=0, max_new_queue_items=1000)
    assert len(values) == 8

    GraphExecutionStateValidator = TypeAdapter(GraphExecutionState)
    # graph should be serialized
    ges = GraphExecutionStateValidator.validate_json(values[0].session)

    # graph values should be populated
    assert ges.graph.get_node("1").prompt == "Banana sushi"
    assert ges.graph.get_node("2").prompt == "Strawberry sushi"
    assert ges.graph.get_node("3").prompt == "Orange sushi"
    assert ges.graph.get_node("4").prompt == "Nissan"

    # session ids should match deserialized graph
    assert [v.session_id for v in values] == [GraphExecutionStateValidator.validate_json(v.session).id for v in values]

    # should unique session ids
    sids = [v.session_id for v in values]
    assert len(sids) == len(set(sids))

    NodeFieldValueValidator = TypeAdapter(list[NodeFieldValue])
    # should have 3 node field values
    assert isinstance(values[0].field_values, str)
    assert len(NodeFieldValueValidator.validate_json(values[0].field_values)) == 3

    # should have batch id and priority
    assert all(v.batch_id == b.batch_id for v in values)
    assert all(v.priority == 0 for v in values)


def test_prepare_values_to_insert_with_priority(batch_data_collection, batch_graph):
    b = Batch(graph=batch_graph, data=batch_data_collection, runs=2)
    values = prepare_values_to_insert(queue_id="default", batch=b, priority=1, max_new_queue_items=1000)
    assert all(v.priority == 1 for v in values)


def test_prepare_values_to_insert_with_max(batch_data_collection, batch_graph):
    b = Batch(graph=batch_graph, data=batch_data_collection, runs=2)
    values = prepare_values_to_insert(queue_id="default", batch=b, priority=1, max_new_queue_items=5)
    assert len(values) == 5


def test_cannot_create_bad_batch_items_length(batch_graph):
    with pytest.raises(ValidationError, match="Zipped batch items must all have the same length"):
        Batch(
            graph=batch_graph,
            data=[
                [
                    BatchDatum(node_path="1", field_name="prompt", items=["Banana sushi"]),  # 1 item
                    BatchDatum(node_path="2", field_name="prompt", items=["Toyota", "Nissan"]),  # 2 items
                ],
            ],
        )


def test_cannot_create_bad_batch_items_type(batch_graph):
    with pytest.raises(ValidationError, match="All items in a batch must have the same type"):
        Batch(
            graph=batch_graph,
            data=[
                [
                    BatchDatum(node_path="1", field_name="prompt", items=["Banana sushi", 123]),
                ]
            ],
        )


def test_cannot_create_bad_batch_unique_ids(batch_graph):
    with pytest.raises(ValidationError, match="Each batch data must have unique node_id and field_name"):
        Batch(
            graph=batch_graph,
            data=[
                [
                    BatchDatum(node_path="1", field_name="prompt", items=["Banana sushi"]),
                ],
                [
                    BatchDatum(node_path="1", field_name="prompt", items=["Banana sushi"]),
                ],
            ],
        )


def test_cannot_create_bad_batch_nodes_exist(
    batch_graph,
):
    with pytest.raises(ValidationError, match=r"Node .* not found in graph"):
        Batch(
            graph=batch_graph,
            data=[
                [
                    BatchDatum(node_path="batman", field_name="prompt", items=["Banana sushi"]),
                ],
            ],
        )


def test_cannot_create_bad_batch_fields_exist(
    batch_graph,
):
    with pytest.raises(ValidationError, match=r"Field .* not found in node"):
        Batch(
            graph=batch_graph,
            data=[
                [
                    BatchDatum(node_path="1", field_name="batman", items=["Banana sushi"]),
                ],
            ],
        )
