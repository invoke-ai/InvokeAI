from .test_invoker import create_edge
from .test_nodes import (
    TestEventService,
    TextToImageTestInvocation,
    PromptTestInvocation,
    PromptCollectionTestInvocation,
)
from invokeai.app.services.invocation_queue import MemoryInvocationQueue
from invokeai.app.services.processor import DefaultInvocationProcessor
from invokeai.app.services.sqlite import SqliteItemStorage, sqlite_memory
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
)
from invokeai.app.invocations.collections import RangeInvocation
from invokeai.app.invocations.math import AddInvocation, MultiplyInvocation
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invocation_stats import InvocationStatsService
from invokeai.app.services.graph import (
    Graph,
    CollectInvocation,
    IterateInvocation,
    GraphExecutionState,
    LibraryGraph,
)
import pytest


@pytest.fixture
def simple_graph():
    g = Graph()
    g.add_node(PromptTestInvocation(id="1", prompt="Banana sushi"))
    g.add_node(TextToImageTestInvocation(id="2"))
    g.add_edge(create_edge("1", "prompt", "2", "prompt"))
    return g


# This must be defined here to avoid issues with the dynamic creation of the union of all invocation types
# Defining it in a separate module will cause the union to be incomplete, and pydantic will not validate
# the test invocations.
@pytest.fixture
def mock_services() -> InvocationServices:
    # NOTE: none of these are actually called by the test invocations
    graph_execution_manager = SqliteItemStorage[GraphExecutionState](
        filename=sqlite_memory, table_name="graph_executions"
    )
    return InvocationServices(
        model_manager=None,  # type: ignore
        events=TestEventService(),
        logger=None,  # type: ignore
        images=None,  # type: ignore
        latents=None,  # type: ignore
        boards=None,  # type: ignore
        board_images=None,  # type: ignore
        queue=MemoryInvocationQueue(),
        graph_library=SqliteItemStorage[LibraryGraph](filename=sqlite_memory, table_name="graphs"),
        graph_execution_manager=graph_execution_manager,
        performance_statistics=InvocationStatsService(graph_execution_manager),
        processor=DefaultInvocationProcessor(),
        configuration=None,  # type: ignore
    )


def invoke_next(g: GraphExecutionState, services: InvocationServices) -> tuple[BaseInvocation, BaseInvocationOutput]:
    n = g.next()
    if n is None:
        return (None, None)

    print(f"invoking {n.id}: {type(n)}")
    o = n.invoke(InvocationContext(services, "1"))
    g.complete(n.id, o)

    return (n, o)


def test_graph_state_executes_in_order(simple_graph, mock_services):
    g = GraphExecutionState(graph=simple_graph)

    n1 = invoke_next(g, mock_services)
    n2 = invoke_next(g, mock_services)
    n3 = g.next()

    assert g.prepared_source_mapping[n1[0].id] == "1"
    assert g.prepared_source_mapping[n2[0].id] == "2"
    assert n3 is None
    assert g.results[n1[0].id].prompt == n1[0].prompt
    assert n2[0].prompt == n1[0].prompt


def test_graph_is_complete(simple_graph, mock_services):
    g = GraphExecutionState(graph=simple_graph)
    _ = invoke_next(g, mock_services)
    _ = invoke_next(g, mock_services)
    _ = g.next()

    assert g.is_complete()


def test_graph_is_not_complete(simple_graph, mock_services):
    g = GraphExecutionState(graph=simple_graph)
    _ = invoke_next(g, mock_services)
    _ = g.next()

    assert not g.is_complete()


# TODO: test completion with iterators/subgraphs


def test_graph_state_expands_iterator(mock_services):
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
        invoke_next(g, mock_services)

    prepared_add_nodes = g.source_prepared_mapping["3"]
    results = set([g.results[n].value for n in prepared_add_nodes])
    expected = set([1, 11, 21])
    assert results == expected


def test_graph_state_collects(mock_services):
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
    _ = invoke_next(g, mock_services)
    _ = invoke_next(g, mock_services)
    _ = invoke_next(g, mock_services)
    _ = invoke_next(g, mock_services)
    _ = invoke_next(g, mock_services)
    n6 = invoke_next(g, mock_services)

    assert isinstance(n6[0], CollectInvocation)

    assert sorted(g.results[n6[0].id].collection) == sorted(test_prompts)


def test_graph_state_prepares_eagerly(mock_services):
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


def test_graph_executes_depth_first(mock_services):
    """Tests that the graph executes depth-first, executing a branch as far as possible before moving to the next branch"""
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
    _ = invoke_next(g, mock_services)
    _ = invoke_next(g, mock_services)
    _ = invoke_next(g, mock_services)
    _ = invoke_next(g, mock_services)

    # Because ordering is not guaranteed, we cannot compare results directly.
    # Instead, we must count the number of results.
    def get_completed_count(g, id):
        ids = [i for i in g.source_prepared_mapping[id]]
        completed_ids = [i for i in g.executed if i in ids]
        return len(completed_ids)

    # Check at each step that the number of executed nodes matches the expectation for depth-first execution
    assert get_completed_count(g, "prompt_iterated") == 1
    assert get_completed_count(g, "prompt_successor") == 0

    _ = invoke_next(g, mock_services)

    assert get_completed_count(g, "prompt_iterated") == 1
    assert get_completed_count(g, "prompt_successor") == 1

    _ = invoke_next(g, mock_services)

    assert get_completed_count(g, "prompt_iterated") == 2
    assert get_completed_count(g, "prompt_successor") == 1

    _ = invoke_next(g, mock_services)

    assert get_completed_count(g, "prompt_iterated") == 2
    assert get_completed_count(g, "prompt_successor") == 2
