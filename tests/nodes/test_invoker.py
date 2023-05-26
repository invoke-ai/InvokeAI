from .test_nodes import ErrorInvocation, ImageTestInvocation, ListPassThroughInvocation, PromptTestInvocation, PromptCollectionTestInvocation, TestEventService, create_edge, wait_until
from invokeai.app.services.processor import DefaultInvocationProcessor
from invokeai.app.services.sqlite import SqliteItemStorage, sqlite_memory
from invokeai.app.services.invocation_queue import MemoryInvocationQueue
from invokeai.app.services.invoker import Invoker
from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.graph import Graph, GraphInvocation, InvalidEdgeError, LibraryGraph, NodeAlreadyInGraphError, NodeNotFoundError, are_connections_compatible, EdgeConnection, CollectInvocation, IterateInvocation, GraphExecutionState
import pytest


@pytest.fixture
def simple_graph():
    g = Graph()
    g.add_node(PromptTestInvocation(id = "1", prompt = "Banana sushi"))
    g.add_node(ImageTestInvocation(id = "2"))
    g.add_edge(create_edge("1", "prompt", "2", "prompt"))
    return g

@pytest.fixture
def mock_services() -> InvocationServices:
    # NOTE: none of these are actually called by the test invocations
    return InvocationServices(
        model_manager = None, # type: ignore
        events = TestEventService(),
        logger = None, # type: ignore
        images = None, # type: ignore
        latents = None, # type: ignore
        queue = MemoryInvocationQueue(),
        graph_library=SqliteItemStorage[LibraryGraph](
            filename=sqlite_memory, table_name="graphs"
        ),
        graph_execution_manager = SqliteItemStorage[GraphExecutionState](filename = sqlite_memory, table_name = 'graph_executions'),
        processor = DefaultInvocationProcessor(),
        restoration = None, # type: ignore
        configuration = None, # type: ignore
    )

@pytest.fixture()
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    return Invoker(
        services = mock_services
    )

def test_can_create_graph_state(mock_invoker: Invoker):
    g = mock_invoker.create_execution_state()
    mock_invoker.stop()

    assert g is not None
    assert isinstance(g, GraphExecutionState)

def test_can_create_graph_state_from_graph(mock_invoker: Invoker, simple_graph):
    g = mock_invoker.create_execution_state(graph = simple_graph)
    mock_invoker.stop()

    assert g is not None
    assert isinstance(g, GraphExecutionState)
    assert g.graph == simple_graph

def test_can_invoke(mock_invoker: Invoker, simple_graph):
    g = mock_invoker.create_execution_state(graph = simple_graph)
    invocation_id = mock_invoker.invoke(g)
    assert invocation_id is not None

    def has_executed_any(g: GraphExecutionState):
        g = mock_invoker.services.graph_execution_manager.get(g.id)
        return len(g.executed) > 0

    wait_until(lambda: has_executed_any(g), timeout = 5, interval = 1)
    mock_invoker.stop()

    g = mock_invoker.services.graph_execution_manager.get(g.id)
    assert len(g.executed) > 0

def test_can_invoke_all(mock_invoker: Invoker, simple_graph):
    g = mock_invoker.create_execution_state(graph = simple_graph)
    invocation_id = mock_invoker.invoke(g, invoke_all = True)
    assert invocation_id is not None

    def has_executed_all(g: GraphExecutionState):
        g = mock_invoker.services.graph_execution_manager.get(g.id)
        return g.is_complete()

    wait_until(lambda: has_executed_all(g), timeout = 5, interval = 1)
    mock_invoker.stop()

    g = mock_invoker.services.graph_execution_manager.get(g.id)
    assert g.is_complete()

def test_handles_errors(mock_invoker: Invoker):
    g = mock_invoker.create_execution_state()
    g.graph.add_node(ErrorInvocation(id = "1"))

    mock_invoker.invoke(g, invoke_all=True)

    def has_executed_all(g: GraphExecutionState):
        g = mock_invoker.services.graph_execution_manager.get(g.id)
        return g.is_complete()

    wait_until(lambda: has_executed_all(g), timeout = 5, interval = 1)
    mock_invoker.stop()

    g = mock_invoker.services.graph_execution_manager.get(g.id)
    assert g.has_error()
    assert g.is_complete()

    assert all((i in g.errors for i in g.source_prepared_mapping['1']))
