import logging
from unittest.mock import Mock

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.item_storage.item_storage_memory import ItemStorageMemory

# This import must happen before other invoke imports or test in other files(!!) break
from .test_nodes import (  # isort: split
    ErrorInvocation,
    PromptTestInvocation,
    TestEventService,
    TextToImageTestInvocation,
    create_edge,
    wait_until,
)

from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.invocation_processor.invocation_processor_default import DefaultInvocationProcessor
from invokeai.app.services.invocation_queue.invocation_queue_memory import MemoryInvocationQueue
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invocation_stats.invocation_stats_default import InvocationStatsService
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import DEFAULT_QUEUE_ID
from invokeai.app.services.shared.graph import Graph, GraphExecutionState, GraphInvocation


@pytest.fixture
def simple_graph():
    g = Graph()
    g.add_node(PromptTestInvocation(id="1", prompt="Banana sushi"))
    g.add_node(TextToImageTestInvocation(id="2"))
    g.add_edge(create_edge("1", "prompt", "2", "prompt"))
    return g


@pytest.fixture
def graph_with_subgraph():
    sub_g = Graph()
    sub_g.add_node(PromptTestInvocation(id="1", prompt="Banana sushi"))
    sub_g.add_node(TextToImageTestInvocation(id="2"))
    sub_g.add_edge(create_edge("1", "prompt", "2", "prompt"))
    g = Graph()
    g.add_node(GraphInvocation(id="1", graph=sub_g))
    return g


# This must be defined here to avoid issues with the dynamic creation of the union of all invocation types
# Defining it in a separate module will cause the union to be incomplete, and pydantic will not validate
# the test invocations.
@pytest.fixture
def mock_services() -> InvocationServices:
    configuration = InvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    return InvocationServices(
        board_image_records=None,  # type: ignore
        board_images=None,  # type: ignore
        board_records=None,  # type: ignore
        boards=None,  # type: ignore
        configuration=configuration,
        events=TestEventService(),
        graph_execution_manager=ItemStorageMemory[GraphExecutionState](),
        image_files=None,  # type: ignore
        image_records=None,  # type: ignore
        images=None,  # type: ignore
        invocation_cache=MemoryInvocationCache(max_cache_size=0),
        latents=None,  # type: ignore
        logger=logging,  # type: ignore
        model_manager=Mock(),  # type: ignore
        download_queue=None,  # type: ignore
        names=None,  # type: ignore
        performance_statistics=InvocationStatsService(),
        processor=DefaultInvocationProcessor(),
        queue=MemoryInvocationQueue(),
        session_processor=None,  # type: ignore
        session_queue=None,  # type: ignore
        urls=None,  # type: ignore
        workflow_records=None,  # type: ignore
    )


@pytest.fixture()
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    return Invoker(services=mock_services)


def test_can_create_graph_state(mock_invoker: Invoker):
    g = mock_invoker.create_execution_state()
    mock_invoker.stop()

    assert g is not None
    assert isinstance(g, GraphExecutionState)


def test_can_create_graph_state_from_graph(mock_invoker: Invoker, simple_graph):
    g = mock_invoker.create_execution_state(graph=simple_graph)
    mock_invoker.stop()

    assert g is not None
    assert isinstance(g, GraphExecutionState)
    assert g.graph == simple_graph


# @pytest.mark.xfail(reason = "Requires fixing following the model manager refactor")
def test_can_invoke(mock_invoker: Invoker, simple_graph):
    g = mock_invoker.create_execution_state(graph=simple_graph)
    invocation_id = mock_invoker.invoke(
        session_queue_batch_id="1",
        session_queue_item_id=1,
        session_queue_id=DEFAULT_QUEUE_ID,
        graph_execution_state=g,
    )
    assert invocation_id is not None

    def has_executed_any(g: GraphExecutionState):
        g = mock_invoker.services.graph_execution_manager.get(g.id)
        return len(g.executed) > 0

    wait_until(lambda: has_executed_any(g), timeout=5, interval=1)
    mock_invoker.stop()

    g = mock_invoker.services.graph_execution_manager.get(g.id)
    assert len(g.executed) > 0


# @pytest.mark.xfail(reason = "Requires fixing following the model manager refactor")
def test_can_invoke_all(mock_invoker: Invoker, simple_graph):
    g = mock_invoker.create_execution_state(graph=simple_graph)
    invocation_id = mock_invoker.invoke(
        session_queue_batch_id="1",
        session_queue_item_id=1,
        session_queue_id=DEFAULT_QUEUE_ID,
        graph_execution_state=g,
        invoke_all=True,
    )
    assert invocation_id is not None

    def has_executed_all(g: GraphExecutionState):
        g = mock_invoker.services.graph_execution_manager.get(g.id)
        return g.is_complete()

    wait_until(lambda: has_executed_all(g), timeout=5, interval=1)
    mock_invoker.stop()

    g = mock_invoker.services.graph_execution_manager.get(g.id)
    assert g.is_complete()


# @pytest.mark.xfail(reason = "Requires fixing following the model manager refactor")
def test_handles_errors(mock_invoker: Invoker):
    g = mock_invoker.create_execution_state()
    g.graph.add_node(ErrorInvocation(id="1"))

    mock_invoker.invoke(
        session_queue_batch_id="1",
        session_queue_item_id=1,
        session_queue_id=DEFAULT_QUEUE_ID,
        graph_execution_state=g,
        invoke_all=True,
    )

    def has_executed_all(g: GraphExecutionState):
        g = mock_invoker.services.graph_execution_manager.get(g.id)
        return g.is_complete()

    wait_until(lambda: has_executed_all(g), timeout=5, interval=1)
    mock_invoker.stop()

    g = mock_invoker.services.graph_execution_manager.get(g.id)
    assert g.has_error()
    assert g.is_complete()

    assert all((i in g.errors for i in g.source_prepared_mapping["1"]))
