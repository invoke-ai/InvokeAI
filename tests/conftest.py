import pytest
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invocation_queue import MemoryInvocationQueue
from invokeai.app.services.sqlite import SqliteItemStorage, sqlite_memory
from invokeai.app.services.graph import LibraryGraph, GraphExecutionState
from invokeai.app.services.processor import DefaultInvocationProcessor

# Ignore these files as they need to be rewritten following the model manager refactor
collect_ignore = ["nodes/test_graph_execution_state.py", "nodes/test_node_graph.py", "test_textual_inversion.py"]

@pytest.fixture(scope="session", autouse=True)
def mock_services():
    # NOTE: none of these are actually called by the test invocations
    return InvocationServices(
        model_manager = None, # type: ignore
        events = None, # type: ignore
        logger = None, # type: ignore
        images = None, # type: ignore
        latents = None, # type: ignore
        board_images=None, # type: ignore
        boards=None, # type: ignore
        queue = MemoryInvocationQueue(),
        graph_library=SqliteItemStorage[LibraryGraph](
            filename=sqlite_memory, table_name="graphs"
        ),
        graph_execution_manager = SqliteItemStorage[GraphExecutionState](filename = sqlite_memory, table_name = 'graph_executions'),
        processor = DefaultInvocationProcessor(),
        restoration = None, # type: ignore
        configuration = None, # type: ignore
    )
