from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.progress_previews.progress_previews_default import MemoryProgressPreviews
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
from invokeai.app.services.shared.execution_effects import ExecutionEffectsRecorder
from invokeai.app.services.shared.graph import CollectInvocation, Graph, GraphExecutionState, IterateInvocation


@pytest.mark.parametrize("invocation_type", [IterateInvocation, CollectInvocation])
def test_after_run_node_callback_receives_control_node_inputs(monkeypatch: pytest.MonkeyPatch, invocation_type):
    invocation = invocation_type(id="control", collection=[1, 2, 3])
    callback_collections: list[list[int]] = []
    callback_invocations = []
    session = GraphExecutionState(graph=Graph())
    session.execution_graph.add_node(invocation)
    session._register_prepared_exec_node(invocation.id, "source")
    session._prepared_registry().set_iteration_path(invocation.id, (2,))
    session.indegree[invocation.id] = 0
    context_data = []

    services = SimpleNamespace(
        configuration=SimpleNamespace(node_cache_size=0),
        progress_previews=MemoryProgressPreviews(),
        events=Mock(),
        logger=Mock(),
        performance_statistics=Mock(),
        invocation_cache=MemoryInvocationCache(max_cache_size=0),
    )
    services.performance_statistics.collect_stats.return_value = nullcontext()
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        lambda data, services, is_canceled: context_data.append(data)
        or SimpleNamespace(
            execution_effects=ExecutionEffectsRecorder(
                source_node_id=data.invocation.id,
                frame_path=data.execution_frame,
            )
        ),
    )

    def on_after_run_node(invocation, queue_item, output):
        callback_invocations.append(invocation)
        callback_collections.append(list(invocation.collection))

    runner = DefaultSessionRunner(on_after_run_node_callbacks=[on_after_run_node])
    runner.start(services=services, cancel_event=Event())
    queue_item = Mock(session=session, session_id="session")

    runner.run_node(invocation, queue_item)

    assert callback_collections == [[1, 2, 3]]
    assert callback_invocations == [invocation]
    assert callback_invocations[0] is invocation
    assert invocation.collection == []
    assert context_data[0].execution_frame == (2,)
