"""The session runner releases what an invocation's context held, however the invocation ends."""

from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from invokeai.app.invocations.primitives import IntegerInvocation
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.progress_previews.progress_previews_default import MemoryProgressPreviews
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
from invokeai.app.services.shared.execution_effects import ExecutionEffectsRecorder
from invokeai.app.services.shared.graph import Graph, GraphExecutionState


@pytest.mark.parametrize("outcome", ["completed", "failed", "canceled"])
def test_resources_held_during_an_invocation_are_released_when_it_ends(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    graph = Graph()
    graph.add_node(IntegerInvocation(id="source", value=1))
    session = GraphExecutionState(graph=graph)
    invocation = IntegerInvocation(id="node", value=1)
    session.execution_graph.add_node(invocation)
    session._register_prepared_exec_node(invocation.id, "source")
    session.indegree[invocation.id] = 0

    services = SimpleNamespace(
        configuration=SimpleNamespace(node_cache_size=0, multiuser=False),
        progress_previews=MemoryProgressPreviews(),
        events=Mock(),
        logger=Mock(),
        performance_statistics=Mock(),
        session_queue=Mock(),
        invocation_cache=MemoryInvocationCache(max_cache_size=0),
    )
    services.performance_statistics.collect_stats.return_value = nullcontext()
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        lambda data, services, is_canceled: SimpleNamespace(
            _data=data,
            execution_effects=ExecutionEffectsRecorder(source_node_id=data.invocation.id, frame_path=()),
        ),
    )

    timeline: list[str] = []
    run = IntegerInvocation.invoke_internal_with_effects

    def run_holding_a_resource(self, context, services):
        # What a denoise does on its first preview: hold something until the invocation ends.
        context._data.invocation_resources.callback(timeline.append, "released")
        timeline.append("invoked")
        if outcome == "failed":
            raise RuntimeError("the node failed")
        if outcome == "canceled":
            raise CanceledException
        return run(self, context, services)

    monkeypatch.setattr(IntegerInvocation, "invoke_internal_with_effects", run_holding_a_resource)
    runner = DefaultSessionRunner()
    runner.start(services=services, cancel_event=Event())

    runner.run_node(invocation, Mock(session=session, session_id="session", item_id=1))

    assert timeline == ["invoked", "released"]
    assert services.session_queue.fail_queue_item.called is (outcome == "failed")
