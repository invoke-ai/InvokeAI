"""Attention inside a node is attributed to the node's type, so the pinned-kernel report (`attention_backend`) can say
which node a kernel was not possible for, whatever the model family."""

from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import invokeai.backend.util.attention as attention
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.primitives import IntegerOutput
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.progress_previews.progress_previews_default import MemoryProgressPreviews
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
from invokeai.app.services.shared.execution_effects import ExecutionEffectsRecorder
from invokeai.app.services.shared.graph import Graph, GraphExecutionState


def _run_one_node(monkeypatch: pytest.MonkeyPatch, invoke) -> Mock:
    """Run an `add` node whose `invoke` is replaced, through the runner's own `run_node`. Returns the queue double."""
    monkeypatch.setattr(AddInvocation, "invoke", invoke)
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2, use_cache=False))
    session = GraphExecutionState(graph=graph)
    invocation = session.next()

    services = SimpleNamespace(
        configuration=SimpleNamespace(node_cache_size=0, offload_text_encoders_to_idle_gpus=False),
        progress_previews=MemoryProgressPreviews(),
        events=Mock(),
        logger=Mock(),
        performance_statistics=Mock(),
        invocation_cache=MemoryInvocationCache(max_cache_size=0),
        session_queue=Mock(),
    )
    services.performance_statistics.collect_stats.return_value = nullcontext()
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        lambda data, services, is_canceled: SimpleNamespace(
            execution_effects=ExecutionEffectsRecorder(
                source_node_id=data.invocation.id, frame_path=data.execution_frame
            )
        ),
    )
    runner = DefaultSessionRunner()
    runner.start(services=services, cancel_event=Event())
    runner.run_node(invocation, Mock(session=session, session_id="session", item_id=1))
    return services.session_queue


def test_a_nodes_attention_is_attributed_to_its_type_and_only_while_it_runs(monkeypatch):
    seen: list[str | None] = []
    original = AddInvocation.invoke

    # Annotated: the session checks a node's output against the return type `invoke` declares.
    def invoke(self, context) -> IntegerOutput:
        seen.append(attention._attention_caller.get())
        return original(self, context)

    queue = _run_one_node(monkeypatch, invoke)

    assert seen == ["add"]
    assert attention._attention_caller.get() is None
    queue.fail_queue_item.assert_not_called()


def test_a_failing_node_leaves_no_caller_behind(monkeypatch):
    """The next node, or attention outside any node on this worker thread, must not be blamed on it."""
    seen: list[str | None] = []

    def invoke(self, context) -> IntegerOutput:
        seen.append(attention._attention_caller.get())
        raise RuntimeError("boom")

    queue = _run_one_node(monkeypatch, invoke)

    assert seen == ["add"]
    queue.fail_queue_item.assert_called_once()
    assert queue.fail_queue_item.call_args.args[1:3] == ("RuntimeError", "boom"), "the node's own error"
    assert attention._attention_caller.get() is None
