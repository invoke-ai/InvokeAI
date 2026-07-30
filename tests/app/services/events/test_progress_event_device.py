"""Tests for the device reported by InvocationProgressEvent.

The UI labels progress with the GPU executing the session. The queue item's persisted `device` is
the authority: the worker thread's session device is temporarily re-pinned to a borrowed idle GPU
during offloaded encoder nodes, and reporting that would make the device badge jump to the borrowed
GPU and back within a single queue item.
"""

from collections.abc import Iterator
from datetime import datetime

import pytest

from invokeai.app.services.events.events_common import InvocationProgressEvent
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.backend.util.devices import TorchDevice
from tests.test_nodes import PromptTestInvocation


@pytest.fixture(autouse=True)
def clear_session_device() -> Iterator[None]:
    try:
        yield
    finally:
        TorchDevice.clear_session_device()


def _build_queue_item(device: str | None) -> tuple[SessionQueueItem, PromptTestInvocation]:
    invocation = PromptTestInvocation(id="prompt", prompt="test")
    graph = Graph()
    graph.add_node(invocation)
    session = GraphExecutionState(graph=graph)
    session.prepared_source_mapping[invocation.id] = invocation.id
    now = datetime.now()
    queue_item = SessionQueueItem(
        item_id=1,
        status="in_progress",
        priority=0,
        batch_id="batch-1",
        session_id=session.id,
        created_at=now,
        updated_at=now,
        started_at=None,
        completed_at=None,
        queue_id="default",
        user_id="system",
        session=session,
        device=device,
    )
    return queue_item, invocation


def test_progress_event_reports_queue_item_device_not_thread_local():
    """During an idle-GPU-offloaded encoder the thread is pinned to the borrowed GPU; the event
    must still report the queue item's execution device."""
    queue_item, invocation = _build_queue_item(device="cuda:0")
    TorchDevice.set_session_device("cuda:1")

    event = InvocationProgressEvent.build(queue_item=queue_item, invocation=invocation, message="encoding")

    assert event.device == "cuda:0"


def test_progress_event_falls_back_to_thread_local_in_legacy_mode():
    """Legacy single-device mode tags queue items with device=None; the worker thread's pinned
    device is the only signal available."""
    queue_item, invocation = _build_queue_item(device=None)
    TorchDevice.set_session_device("cuda:1")

    event = InvocationProgressEvent.build(queue_item=queue_item, invocation=invocation, message="encoding")

    assert event.device == "cuda:1"


def test_progress_event_omits_non_cuda_device():
    queue_item, invocation = _build_queue_item(device="cpu")
    event = InvocationProgressEvent.build(queue_item=queue_item, invocation=invocation, message="encoding")
    assert event.device is None
