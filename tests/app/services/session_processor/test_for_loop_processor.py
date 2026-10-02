import asyncio
from contextlib import contextmanager
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.services.events.events_common import QueueItemStatusChangedEvent
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.progress_previews.progress_previews_default import MemoryProgressPreviews
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
)
from invokeai.app.services.session_queue.session_queue_common import (
    BatchStatus,
    SessionQueueItem,
    SessionQueueItemNotFoundError,
    SessionQueueStatus,
)
from invokeai.app.services.shared.execution_effects import ExecutionEffectsRecorder, ExecutionInterface
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.invocation_context import InvocationContext
from tests.test_nodes import create_edge, create_loop_linkage


@invocation_output("test_for_processor_body_output")
class ForProcessorBodyOutput(BaseInvocationOutput):
    value: int = OutputField(description="The loop body value")


@invocation("test_for_processor_body", version="1.0.0")
class ForProcessorBodyInvocation(BaseInvocation):
    value: int = InputField(default=0, description="The current loop item")
    fail_on: int | None = InputField(default=None, description="The value that raises an exception")

    def invoke(self, context: InvocationContext) -> ForProcessorBodyOutput:
        if self.value == self.fail_on:
            raise ValueError(f"Refusing loop value {self.value}")
        return ForProcessorBodyOutput(value=self.value)


@invocation_output("test_for_processor_after_output")
class ForProcessorAfterOutput(BaseInvocationOutput):
    values: list[Any] = OutputField(description="The completed loop values")


@invocation("test_for_processor_after", version="1.0.0")
class ForProcessorAfterInvocation(BaseInvocation):
    values: list[Any] = InputField(default_factory=list, description="The completed loop values")

    def invoke(self, context: InvocationContext) -> ForProcessorAfterOutput:
        return ForProcessorAfterOutput(values=self.values)


class _Logger:
    def debug(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass


class _Stats:
    @contextmanager
    def collect_stats(self, invocation: BaseInvocation, graph_execution_state_id: str):
        yield

    def log_stats(self, graph_execution_state_id: str) -> None:
        pass

    def reset_stats(self, graph_execution_state_id: str) -> None:
        pass


class _Events:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.completed: list[str] = []
        self.completed_outputs: list[tuple[str, BaseInvocationOutput]] = []
        self.errors: list[tuple[str, str, str]] = []

    def emit_invocation_started(self, queue_item, invocation) -> None:
        self.started.append(invocation.id)

    def emit_invocation_complete(self, invocation, queue_item, output) -> None:
        self.completed.append(invocation.id)
        self.completed_outputs.append((invocation.id, output.model_copy(deep=True)))

    def emit_invocation_error(self, queue_item, invocation, error_type, error_message, error_traceback) -> None:
        self.errors.append((invocation.id, error_type, error_message))


class _ProcessorQueue:
    """Small synchronized queue that preserves the queue mutation/event boundary needed here."""

    def __init__(self, item: SessionQueueItem) -> None:
        self._item = item
        self._lock = Lock()
        self._dequeued = False
        self.terminal = Event()
        self.status_events: list[QueueItemStatusChangedEvent] = []
        self.on_status_changed: Callable[[QueueItemStatusChangedEvent], None] | None = None
        self.session_updates: list[GraphExecutionState] = []
        self.completed_item_ids: list[int] = []
        self.canceled_item_ids: list[int] = []
        self.failed_item_ids: list[int] = []

    def dequeue(self, device: str | None = None) -> SessionQueueItem | None:
        with self._lock:
            if self._dequeued:
                return None
            self._dequeued = True
            self._item.status = "in_progress"
            self._item.device = device
            return self._item

    def get_queue_item(self, item_id: int) -> SessionQueueItem:
        if item_id != self._item.item_id:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        return self._item

    def set_queue_item_session(self, item_id: int, session: GraphExecutionState) -> SessionQueueItem:
        item = self.get_queue_item(item_id)
        item.session = session
        self.session_updates.append(session)
        return item

    def _status_event(self) -> QueueItemStatusChangedEvent:
        return QueueItemStatusChangedEvent.build(
            self._item,
            BatchStatus(
                queue_id=self._item.queue_id,
                batch_id=self._item.batch_id,
                origin=self._item.origin,
                destination=self._item.destination,
                pending=0,
                in_progress=1 if self._item.status == "in_progress" else 0,
                waiting=0,
                completed=1 if self._item.status == "completed" else 0,
                failed=1 if self._item.status == "failed" else 0,
                canceled=1 if self._item.status == "canceled" else 0,
                total=1,
            ),
            SessionQueueStatus(
                queue_id=self._item.queue_id,
                item_id=self._item.item_id,
                batch_id=self._item.batch_id,
                session_id=self._item.session_id,
                pending=0,
                in_progress=1 if self._item.status == "in_progress" else 0,
                waiting=0,
                completed=1 if self._item.status == "completed" else 0,
                failed=1 if self._item.status == "failed" else 0,
                canceled=1 if self._item.status == "canceled" else 0,
                total=1,
            ),
        )

    def _emit_status_changed(self) -> None:
        event = self._status_event()
        self.status_events.append(event)
        if self.on_status_changed is not None:
            self.on_status_changed(event)

    def complete_queue_item(self, item_id: int, queue_item: SessionQueueItem | None = None) -> SessionQueueItem:
        item = self.get_queue_item(item_id)
        item.status = "completed"
        self.completed_item_ids.append(item_id)
        self.terminal.set()
        return item

    def cancel_queue_item(self, item_id: int) -> SessionQueueItem:
        item = self.get_queue_item(item_id)
        with self._lock:
            if item.status == "canceled":
                return item
            item.status = "canceled"
            self.canceled_item_ids.append(item_id)
        self._emit_status_changed()
        self.terminal.set()
        return item

    def fail_queue_item(
        self, item_id: int, error_type: str, error_message: str, error_traceback: str
    ) -> SessionQueueItem:
        item = self.get_queue_item(item_id)
        item.status = "failed"
        item.error_type = error_type
        item.error_message = error_message
        item.error_traceback = error_traceback
        self.failed_item_ids.append(item_id)
        self.terminal.set()
        return item


def _build_graph(*, fail_on: int | None = None) -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=[1, 2, 3]))
    graph.add_node(ForProcessorBodyInvocation(id="body", fail_on=fail_on))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_node(ForProcessorAfterInvocation(id="after"))
    graph.add_edge(create_edge("for", "item", "body", "value"))
    graph.add_edge(create_edge("body", "value", "return", "output"))
    graph.add_edge(create_edge("for", "output_collection", "after", "values"))
    graph.add_edge(create_loop_linkage("for", "return"))
    return graph


def _build_item(*, fail_on: int | None = None) -> SessionQueueItem:
    session = GraphExecutionState(graph=_build_graph(fail_on=fail_on))
    return SessionQueueItem(
        item_id=1,
        batch_id="batch-1",
        session_id=session.id,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        started_at=None,
        completed_at=None,
        queue_id="default",
        session=session,
    )


def _build_processor(
    monkeypatch: pytest.MonkeyPatch,
    queue: _ProcessorQueue,
    events: _Events,
    on_after_run_node: Callable | None = None,
) -> DefaultSessionProcessor:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        lambda data, services, is_canceled: (
            effects := ExecutionEffectsRecorder(
                source_node_id=data.invocation.id,
                frame_path=data.execution_frame,
                state_id=data.execution_state_id,
                frame_id=data.execution_frame_id,
                workflow_call_depth=data.execution_workflow_call_depth,
            )
        )
        and SimpleNamespace(execution_effects=effects, execution=ExecutionInterface(effects)),
    )
    config = SimpleNamespace(
        generation_devices=[],
        profile_graphs=False,
        multiuser=False,
        node_cache_size=0,
        offload_text_encoders_to_idle_gpus=False,
    )
    services = SimpleNamespace(
        configuration=config,
        events=events,
        logger=_Logger(),
        performance_statistics=_Stats(),
        progress_previews=MemoryProgressPreviews(),
        session_queue=queue,
        invocation_cache=MemoryInvocationCache(max_cache_size=0),
        image_moves=None,
    )
    runner = DefaultSessionRunner(
        on_after_run_node_callbacks=[on_after_run_node] if on_after_run_node is not None else []
    )
    processor = DefaultSessionProcessor(session_runner=runner, polling_interval=0)
    if on_after_run_node is not None:
        callback_target: dict[str, DefaultSessionProcessor] = {}
        queue.on_status_changed = lambda event: asyncio.run(
            callback_target["processor"]._on_queue_item_status_changed((event.__event_name__, event))
        )
        callback_target["processor"] = processor
    processor.start(SimpleNamespace(services=services))
    return processor


def _stop_processor(processor: DefaultSessionProcessor) -> None:
    processor.stop()
    for worker in processor._workers:
        assert worker.thread is not None
        worker.thread.join(timeout=5)
        assert not worker.thread.is_alive()


def _completed_source_ids(item: SessionQueueItem, events: _Events) -> list[str]:
    return [item.session.prepared_source_mapping[exec_id] for exec_id in events.completed]


def test_processor_completes_for_loop_in_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    item = _build_item()
    queue = _ProcessorQueue(item)
    events = _Events()
    processor = _build_processor(monkeypatch, queue, events)
    try:
        assert queue.terminal.wait(timeout=5)
    finally:
        _stop_processor(processor)

    assert item.status == "completed"
    assert queue.completed_item_ids == [item.item_id]
    assert not queue.canceled_item_ids
    assert not queue.failed_item_ids
    assert item.session.is_complete()
    completed_source_ids = _completed_source_ids(item, events)
    assert completed_source_ids.count("for") == 4
    assert completed_source_ids.count("body") == 3
    assert completed_source_ids.count("return") == 3
    assert completed_source_ids[-1] == "after"

    final_for_outputs = [
        output
        for exec_id, output in events.completed_outputs
        if item.session.prepared_source_mapping[exec_id] == "for"
        and getattr(output, "output_collection", None) == [1, 2, 3]
    ]
    assert final_for_outputs


def test_processor_cancellation_event_stops_for_loop_without_final_output(monkeypatch: pytest.MonkeyPatch) -> None:
    item = _build_item()
    queue = _ProcessorQueue(item)
    events = _Events()

    def cancel_after_first_return(invocation, queue_item, output) -> None:
        source_id = queue_item.session.prepared_source_mapping[invocation.id]
        if source_id == "return" and _completed_source_ids(queue_item, events).count("return") == 1:
            queue.cancel_queue_item(queue_item.item_id)

    processor = _build_processor(monkeypatch, queue, events, on_after_run_node=cancel_after_first_return)
    try:
        assert queue.terminal.wait(timeout=5)
    finally:
        _stop_processor(processor)

    assert queue.status_events and queue.status_events[-1].status == "canceled"
    assert item.status == "canceled"
    assert queue.canceled_item_ids == [item.item_id]
    assert queue.completed_item_ids == []
    assert queue.failed_item_ids == []
    assert not item.session.is_complete()
    completed_source_ids = _completed_source_ids(item, events)
    assert completed_source_ids.count("for") == 1
    assert completed_source_ids.count("body") == 1
    assert completed_source_ids.count("return") == 1
    assert "after" not in completed_source_ids
    assert "after" not in item.session.source_prepared_mapping
    assert not item.session.finalized_loop_contexts


def test_processor_body_exception_fails_for_item_without_after_loop_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    item = _build_item(fail_on=2)
    queue = _ProcessorQueue(item)
    events = _Events()
    processor = _build_processor(monkeypatch, queue, events)
    try:
        assert queue.terminal.wait(timeout=5)
    finally:
        _stop_processor(processor)

    assert item.status == "failed"
    assert queue.failed_item_ids == [item.item_id]
    assert queue.completed_item_ids == []
    assert queue.canceled_item_ids == []
    assert item.error_type == "ValueError"
    assert item.error_message == "Refusing loop value 2"
    assert len(item.session.errors) == 1
    completed_source_ids = _completed_source_ids(item, events)
    assert completed_source_ids.count("for") == 2
    assert completed_source_ids.count("body") == 1
    assert completed_source_ids.count("return") == 1
    assert "after" not in completed_source_ids
    assert "after" not in item.session.source_prepared_mapping
    assert not item.session.finalized_loop_contexts
