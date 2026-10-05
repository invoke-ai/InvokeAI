import asyncio
import json
import uuid
from contextlib import contextmanager
from copy import deepcopy
from threading import Condition, Event
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from fastapi_events.handlers.local import local_handler

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.collections import CollectionConcatInvocation, RangeInvocation
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.events.events_common import EventBase, QueueItemStatusChangedEvent
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
)
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.execution_effects import ExecutionEffectsRecorder, ExecutionInterface
from invokeai.app.services.shared.execution_state_migration import dump_execution_state
from invokeai.app.services.shared.graph import (
    CollectInvocation,
    Graph,
    GraphExecutionState,
    IterateInvocation,
    _ExecutionScheduler,
    _GenericGraphSchedulerAdapter,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from tests.test_nodes import create_edge, create_loop_linkage


@invocation_output("test_for_sqlite_body_output")
class ForSqliteBodyOutput(BaseInvocationOutput):
    value: int = OutputField(description="The loop body value")


@invocation("test_for_sqlite_body", version="1.0.0")
class ForSqliteBodyInvocation(BaseInvocation):
    value: int = InputField(default=0, description="The current loop item")
    fail_on: int | None = InputField(default=None, description="The value that raises an exception")

    def invoke(self, context: InvocationContext) -> ForSqliteBodyOutput:
        if self.value == self.fail_on:
            raise ValueError(f"Refusing loop value {self.value}")
        return ForSqliteBodyOutput(value=self.value)


@invocation_output("test_for_sqlite_collection_adapter_output")
class ForSqliteCollectionAdapterOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(description="The inner loop collection")


@invocation("test_for_sqlite_collection_adapter", version="1.0.0")
class ForSqliteCollectionAdapterInvocation(BaseInvocation):
    value: Any = InputField(default=None, description="The outer loop item")

    def invoke(self, context: InvocationContext) -> ForSqliteCollectionAdapterOutput:
        return ForSqliteCollectionAdapterOutput(collection=self.value)


@invocation_output("test_for_sqlite_after_output")
class ForSqliteAfterOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(description="The completed loop collection")


@invocation("test_for_sqlite_after", version="1.0.0")
class ForSqliteAfterInvocation(BaseInvocation):
    collection: list[Any] = InputField(default_factory=list, description="The completed loop collection")

    def invoke(self, context: InvocationContext) -> ForSqliteAfterOutput:
        return ForSqliteAfterOutput(collection=self.collection)


def _build_nested_graph(*, fail_on: int | None = None) -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=[[1, 2], [3, 4]]))
    graph.add_node(ForSqliteCollectionAdapterInvocation(id="adapter"))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(ForSqliteBodyInvocation(id="body", fail_on=fail_on))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_node(ForSqliteAfterInvocation(id="after"))
    graph.add_edge(create_edge("for", "item", "adapter", "value"))
    graph.add_edge(create_edge("adapter", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "body", "value"))
    graph.add_edge(create_edge("body", "value", "collect", "item"))
    graph.add_edge(create_edge("collect", "collection", "return", "output"))
    graph.add_edge(create_edge("for", "output_collection", "after", "collection"))
    graph.add_edge(create_loop_linkage("for", "return"))
    return graph


def _build_iterate_collect_graph() -> Graph:
    graph = Graph()
    graph.add_node(RangeInvocation(id="range", start=0, stop=3, step=1))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(create_edge("range", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "collect", "item"))
    return graph


def _build_direct_planner_iterate_body_collect_graph() -> Graph:
    graph = Graph()
    graph.add_node(CollectionConcatInvocation(id="source", first=[0, 1, 2]))
    graph.add_node(IterateInvocation(id="iterate"))
    graph.add_node(ForSqliteBodyInvocation(id="body"))
    graph.add_node(CollectInvocation(id="collect"))
    graph.add_edge(create_edge("source", "collection", "iterate", "collection"))
    graph.add_edge(create_edge("iterate", "item", "body", "value"))
    graph.add_edge(create_edge("body", "value", "collect", "item"))
    return graph


def _build_flat_for_graph() -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="for", collection=[0, 1, 2]))
    graph.add_node(ForSqliteBodyInvocation(id="body"))
    graph.add_node(ForReturnInvocation(id="return"))
    graph.add_node(ForSqliteAfterInvocation(id="after"))
    graph.add_edge(create_edge("for", "item", "body", "value"))
    graph.add_edge(create_edge("body", "value", "return", "output"))
    graph.add_edge(create_edge("for", "output_collection", "after", "collection"))
    graph.add_edge(create_loop_linkage("for", "return"))
    return graph


def _build_nested_for_graph(*, fail_on: int | None = None) -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="outer_for", collection=[[1, 2], [3, 4]]))
    graph.add_node(ForSqliteCollectionAdapterInvocation(id="inner_collection"))
    graph.add_node(ForInvocation(id="inner_for"))
    graph.add_node(ForSqliteBodyInvocation(id="inner_body", fail_on=fail_on))
    graph.add_node(ForReturnInvocation(id="inner_return"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    graph.add_node(ForSqliteAfterInvocation(id="after"))
    graph.add_edge(create_edge("outer_for", "item", "inner_collection", "value"))
    graph.add_edge(create_edge("inner_collection", "collection", "inner_for", "collection"))
    graph.add_edge(create_edge("inner_for", "item", "inner_body", "value"))
    graph.add_edge(create_edge("inner_body", "value", "inner_return", "output"))
    graph.add_edge(create_edge("inner_for", "output_collection", "outer_return", "output"))
    graph.add_edge(create_edge("outer_for", "output_collection", "after", "collection"))
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    graph.add_edge(create_loop_linkage("inner_for", "inner_return"))
    return graph


def _build_two_sibling_nested_for_fan_in_graph() -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="outer_for", collection=[[1, 2], [3, 4]]))
    graph.add_node(ForInvocation(id="left_for"))
    graph.add_node(ForInvocation(id="right_for"))
    graph.add_node(ForSqliteBodyInvocation(id="left_body"))
    graph.add_node(ForSqliteBodyInvocation(id="right_body"))
    graph.add_node(ForReturnInvocation(id="left_return"))
    graph.add_node(ForReturnInvocation(id="right_return"))
    graph.add_node(CollectionConcatInvocation(id="join"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    graph.add_node(ForSqliteAfterInvocation(id="after"))
    graph.add_edge(create_edge("outer_for", "item", "left_for", "collection"))
    graph.add_edge(create_edge("outer_for", "item", "right_for", "collection"))
    graph.add_edge(create_edge("left_for", "item", "left_body", "value"))
    graph.add_edge(create_edge("right_for", "item", "right_body", "value"))
    graph.add_edge(create_edge("left_body", "value", "left_return", "output"))
    graph.add_edge(create_edge("right_body", "value", "right_return", "output"))
    graph.add_edge(create_edge("left_for", "output_collection", "join", "first"))
    graph.add_edge(create_edge("right_for", "output_collection", "join", "second"))
    graph.add_edge(create_edge("join", "collection", "outer_return", "output"))
    graph.add_edge(create_edge("outer_for", "output_collection", "after", "collection"))
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    graph.add_edge(create_loop_linkage("left_for", "left_return"))
    graph.add_edge(create_loop_linkage("right_for", "right_return"))
    return graph


def _build_three_level_nested_for_graph() -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="outer_for", collection=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    graph.add_node(ForInvocation(id="middle_for"))
    graph.add_node(ForInvocation(id="inner_for"))
    graph.add_node(ForSqliteBodyInvocation(id="body"))
    graph.add_node(ForReturnInvocation(id="inner_return"))
    graph.add_node(ForReturnInvocation(id="middle_return"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    graph.add_node(ForSqliteAfterInvocation(id="after"))
    graph.add_edge(create_edge("outer_for", "item", "middle_for", "collection"))
    graph.add_edge(create_edge("middle_for", "item", "inner_for", "collection"))
    graph.add_edge(create_edge("inner_for", "item", "body", "value"))
    graph.add_edge(create_edge("body", "value", "inner_return", "output"))
    graph.add_edge(create_edge("inner_for", "output_collection", "middle_return", "output"))
    graph.add_edge(create_edge("middle_for", "output_collection", "outer_return", "output"))
    graph.add_edge(create_edge("outer_for", "output_collection", "after", "collection"))
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    graph.add_edge(create_loop_linkage("middle_for", "middle_return"))
    graph.add_edge(create_loop_linkage("inner_for", "inner_return"))
    return graph


def _build_four_level_nested_for_graph() -> Graph:
    graph = Graph()
    graph.add_node(ForInvocation(id="outer_for", collection=[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]))
    graph.add_node(ForInvocation(id="second_for"))
    graph.add_node(ForInvocation(id="third_for"))
    graph.add_node(ForInvocation(id="deepest_for"))
    graph.add_node(ForSqliteBodyInvocation(id="body"))
    graph.add_node(ForReturnInvocation(id="deepest_return"))
    graph.add_node(ForReturnInvocation(id="third_return"))
    graph.add_node(ForReturnInvocation(id="second_return"))
    graph.add_node(ForReturnInvocation(id="outer_return"))
    graph.add_node(ForSqliteAfterInvocation(id="after"))
    graph.add_edge(create_edge("outer_for", "item", "second_for", "collection"))
    graph.add_edge(create_edge("second_for", "item", "third_for", "collection"))
    graph.add_edge(create_edge("third_for", "item", "deepest_for", "collection"))
    graph.add_edge(create_edge("deepest_for", "item", "body", "value"))
    graph.add_edge(create_edge("body", "value", "deepest_return", "output"))
    graph.add_edge(create_edge("deepest_for", "output_collection", "third_return", "output"))
    graph.add_edge(create_edge("third_for", "output_collection", "second_return", "output"))
    graph.add_edge(create_edge("second_for", "output_collection", "outer_return", "output"))
    graph.add_edge(create_edge("outer_for", "output_collection", "after", "collection"))
    graph.add_edge(create_loop_linkage("outer_for", "outer_return"))
    graph.add_edge(create_loop_linkage("second_for", "second_return"))
    graph.add_edge(create_loop_linkage("third_for", "third_return"))
    graph.add_edge(create_loop_linkage("deepest_for", "deepest_return"))
    return graph


class _RecordingRegisteredEventService(EventServiceBase):
    def __init__(self) -> None:
        self._events: list[EventBase] = []
        self._events_condition = Condition()

    def dispatch(self, event: EventBase) -> None:
        with self._events_condition:
            self._events.append(event)
            self._events_condition.notify_all()
        asyncio.run(local_handler.handle((event.__event_name__, event)))

    def wait_for_status(self, item_id: int, status: str, timeout: float = 5) -> bool:
        def has_status() -> bool:
            return any(
                isinstance(event, QueueItemStatusChangedEvent) and event.item_id == item_id and event.status == status
                for event in self._events
            )

        with self._events_condition:
            if has_status():
                return True
            return self._events_condition.wait_for(has_status, timeout=timeout)


@pytest.fixture
def registered_event_bus() -> Iterator[_RecordingRegisteredEventService]:
    yield _RecordingRegisteredEventService()


def _stop_processor(processor: DefaultSessionProcessor) -> None:
    processor.stop()
    for worker in processor._workers:
        assert worker.thread is not None
        worker.thread.join(timeout=5)
        assert not worker.thread.is_alive()


def _insert_session(queue: SqliteSessionQueue, graph: Graph, *, versioned: bool = False) -> int:
    session = GraphExecutionState(graph=graph)
    session_json = (
        json.dumps(dump_execution_state(session))
        if versioned
        else session.model_dump_json(warnings=False, exclude_none=True)
    )
    with queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, field_values, priority,
                workflow, origin, destination, retried_from_item_id, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                session_json,
                session.id,
                str(uuid.uuid4()),
                None,
                0,
                None,
                None,
                None,
                None,
                "system",
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def _build_test_invocation_context(data, services, is_canceled):
    recorder = ExecutionEffectsRecorder(
        source_node_id=data.invocation.id,
        frame_path=data.execution_frame,
        state_id=getattr(data, "execution_state_id", None),
        frame_id=getattr(data, "execution_frame_id", None),
        workflow_call_depth=getattr(data, "execution_workflow_call_depth", None),
    )
    return SimpleNamespace(
        execution_effects=recorder,
        effects=recorder,
        execution=ExecutionInterface(recorder),
    )


class _Stats:
    @contextmanager
    def collect_stats(self, invocation, graph_execution_state_id):
        yield

    def log_stats(self, graph_execution_state_id) -> None:
        pass

    def reset_stats(self, graph_execution_state_id) -> None:
        pass


@pytest.mark.parametrize("outcome", ["success", "canceled", "failure"])
def test_processor_sqlite_queue_nested_iterate_for_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
    outcome: str,
) -> None:
    test_timeout = 30
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    returns_seen = 0

    def cancel_after_first_return(invocation, queue_item, output) -> None:
        nonlocal returns_seen
        if queue_item.session.prepared_source_mapping[invocation.id] != "return":
            return
        returns_seen += 1
        if outcome == "canceled" and returns_seen == 1:
            queue.cancel_queue_item(queue_item.item_id)

    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(on_after_run_node_callbacks=[cancel_after_first_return]),
        polling_interval=0,
    )
    graph = _build_nested_graph(fail_on=2 if outcome == "failure" else None)
    item_id = _insert_session(queue, graph)
    status_handler_called = Event()
    original_status_handler = processor._on_queue_item_status_changed

    async def recording_status_handler(event) -> None:
        if event[1].item_id == item_id and event[1].status == "canceled":
            status_handler_called.set()
        await original_status_handler(event)

    processor._on_queue_item_status_changed = recording_status_handler  # type: ignore[method-assign]
    try:
        processor.start(mock_invoker)

        expected_status = {
            "success": "completed",
            "canceled": "canceled",
            "failure": "failed",
        }[outcome]
        assert registered_event_bus.wait_for_status(item_id, expected_status, timeout=test_timeout)

        queue_item = queue.get_queue_item(item_id)
        assert queue_item.status == expected_status
        assert queue.get_current("default") is None
        assert any(
            isinstance(event, QueueItemStatusChangedEvent)
            and event.item_id == item_id
            and event.status == expected_status
            for event in registered_event_bus._events
        )

        if outcome == "success":
            assert queue_item.session.is_complete()
            [after_exec_id] = queue_item.session.source_prepared_mapping["after"]
            assert queue_item.session.results[after_exec_id].collection == [[1, 2], [3, 4]]
            assert returns_seen == 2
        else:
            assert "after" not in queue_item.session.source_prepared_mapping
            assert ("for", ()) not in queue_item.session.finalized_loop_contexts
            if outcome == "canceled":
                assert returns_seen == 1
                assert status_handler_called.wait(timeout=5)
                assert not queue_item.session.is_complete()
            else:
                assert queue_item.session.has_error()
                assert queue_item.error_type == "ValueError"
                assert queue_item.error_message == "Refusing loop value 2"
                assert returns_seen == 0
    finally:
        _stop_processor(processor)


def test_processor_sqlite_nested_iterate_for_cancel_retry_reloads_fresh_stream_state(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    item_id = _insert_session(queue, _build_nested_graph(), versioned=True)
    session_persisted = Event()
    returns_seen = 0

    def cancel_after_first_inner_return(invocation, queue_item, output) -> None:
        nonlocal returns_seen
        if queue_item.session.prepared_source_mapping[invocation.id] != "return":
            return
        returns_seen += 1
        if returns_seen == 1:
            queue.cancel_queue_item(queue_item.item_id)

    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_node_callbacks=[cancel_after_first_inner_return],
            on_after_run_session_callbacks=[lambda queue_item: session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "canceled")
        assert session_persisted.wait(timeout=5)
    finally:
        _stop_processor(processor)

    canceled_item = queue.get_queue_item(item_id)
    assert canceled_item.status == "canceled"
    assert returns_seen == 1
    assert not canceled_item.session.is_complete()
    assert "after" not in canceled_item.session.source_prepared_mapping
    canceled_snapshot = dump_execution_state(canceled_item.session)

    # Simulate an interrupted process after the partial state was persisted. Startup must cancel
    # the stale row while preserving the nested frame/stream snapshot for inspection or retry.
    with queue._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'in_progress' WHERE item_id = ?", (item_id,))
    restarted_queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    restarted_queue.start(mock_invoker)
    reloaded_canceled_item = restarted_queue.get_queue_item(item_id)
    assert reloaded_canceled_item.status == "canceled"
    assert dump_execution_state(reloaded_canceled_item.session) == canceled_snapshot
    assert restarted_queue.dequeue() is None

    retry_result = queue.retry_items_by_id("default", [item_id])
    assert retry_result.retried_item_ids == [item_id]
    retried_item = next(
        queue_item for queue_item in queue.list_all_queue_items("default") if queue_item.retried_from_item_id == item_id
    )
    assert retried_item.status == "pending"
    assert retried_item.item_id != item_id
    assert retried_item.session.id != canceled_item.session.id
    assert retried_item.session.results == {}
    assert retried_item.session.execution_refs == {}
    assert retried_item.session.execution_tokens == {}
    assert retried_item.session.execution_effects == {}
    assert not retried_item.session._generic_runtime().streams

    retry_session_persisted = Event()
    retry_processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_session_callbacks=[lambda queue_item: retry_session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        retry_processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(retried_item.item_id, "completed")
        assert retry_session_persisted.wait(timeout=5)
    finally:
        _stop_processor(retry_processor)

    canceled_item = queue.get_queue_item(item_id)
    completed_item = queue.get_queue_item(retried_item.item_id)
    assert dump_execution_state(canceled_item.session) == canceled_snapshot
    assert canceled_item.status == "canceled"
    assert completed_item.status == "completed"
    assert completed_item.session.is_complete()
    [after_execution_id] = completed_item.session.source_prepared_mapping["after"]
    assert completed_item.session.results[after_execution_id].collection == [[1, 2], [3, 4]]
    completed_streams = [
        stream for stream in completed_item.session._generic_runtime().streams.values() if stream.owner_id == "iterate"
    ]
    assert len(completed_streams) == 2
    assert all(stream.closed for stream in completed_streams)
    assert {stream.values for stream in completed_streams} == {(1, 2), (3, 4)}
    assert all(stream.stream_id.startswith(f"{completed_item.session.id}:iterate:") for stream in completed_streams)


def test_processor_sqlite_three_level_nested_for_cancel_restart_retry_isolates_execution_state(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    item_id = _insert_session(queue, _build_three_level_nested_for_graph(), versioned=True)
    session_persisted = Event()
    nested_returns_seen = 0

    def cancel_after_first_inner_return(invocation, queue_item, output) -> None:
        nonlocal nested_returns_seen
        if queue_item.session.prepared_source_mapping[invocation.id] != "inner_return":
            return
        nested_returns_seen += 1
        if nested_returns_seen == 1:
            queue.cancel_queue_item(queue_item.item_id)

    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_node_callbacks=[cancel_after_first_inner_return],
            on_after_run_session_callbacks=[lambda queue_item: session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "canceled")
        assert session_persisted.wait(timeout=5)
    finally:
        _stop_processor(processor)

    canceled_item = queue.get_queue_item(item_id)
    assert canceled_item.status == "canceled"
    assert nested_returns_seen == 1
    assert not canceled_item.session.is_complete()
    assert "after" not in canceled_item.session.source_prepared_mapping
    [inner_return_execution_id] = [
        execution_id
        for execution_id in canceled_item.session.source_prepared_mapping["inner_return"]
        if execution_id in canceled_item.session.results
    ]
    assert canceled_item.session.results[inner_return_execution_id].output == 1
    canceled_snapshot = dump_execution_state(canceled_item.session)
    canceled_execution_ids = {
        execution_id
        for execution_ids in canceled_item.session.source_prepared_mapping.values()
        for execution_id in execution_ids
    }
    canceled_frame_ids = {
        reference.frame.frame_id
        for reference in canceled_item.session.execution_refs.values()
        if reference.frame.frame_id
    }
    assert canceled_frame_ids
    assert all(
        reference.state_id == canceled_item.session.id and reference.frame.state_id == canceled_item.session.id
        for reference in canceled_item.session.execution_refs.values()
    )

    with queue._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'in_progress' WHERE item_id = ?", (item_id,))
    restarted_queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    restarted_queue.start(mock_invoker)
    mock_invoker.services.session_queue = restarted_queue
    reloaded_canceled_item = restarted_queue.get_queue_item(item_id)
    assert reloaded_canceled_item.status == "canceled"
    assert dump_execution_state(reloaded_canceled_item.session) == canceled_snapshot
    assert restarted_queue.dequeue() is None

    assert mock_invoker.services.session_queue is restarted_queue
    retry_result = restarted_queue.retry_items_by_id("default", [item_id])
    assert retry_result.retried_item_ids == [item_id]
    [retried_item] = [
        queue_item
        for queue_item in restarted_queue.list_all_queue_items("default")
        if queue_item.retried_from_item_id == item_id
    ]
    assert retried_item.status == "pending"
    assert retried_item.item_id != item_id
    assert retried_item.session.id != canceled_item.session.id
    assert retried_item.session.results == {}
    assert retried_item.session.execution_refs == {}
    assert retried_item.session.execution_tokens == {}
    assert retried_item.session.execution_effects == {}
    retry_runtime = retried_item.session._generic_runtime()
    assert not retry_runtime.gates
    assert not retry_runtime.streams
    assert not retry_runtime.continuations

    retry_session_persisted = Event()
    retry_processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_session_callbacks=[lambda queue_item: retry_session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        retry_processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(retried_item.item_id, "completed")
        assert retry_session_persisted.wait(timeout=5)
    finally:
        _stop_processor(retry_processor)

    canceled_item = restarted_queue.get_queue_item(item_id)
    completed_item = restarted_queue.get_queue_item(retried_item.item_id)
    assert canceled_item.status == "canceled"
    assert dump_execution_state(canceled_item.session) == canceled_snapshot
    assert completed_item.status == "completed"
    assert completed_item.session.is_complete()
    completed_execution_ids = {
        execution_id
        for execution_ids in completed_item.session.source_prepared_mapping.values()
        for execution_id in execution_ids
    }
    assert canceled_execution_ids.isdisjoint(completed_execution_ids)
    completed_frame_ids = {
        reference.frame.frame_id
        for reference in completed_item.session.execution_refs.values()
        if reference.frame.frame_id
    }
    assert completed_frame_ids
    assert all(
        reference.state_id == completed_item.session.id and reference.frame.state_id == completed_item.session.id
        for reference in completed_item.session.execution_refs.values()
    )
    assert canceled_frame_ids.isdisjoint(completed_frame_ids)
    [after_execution_id] = completed_item.session.source_prepared_mapping["after"]
    assert completed_item.session.results[after_execution_id].collection == [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]
    completed_runtime = completed_item.session._generic_runtime()
    assert not completed_runtime.gates
    assert not completed_runtime.streams


def test_processor_sqlite_four_level_nested_for_cancel_restart_retry_isolates_execution_state(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    item_id = _insert_session(queue, _build_four_level_nested_for_graph(), versioned=True)
    session_persisted = Event()
    deepest_returns_seen = 0

    def cancel_after_first_deepest_return(invocation, queue_item, output) -> None:
        del output
        nonlocal deepest_returns_seen
        if queue_item.session.prepared_source_mapping[invocation.id] != "deepest_return":
            return
        deepest_returns_seen += 1
        if deepest_returns_seen == 1:
            queue.cancel_queue_item(queue_item.item_id)

    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_node_callbacks=[cancel_after_first_deepest_return],
            on_after_run_session_callbacks=[lambda queue_item: session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "canceled")
        assert session_persisted.wait(timeout=5)
    finally:
        _stop_processor(processor)

    canceled_item = queue.get_queue_item(item_id)
    assert canceled_item.status == "canceled"
    assert deepest_returns_seen == 1
    assert not canceled_item.session.is_complete()
    assert "after" not in canceled_item.session.source_prepared_mapping
    [deepest_return_execution_id] = [
        execution_id
        for execution_id in canceled_item.session.source_prepared_mapping["deepest_return"]
        if execution_id in canceled_item.session.results
    ]
    assert canceled_item.session.results[deepest_return_execution_id].output == 1
    canceled_snapshot = dump_execution_state(canceled_item.session)
    canceled_execution_ids = {
        execution_id
        for execution_ids in canceled_item.session.source_prepared_mapping.values()
        for execution_id in execution_ids
    }
    canceled_frame_ids = {
        reference.frame.frame_id
        for reference in canceled_item.session.execution_refs.values()
        if reference.frame.frame_id
    }
    assert canceled_frame_ids
    assert all(
        reference.state_id == canceled_item.session.id and reference.frame.state_id == canceled_item.session.id
        for reference in canceled_item.session.execution_refs.values()
    )

    with queue._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'in_progress' WHERE item_id = ?", (item_id,))
    restarted_queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    restarted_queue.start(mock_invoker)
    mock_invoker.services.session_queue = restarted_queue
    reloaded_canceled_item = restarted_queue.get_queue_item(item_id)
    assert reloaded_canceled_item.status == "canceled"
    assert dump_execution_state(reloaded_canceled_item.session) == canceled_snapshot
    assert restarted_queue.dequeue() is None

    assert mock_invoker.services.session_queue is restarted_queue
    retry_result = restarted_queue.retry_items_by_id("default", [item_id])
    assert retry_result.retried_item_ids == [item_id]
    [retried_item] = [
        queue_item
        for queue_item in restarted_queue.list_all_queue_items("default")
        if queue_item.retried_from_item_id == item_id
    ]
    assert retried_item.status == "pending"
    assert retried_item.item_id != item_id
    assert retried_item.session.id != canceled_item.session.id
    assert retried_item.session.results == {}
    assert retried_item.session.execution_refs == {}
    assert retried_item.session.execution_tokens == {}
    assert retried_item.session.execution_effects == {}
    retry_runtime = retried_item.session._generic_runtime()
    assert not retry_runtime.gates
    assert not retry_runtime.streams
    assert not retry_runtime.continuations

    retry_session_persisted = Event()
    retry_processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_session_callbacks=[lambda queue_item: retry_session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        retry_processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(retried_item.item_id, "completed")
        assert retry_session_persisted.wait(timeout=5)
    finally:
        _stop_processor(retry_processor)

    canceled_item = restarted_queue.get_queue_item(item_id)
    completed_item = restarted_queue.get_queue_item(retried_item.item_id)
    assert canceled_item.status == "canceled"
    assert dump_execution_state(canceled_item.session) == canceled_snapshot
    assert completed_item.status == "completed"
    assert completed_item.session.is_complete()
    completed_execution_ids = {
        execution_id
        for execution_ids in completed_item.session.source_prepared_mapping.values()
        for execution_id in execution_ids
    }
    assert canceled_execution_ids.isdisjoint(completed_execution_ids)
    completed_frame_ids = {
        reference.frame.frame_id
        for reference in completed_item.session.execution_refs.values()
        if reference.frame.frame_id
    }
    assert completed_frame_ids
    assert all(
        reference.state_id == completed_item.session.id and reference.frame.state_id == completed_item.session.id
        for reference in completed_item.session.execution_refs.values()
    )
    assert canceled_frame_ids.isdisjoint(completed_frame_ids)
    [after_execution_id] = completed_item.session.source_prepared_mapping["after"]
    assert completed_item.session.results[after_execution_id].collection == [
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    ]
    completed_runtime = completed_item.session._generic_runtime()
    assert not completed_runtime.gates
    assert not completed_runtime.streams
    assert completed_runtime.continuations
    assert all(
        continuation.frame.state_id == completed_item.session.id
        and continuation.frame.frame_id not in canceled_frame_ids
        for continuation in completed_runtime.continuations.values()
    )


def test_processor_sqlite_two_sibling_nested_for_cancel_root_retry_cleans_identity(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
) -> None:
    test_timeout = 30
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    child_ids = {"left_for", "right_for"}
    cancel_requested = False

    def cancel_after_first_child_finalization(invocation, queue_item, output) -> None:
        del invocation, output
        nonlocal cancel_requested
        finalized_children = {
            source_id
            for source_id, _parent_path in queue_item.session.finalized_loop_contexts
            if source_id in child_ids
        }
        if not cancel_requested and len(finalized_children) == 1:
            cancel_requested = True
            queue.cancel_queue_item(queue_item.item_id)

    item_id = _insert_session(queue, _build_two_sibling_nested_for_fan_in_graph(), versioned=True)
    session_persisted = Event()
    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_node_callbacks=[cancel_after_first_child_finalization],
            on_after_run_session_callbacks=[lambda queue_item: session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "canceled", timeout=test_timeout)
        assert session_persisted.wait(timeout=test_timeout)
    finally:
        _stop_processor(processor)

    canceled_item = queue.get_queue_item(item_id)
    assert canceled_item.status == "canceled"
    assert cancel_requested
    assert not canceled_item.session.is_complete()
    assert len({source_id for source_id, _ in canceled_item.session.finalized_loop_contexts} & child_ids) == 1
    assert "outer_return" not in canceled_item.session.source_prepared_mapping
    assert "after" not in canceled_item.session.source_prepared_mapping
    canceled_snapshot = dump_execution_state(canceled_item.session)
    canceled_execution_ids = set(canceled_item.session.prepared_source_mapping)
    canceled_frame_ids = {
        reference.frame.frame_id
        for reference in canceled_item.session.execution_refs.values()
        if reference.frame.frame_id
    }
    assert canceled_frame_ids
    assert all(
        reference.state_id == canceled_item.session.id and reference.frame.state_id == canceled_item.session.id
        for reference in canceled_item.session.execution_refs.values()
    )

    with queue._db.transaction() as cursor:
        cursor.execute("UPDATE session_queue SET status = 'in_progress' WHERE item_id = ?", (item_id,))
    restarted_queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    restarted_queue.start(mock_invoker)
    mock_invoker.services.session_queue = restarted_queue
    reloaded_canceled_item = restarted_queue.get_queue_item(item_id)
    assert reloaded_canceled_item.status == "canceled"
    assert dump_execution_state(reloaded_canceled_item.session) == canceled_snapshot
    assert restarted_queue.dequeue() is None

    assert mock_invoker.services.session_queue is restarted_queue
    retry_result = restarted_queue.retry_items_by_id("default", [item_id])
    assert retry_result.retried_item_ids == [item_id]
    [retried_item] = [
        queue_item
        for queue_item in restarted_queue.list_all_queue_items("default")
        if queue_item.retried_from_item_id == item_id
    ]
    assert retried_item.session.id != canceled_item.session.id
    assert retried_item.session.results == {}
    assert retried_item.session.execution_refs == {}
    assert retried_item.session.execution_tokens == {}
    assert retried_item.session.execution_effects == {}
    assert not retried_item.session._generic_runtime().gates
    assert not retried_item.session._generic_runtime().streams
    assert not retried_item.session._generic_runtime().continuations

    retry_session_persisted = Event()
    retry_processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_session_callbacks=[lambda queue_item: retry_session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        retry_processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(retried_item.item_id, "completed", timeout=test_timeout)
        assert retry_session_persisted.wait(timeout=test_timeout)
    finally:
        _stop_processor(retry_processor)

    canceled_item = restarted_queue.get_queue_item(item_id)
    completed_item = restarted_queue.get_queue_item(retried_item.item_id)
    assert canceled_item.status == "canceled"
    assert dump_execution_state(canceled_item.session) == canceled_snapshot
    assert completed_item.status == "completed"
    assert completed_item.session.is_complete()
    completed_execution_ids = set(completed_item.session.prepared_source_mapping)
    assert canceled_execution_ids.isdisjoint(completed_execution_ids)
    completed_frame_ids = {
        reference.frame.frame_id
        for reference in completed_item.session.execution_refs.values()
        if reference.frame.frame_id
    }
    assert completed_frame_ids
    assert all(
        reference.state_id == completed_item.session.id and reference.frame.state_id == completed_item.session.id
        for reference in completed_item.session.execution_refs.values()
    )
    assert canceled_frame_ids.isdisjoint(completed_frame_ids)
    [after_execution_id] = completed_item.session.source_prepared_mapping["after"]
    assert completed_item.session.results[after_execution_id].collection == [
        [1, 2, 1, 2],
        [3, 4, 3, 4],
    ]
    completed_runtime = completed_item.session._generic_runtime()
    assert not completed_runtime.gates
    assert not completed_runtime.streams
    assert completed_runtime.continuations
    assert all(
        continuation.frame.state_id == completed_item.session.id
        for continuation in completed_runtime.continuations.values()
    )


def test_processor_sqlite_two_sibling_nested_for_failure_cleans_runtime(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    captured_sessions: list[GraphExecutionState] = []
    original_set_queue_item_session = queue.set_queue_item_session

    def capture_set_queue_item_session(item_id: int, session: GraphExecutionState):
        captured_sessions.append(session)
        return original_set_queue_item_session(item_id, session)

    monkeypatch.setattr(queue, "set_queue_item_session", capture_set_queue_item_session)

    graph = _build_two_sibling_nested_for_fan_in_graph()
    graph.nodes["left_body"].fail_on = 1
    item_id = _insert_session(queue, graph, versioned=True)
    processor = DefaultSessionProcessor(polling_interval=0)
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "failed")
    finally:
        _stop_processor(processor)

    assert captured_sessions
    live_failed_session = captured_sessions[0]
    live_runtime = live_failed_session._generic_runtime()
    assert not live_runtime.gates
    assert not live_runtime.streams
    assert not live_runtime.continuations
    assert not live_failed_session._ready_queues
    assert not live_failed_session._ready_node_ids
    assert live_failed_session._active_class is None
    assert live_failed_session._generic_graph_scheduler is None
    assert not isinstance(live_failed_session._execution_scheduler, _GenericGraphSchedulerAdapter)
    assert live_failed_session.execution_refs
    assert live_failed_session.execution_tokens
    assert live_failed_session.execution_effects

    failed_item = queue.get_queue_item(item_id)
    assert failed_item.status == "failed"
    assert failed_item.error_type == "ValueError"
    assert failed_item.error_message == "Refusing loop value 1"
    assert queue.get_current("default") is None

    session = failed_item.session
    assert session.execution_refs
    assert all(
        session.execution_refs[execution_id].source_node_id == execution_ref.source_node_id
        and session.execution_refs[execution_id].frame == execution_ref.frame
        and session.execution_refs[execution_id].state_id == execution_ref.state_id
        for execution_id, execution_ref in live_failed_session.execution_refs.items()
    )
    assert session.execution_tokens == {}
    assert session.execution_effects
    assert set(live_failed_session.execution_effects) <= set(session.execution_effects)
    assert session.has_error()
    assert ("outer_for", ()) not in session.finalized_loop_contexts
    for source_id in ("join", "outer_return", "after"):
        assert not any(
            execution_id in session.results for execution_id in session.source_prepared_mapping.get(source_id, ())
        )

    runtime = session._generic_runtime()
    assert not runtime.gates
    assert not runtime.streams
    assert not runtime.continuations


def test_processor_sqlite_three_level_nested_for_failure_cleans_runtime(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    body_scheduler_seen: list[str] = []
    original_set_queue_item_session = queue.set_queue_item_session
    failed_persistence_snapshot: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]] | None = None

    def record_generic_scheduler_before_body(invocation, queue_item) -> None:
        if queue_item.session.prepared_source_mapping[invocation.id] == "body":
            assert isinstance(queue_item.session._execution_scheduler, _GenericGraphSchedulerAdapter)
            body_scheduler_seen.append(invocation.id)

    def capture_set_queue_item_session(item_id: int, session: GraphExecutionState):
        nonlocal failed_persistence_snapshot
        if session.has_error():
            assert session._generic_execution_runtime is not None
            runtime = session._generic_execution_runtime
            assert not runtime.gates
            assert not runtime.streams
            assert not runtime.continuations
            assert not session._ready_queues
            assert not session._ready_node_ids
            assert session._active_class is None
            assert session._generic_graph_scheduler is None
            assert not isinstance(session._execution_scheduler, _GenericGraphSchedulerAdapter)
            snapshot_session = session.model_copy(deep=True)
            snapshot_session._rehydrate_execution_refs()
            durable_snapshot = dump_execution_state(snapshot_session)
            failed_persistence_snapshot = (
                deepcopy(durable_snapshot["execution_refs"]),
                deepcopy(durable_snapshot["execution_tokens"]),
                deepcopy(durable_snapshot["execution_effects"]),
                deepcopy(session.errors),
            )
        return original_set_queue_item_session(item_id, session)

    monkeypatch.setattr(queue, "set_queue_item_session", capture_set_queue_item_session)

    graph = _build_three_level_nested_for_graph()
    graph.nodes["body"].fail_on = 1
    item_id = _insert_session(queue, graph, versioned=True)
    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(on_before_run_node_callbacks=[record_generic_scheduler_before_body]),
        polling_interval=0,
    )
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "failed")
    finally:
        _stop_processor(processor)

    assert body_scheduler_seen
    assert failed_persistence_snapshot is not None
    execution_refs_snapshot, execution_tokens_snapshot, execution_effects_snapshot, errors_snapshot = (
        failed_persistence_snapshot
    )
    assert execution_refs_snapshot == {}
    assert execution_tokens_snapshot == {}
    assert execution_effects_snapshot

    failed_item = queue.get_queue_item(item_id)
    assert failed_item.status == "failed"
    assert failed_item.error_type == "ValueError"
    assert failed_item.error_message == "Refusing loop value 1"
    assert queue.get_current("default") is None

    reloaded_queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    reloaded_queue.start(mock_invoker)
    reloaded_item = reloaded_queue.get_queue_item(item_id)
    assert reloaded_item.status == "failed"
    assert reloaded_queue.get_current("default") is None

    session = reloaded_item.session
    durable_snapshot = dump_execution_state(session)
    assert durable_snapshot["execution_refs"] == execution_refs_snapshot
    assert durable_snapshot["execution_tokens"] == execution_tokens_snapshot
    assert durable_snapshot["execution_effects"] == execution_effects_snapshot
    assert session.errors == errors_snapshot
    assert session.has_error()
    assert reloaded_item.error_type == "ValueError"
    assert reloaded_item.error_message == "Refusing loop value 1"
    assert ("outer_for", ()) not in session.finalized_loop_contexts
    for source_id in ("inner_return", "middle_return", "outer_return", "after"):
        assert not any(
            execution_id in session.results for execution_id in session.source_prepared_mapping.get(source_id, ())
        )

    runtime = session._generic_runtime()
    assert not runtime.gates
    assert not runtime.streams
    assert not runtime.continuations


def test_processor_sqlite_four_level_nested_for_failure_cleans_runtime(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    body_scheduler_seen: list[str] = []
    original_set_queue_item_session = queue.set_queue_item_session
    failed_persistence_snapshot: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]] | None = None

    def record_generic_scheduler_before_body(invocation, queue_item) -> None:
        if queue_item.session.prepared_source_mapping[invocation.id] == "body":
            assert isinstance(queue_item.session._execution_scheduler, _GenericGraphSchedulerAdapter)
            body_scheduler_seen.append(invocation.id)

    def capture_set_queue_item_session(item_id: int, session: GraphExecutionState):
        nonlocal failed_persistence_snapshot
        if session.has_error():
            assert session._generic_execution_runtime is not None
            runtime = session._generic_execution_runtime
            assert not runtime.gates
            assert not runtime.streams
            assert not runtime.continuations
            assert not session._ready_queues
            assert not session._ready_node_ids
            assert session._active_class is None
            assert session._generic_graph_scheduler is None
            assert not isinstance(session._execution_scheduler, _GenericGraphSchedulerAdapter)
            snapshot_session = session.model_copy(deep=True)
            snapshot_session._rehydrate_execution_refs()
            durable_snapshot = dump_execution_state(snapshot_session)
            failed_persistence_snapshot = (
                deepcopy(durable_snapshot["execution_refs"]),
                deepcopy(durable_snapshot["execution_tokens"]),
                deepcopy(durable_snapshot["execution_effects"]),
                deepcopy(session.errors),
            )
        return original_set_queue_item_session(item_id, session)

    monkeypatch.setattr(queue, "set_queue_item_session", capture_set_queue_item_session)

    graph = _build_four_level_nested_for_graph()
    graph.nodes["body"].fail_on = 1
    item_id = _insert_session(queue, graph, versioned=True)
    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(on_before_run_node_callbacks=[record_generic_scheduler_before_body]),
        polling_interval=0,
    )
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "failed")
    finally:
        _stop_processor(processor)

    assert body_scheduler_seen
    assert failed_persistence_snapshot is not None
    execution_refs_snapshot, execution_tokens_snapshot, execution_effects_snapshot, errors_snapshot = (
        failed_persistence_snapshot
    )
    assert execution_refs_snapshot == {}
    assert execution_tokens_snapshot == {}
    assert execution_effects_snapshot

    failed_item = queue.get_queue_item(item_id)
    assert failed_item.status == "failed"
    assert failed_item.error_type == "ValueError"
    assert failed_item.error_message == "Refusing loop value 1"
    assert queue.get_current("default") is None

    reloaded_queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    reloaded_queue.start(mock_invoker)
    reloaded_item = reloaded_queue.get_queue_item(item_id)
    assert reloaded_item.status == "failed"
    assert reloaded_queue.get_current("default") is None

    session = reloaded_item.session
    durable_snapshot = dump_execution_state(session)
    assert durable_snapshot["execution_refs"] == execution_refs_snapshot
    assert durable_snapshot["execution_tokens"] == execution_tokens_snapshot
    assert durable_snapshot["execution_effects"] == execution_effects_snapshot
    assert session.errors == errors_snapshot
    assert session.has_error()
    assert reloaded_item.error_type == "ValueError"
    assert reloaded_item.error_message == "Refusing loop value 1"

    expected_sources = {
        "outer_for": 1,
        "second_for": 1,
        "third_for": 1,
        "deepest_for": 1,
        "body": 1,
        "deepest_return": 1,
    }
    assert {source_id: len(execution_ids) for source_id, execution_ids in session.source_prepared_mapping.items()} == (
        expected_sources
    )
    assert {
        reference.source_node_id: reference.frame.iteration_path for reference in session.execution_refs.values()
    } == {
        "outer_for": (0,),
        "second_for": (0, 0),
        "third_for": (0, 0, 0),
        "deepest_for": (0, 0, 0, 0),
        "body": (0, 0, 0, 0),
        "deepest_return": (0, 0, 0, 0),
    }
    assert len(session.execution_refs) == 6
    assert session.execution_tokens == {}
    assert len(session.execution_effects) == 4
    assert all(
        reference.state_id == session.id
        and reference.frame.state_id == session.id
        and reference.frame.frame_id
        and reference.frame.iteration_path
        for reference in session.execution_refs.values()
    )
    assert len(session.errors) == 1
    assert next(iter(session.errors.values())) == "ValueError: Refusing loop value 1"
    assert ("outer_for", ()) not in session.finalized_loop_contexts
    for source_id in ("deepest_return", "third_return", "second_return", "outer_return", "after"):
        assert not any(
            execution_id in session.results for execution_id in session.source_prepared_mapping.get(source_id, ())
        )

    runtime = session._generic_runtime()
    assert not runtime.gates
    assert not runtime.streams
    assert not runtime.continuations


@pytest.mark.parametrize("outcome", ["success", "canceled", "failure"])
def test_processor_sqlite_queue_nested_for_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
    outcome: str,
) -> None:
    test_timeout = 30
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    inner_returns_seen = 0

    def cancel_after_first_inner_return(invocation, queue_item, output) -> None:
        nonlocal inner_returns_seen
        if queue_item.session.prepared_source_mapping[invocation.id] != "inner_return":
            return
        inner_returns_seen += 1
        if outcome == "canceled" and inner_returns_seen == 1:
            queue.cancel_queue_item(queue_item.item_id)

    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(on_after_run_node_callbacks=[cancel_after_first_inner_return]),
        polling_interval=0,
    )
    graph = _build_nested_for_graph(fail_on=3 if outcome == "failure" else None)
    item_id = _insert_session(queue, graph)
    status_handler_called = Event()
    original_status_handler = processor._on_queue_item_status_changed

    async def recording_status_handler(event) -> None:
        if event[1].item_id == item_id and event[1].status == "canceled":
            status_handler_called.set()
        await original_status_handler(event)

    processor._on_queue_item_status_changed = recording_status_handler  # type: ignore[method-assign]
    try:
        processor.start(mock_invoker)

        expected_status = {
            "success": "completed",
            "canceled": "canceled",
            "failure": "failed",
        }[outcome]
        assert registered_event_bus.wait_for_status(item_id, expected_status, timeout=test_timeout)

        queue_item = queue.get_queue_item(item_id)
        assert queue_item.status == expected_status
        assert queue.get_current("default") is None
        assert any(
            isinstance(event, QueueItemStatusChangedEvent)
            and event.item_id == item_id
            and event.status == expected_status
            for event in registered_event_bus._events
        )

        if outcome == "success":
            assert queue_item.session.is_complete()
            [after_exec_id] = queue_item.session.source_prepared_mapping["after"]
            assert queue_item.session.results[after_exec_id].collection == [[1, 2], [3, 4]]
            assert (
                len(
                    [
                        exec_id
                        for exec_id in queue_item.session.source_prepared_mapping["outer_return"]
                        if exec_id in queue_item.session.results
                    ]
                )
                == 2
            )
            assert inner_returns_seen == 4
        else:
            assert "after" not in queue_item.session.source_prepared_mapping
            assert ("outer_for", ()) not in queue_item.session.finalized_loop_contexts
            assert not any(
                getattr(queue_item.session.results.get(exec_id), "output_collection", [])
                for exec_id in queue_item.session.source_prepared_mapping.get("outer_for", [])
            )
            if outcome == "canceled":
                assert inner_returns_seen == 1
                assert not any(
                    exec_id in queue_item.session.results
                    for exec_id in queue_item.session.source_prepared_mapping.get("outer_return", [])
                )
                assert status_handler_called.wait(timeout=test_timeout)
                assert not queue_item.session.is_complete()
            else:
                assert queue_item.session.has_error()
                assert queue_item.error_type == "ValueError"
                assert queue_item.error_message == "Refusing loop value 3"
                assert (
                    len(
                        [
                            exec_id
                            for exec_id in queue_item.session.source_prepared_mapping["outer_return"]
                            if exec_id in queue_item.session.results
                        ]
                    )
                    == 1
                )
                assert inner_returns_seen == 2
    finally:
        _stop_processor(processor)


def test_processor_sqlite_iterate_collect_cancel_retry_does_not_leak_stream_state(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
) -> None:
    test_timeout = 30
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    item_id = _insert_session(queue, _build_iterate_collect_graph())
    session_persisted = Event()
    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_node_callbacks=[
                lambda invocation, queue_item, output: queue.cancel_queue_item(queue_item.item_id)
                if queue_item.session.prepared_source_mapping[invocation.id] == "iterate" and invocation.index == 0
                else None
            ],
            on_after_run_session_callbacks=[lambda queue_item: session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "canceled", timeout=test_timeout)
        assert session_persisted.wait(timeout=test_timeout)
    finally:
        _stop_processor(processor)

    canceled_item_before_retry = queue.get_queue_item(item_id)
    assert canceled_item_before_retry.status == "canceled"
    assert not canceled_item_before_retry.session.is_complete()
    assert "collect" not in canceled_item_before_retry.session.source_prepared_mapping
    [iterate_execution_id] = [
        exec_node_id
        for exec_node_id in canceled_item_before_retry.session.source_prepared_mapping["iterate"]
        if exec_node_id in canceled_item_before_retry.session.results
    ]
    assert canceled_item_before_retry.session.results[iterate_execution_id].index == 0
    assert sum(len(effects) for effects in canceled_item_before_retry.session.execution_effects.values()) == 1
    canceled_streams = [
        stream
        for stream in canceled_item_before_retry.session._generic_runtime().streams.values()
        if stream.owner_id == "iterate"
    ]
    assert len(canceled_streams) == 1
    assert canceled_streams[0].values == (0,)
    assert not canceled_streams[0].closed
    canceled_stream_id = canceled_streams[0].stream_id

    retry_result = queue.retry_items_by_id("default", [item_id])
    assert retry_result.retried_item_ids == [item_id]
    [retried_item] = [
        queue_item for queue_item in queue.list_all_queue_items("default") if queue_item.retried_from_item_id == item_id
    ]
    assert retried_item.status == "pending"
    assert retried_item.item_id != item_id
    assert retried_item.retried_from_item_id == item_id
    assert retried_item.session.id != canceled_item_before_retry.session.id
    assert retried_item.session.results == {}
    assert retried_item.session.execution_refs == {}
    assert retried_item.session.execution_tokens == {}
    assert retried_item.session.execution_effects == {}
    assert not retried_item.session._generic_runtime().streams

    retry_session_persisted = Event()
    retry_processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_session_callbacks=[lambda queue_item: retry_session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        retry_processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(retried_item.item_id, "completed", timeout=test_timeout)
        assert retry_session_persisted.wait(timeout=test_timeout)
    finally:
        _stop_processor(retry_processor)

    canceled_item = queue.get_queue_item(item_id)
    completed_item = queue.get_queue_item(retried_item.item_id)
    assert canceled_item.status == "canceled"
    assert completed_item.status == "completed"
    assert completed_item.session.is_complete()
    [collect_execution_id] = completed_item.session.source_prepared_mapping["collect"]
    assert completed_item.session.results[collect_execution_id].collection == [0, 1, 2]
    completed_streams = [
        stream for stream in completed_item.session._generic_runtime().streams.values() if stream.owner_id == "iterate"
    ]
    assert len(completed_streams) == 1
    assert completed_streams[0].values == (0, 1, 2)
    assert completed_streams[0].closed
    assert completed_streams[0].stream_id != canceled_stream_id
    assert completed_streams[0].stream_id.startswith(f"{completed_item.session.id}:iterate:")
    assert sum(len(effects) for effects in completed_item.session.execution_effects.values()) == 4


@pytest.mark.parametrize("cancel_after", ["source", "iterate"])
def test_processor_sqlite_direct_planner_cancel_retry_isolates_execution_state(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
    cancel_after: str,
) -> None:
    test_timeout = 30
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    item_id = _insert_session(queue, _build_direct_planner_iterate_body_collect_graph(), versioned=True)
    session_persisted = Event()

    def cancel_at_boundary(invocation, queue_item, output) -> None:
        source_id = queue_item.session.prepared_source_mapping[invocation.id]
        if source_id == cancel_after and (source_id != "iterate" or invocation.index == 0):
            queue.cancel_queue_item(queue_item.item_id)

    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_node_callbacks=[cancel_at_boundary],
            on_after_run_session_callbacks=[lambda queue_item: session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "canceled", timeout=test_timeout)
        assert session_persisted.wait(timeout=test_timeout)
    finally:
        _stop_processor(processor)

    canceled_item = queue.get_queue_item(item_id)
    assert canceled_item.status == "canceled"
    assert not canceled_item.session.is_complete()
    completed_sources = [
        canceled_item.session.prepared_source_mapping[execution_id] for execution_id in canceled_item.session.results
    ]
    if cancel_after == "source":
        assert completed_sources == ["source"]
        assert "iterate" not in canceled_item.session.source_prepared_mapping
        assert "body" not in canceled_item.session.source_prepared_mapping
        assert "collect" not in canceled_item.session.source_prepared_mapping
        assert not canceled_item.session._generic_runtime().streams
        canceled_stream_id = None
    else:
        assert completed_sources == ["source", "iterate"]
        assert len(canceled_item.session.source_prepared_mapping["iterate"]) == 3
        assert len(canceled_item.session.source_prepared_mapping["body"]) == 3
        assert len(canceled_item.session.source_prepared_mapping["collect"]) == 1
        assert not any(
            execution_id in canceled_item.session.results
            for execution_id in canceled_item.session.source_prepared_mapping["body"]
        )
        canceled_streams = [
            stream
            for stream in canceled_item.session._generic_runtime().streams.values()
            if stream.owner_id == "iterate"
        ]
        assert len(canceled_streams) == 1
        assert canceled_streams[0].values == (0,)
        assert not canceled_streams[0].closed
        canceled_stream_id = canceled_streams[0].stream_id

    retry_result = queue.retry_items_by_id("default", [item_id])
    assert retry_result.retried_item_ids == [item_id]
    retried_item = next(
        queue_item for queue_item in queue.list_all_queue_items("default") if queue_item.retried_from_item_id == item_id
    )
    assert retried_item.status == "pending"
    assert retried_item.item_id != item_id
    assert retried_item.session.id != canceled_item.session.id
    assert retried_item.session.results == {}
    assert retried_item.session.execution_refs == {}
    assert retried_item.session.execution_tokens == {}
    assert retried_item.session.execution_effects == {}
    assert not retried_item.session._generic_runtime().streams

    retry_session_persisted = Event()
    retry_processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_session_callbacks=[lambda queue_item: retry_session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        retry_processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(retried_item.item_id, "completed", timeout=test_timeout)
        assert retry_session_persisted.wait(timeout=test_timeout)
    finally:
        _stop_processor(retry_processor)

    completed_item = queue.get_queue_item(retried_item.item_id)
    assert queue.get_queue_item(item_id).status == "canceled"
    assert completed_item.status == "completed"
    assert completed_item.session.is_complete()
    completed_sources = [
        completed_item.session.prepared_source_mapping[execution_id] for execution_id in completed_item.session.results
    ]
    assert completed_sources[0] == "source"
    assert completed_sources.count("iterate") == 3
    assert completed_sources.count("body") == 3
    assert completed_sources[-1] == "collect"
    [collect_execution_id] = completed_item.session.source_prepared_mapping["collect"]
    assert completed_item.session.results[collect_execution_id].collection == [0, 1, 2]
    canceled_execution_ids = set(canceled_item.session.prepared_source_mapping)
    retried_execution_ids = set(completed_item.session.prepared_source_mapping)
    assert canceled_execution_ids.isdisjoint(retried_execution_ids)
    assert all(
        reference.frame.state_id == completed_item.session.id
        for reference in completed_item.session.execution_refs.values()
    )
    completed_streams = [
        stream for stream in completed_item.session._generic_runtime().streams.values() if stream.owner_id == "iterate"
    ]
    assert len(completed_streams) == 1
    assert completed_streams[0].values == (0, 1, 2)
    assert completed_streams[0].closed
    if canceled_stream_id is not None:
        assert completed_streams[0].stream_id != canceled_stream_id
    assert completed_streams[0].stream_id.startswith(f"{completed_item.session.id}:iterate:")


@pytest.mark.parametrize("force_compatibility_scheduler", [False, True])
def test_processor_sqlite_flat_for_cancel_retry_isolates_execution_state(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
    force_compatibility_scheduler: bool,
) -> None:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        _build_test_invocation_context,
    )
    if force_compatibility_scheduler:
        monkeypatch.setattr(GraphExecutionState, "_can_use_generic_for_scheduler", lambda self: False)

    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    item_id = _insert_session(queue, _build_flat_for_graph())
    session_persisted = Event()
    completed_sources: list[str] = []

    def cancel_after_first_return(invocation, queue_item, output) -> None:
        source_id = queue_item.session.prepared_source_mapping[invocation.id]
        completed_sources.append(source_id)
        if source_id == "return":
            queue.cancel_queue_item(queue_item.item_id)

    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_node_callbacks=[cancel_after_first_return],
            on_after_run_session_callbacks=[lambda queue_item: session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "canceled")
        assert session_persisted.wait(timeout=5)
    finally:
        _stop_processor(processor)

    canceled_item = queue.get_queue_item(item_id)
    assert canceled_item.status == "canceled"
    assert completed_sources == ["for", "body", "return"]
    assert not canceled_item.session.is_complete()
    assert "after" not in canceled_item.session.source_prepared_mapping
    assert not canceled_item.session.finalized_loop_contexts
    canceled_snapshot = dump_execution_state(canceled_item.session)
    if force_compatibility_scheduler:
        assert isinstance(canceled_item.session._execution_scheduler, _ExecutionScheduler)
    else:
        assert isinstance(canceled_item.session._execution_scheduler, _GenericGraphSchedulerAdapter)

    retry_result = queue.retry_items_by_id("default", [item_id])
    assert retry_result.retried_item_ids == [item_id]
    [retried_item] = [
        queue_item for queue_item in queue.list_all_queue_items("default") if queue_item.retried_from_item_id == item_id
    ]
    assert retried_item.status == "pending"
    assert retried_item.item_id != item_id
    assert retried_item.session.id != canceled_item.session.id
    assert retried_item.session.results == {}
    assert retried_item.session.execution_refs == {}
    assert retried_item.session.execution_tokens == {}
    assert retried_item.session.execution_effects == {}
    retry_runtime = retried_item.session._generic_runtime()
    assert not retry_runtime.gates
    assert not retry_runtime.streams
    assert not retry_runtime.continuations

    retry_session_persisted = Event()
    retry_processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_session_callbacks=[lambda queue_item: retry_session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        retry_processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(retried_item.item_id, "completed")
        assert retry_session_persisted.wait(timeout=5)
    finally:
        _stop_processor(retry_processor)

    canceled_item = queue.get_queue_item(item_id)
    completed_item = queue.get_queue_item(retried_item.item_id)
    assert canceled_item.status == "canceled"
    assert dump_execution_state(canceled_item.session) == canceled_snapshot
    assert completed_item.status == "completed"
    assert completed_item.session.id != canceled_item.session.id
    assert completed_item.session.is_complete()
    [after_execution_id] = completed_item.session.source_prepared_mapping["after"]
    assert completed_item.session.results[after_execution_id].collection == [0, 1, 2]
    assert all(
        reference.state_id == completed_item.session.id for reference in completed_item.session.execution_refs.values()
    )
