import asyncio
import uuid
from contextlib import contextmanager
from threading import Condition, Event
from typing import Iterator

import pytest
from fastapi_events.handlers.local import local_handler

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.primitives import BooleanInvocation
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.events.events_common import EventBase, QueueItemStatusChangedEvent
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
)
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.test_nodes import create_edge


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


class _Stats:
    @contextmanager
    def collect_stats(self, invocation, graph_execution_state_id):
        yield

    def log_stats(self, graph_execution_state_id) -> None:
        pass

    def reset_stats(self, graph_execution_state_id) -> None:
        pass


def _build_if_graph(*, condition: bool = True) -> Graph:
    graph = Graph()
    graph.add_node(BooleanInvocation(id="condition", value=condition))
    graph.add_node(AddInvocation(id="true_branch", a=2, b=3))
    graph.add_node(AddInvocation(id="false_branch", a=10, b=20))
    graph.add_node(IfInvocation(id="if"))
    graph.add_node(AddInvocation(id="sink", b=1))
    graph.add_edge(create_edge("condition", "value", "if", "condition"))
    graph.add_edge(create_edge("true_branch", "value", "if", "true_input"))
    graph.add_edge(create_edge("false_branch", "value", "if", "false_input"))
    graph.add_edge(create_edge("if", "value", "sink", "a"))
    return graph


def _insert_session(queue: SqliteSessionQueue, graph: Graph) -> int:
    session = GraphExecutionState(graph=graph)
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
                session.model_dump_json(warnings=False, exclude_none=True),
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


def _source_ids(session: GraphExecutionState, execution_ids: list[str] | set[str]) -> list[str]:
    return [session.prepared_source_mapping[execution_id] for execution_id in execution_ids]


def _assert_if_activation_token(
    session: GraphExecutionState,
    *,
    expected_field: str = "true_input",
) -> None:
    activation_tokens = [token for token in session.execution_tokens.values() if token.token_kind == "activation"]
    assert len(activation_tokens) == 1
    token = activation_tokens[0]
    [if_execution_id] = session.source_prepared_mapping["if"]
    assert token.owner_node_id == if_execution_id
    assert token.reference_id == session.execution_refs[if_execution_id].reference_id
    assert token.frame == session.execution_refs[if_execution_id].frame
    assert token.token_id == f"{token.reference_id}:activation:{expected_field}"
    assert token.token_id in session.execution_tokens
    assert token.port == expected_field
    assert token.value == expected_field


def _stop_processor(processor: DefaultSessionProcessor) -> None:
    processor.stop()
    for worker in processor._workers:
        assert worker.thread is not None
        worker.thread.join(timeout=5)
        assert not worker.thread.is_alive()


def _completed_source_ids(session: GraphExecutionState) -> list[str]:
    return _source_ids(session, session.results)


@pytest.fixture
def registered_event_bus() -> Iterator[_RecordingRegisteredEventService]:
    yield _RecordingRegisteredEventService()


@pytest.mark.parametrize(
    ("condition", "selected_branch", "unselected_branch", "selected_field"),
    [
        (True, "true_branch", "false_branch", "true_input"),
        (False, "false_branch", "true_branch", "false_input"),
    ],
)
def test_if_cancellation_preserves_activation_and_never_resumes_unselected_branch(
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
    condition: bool,
    selected_branch: str,
    unselected_branch: str,
    selected_field: str,
) -> None:
    test_timeout = 30
    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    processor: DefaultSessionProcessor | None = None
    item_id = _insert_session(queue, _build_if_graph(condition=condition))
    completed_sources: list[str] = []
    session_persisted = Event()

    def cancel_after_if(invocation, queue_item, output) -> None:
        source_id = queue_item.session.prepared_source_mapping[invocation.id]
        completed_sources.append(source_id)
        if source_id == "if":
            queue.cancel_queue_item(queue_item.item_id)

    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_node_callbacks=[cancel_after_if],
            on_after_run_session_callbacks=[lambda queue_item: session_persisted.set()],
        ),
        polling_interval=0,
    )
    try:
        processor.start(mock_invoker)
        assert registered_event_bus.wait_for_status(item_id, "canceled", timeout=test_timeout)
        assert session_persisted.wait(timeout=test_timeout)

        queue_item = queue.get_queue_item(item_id)
        assert queue_item.status == "canceled"
        assert queue.get_current("default") is None
        assert completed_sources == ["condition", selected_branch, "if"]
        assert _completed_source_ids(queue_item.session) == completed_sources
        assert unselected_branch not in completed_sources
        assert "sink" not in completed_sources
        assert not queue_item.session.is_complete()
        _assert_if_activation_token(queue_item.session, expected_field=selected_field)
    finally:
        _stop_processor(processor)


@pytest.mark.parametrize(
    (
        "condition",
        "selected_branch",
        "unselected_branch",
        "selected_field",
        "sink_value",
        "cancel_after_source",
    ),
    [
        (True, "true_branch", "false_branch", "true_input", 6, "if"),
        (False, "false_branch", "true_branch", "false_input", 31, "if"),
        (True, "true_branch", "false_branch", "true_input", 6, "true_branch"),
        (False, "false_branch", "true_branch", "false_input", 31, "false_branch"),
    ],
)
def test_if_retry_starts_fresh_and_preserves_selected_output_without_stale_activation(
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
    condition: bool,
    selected_branch: str,
    unselected_branch: str,
    selected_field: str,
    sink_value: int,
    cancel_after_source: str,
) -> None:
    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    item_id = _insert_session(queue, _build_if_graph(condition=condition))
    session_persisted = Event()
    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_node_callbacks=[
                lambda invocation, queue_item, output: queue.cancel_queue_item(queue_item.item_id)
                if queue_item.session.prepared_source_mapping[invocation.id] == cancel_after_source
                else None
            ],
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
    _assert_if_activation_token(canceled_item.session, expected_field=selected_field)
    retry_result = queue.retry_items_by_id("default", [item_id])
    assert retry_result.retried_item_ids == [item_id]

    retried_item = queue.dequeue()
    assert retried_item is not None
    assert retried_item.item_id != item_id
    assert retried_item.retried_from_item_id == item_id
    assert retried_item.session.id != canceled_item.session.id
    assert retried_item.session.results == {}
    assert retried_item.session.execution_tokens == {}
    assert retried_item.session.execution_refs == {}

    runner = DefaultSessionRunner()
    runner.start(services=mock_invoker.services, cancel_event=Event())
    runner.run(retried_item)

    completed_item = queue.get_queue_item(retried_item.item_id)
    assert queue.get_queue_item(item_id).status == "canceled"
    assert completed_item.status == "completed"
    assert completed_item.session.is_complete()
    completed_sources = _completed_source_ids(completed_item.session)
    assert completed_sources == ["condition", selected_branch, "if", "sink"]
    assert unselected_branch not in completed_sources
    [sink_execution_id] = completed_item.session.source_prepared_mapping["sink"]
    assert completed_item.session.results[sink_execution_id].value == sink_value
    _assert_if_activation_token(completed_item.session, expected_field=selected_field)


@pytest.mark.parametrize(
    ("condition", "selected_branch", "unselected_branch", "selected_field"),
    [
        (True, "true_branch", "false_branch", "true_input"),
        (False, "false_branch", "true_branch", "false_input"),
    ],
)
def test_if_early_cancellation_before_resolution_does_not_run_any_branch_continuation(
    mock_invoker: Invoker,
    registered_event_bus: _RecordingRegisteredEventService,
    condition: bool,
    selected_branch: str,
    unselected_branch: str,
    selected_field: str,
) -> None:
    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    mock_invoker.services.events = registered_event_bus
    mock_invoker.services.session_queue = queue
    mock_invoker.services.performance_statistics = _Stats()
    queue.start(mock_invoker)

    item_id = _insert_session(queue, _build_if_graph(condition=condition))
    session_persisted = Event()
    completed_sources: list[str] = []

    def cancel_after_selected(invocation, queue_item, output) -> None:
        source_id = queue_item.session.prepared_source_mapping[invocation.id]
        completed_sources.append(source_id)
        if source_id == selected_branch:
            queue.cancel_queue_item(queue_item.item_id)

    processor = DefaultSessionProcessor(
        session_runner=DefaultSessionRunner(
            on_after_run_node_callbacks=[cancel_after_selected],
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
    assert queue.get_current("default") is None
    assert completed_sources == ["condition", selected_branch]
    assert _completed_source_ids(canceled_item.session) == completed_sources
    assert unselected_branch not in completed_sources
    assert "if" not in completed_sources
    assert "sink" not in completed_sources
    assert "if" not in _completed_source_ids(canceled_item.session)
    _assert_if_activation_token(canceled_item.session, expected_field=selected_field)
    assert not canceled_item.session.is_complete()
