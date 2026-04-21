"""Tests for workflow-call retry semantics in the session queue."""

from datetime import datetime

import pytest

from invokeai.app.services.events.events_common import QueueItemsRetriedEvent
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.test_nodes import TestEventService


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


@pytest.fixture
def event_bus(mock_invoker: Invoker) -> TestEventService:
    assert isinstance(mock_invoker.services.events, TestEventService)
    return mock_invoker.services.events


def _build_queue_item(
    *,
    item_id: int,
    session: GraphExecutionState,
    user_id: str,
    status: str,
    root_item_id: int | None = None,
    retried_from_item_id: int | None = None,
) -> SessionQueueItem:
    now = datetime.now()
    return SessionQueueItem(
        item_id=item_id,
        status=status,
        priority=0,
        batch_id=f"batch-{item_id}",
        origin=None,
        destination=None,
        session_id=session.id,
        error_type=None,
        error_message=None,
        error_traceback=None,
        created_at=now,
        updated_at=now,
        started_at=None,
        completed_at=None,
        queue_id="default",
        user_id=user_id,
        user_display_name=None,
        user_email=None,
        field_values=None,
        retried_from_item_id=retried_from_item_id,
        workflow_call_id=None,
        parent_item_id=None,
        parent_session_id=None,
        root_item_id=root_item_id,
        workflow_call_depth=None,
        session=session,
        workflow=None,
    )


def test_retry_items_by_id_retries_root_once_for_child_chain_item(
    session_queue: SqliteSessionQueue, event_bus: TestEventService, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())

    root_item = _build_queue_item(item_id=10, session=root_session, user_id="user-1", status="failed")
    child_item = _build_queue_item(
        item_id=11,
        session=child_session,
        user_id="user-1",
        status="failed",
        root_item_id=root_item.item_id,
    )

    items = {root_item.item_id: root_item, child_item.item_id: child_item}
    monkeypatch.setattr(session_queue, "get_queue_item", lambda item_id: items[item_id])

    retry_result = session_queue.retry_items_by_id("default", [child_item.item_id, root_item.item_id])

    assert retry_result.retried_item_ids == [root_item.item_id]

    all_items = session_queue.list_all_queue_items("default")
    retried_items = [item for item in all_items if item.retried_from_item_id == root_item.item_id]
    assert len(retried_items) == 1
    assert retried_items[0].status == "pending"
    assert retried_items[0].workflow_call_id is None
    assert retried_items[0].parent_item_id is None
    assert retried_items[0].root_item_id is None

    retry_events = [event for event in event_bus.events if isinstance(event, QueueItemsRetriedEvent)]
    assert len(retry_events) == 1
    assert retry_events[0].retried_item_ids == [root_item.item_id]
    assert retry_events[0].user_ids == ["user-1"]


def test_retry_items_by_id_emits_unique_owner_ids_for_multiple_roots(
    session_queue: SqliteSessionQueue, event_bus: TestEventService, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root_item = _build_queue_item(
        item_id=20, session=GraphExecutionState(graph=Graph()), user_id="user-1", status="failed"
    )
    second_root_item = _build_queue_item(
        item_id=21, session=GraphExecutionState(graph=Graph()), user_id="user-2", status="canceled"
    )

    items = {
        first_root_item.item_id: first_root_item,
        second_root_item.item_id: second_root_item,
    }
    monkeypatch.setattr(session_queue, "get_queue_item", lambda item_id: items[item_id])

    retry_result = session_queue.retry_items_by_id("default", [first_root_item.item_id, second_root_item.item_id])

    assert retry_result.retried_item_ids == [first_root_item.item_id, second_root_item.item_id]

    retry_events = [event for event in event_bus.events if isinstance(event, QueueItemsRetriedEvent)]
    assert len(retry_events) == 1
    assert retry_events[0].user_ids == ["user-1", "user-2"]
