"""Tests for workflow-call retry semantics in the session queue."""

import uuid

import pytest

from invokeai.app.services.events.events_common import QueueItemsRetriedEvent
from invokeai.app.services.invoker import Invoker
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


def _insert_queue_item(
    session_queue: SqliteSessionQueue,
    *,
    session: GraphExecutionState,
    status: str,
    user_id: str,
    root_item_id: int | None = None,
    project_id: str | None = None,
    queue_id: str = "default",
) -> int:
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id,
                session,
                session_id,
                batch_id,
                field_values,
                priority,
                workflow,
                origin,
                destination,
                retried_from_item_id,
                user_id,
                workflow_call_id,
                parent_item_id,
                parent_session_id,
                root_item_id,
                workflow_call_depth,
                project_id,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                queue_id,
                session.model_dump_json(warnings=False),
                session.id,
                str(uuid.uuid4()),
                None,
                0,
                None,
                None,
                None,
                None,
                user_id,
                None,
                None,
                None,
                root_item_id,
                None,
                project_id,
                status,
            ),
        )
        return cursor.lastrowid


def test_retry_items_by_id_retries_root_once_for_child_chain_item(
    session_queue: SqliteSessionQueue, event_bus: TestEventService
) -> None:
    root_session = GraphExecutionState(graph=Graph())
    child_session = GraphExecutionState(graph=Graph())

    root_item_id = _insert_queue_item(session_queue, session=root_session, user_id="user-1", status="failed")
    child_item_id = _insert_queue_item(
        session_queue,
        session=child_session,
        user_id="user-1",
        status="failed",
        root_item_id=root_item_id,
    )

    retry_result = session_queue.retry_items_by_id("default", [child_item_id, root_item_id])

    assert retry_result.retried_item_ids == [root_item_id]

    all_items = session_queue.list_all_queue_items("default")
    retried_items = [item for item in all_items if item.retried_from_item_id == root_item_id]
    assert len(retried_items) == 1
    assert retried_items[0].status == "pending"
    assert retried_items[0].workflow_call_id is None
    assert retried_items[0].parent_item_id is None
    assert retried_items[0].root_item_id is None

    retry_events = [event for event in event_bus.events if isinstance(event, QueueItemsRetriedEvent)]
    assert len(retry_events) == 1
    assert retry_events[0].retried_item_ids == [root_item_id]
    assert retry_events[0].user_ids == ["user-1"]
    assert retry_events[0].retried_item_ids_by_user == {"user-1": [root_item_id]}


def test_retry_items_by_id_emits_unique_owner_ids_for_multiple_roots(
    session_queue: SqliteSessionQueue, event_bus: TestEventService
) -> None:
    first_root_item_id = _insert_queue_item(
        session_queue, session=GraphExecutionState(graph=Graph()), user_id="user-1", status="failed"
    )
    second_root_item_id = _insert_queue_item(
        session_queue, session=GraphExecutionState(graph=Graph()), user_id="user-2", status="canceled"
    )

    retry_result = session_queue.retry_items_by_id("default", [first_root_item_id, second_root_item_id])

    assert retry_result.retried_item_ids == [first_root_item_id, second_root_item_id]

    retry_events = [event for event in event_bus.events if isinstance(event, QueueItemsRetriedEvent)]
    assert len(retry_events) == 1
    assert retry_events[0].user_ids == ["user-1", "user-2"]
    assert retry_events[0].retried_item_ids_by_user == {
        "user-1": [first_root_item_id],
        "user-2": [second_root_item_id],
    }


def test_retried_items_inherit_the_project_of_the_root(
    session_queue: SqliteSessionQueue,
) -> None:
    root_item_id = _insert_queue_item(
        session_queue,
        session=GraphExecutionState(graph=Graph()),
        user_id="user-1",
        status="failed",
        project_id="p1",
    )
    session_queue.retry_items_by_id("default", [root_item_id])

    retried = [
        item for item in session_queue.list_all_queue_items("default") if item.retried_from_item_id == root_item_id
    ]
    assert [item.project_id for item in retried] == ["p1"]
