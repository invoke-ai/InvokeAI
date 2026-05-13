"""Tests for session queue clear() user_id scoping."""

import uuid

import pytest

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    """Create a SqliteSessionQueue backed by the mock invoker's in-memory database."""
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


def _insert_queue_item(session_queue: SqliteSessionQueue, queue_id: str, user_id: str) -> None:
    """Directly insert a minimal queue item for the given user."""
    session_id = str(uuid.uuid4())
    batch_id = str(uuid.uuid4())
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination, retried_from_item_id, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (queue_id, "{}", session_id, batch_id, None, 0, None, None, None, None, user_id),
        )


def _count_items(session_queue: SqliteSessionQueue, queue_id: str, user_id: str | None = None) -> int:
    """Count items in the queue, optionally filtered by user_id."""
    with session_queue._db.transaction() as cursor:
        if user_id is not None:
            cursor.execute(
                "SELECT COUNT(*) FROM session_queue WHERE queue_id = ? AND user_id = ?",
                (queue_id, user_id),
            )
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM session_queue WHERE queue_id = ?",
                (queue_id,),
            )
        return cursor.fetchone()[0]


def test_clear_with_user_id_only_deletes_own_items(session_queue: SqliteSessionQueue) -> None:
    """Non-admin clear (user_id provided) should only remove that user's items."""
    queue_id = "default"
    user_a = "user_a"
    user_b = "user_b"

    _insert_queue_item(session_queue, queue_id, user_a)
    _insert_queue_item(session_queue, queue_id, user_a)
    _insert_queue_item(session_queue, queue_id, user_b)

    result = session_queue.clear(queue_id, user_id=user_a)

    assert result.deleted == 2
    assert _count_items(session_queue, queue_id, user_a) == 0
    assert _count_items(session_queue, queue_id, user_b) == 1


def test_clear_without_user_id_deletes_all_items(session_queue: SqliteSessionQueue) -> None:
    """Admin clear (no user_id) should remove all items in the queue."""
    queue_id = "default"

    _insert_queue_item(session_queue, queue_id, "user_a")
    _insert_queue_item(session_queue, queue_id, "user_b")
    _insert_queue_item(session_queue, queue_id, "user_c")

    result = session_queue.clear(queue_id)

    assert result.deleted == 3
    assert _count_items(session_queue, queue_id) == 0


def test_clear_with_user_id_does_not_affect_other_queues(session_queue: SqliteSessionQueue) -> None:
    """Clearing one queue should not affect items in another queue."""
    queue_a = "queue_a"
    queue_b = "queue_b"
    user_id = "user_x"

    _insert_queue_item(session_queue, queue_a, user_id)
    _insert_queue_item(session_queue, queue_b, user_id)

    result = session_queue.clear(queue_a, user_id=user_id)

    assert result.deleted == 1
    assert _count_items(session_queue, queue_a) == 0
    assert _count_items(session_queue, queue_b) == 1


def test_clear_returns_zero_when_no_matching_items(session_queue: SqliteSessionQueue) -> None:
    """Clear should return 0 deleted when there are no items for the given user."""
    queue_id = "default"

    _insert_queue_item(session_queue, queue_id, "user_b")

    result = session_queue.clear(queue_id, user_id="user_a")

    assert result.deleted == 0
    assert _count_items(session_queue, queue_id) == 1
