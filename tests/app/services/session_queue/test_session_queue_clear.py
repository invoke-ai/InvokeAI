"""Tests for session queue clear() user_id scoping."""

import uuid

import pytest

from invokeai.app.services.events.events_common import QueueClearedEvent, QueueItemStatusChangedEvent
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.test_nodes import PromptTestInvocation


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    """Create a SqliteSessionQueue backed by the mock invoker's in-memory database."""
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


def _insert_queue_item(session_queue: SqliteSessionQueue, queue_id: str, user_id: str, status: str = "pending") -> int:
    """Directly insert a minimal queue item for the given user; returns its item_id.

    The session payload must be a valid GraphExecutionState: canceling an in-progress item
    re-hydrates the full SessionQueueItem row (see _set_queue_item_status).
    """
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt="test"))
    session = GraphExecutionState(graph=graph)
    batch_id = str(uuid.uuid4())
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination, retried_from_item_id, user_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                queue_id,
                session.model_dump_json(),
                session.id,
                batch_id,
                None,
                0,
                None,
                None,
                None,
                None,
                user_id,
                status,
            ),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid


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


def _status_of(session_queue: SqliteSessionQueue, item_id: int) -> str | None:
    with session_queue._db.transaction() as cursor:
        cursor.execute("SELECT status FROM session_queue WHERE item_id = ?", (item_id,))
        row = cursor.fetchone()
    return row[0] if row is not None else None


def test_user_scoped_clear_cancels_only_that_users_in_progress_items(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """With multiple workers, several items can be in_progress at once — one per user here.
    Alice's clear must cancel ALL of Alice's running items (each via its own status-changed
    event, which is what signals the worker running that exact item) and must leave Bob's
    running item untouched: neither canceled, nor deleted, nor abandoned (JPPhoto merge
    blocker, 2026-07-22)."""
    queue_id = "default"
    alice_running_1 = _insert_queue_item(session_queue, queue_id, "alice", status="in_progress")
    alice_running_2 = _insert_queue_item(session_queue, queue_id, "alice", status="in_progress")
    _insert_queue_item(session_queue, queue_id, "alice", status="pending")
    bob_running = _insert_queue_item(session_queue, queue_id, "bob", status="in_progress")

    events = mock_invoker.services.events
    events.events.clear()

    result = session_queue.clear(queue_id, user_id="alice")

    assert result.deleted == 3
    # Bob's running item is untouched: still present, still in_progress.
    assert _status_of(session_queue, bob_running) == "in_progress"
    # Each of Alice's running items was individually canceled (status-changed emitted) before
    # its row was deleted; Bob's was not.
    canceled_item_ids = {
        e.item_id for e in events.events if isinstance(e, QueueItemStatusChangedEvent) and e.status == "canceled"
    }
    assert {alice_running_1, alice_running_2} <= canceled_item_ids
    assert bob_running not in canceled_item_ids
    # The cleared event carries the scoping user id for the processor's worker filter.
    cleared_events = [e for e in events.events if isinstance(e, QueueClearedEvent)]
    assert [(e.queue_id, e.user_id) for e in cleared_events] == [(queue_id, "alice")]


def test_admin_clear_cancels_all_in_progress_items(session_queue: SqliteSessionQueue, mock_invoker: Invoker) -> None:
    """An unscoped (admin) clear cancels every running item before deleting the rows."""
    queue_id = "default"
    alice_running = _insert_queue_item(session_queue, queue_id, "alice", status="in_progress")
    bob_running = _insert_queue_item(session_queue, queue_id, "bob", status="in_progress")

    events = mock_invoker.services.events
    events.events.clear()

    result = session_queue.clear(queue_id)

    assert result.deleted == 2
    canceled_item_ids = {
        e.item_id for e in events.events if isinstance(e, QueueItemStatusChangedEvent) and e.status == "canceled"
    }
    assert {alice_running, bob_running} <= canceled_item_ids
    cleared_events = [e for e in events.events if isinstance(e, QueueClearedEvent)]
    assert [(e.queue_id, e.user_id) for e in cleared_events] == [(queue_id, None)]
