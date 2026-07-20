"""Tests that bulk cancel/delete operations emit queue_items_canceled.

A bulk cancel (e.g. cancel-all-except-current) updates rows in a single SQL statement and emits no
per-item queue_item_status_changed events. Without a bulk event, other connected clients never
learn that pending items left the queue — an owner's badge kept showing stale counts after an
admin canceled their items. The queue_items_canceled event is that signal; these tests are the
regression tests for its emission.
"""

import uuid

import pytest

from invokeai.app.services.events.events_common import QueueItemsCanceledEvent
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


def _insert_queue_item(
    session_queue: SqliteSessionQueue,
    queue_id: str,
    user_id: str,
    batch_id: str | None = None,
    destination: str | None = None,
) -> int:
    """Directly insert a minimal pending queue item for the given user and return its item_id."""
    session_id = str(uuid.uuid4())
    batch_id = batch_id or str(uuid.uuid4())
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination, retried_from_item_id, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (queue_id, "{}", session_id, batch_id, None, 0, None, None, destination, None, user_id),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid


def _insert_dequeueable_queue_item(
    session_queue: SqliteSessionQueue,
    queue_id: str,
    user_id: str,
    destination: str | None = None,
) -> int:
    """Insert a pending queue item with a real parseable session, for tests that dequeue it."""
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt="test"))
    session = GraphExecutionState(graph=graph)
    session_json = session.model_dump_json(warnings=False, exclude_none=True)
    batch_id = str(uuid.uuid4())
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination, retried_from_item_id, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (queue_id, session_json, session.id, batch_id, None, 0, None, None, destination, None, user_id),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid


def _canceled_events(mock_invoker: Invoker) -> list[QueueItemsCanceledEvent]:
    return [e for e in mock_invoker.services.events.events if isinstance(e, QueueItemsCanceledEvent)]


def test_cancel_all_except_current_emits_queue_items_canceled_grouped_by_owner(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """An admin cancel (no user_id) affects every user's pending items; each owner's client needs
    its own item ids so its queue list and badge counts refetch."""
    a1 = _insert_queue_item(session_queue, "default", "user_a")
    a2 = _insert_queue_item(session_queue, "default", "user_a")
    b1 = _insert_queue_item(session_queue, "default", "user_b")

    result = session_queue.cancel_all_except_current("default")

    assert result.canceled == 3
    events = _canceled_events(mock_invoker)
    assert len(events) == 1
    event = events[0]
    assert event.queue_id == "default"
    assert event.canceled_item_ids_by_user == {"user_a": [a1, a2], "user_b": [b1]}
    assert sorted(event.canceled_item_ids) == sorted([a1, a2, b1])
    assert sorted(event.user_ids) == ["user_a", "user_b"]


def test_cancel_all_except_current_scoped_to_user_only_names_their_items(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """A non-admin cancel is scoped to the caller; other users' items are untouched and must not
    appear in the event."""
    a1 = _insert_queue_item(session_queue, "default", "user_a")
    _insert_queue_item(session_queue, "default", "user_b")

    result = session_queue.cancel_all_except_current("default", user_id="user_a")

    assert result.canceled == 1
    events = _canceled_events(mock_invoker)
    assert len(events) == 1
    assert events[0].canceled_item_ids_by_user == {"user_a": [a1]}


def test_delete_all_except_current_emits_queue_items_canceled(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """Bulk delete removes rows entirely — same notification requirement as bulk cancel."""
    a1 = _insert_queue_item(session_queue, "default", "user_a")
    b1 = _insert_queue_item(session_queue, "default", "user_b")

    result = session_queue.delete_all_except_current("default")

    assert result.deleted == 2
    events = _canceled_events(mock_invoker)
    assert len(events) == 1
    assert events[0].canceled_item_ids_by_user == {"user_a": [a1], "user_b": [b1]}


def test_no_event_when_nothing_was_canceled(session_queue: SqliteSessionQueue, mock_invoker: Invoker) -> None:
    """An empty result must not broadcast a pointless refetch signal to every client."""
    result = session_queue.cancel_all_except_current("default")

    assert result.canceled == 0
    assert _canceled_events(mock_invoker) == []


def test_cancel_by_batch_ids_emits_queue_items_canceled(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """Canceling batches bulk-updates pending rows with no per-item events; other clients need
    the bulk event to learn the items left the queue."""
    batch_id = str(uuid.uuid4())
    a1 = _insert_queue_item(session_queue, "default", "user_a", batch_id=batch_id)
    a2 = _insert_queue_item(session_queue, "default", "user_a", batch_id=batch_id)
    _insert_queue_item(session_queue, "default", "user_b")  # different batch - untouched

    result = session_queue.cancel_by_batch_ids("default", [batch_id])

    assert result.canceled == 2
    events = _canceled_events(mock_invoker)
    assert len(events) == 1
    assert events[0].canceled_item_ids_by_user == {"user_a": [a1, a2]}


def test_cancel_by_destination_emits_queue_items_canceled(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """Discarding a canvas staging area cancels by destination; the bulk event keeps other
    clients' queue badges in sync."""
    a1 = _insert_queue_item(session_queue, "default", "user_a", destination="canvas:sess-1")
    b1 = _insert_queue_item(session_queue, "default", "user_b", destination="canvas:sess-1")
    _insert_queue_item(session_queue, "default", "user_b", destination="other")

    result = session_queue.cancel_by_destination("default", "canvas:sess-1")

    assert result.canceled == 2
    events = _canceled_events(mock_invoker)
    assert len(events) == 1
    assert events[0].canceled_item_ids_by_user == {"user_a": [a1], "user_b": [b1]}


def test_delete_by_destination_emits_queue_items_canceled(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """Bulk delete by destination removes rows entirely - same notification requirement."""
    a1 = _insert_queue_item(session_queue, "default", "user_a", destination="canvas:sess-1")
    _insert_queue_item(session_queue, "default", "user_a", destination="other")

    result = session_queue.delete_by_destination("default", "canvas:sess-1")

    assert result.deleted == 1
    events = _canceled_events(mock_invoker)
    assert len(events) == 1
    assert events[0].canceled_item_ids_by_user == {"user_a": [a1]}


def test_delete_by_destination_excludes_separately_canceled_current_item(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """The in-progress current item matching the destination is canceled via cancel_queue_item,
    which emits its own per-item queue_item_status_changed - the bulk event must not signal the
    same item a second time. The row is still deleted and still counted in the returned total."""
    current_id = _insert_dequeueable_queue_item(session_queue, "default", "user_a", destination="canvas:sess-1")
    pending_id = _insert_queue_item(session_queue, "default", "user_a", destination="canvas:sess-1")

    in_progress = session_queue.dequeue()
    assert in_progress is not None and in_progress.item_id == current_id

    result = session_queue.delete_by_destination("default", "canvas:sess-1")

    assert result.deleted == 2
    events = _canceled_events(mock_invoker)
    assert len(events) == 1
    assert events[0].canceled_item_ids_by_user == {"user_a": [pending_id]}


def test_cancel_by_queue_id_emits_queue_items_canceled(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """Canceling a whole queue bulk-updates every user's non-terminal items."""
    a1 = _insert_queue_item(session_queue, "default", "user_a")
    b1 = _insert_queue_item(session_queue, "default", "user_b")

    result = session_queue.cancel_by_queue_id("default")

    assert result.canceled == 2
    events = _canceled_events(mock_invoker)
    assert len(events) == 1
    assert events[0].canceled_item_ids_by_user == {"user_a": [a1], "user_b": [b1]}
