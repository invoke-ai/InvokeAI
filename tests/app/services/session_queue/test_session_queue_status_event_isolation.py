"""Regression tests for the cross-user identifier leak in QueueItemStatusChangedEvent.

When user A's queue item changes status while user B's item is currently in_progress,
the embedded SessionQueueStatus inside the event must NOT expose B's item_id,
session_id, or batch_id. The full event ships to user:{A.user_id} and admin rooms,
so unredacted fields would let owner A learn user B's identifiers.
"""

import uuid

import pytest

from invokeai.app.services.events.events_common import QueueItemStatusChangedEvent
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.test_nodes import PromptTestInvocation, TestEventService


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


def _insert_queue_item(session_queue: SqliteSessionQueue, user_id: str) -> int:
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt="test"))
    session = GraphExecutionState(graph=graph)
    session_json = session.model_dump_json(warnings=False, exclude_none=True)
    batch_id = str(uuid.uuid4())
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, field_values,
                priority, workflow, origin, destination, retried_from_item_id, user_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("default", session_json, session.id, batch_id, None, 0, None, None, None, None, user_id),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def _last_status_event_for_item(event_bus: TestEventService, item_id: int) -> QueueItemStatusChangedEvent:
    matches = [
        e for e in event_bus.events if isinstance(e, QueueItemStatusChangedEvent) and e.item_id == item_id
    ]
    assert matches, f"No QueueItemStatusChangedEvent found for item {item_id}"
    return matches[-1]


def test_event_redacts_other_users_current_item_identifiers(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """When user A's pending item is canceled while user B's item is in_progress, the
    embedded queue_status in A's status-changed event must not expose B's identifiers."""
    user_a = "user-a"
    user_b = "user-b"

    a_item_id = _insert_queue_item(session_queue, user_id=user_a)
    b_item_id = _insert_queue_item(session_queue, user_id=user_b)

    # Make user B's item the in-progress one. We must dequeue B first; FIFO would dequeue A
    # because it was inserted first, so reverse the insertion: cancel A's, re-insert as new.
    # Simpler: dequeue twice. First dequeue picks A (older); promote B by inserting in
    # right order means we need B to be the in_progress item when A's event fires.
    # Cancel A first to make it ineligible, then dequeue B.
    # Actually we need A to be pending when its status changes — so we must dequeue B first.
    # Re-do: insert B BEFORE A by temporarily inserting A second. Recreate cleanly:
    session_queue.delete_queue_item(a_item_id)
    session_queue.delete_queue_item(b_item_id)
    b_item_id = _insert_queue_item(session_queue, user_id=user_b)
    a_item_id = _insert_queue_item(session_queue, user_id=user_a)

    in_progress = session_queue.dequeue()
    assert in_progress is not None and in_progress.item_id == b_item_id
    assert in_progress.user_id == user_b

    event_bus: TestEventService = mock_invoker.services.events
    event_bus.events.clear()

    # Now cancel user A's pending item. The emitted event for A must not leak B's
    # current-item identifiers via the embedded queue_status.
    canceled = session_queue.cancel_queue_item(a_item_id)
    assert canceled.user_id == user_a

    a_event = _last_status_event_for_item(event_bus, a_item_id)
    assert a_event.user_id == user_a
    assert a_event.queue_status.item_id is None, "must not leak other user's current item_id"
    assert a_event.queue_status.session_id is None, "must not leak other user's current session_id"
    assert a_event.queue_status.batch_id is None, "must not leak other user's current batch_id"
    # Aggregate counts in the embedded status are global and OK to share.
    assert a_event.queue_status.in_progress == 1
    assert a_event.queue_status.canceled == 1


def test_event_preserves_owner_current_item_identifiers(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """When the current in-progress item belongs to the same user as the changed item, the
    embedded queue_status must continue to expose the identifiers (no over-redaction)."""
    user_a = "user-a"

    a_item_id = _insert_queue_item(session_queue, user_id=user_a)

    in_progress = session_queue.dequeue()
    assert in_progress is not None and in_progress.item_id == a_item_id

    event_bus: TestEventService = mock_invoker.services.events
    event_bus.events.clear()

    completed = session_queue.complete_queue_item(a_item_id)
    assert completed.user_id == user_a

    # The event for A's transition fires AFTER the row is marked completed, so by the time
    # _set_queue_item_status reads get_current it returns None — there is no in-progress
    # item to leak. queue_status fields should therefore be None.
    a_event = _last_status_event_for_item(event_bus, a_item_id)
    assert a_event.user_id == user_a
    assert a_event.queue_status.item_id is None  # no in-progress item at all
    assert a_event.queue_status.completed == 1


def test_event_preserves_identifiers_when_current_item_is_the_changed_item(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    """The dequeue() transition makes the changed item itself the in-progress current item.
    queue_status must expose its identifiers since they belong to the event's owner."""
    user_a = "user-a"
    a_item_id = _insert_queue_item(session_queue, user_id=user_a)

    event_bus: TestEventService = mock_invoker.services.events
    event_bus.events.clear()

    in_progress = session_queue.dequeue()
    assert in_progress is not None and in_progress.item_id == a_item_id

    a_event = _last_status_event_for_item(event_bus, a_item_id)
    assert a_event.status == "in_progress"
    assert a_event.user_id == user_a
    # Current item == changed item == owned by user_a → no redaction
    assert a_event.queue_status.item_id == a_item_id
    assert a_event.queue_status.session_id == in_progress.session_id
    assert a_event.queue_status.batch_id == in_progress.batch_id
