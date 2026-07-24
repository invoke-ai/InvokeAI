"""Regression tests for multi-GPU bulk cancellation.

With one session-processor worker per device, several queue items can be `in_progress` at the same
time. The bulk-cancel APIs must cancel ALL matching in-progress items (each emitting a cancel event
so its worker stops), not just the single `get_current()` item. See JPPhoto's review on PR #9263.
"""

import uuid

import pytest

from invokeai.app.services.events.events_common import QueueItemStatusChangedEvent
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItemNotFoundError
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from tests.test_nodes import PromptTestInvocation, TestEventService


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


def _insert(
    session_queue: SqliteSessionQueue,
    batch_id: str,
    destination: str | None = None,
    user_id: str = "system",
    queue_id: str = "default",
    parent_item_id: int | None = None,
    status: str = "pending",
) -> int:
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt="test"))
    session = GraphExecutionState(graph=graph)
    session_json = session.model_dump_json(warnings=False, exclude_none=True)
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, field_values, priority,
                workflow, origin, destination, retried_from_item_id, user_id, parent_item_id, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                queue_id,
                session_json,
                session.id,
                batch_id,
                None,
                0,
                None,
                None,
                destination,
                None,
                user_id,
                parent_item_id,
                status,
            ),
        )
        return cursor.lastrowid


def _canceled_event_item_ids(mock_invoker: Invoker) -> set[int]:
    event_bus: TestEventService = mock_invoker.services.events
    return {
        e.item_id for e in event_bus.events if isinstance(e, QueueItemStatusChangedEvent) and e.status == "canceled"
    }


def _dequeue_two_on_separate_devices(session_queue: SqliteSessionQueue) -> tuple[int, int]:
    a = session_queue.dequeue(device="cuda:0")
    b = session_queue.dequeue(device="cuda:1")
    assert a is not None and b is not None
    assert a.status == "in_progress" and b.status == "in_progress"
    return a.item_id, b.item_id


def test_cancel_by_batch_ids_cancels_all_in_progress(session_queue: SqliteSessionQueue, mock_invoker: Invoker):
    batch_id = str(uuid.uuid4())
    _insert(session_queue, batch_id=batch_id)
    _insert(session_queue, batch_id=batch_id)
    id_a, id_b = _dequeue_two_on_separate_devices(session_queue)

    result = session_queue.cancel_by_batch_ids("default", [batch_id])

    assert result.canceled == 2
    assert session_queue.get_queue_item(id_a).status == "canceled"
    assert session_queue.get_queue_item(id_b).status == "canceled"
    # Each worker must have received a cancel event for its item.
    assert {id_a, id_b} <= _canceled_event_item_ids(mock_invoker)


def test_cancel_by_destination_cancels_all_in_progress(session_queue: SqliteSessionQueue, mock_invoker: Invoker):
    _insert(session_queue, batch_id=str(uuid.uuid4()), destination="canvas")
    _insert(session_queue, batch_id=str(uuid.uuid4()), destination="canvas")
    id_a, id_b = _dequeue_two_on_separate_devices(session_queue)

    result = session_queue.cancel_by_destination("default", "canvas")

    assert result.canceled == 2
    assert session_queue.get_queue_item(id_a).status == "canceled"
    assert session_queue.get_queue_item(id_b).status == "canceled"
    assert {id_a, id_b} <= _canceled_event_item_ids(mock_invoker)


def test_cancel_by_queue_id_cancels_all_in_progress(session_queue: SqliteSessionQueue, mock_invoker: Invoker):
    _insert(session_queue, batch_id=str(uuid.uuid4()))
    _insert(session_queue, batch_id=str(uuid.uuid4()))
    id_a, id_b = _dequeue_two_on_separate_devices(session_queue)

    result = session_queue.cancel_by_queue_id("default")

    assert result.canceled == 2
    assert session_queue.get_queue_item(id_a).status == "canceled"
    assert session_queue.get_queue_item(id_b).status == "canceled"
    assert {id_a, id_b} <= _canceled_event_item_ids(mock_invoker)


def test_delete_by_destination_cancels_all_in_progress(session_queue: SqliteSessionQueue, mock_invoker: Invoker):
    """delete_by_destination must signal every running worker (not just get_current()) before
    deleting their rows, or the un-canceled workers keep running and then fail to update a deleted
    row."""
    _insert(session_queue, batch_id=str(uuid.uuid4()), destination="canvas")
    _insert(session_queue, batch_id=str(uuid.uuid4()), destination="canvas")
    id_a, id_b = _dequeue_two_on_separate_devices(session_queue)

    result = session_queue.delete_by_destination("default", "canvas")

    assert result.deleted == 2
    # Both in-progress workers were signaled to cancel before deletion.
    assert {id_a, id_b} <= _canceled_event_item_ids(mock_invoker)
    # Rows are gone.
    for item_id in (id_a, id_b):
        with pytest.raises(SessionQueueItemNotFoundError):
            session_queue.get_queue_item(item_id)


def _make_workflow_chain(session_queue: SqliteSessionQueue, device: str) -> tuple[int, int, int]:
    """Create a workflow-call chain mid-execution: a 'waiting' parent, a child dequeued to
    'in_progress' on `device`, and a 'pending' next child. Returns (parent, running, pending) ids."""
    parent_id = _insert(session_queue, batch_id=str(uuid.uuid4()), status="waiting")
    # Insert directly as in_progress with the device, rather than via dequeue(): dequeue picks the
    # globally best pending item, which in multi-chain setups may belong to another chain.
    running_child_id = _insert(session_queue, batch_id=str(uuid.uuid4()), parent_item_id=parent_id)
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            "UPDATE session_queue SET status = 'in_progress', device = ? WHERE item_id = ?",
            (device, running_child_id),
        )
    pending_child_id = _insert(session_queue, batch_id=str(uuid.uuid4()), parent_item_id=parent_id)
    return parent_id, running_child_id, pending_child_id


def test_cancel_all_except_current_preserves_every_active_workflow_chain(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
):
    """With one in-progress item per GPU, 'cancel all except current' must leave EVERY active
    worker's workflow-call chain intact — not just the chain of one arbitrarily selected item."""
    parent_a, running_a, pending_a = _make_workflow_chain(session_queue, "cuda:0")
    parent_b, running_b, pending_b = _make_workflow_chain(session_queue, "cuda:1")
    unrelated = _insert(session_queue, batch_id=str(uuid.uuid4()))

    result = session_queue.cancel_all_except_current("default")

    for item_id in (parent_a, parent_b):
        assert session_queue.get_queue_item(item_id).status == "waiting"
    for item_id in (pending_a, pending_b):
        assert session_queue.get_queue_item(item_id).status == "pending"
    for item_id in (running_a, running_b):
        assert session_queue.get_queue_item(item_id).status == "in_progress"
    assert session_queue.get_queue_item(unrelated).status == "canceled"
    assert result.canceled == 1


def test_delete_all_except_current_preserves_every_active_workflow_chain(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
):
    parent_a, running_a, pending_a = _make_workflow_chain(session_queue, "cuda:0")
    parent_b, running_b, pending_b = _make_workflow_chain(session_queue, "cuda:1")
    unrelated = _insert(session_queue, batch_id=str(uuid.uuid4()))

    result = session_queue.delete_all_except_current("default")

    for item_id in (parent_a, running_a, pending_a, parent_b, running_b, pending_b):
        # Still present (get_queue_item raises for deleted rows) with its status untouched.
        assert session_queue.get_queue_item(item_id).status in ("waiting", "in_progress", "pending")
    with pytest.raises(SessionQueueItemNotFoundError):
        session_queue.get_queue_item(unrelated)
    assert result.deleted == 1


def test_bulk_cancel_tolerates_rows_deleted_mid_cancel(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
):
    """A concurrent clear/delete can remove an in-progress row between _cancel_in_progress_matching's
    id SELECT and its per-item status update. The bulk cancel must complete instead of raising
    SessionQueueItemNotFoundError (which the router would surface as a 500)."""
    batch_id = str(uuid.uuid4())
    _insert(session_queue, batch_id=batch_id)
    _insert(session_queue, batch_id=batch_id)
    _dequeue_two_on_separate_devices(session_queue)

    original_set_status = session_queue._set_queue_item_status

    def delete_then_set_status(item_id: int, status: str, **kwargs):
        # Simulate the concurrent deletion landing just before the per-item cancel.
        with session_queue._db.transaction() as cursor:
            cursor.execute("DELETE FROM session_queue WHERE item_id = ?", (item_id,))
        return original_set_status(item_id, status, **kwargs)

    monkeypatch.setattr(session_queue, "_set_queue_item_status", delete_then_set_status)

    result = session_queue.cancel_by_batch_ids("default", [batch_id])

    # Both rows vanished before they could be canceled, so nothing was actually canceled — the
    # point is that the request completed.
    assert result.canceled == 0


def test_cancel_by_batch_ids_respects_user_scope(session_queue: SqliteSessionQueue, mock_invoker: Invoker):
    """A user-scoped cancel must not cancel another user's in-progress item in the same batch."""
    batch_id = str(uuid.uuid4())
    _insert(session_queue, batch_id=batch_id, user_id="alice")
    _insert(session_queue, batch_id=batch_id, user_id="bob")
    alice_item = session_queue.dequeue(device="cuda:0")
    bob_item = session_queue.dequeue(device="cuda:1")
    assert alice_item is not None and bob_item is not None

    result = session_queue.cancel_by_batch_ids("default", [batch_id], user_id="alice")

    assert result.canceled == 1
    assert session_queue.get_queue_item(alice_item.item_id).status == "canceled"
    assert session_queue.get_queue_item(bob_item.item_id).status == "in_progress"
    assert _canceled_event_item_ids(mock_invoker) == {alice_item.item_id}
