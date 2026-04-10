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


def _insert_queue_item(
    session_queue: SqliteSessionQueue,
    queue_id: str = "default",
    destination: str | None = None,
) -> int:
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt="test"))
    session = GraphExecutionState(graph=graph)
    session_json = session.model_dump_json(warnings=False, exclude_none=True)
    batch_id = str(uuid.uuid4())
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
                user_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (queue_id, session_json, session.id, batch_id, None, 0, None, None, destination, None, "system"),
        )
        return cursor.lastrowid


def test_status_sequence_increments_for_queue_item_lifecycle(
    session_queue: SqliteSessionQueue, mock_invoker: Invoker
) -> None:
    item_id = _insert_queue_item(session_queue)

    pending_item = session_queue.get_queue_item(item_id)
    assert pending_item.status == "pending"
    assert pending_item.status_sequence == 0

    in_progress_item = session_queue.dequeue()
    assert in_progress_item is not None
    assert in_progress_item.item_id == item_id
    assert in_progress_item.status == "in_progress"
    assert in_progress_item.status_sequence == 1

    completed_item = session_queue.complete_queue_item(item_id)
    assert completed_item.status == "completed"
    assert completed_item.status_sequence == 2

    event_bus: TestEventService = mock_invoker.services.events
    status_events = [event for event in event_bus.events if isinstance(event, QueueItemStatusChangedEvent)]

    assert len(status_events) == 2
    assert [event.status for event in status_events] == ["in_progress", "completed"]
    assert [event.status_sequence for event in status_events] == [1, 2]


def test_status_sequence_increments_for_bulk_cancel_paths(session_queue: SqliteSessionQueue) -> None:
    first_item_id = _insert_queue_item(session_queue)
    second_item_id = _insert_queue_item(session_queue)

    result = session_queue.cancel_all_except_current("default")

    assert result.canceled == 2
    assert session_queue.get_queue_item(first_item_id).status == "canceled"
    assert session_queue.get_queue_item(first_item_id).status_sequence == 1
    assert session_queue.get_queue_item(second_item_id).status == "canceled"
    assert session_queue.get_queue_item(second_item_id).status_sequence == 1
