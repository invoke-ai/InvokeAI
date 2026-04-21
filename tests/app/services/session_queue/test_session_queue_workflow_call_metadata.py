"""Tests for workflow-call relationship metadata on session_queue items."""

import uuid

import pytest

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


def test_get_queue_item_round_trips_workflow_call_metadata(session_queue: SqliteSessionQueue) -> None:
    session = GraphExecutionState(graph=Graph())
    session_json = session.model_dump_json(warnings=False)

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
                workflow_call_depth
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                "user-1",
                "workflow-call-1",
                11,
                "parent-session-1",
                7,
                3,
            ),
        )
        item_id = cursor.lastrowid

    queue_item = session_queue.get_queue_item(item_id)

    assert queue_item.workflow_call_id == "workflow-call-1"
    assert queue_item.parent_item_id == 11
    assert queue_item.parent_session_id == "parent-session-1"
    assert queue_item.root_item_id == 7
    assert queue_item.workflow_call_depth == 3
