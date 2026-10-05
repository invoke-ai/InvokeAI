"""Cursor pages of list_queue_items stay inside the requested queue, status and destination."""

import json
import uuid

import pytest
from pydantic_core import to_jsonable_python

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState

_SESSION_JSON = json.dumps(to_jsonable_python(GraphExecutionState(graph=Graph()).model_dump()))


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    return SqliteSessionQueue(db=mock_invoker.services.board_records._db)


def _insert(session_queue: SqliteSessionQueue, queue_id: str, status: str, destination: str, priority: int = 0) -> int:
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (queue_id, session, session_id, batch_id, priority, status, destination)
            VALUES (?, ?, ?, 'batch', ?, ?, ?);
            """,
            (queue_id, _SESSION_JSON, str(uuid.uuid4()), priority, status, destination),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def test_cursor_page_excludes_items_outside_the_filters(session_queue: SqliteSessionQueue) -> None:
    first = _insert(session_queue, "q1", "pending", "canvas", priority=1)
    second = _insert(session_queue, "q1", "pending", "canvas", priority=1)
    _insert(session_queue, "q2", "pending", "canvas", priority=1)
    _insert(session_queue, "q1", "completed", "canvas", priority=1)
    _insert(session_queue, "q1", "pending", "gallery", priority=1)
    lower = _insert(session_queue, "q1", "pending", "canvas", priority=0)
    _insert(session_queue, "q2", "pending", "canvas", priority=0)

    page = session_queue.list_queue_items(
        "q1", limit=10, priority=1, cursor=first, status="pending", destination="canvas"
    )

    assert [item.item_id for item in page.items] == [second, lower]
    assert not page.has_more
