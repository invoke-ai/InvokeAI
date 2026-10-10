"""Queue listings stay inside their filters: cursor pages of list_queue_items, and the item id listing whose
limited form is the head of its full order."""

import uuid
from typing import Any

import pytest
from sqlalchemy import insert

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import QUEUE_ITEM_STATUS
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.session_queue import session_queue as session_queue_table
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.pagination import SQLiteDirection

_SESSION_JSON = GraphExecutionState(graph=Graph()).model_dump_json()


@pytest.fixture
def session_queue(mock_invoker: Invoker, database: Database) -> SessionQueue:
    queue = SessionQueue(database)
    queue.start(mock_invoker)
    return queue


def _insert_row(session_queue: SessionQueue, **values: Any) -> int:
    row = {"session": _SESSION_JSON, "session_id": str(uuid.uuid4()), "batch_id": "batch", **values}
    with session_queue._queries._database.begin(write=True) as conn:
        return int(conn.execute(insert(session_queue_table).values(row)).inserted_primary_key[0])


def _insert(
    session_queue: SessionQueue, queue_id: str, status: QUEUE_ITEM_STATUS, destination: str, priority: int = 0
) -> int:
    return _insert_row(session_queue, queue_id=queue_id, status=status, destination=destination, priority=priority)


def test_cursor_page_excludes_items_outside_the_filters(session_queue: SessionQueue) -> None:
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


PREFIX = "webv2:p:project-1:q:"
# (created_at, user_id, origin), inserted in this order. One enqueue writes its rows within the same
# millisecond, so equal timestamps are ordinary; the fourth row's clock is behind the third's.
LISTED_ROWS = [
    ("2026-10-01 10:00:00.000", "alice", f"{PREFIX}a"),
    ("2026-10-01 10:00:00.000", "bob", "webv2:q:b"),
    ("2026-10-01 10:00:00.000", "alice", f"{PREFIX}c"),
    ("2026-09-30 09:00:00.000", "bob", f"{PREFIX}d"),
    ("2026-10-01 10:00:05.000", "alice", "webv2:util:e"),
    ("2026-10-01 10:00:05.000", "bob", f"{PREFIX}f"),
    ("2026-10-01 10:00:05.000", "alice", f"{PREFIX}g"),
]


def _insert_listed(session_queue: SessionQueue, queue_id: str, created_at: str, user_id: str, origin: str) -> int:
    return _insert_row(session_queue, queue_id=queue_id, created_at=created_at, user_id=user_id, origin=origin)


@pytest.mark.parametrize(
    ("direction", "filters", "expected_rows"),
    [
        # Newest first; rows enqueued together keep their enqueue order in either direction.
        (SQLiteDirection.Descending, {}, [5, 6, 7, 1, 2, 3, 4]),
        (SQLiteDirection.Ascending, {}, [4, 1, 2, 3, 5, 6, 7]),
        (SQLiteDirection.Descending, {"origin_prefix": PREFIX}, [6, 7, 1, 3, 4]),
        (SQLiteDirection.Ascending, {"origin_prefix": PREFIX}, [4, 1, 3, 6, 7]),
        (SQLiteDirection.Descending, {"user_id": "alice"}, [5, 7, 1, 3]),
        (SQLiteDirection.Ascending, {"user_id": "alice"}, [1, 3, 5, 7]),
        (SQLiteDirection.Descending, {"user_id": "alice", "origin_prefix": PREFIX}, [7, 1, 3]),
        (SQLiteDirection.Ascending, {"user_id": "alice", "origin_prefix": PREFIX}, [1, 3, 7]),
    ],
)
def test_limited_item_ids_are_the_head_of_the_filtered_order(
    session_queue: SessionQueue,
    direction: SQLiteDirection,
    filters: dict[str, str],
    expected_rows: list[int],
) -> None:
    item_ids = [_insert_listed(session_queue, "default", *row) for row in LISTED_ROWS]
    # Newer than every listed row and matching every filter, but in another queue.
    _insert_listed(session_queue, "other", "2026-10-02 00:00:00.000", "alice", f"{PREFIX}z")
    expected = [item_ids[row - 1] for row in expected_rows]

    unlimited = session_queue.get_queue_item_ids("default", direction, **filters)
    assert unlimited.item_ids == expected
    assert unlimited.total_count == len(expected)

    for limit in range(1, len(expected) + 2):
        limited = session_queue.get_queue_item_ids("default", direction, **filters, limit=limit)
        assert limited.item_ids == expected[:limit], limit
        # The number of ids returned, not of every matching item.
        assert limited.total_count == min(limit, len(expected)), limit
