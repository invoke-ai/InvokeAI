"""Tests for list_queue_items() cursor pagination staying inside the requested scope."""

import uuid
from typing import Optional

import pytest
from sqlalchemy import insert

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import QUEUE_ITEM_STATUS
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.session_queue import session_queue as session_queue_table
from invokeai.app.services.shared.graph import Graph, GraphExecutionState

_SESSION_JSON = GraphExecutionState(graph=Graph()).model_dump_json()
# Far more pages than any test here inserts rows for. A cursor that stops advancing must fail the
# test by name instead of paging forever.
_MAX_PAGES = 50


@pytest.fixture
def session_queue(mock_invoker: Invoker, database: Database) -> SessionQueue:
    queue = SessionQueue(database)
    queue.start(mock_invoker)
    return queue


def _insert(
    database: Database,
    queue_id: str,
    status: QUEUE_ITEM_STATUS = "pending",
    destination: str = "canvas",
    priority: int = 0,
) -> int:
    values = {
        "queue_id": queue_id,
        "session": _SESSION_JSON,
        "session_id": str(uuid.uuid4()),
        "batch_id": str(uuid.uuid4()),
        "priority": priority,
        "destination": destination,
        "status": status,
    }
    with database.begin(write=True) as conn:
        return int(conn.execute(insert(session_queue_table).values(values)).inserted_primary_key[0])


def _walk_pages(
    session_queue: SessionQueue,
    queue_id: str,
    limit: int,
    status: Optional[QUEUE_ITEM_STATUS] = None,
    destination: Optional[str] = None,
) -> list[int]:
    """Follow the cursor the way a client does: resume after the last item of the previous page."""
    item_ids: list[int] = []
    cursor: Optional[int] = None
    priority = 0
    for _ in range(_MAX_PAGES):
        page = session_queue.list_queue_items(
            queue_id=queue_id, limit=limit, priority=priority, cursor=cursor, status=status, destination=destination
        )
        item_ids.extend(item.item_id for item in page.items)
        if not page.has_more:
            return item_ids
        cursor = page.items[-1].item_id
        priority = page.items[-1].priority
    pytest.fail(f"list_queue_items did not finish paging within {_MAX_PAGES} pages; collected {item_ids[:20]}")


def test_later_pages_stay_inside_the_requested_scope(session_queue: SessionQueue, database: Database) -> None:
    in_scope: list[int] = []
    for _ in range(4):
        in_scope.append(_insert(database, "default", destination="canvas"))
        # Same priority and a higher item_id than every in-scope row before them, so a cursor
        # predicate that escapes the filters would pull each of these onto the next page.
        _insert(database, "other_queue", destination="canvas")
        _insert(database, "default", status="completed", destination="canvas")
        _insert(database, "default", destination="gallery")

    item_ids = _walk_pages(session_queue, "default", limit=2, status="pending", destination="canvas")

    assert item_ids == in_scope


def test_pages_continue_across_priority_levels(session_queue: SessionQueue, database: Database) -> None:
    low = [_insert(database, "default", priority=0) for _ in range(3)]
    high = [_insert(database, "default", priority=10) for _ in range(3)]
    _insert(database, "other_queue", priority=10)
    _insert(database, "other_queue", priority=0)

    item_ids = _walk_pages(session_queue, "default", limit=2)

    # Highest priority first, oldest first within a priority, each item exactly once.
    assert item_ids == high + low
