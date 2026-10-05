"""Queue refresh reads stay index-only, and point reads stay off the listing index.

Every session_queue column these reads use except queue_id is stored after the session blob, so a
plan that reads table rows walks each row's overflow pages. The database keeps no table statistics,
so the planner treats `queue_id = ?` as selective; reads that must cost their own rows rather than
the queue's history keep queue_id off the listing index.
"""

from collections.abc import Callable

import pytest

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import Batch
from invokeai.app.services.session_queue.session_queue_sqlite import (
    PENDING_QUEUE_ITEM_COUNT_QUERY,
    SqliteSessionQueue,
)
from invokeai.app.services.shared.graph import Graph
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from tests.test_nodes import PromptTestInvocation

LISTING_INDEX = "COVERING INDEX idx_session_queue_listing"


@pytest.fixture
def session_queue(mock_invoker: Invoker) -> SqliteSessionQueue:
    queue = SqliteSessionQueue(db=mock_invoker.services.board_records._db)
    queue.start(mock_invoker)
    return queue


def _plans(session_queue: SqliteSessionQueue, read: Callable[[], object]) -> list[str]:
    """The query plan of each statement `read` executes, as the service issued it."""
    conn = session_queue._db._conn
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        read()
    finally:
        conn.set_trace_callback(None)
    return [
        " | ".join(row["detail"] for row in conn.execute(f"EXPLAIN QUERY PLAN {statement}"))
        for statement in statements
        if "session_queue" in statement
    ]


def test_queue_listing_and_counts_read_only_the_listing_index(session_queue: SqliteSessionQueue) -> None:
    newest_first = _plans(
        session_queue,
        lambda: session_queue.get_queue_item_ids("default", SQLiteDirection.Descending, origin_prefix="webv2:p:1:q:"),
    )
    assert len(newest_first) == 1
    assert LISTING_INDEX in newest_first[0]
    assert "TEMP B-TREE" not in newest_first[0]

    # Oldest first sorts only items that share a created_at, by item_id.
    oldest_first = _plans(session_queue, lambda: session_queue.get_queue_item_ids("default", SQLiteDirection.Ascending))
    assert len(oldest_first) == 1
    assert LISTING_INDEX in oldest_first[0]
    assert "TEMP B-TREE" not in oldest_first[0].replace("TEMP B-TREE FOR LAST TERM OF ORDER BY", "")

    status = _plans(
        session_queue,
        lambda: session_queue.get_queue_status(
            "default", user_id="alice", acting_user_id="alice", origin_prefix="webv2:p:1:q:"
        ),
    )
    counts = [plan for plan in status if "idx_session_queue_listing" in plan]
    assert len(counts) == 2, status
    assert all(LISTING_INDEX in plan for plan in counts)


def test_point_reads_do_not_scan_the_queue_through_the_listing_index(session_queue: SqliteSessionQueue) -> None:
    # A handful of ids already tips an unguarded plan onto the listing index; a page of them is typical.
    summaries = _plans(
        session_queue, lambda: session_queue.get_queue_item_summaries_by_ids("default", list(range(1, 51)))
    )
    assert len(summaries) == 1
    assert "PRIMARY KEY" in summaries[0]

    with session_queue._db.transaction() as cursor:
        plan = cursor.execute(f"EXPLAIN QUERY PLAN {PENDING_QUEUE_ITEM_COUNT_QUERY}", ("default",)).fetchall()
    assert "idx_session_queue_listing" not in " | ".join(row["detail"] for row in plan)


def test_batch_lookups_stay_on_the_batch_index(session_queue: SqliteSessionQueue) -> None:
    """A batch is a handful of rows; a user_id index would read the user's whole history."""
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt="test"))
    enqueue = _plans(
        session_queue,
        lambda: session_queue._enqueue_batch("default", Batch(graph=graph, runs=2), False, "alice"),
    )
    # The batch-id uniqueness probe and the readback of the inserted item ids.
    batch_reads = [plan for plan in enqueue if "batch_id" in plan or "user_id" in plan]
    assert len(batch_reads) == 2, enqueue
    assert all("idx_session_queue_batch_id (batch_id=?)" in plan for plan in batch_reads), enqueue

    owner_status = _plans(session_queue, lambda: session_queue.get_batch_status("default", "batch", user_id="alice"))
    assert len(owner_status) == 1
    assert "idx_session_queue_batch_id (batch_id=?)" in owner_status[0]
