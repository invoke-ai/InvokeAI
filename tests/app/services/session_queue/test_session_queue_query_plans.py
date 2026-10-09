"""Queue refresh reads stay index-only, and point reads stay off the listing index.

Every session_queue column these reads use except queue_id is stored after the session blob, so a
plan that reads table rows walks each row's overflow pages. The database keeps no table statistics,
so the planner treats `queue_id = ?` as selective; reads that must cost their own rows rather than
the queue's history keep queue_id off the listing index.
"""

import re
from collections.abc import Callable
from functools import partial

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

    # The recent window reads the index in order with no sort in between, so LIMIT can stop the walk
    # (test_limited_newest_first_listing_stops_after_the_window shows that it does).
    recent_window = _plans(
        session_queue, lambda: session_queue.get_queue_item_ids("default", SQLiteDirection.Descending, limit=50)
    )
    assert len(recent_window) == 1
    assert LISTING_INDEX in recent_window[0]
    assert "TEMP B-TREE" not in recent_window[0]

    # Oldest first sorts only items that share a created_at, by item_id.
    oldest_first = _plans(session_queue, lambda: session_queue.get_queue_item_ids("default", SQLiteDirection.Ascending))
    assert len(oldest_first) == 1
    assert LISTING_INDEX in oldest_first[0]
    # SQLite words that partial sort differently by version; a full sort reads "FOR ORDER BY".
    partial_sort = re.compile(r"TEMP B-TREE FOR (LAST TERM|RIGHT PART) OF ORDER BY")
    assert "TEMP B-TREE" not in partial_sort.sub("", oldest_first[0])

    status = _plans(
        session_queue,
        lambda: session_queue.get_queue_status(
            "default", user_id="alice", acting_user_id="alice", origin_prefix="webv2:p:1:q:"
        ),
    )
    # The global and per-user counts come from one pass over the scope's history.
    counts = [plan for plan in status if "idx_session_queue_listing" in plan]
    assert len(counts) == 1, status
    assert LISTING_INDEX in counts[0]


def _vm_steps(session_queue: SqliteSessionQueue, read: Callable[[], object]) -> int:
    """Virtual machine steps (in units of 10) the service's statements execute during `read`."""
    conn = session_queue._db._conn
    steps = 0

    def count() -> int:
        nonlocal steps
        steps += 1
        return 0

    conn.set_progress_handler(count, 10)
    try:
        read()
    finally:
        conn.set_progress_handler(None, 0)
    return steps


def _append_history(session_queue: SqliteSessionQueue, start: int, count: int) -> None:
    with session_queue._db.transaction() as cursor:
        cursor.executemany(
            """--sql
            INSERT INTO session_queue (queue_id, session, session_id, batch_id, created_at)
            VALUES ('default', '{}', ?, 'batch', ?);
            """,
            [(f"session-{index}", f"2026-01-01 00:00:00.{index:06d}") for index in range(start, start + count)],
        )


def test_limited_newest_first_listing_stops_after_the_window(session_queue: SqliteSessionQueue) -> None:
    def recent_window() -> object:
        return session_queue.get_queue_item_ids("default", SQLiteDirection.Descending, limit=50)

    _append_history(session_queue, 0, 100)
    small = _vm_steps(session_queue, recent_window)
    _append_history(session_queue, 100, 4_900)
    large = _vm_steps(session_queue, recent_window)
    unlimited = _vm_steps(session_queue, lambda: session_queue.get_queue_item_ids("default"))

    # Fifty times the history costs the limited read about the same; the full listing shows the
    # measure does see history.
    assert large <= 2 * small, (small, large)
    assert unlimited >= 20 * large, (large, unlimited)


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


def test_bulk_cancel_and_clear_find_their_rows_through_the_status_indexes(session_queue: SqliteSessionQueue) -> None:
    """The rows a bulk cancel touches are the live ones, a handful next to the queue's history."""
    status_seek = "idx_session_queue_round_robin_pending (status=?"
    user_status_seek = "idx_session_queue_round_robin_pending (status=? AND user_id=?)"

    # The live-row select, the bulk UPDATE, and the in-progress lookup for the per-item cancels.
    whole_queue = _plans(session_queue, lambda: session_queue.cancel_by_queue_id("default"))
    assert len(whole_queue) == 3, whole_queue
    assert all(status_seek in plan for plan in whole_queue), whole_queue

    # A user's filter must neither keep the statements on the listing index nor move them onto a
    # user index that reads the user's whole history.
    user_scoped = _plans(
        session_queue,
        lambda: session_queue.cancel_by_queue_id("default", user_id="alice", origin_prefix="webv2:p:1:q:"),
    )
    assert len(user_scoped) == 3, user_scoped
    assert all(user_status_seek in plan for plan in user_scoped), user_scoped

    # After the current chain lookup: the live-row select and the bulk UPDATE or DELETE.
    for bulk in (session_queue.cancel_all_except_current, session_queue.delete_all_except_current):
        plans = _plans(session_queue, partial(bulk, "default", user_id="alice"))
        assert len(plans) == 4, plans
        assert all(user_status_seek in plan for plan in plans[2:]), plans

    # The in-progress lookup before the delete; the delete itself visits every row in scope.
    cleared = _plans(session_queue, lambda: session_queue.clear("default", user_id="alice"))
    assert len(cleared) == 2, cleared
    assert user_status_seek in cleared[0], cleared

    # The batch's rows come from the batch index; only the in-progress lookup seeks by status.
    batch = _plans(session_queue, lambda: session_queue.cancel_by_batch_ids("default", ["batch"], user_id="alice"))
    assert len(batch) == 3, batch
    assert all("idx_session_queue_batch_id (batch_id=?)" in plan for plan in batch[:2]), batch
    assert status_seek in batch[2], batch
