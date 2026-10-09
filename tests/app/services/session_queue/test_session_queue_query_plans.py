"""Queue refresh reads stay index-only, and point reads and bulk changes stay off the listing index.

Every session_queue column these reads use except queue_id is stored after the session blob, so a
plan that reads table rows walks each row's overflow pages. SQLite keeps no table statistics, so the
planner treats `queue_id = ?` as selective; statements that must cost their own rows rather than the
queue's history keep queue_id off the listing index. The plans are SQLite's: the listing index exists
only there.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest
from sqlalchemy import insert

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_common import Batch
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.session_queue import session_queue as session_queue_table
from invokeai.app.services.shared.graph import Graph
from invokeai.app.services.shared.pagination import SQLiteDirection
from tests.fixtures.database import capture_statements, explain_query_plan
from tests.test_nodes import PromptTestInvocation

pytestmark = pytest.mark.sqlite_only

LISTING_INDEX = "idx_session_queue_listing"
COVERING_LISTING_INDEX = f"COVERING INDEX {LISTING_INDEX}"
BATCH_SEEK = "idx_session_queue_batch_id (batch_id=?)"
STATUS_SEEK = "idx_session_queue_round_robin_pending (status=?"
USER_STATUS_SEEK = "idx_session_queue_round_robin_pending (status=? AND user_id=?)"
# A seek on any index that leads with the status: the live rows, never the queue's history.
ANY_STATUS_SEEK = re.compile(r"SEARCH session_queue USING (?:COVERING )?INDEX \w+ \(status=\?")
# The primary key, or the unique index on it, which SQLite may pick for a DELETE.
ID_SEEK = "rowid=?"
# A user filter SQLite may look up through an index (`+column` keeps one off every index).
INDEXED_USER_FILTER = re.compile(r"(?<!\+)session_queue\.user_id = \?")
ORIGIN_PREFIX = "webv2:p:1:q:"


@dataclass(frozen=True)
class Statement:
    sql: str
    parameters: Any
    plan: str

    @property
    def is_write(self) -> bool:
        return self.sql.startswith(("UPDATE", "DELETE"))

    def binds(self, value: str) -> bool:
        return value in self.parameters


@pytest.fixture
def session_queue(mock_invoker: Invoker, database: Database) -> SessionQueue:
    queue = SessionQueue(database)
    queue.start(mock_invoker)
    return queue


def _statements(database: Database, action: Callable[[], object]) -> list[Statement]:
    """Each statement on the queue's rows that `action` runs (its inserts left out), with SQLite's plan for it."""
    with capture_statements(database) as captured:
        action()
    statements = []
    for sql, parameters in captured:
        flat = " ".join(sql.split())
        if "FROM session_queue " not in flat and not flat.startswith("UPDATE session_queue "):
            continue
        statements.append(Statement(flat, parameters, " | ".join(explain_query_plan(database, sql, parameters))))
    assert statements, "the action ran no statement on session_queue"
    return statements


def _plans(statements: list[Statement]) -> list[str]:
    return [statement.plan for statement in statements]


def _enqueue(session_queue: SessionQueue) -> None:
    """Two pending items of alice's, so that a bulk change has rows to change."""
    graph = Graph()
    graph.add_node(PromptTestInvocation(id="prompt", prompt="test"))
    batch = Batch(graph=graph, runs=2, batch_id="batch", origin=f"{ORIGIN_PREFIX}canvas")
    session_queue._enqueue_batch("default", batch, False, "alice")


def test_queue_listing_and_counts_read_only_the_listing_index(session_queue: SessionQueue, database: Database) -> None:
    newest_first = _statements(
        database,
        lambda: session_queue.get_queue_item_ids("default", SQLiteDirection.Descending, origin_prefix=ORIGIN_PREFIX),
    )
    for plan in _plans(newest_first):
        assert COVERING_LISTING_INDEX in plan
        assert "TEMP B-TREE" not in plan

    # The recent window reads the index in order with no sort in between, so LIMIT can stop the walk
    # (test_limited_newest_first_listing_stops_after_the_window shows that it does).
    recent_window = _statements(
        database, lambda: session_queue.get_queue_item_ids("default", SQLiteDirection.Descending, limit=50)
    )
    for plan in _plans(recent_window):
        assert COVERING_LISTING_INDEX in plan
        assert "TEMP B-TREE" not in plan

    # Oldest first sorts only items that share a created_at, by item_id.
    oldest_first = _statements(database, lambda: session_queue.get_queue_item_ids("default", SQLiteDirection.Ascending))
    # SQLite words that partial sort differently by version; a full sort reads "FOR ORDER BY".
    partial_sort = re.compile(r"TEMP B-TREE FOR (LAST TERM|RIGHT PART) OF ORDER BY")
    for plan in _plans(oldest_first):
        assert COVERING_LISTING_INDEX in plan
        assert "TEMP B-TREE" not in partial_sort.sub("", plan)

    status = _statements(
        database,
        lambda: session_queue.get_queue_status(
            "default", user_id="alice", acting_user_id="alice", origin_prefix=ORIGIN_PREFIX
        ),
    )
    # The global and per-user counts come from one pass over the scope's history; the item in
    # progress is found by its status.
    counts = [plan for plan in _plans(status) if LISTING_INDEX in plan]
    assert len(counts) == 1, status
    assert COVERING_LISTING_INDEX in counts[0]
    assert counts[0].count("SEARCH session_queue") == 1, counts[0]


def _vm_steps(database: Database, read: Callable[[], object]) -> int:
    """Virtual machine steps (in units of 10) the statements executed during `read` take."""
    steps = 0

    def count() -> int:
        nonlocal steps
        steps += 1
        return 0

    with database.begin(write=False) as conn:
        driver = conn.connection.driver_connection
    assert driver is not None
    driver.set_progress_handler(count, 10)
    try:
        read()
    finally:
        driver.set_progress_handler(None, 0)
    return steps


def _append_history(database: Database, start: int, count: int) -> None:
    rows = [
        {
            "queue_id": "default",
            "session": "{}",
            "session_id": f"session-{index}",
            "batch_id": "batch",
            "created_at": f"2026-01-01 00:00:00.{index:06d}",
        }
        for index in range(start, start + count)
    ]
    with database.begin(write=True) as conn:
        conn.execute(insert(session_queue_table), rows)


def test_limited_newest_first_listing_stops_after_the_window(session_queue: SessionQueue, database: Database) -> None:
    def recent_window() -> object:
        return session_queue.get_queue_item_ids("default", SQLiteDirection.Descending, limit=50)

    _append_history(database, 0, 100)
    small = _vm_steps(database, recent_window)
    _append_history(database, 100, 4_900)
    large = _vm_steps(database, recent_window)
    unlimited = _vm_steps(database, lambda: session_queue.get_queue_item_ids("default"))

    # Fifty times the history costs the limited read about the same; the full listing shows the
    # measure does see history.
    assert large <= 2 * small, (small, large)
    assert unlimited >= 20 * large, (large, unlimited)


def test_point_reads_do_not_scan_the_queue_through_the_listing_index(
    session_queue: SessionQueue, database: Database
) -> None:
    # A handful of ids already tips an unguarded plan onto the listing index; a page of them is typical.
    summaries = _statements(
        database, lambda: session_queue.get_queue_item_summaries_by_ids("default", list(range(1, 51)))
    )
    for plan in _plans(summaries):
        assert "PRIMARY KEY" in plan
        assert LISTING_INDEX not in plan

    # The admission of every enqueue counts the pending backlog.
    enqueue = _statements(database, lambda: _enqueue(session_queue))
    pending_counts = [statement for statement in enqueue if "count(" in statement.sql]
    assert pending_counts, enqueue
    for statement in pending_counts:
        assert LISTING_INDEX not in statement.plan
        assert STATUS_SEEK in statement.plan


def test_batch_lookups_stay_on_the_batch_index(session_queue: SessionQueue, database: Database) -> None:
    """A batch is a handful of rows; a user_id index would read the user's whole history."""
    enqueue = _statements(database, lambda: _enqueue(session_queue))
    # The batch-id uniqueness probe and the readback of the inserted item ids.
    batch_reads = [statement for statement in enqueue if "batch_id" in statement.sql]
    assert len(batch_reads) == 2, enqueue
    assert all(BATCH_SEEK in statement.plan for statement in batch_reads), enqueue

    owner_status = _statements(database, lambda: session_queue.get_batch_status("default", "batch", user_id="alice"))
    for plan in _plans(owner_status):
        assert BATCH_SEEK in plan


def _assert_writes_go_by_id(statements: list[Statement]) -> None:
    writes = [statement for statement in statements if statement.is_write]
    assert writes, statements
    assert all(ID_SEEK in statement.plan for statement in writes), writes


def test_bulk_cancel_and_clear_find_their_rows_through_the_status_indexes(
    session_queue: SessionQueue, database: Database
) -> None:
    """The rows a bulk cancel touches are the live ones, a handful next to the queue's history."""

    def assert_rows_found_by_status(statements: list[Statement]) -> None:
        for statement in statements:
            assert LISTING_INDEX not in statement.plan, statements
            if statement.is_write:
                continue
            assert ANY_STATUS_SEEK.search(statement.plan), statement
            # A user's filter must not move the read onto a user index that reads the user's whole history.
            if INDEXED_USER_FILTER.search(statement.sql):
                assert USER_STATUS_SEEK in statement.plan, statement
        _assert_writes_go_by_id(statements)

    _enqueue(session_queue)
    whole_queue = _statements(database, lambda: session_queue.cancel_by_queue_id("default"))
    assert_rows_found_by_status(whole_queue)
    assert all(STATUS_SEEK in statement.plan for statement in whole_queue if not statement.is_write), whole_queue

    _enqueue(session_queue)
    user_scoped = _statements(
        database, lambda: session_queue.cancel_by_queue_id("default", user_id="alice", origin_prefix=ORIGIN_PREFIX)
    )
    assert_rows_found_by_status(user_scoped)
    assert all(USER_STATUS_SEEK in statement.plan for statement in user_scoped if not statement.is_write)

    # Beside the live rows, these look up the chains of the items running now, by their status.
    for bulk in (session_queue.cancel_all_except_current, session_queue.delete_all_except_current):
        _enqueue(session_queue)
        assert_rows_found_by_status(_statements(database, partial(bulk, "default", user_id="alice")))

    # The in-progress lookup seeks by status; the delete itself visits every row in scope.
    _enqueue(session_queue)
    cleared = _statements(database, lambda: session_queue.clear("default", user_id="alice"))
    in_progress = [statement for statement in cleared if statement.binds("in_progress")]
    assert in_progress, cleared
    assert all(USER_STATUS_SEEK in statement.plan for statement in in_progress), cleared
    assert all(LISTING_INDEX not in statement.plan for statement in cleared), cleared
    _assert_writes_go_by_id(cleared)

    # The batch's live rows come from the batch index; only the in-progress lookup seeks by status.
    _enqueue(session_queue)
    batch = _statements(database, lambda: session_queue.cancel_by_batch_ids("default", ["batch"], user_id="alice"))
    reads = [statement for statement in batch if not statement.is_write]
    live_rows = [statement for statement in reads if statement.binds("pending")]
    in_progress = [statement for statement in reads if statement.binds("in_progress")]
    assert live_rows and in_progress, batch
    assert all(BATCH_SEEK in statement.plan for statement in live_rows), batch
    assert all(STATUS_SEEK in statement.plan for statement in in_progress), batch
    _assert_writes_go_by_id(batch)
