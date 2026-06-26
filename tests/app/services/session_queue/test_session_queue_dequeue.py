"""Tests for session queue dequeue() ordering: FIFO and round-robin modes."""

import json
import uuid
from typing import Optional

import pytest
from pydantic_core import to_jsonable_python

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_queue.session_queue_sqlite import (
    ROUND_ROBIN_DEQUEUE_QUERY,
    SqliteSessionQueue,
)
from invokeai.app.services.shared.graph import Graph, GraphExecutionState

_EMPTY_SESSION_JSON = json.dumps(to_jsonable_python(GraphExecutionState(graph=Graph()).model_dump()))


@pytest.fixture
def session_queue_fifo(mock_invoker: Invoker) -> SqliteSessionQueue:
    """Queue backed by a single-user (FIFO) invoker."""
    # Default config has multiuser=False, so FIFO is always used.
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


@pytest.fixture
def session_queue_round_robin(mock_invoker: Invoker) -> SqliteSessionQueue:
    """Queue backed by a multiuser invoker with round_robin mode."""
    mock_invoker.services.configuration = InvokeAIAppConfig(
        use_memory_db=True,
        node_cache_size=0,
        multiuser=True,
        session_queue_mode="round_robin",
    )
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)
    return queue


def _insert_queue_item(
    session_queue: SqliteSessionQueue,
    queue_id: str,
    user_id: str,
    priority: int = 0,
) -> int:
    """Directly insert a minimal queue item and return its item_id."""
    session_id = str(uuid.uuid4())
    batch_id = str(uuid.uuid4())
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination, retried_from_item_id, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (queue_id, _EMPTY_SESSION_JSON, session_id, batch_id, None, priority, None, None, None, None, user_id),
        )
        return cursor.lastrowid  # type: ignore[return-value]


def _dequeue_user_ids(session_queue: SqliteSessionQueue, count: int) -> list[Optional[str]]:
    """Dequeue `count` items and return the list of user_ids in dequeue order."""
    result = []
    for _ in range(count):
        item = session_queue.dequeue()
        result.append(item.user_id if item is not None else None)
    return result


# ---------------------------------------------------------------------------
# FIFO tests
# ---------------------------------------------------------------------------


def test_fifo_single_user_order(session_queue_fifo: SqliteSessionQueue) -> None:
    """FIFO: items from a single user are dequeued in insertion order."""
    queue_id = "default"
    _insert_queue_item(session_queue_fifo, queue_id, "user_a")
    _insert_queue_item(session_queue_fifo, queue_id, "user_a")
    _insert_queue_item(session_queue_fifo, queue_id, "user_a")

    user_ids = _dequeue_user_ids(session_queue_fifo, 3)
    assert user_ids == ["user_a", "user_a", "user_a"]


def test_fifo_multi_user_preserves_insertion_order(session_queue_fifo: SqliteSessionQueue) -> None:
    """FIFO: jobs from multiple users are dequeued in strict insertion order, not interleaved."""
    queue_id = "default"
    # Insert A1, A2, B1, C1, C2, A3 – FIFO should preserve this exact order.
    _insert_queue_item(session_queue_fifo, queue_id, "user_a")
    _insert_queue_item(session_queue_fifo, queue_id, "user_a")
    _insert_queue_item(session_queue_fifo, queue_id, "user_b")
    _insert_queue_item(session_queue_fifo, queue_id, "user_c")
    _insert_queue_item(session_queue_fifo, queue_id, "user_c")
    _insert_queue_item(session_queue_fifo, queue_id, "user_a")

    user_ids = _dequeue_user_ids(session_queue_fifo, 6)
    assert user_ids == ["user_a", "user_a", "user_b", "user_c", "user_c", "user_a"]


def test_fifo_priority_respected(session_queue_fifo: SqliteSessionQueue) -> None:
    """FIFO: higher-priority items are dequeued before lower-priority ones."""
    queue_id = "default"
    _insert_queue_item(session_queue_fifo, queue_id, "user_a", priority=0)
    _insert_queue_item(session_queue_fifo, queue_id, "user_a", priority=10)

    user_ids = _dequeue_user_ids(session_queue_fifo, 2)
    # Both are user_a; second inserted item has higher priority and should come first.
    assert user_ids == ["user_a", "user_a"]


def test_fifo_returns_none_when_empty(session_queue_fifo: SqliteSessionQueue) -> None:
    """FIFO: dequeue returns None when the queue is empty."""
    assert session_queue_fifo.dequeue() is None


# ---------------------------------------------------------------------------
# Round-robin tests
# ---------------------------------------------------------------------------


def test_round_robin_interleaves_users(session_queue_round_robin: SqliteSessionQueue) -> None:
    """Round-robin: jobs from multiple users are interleaved one per user per round.

    Queue insertion order (matching the issue example):
        A job 1, A job 2, B job 1, C job 1, C job 2, A job 3

    Expected dequeue order:
        A job 1, B job 1, C job 1, A job 2, C job 2, A job 3
    """
    queue_id = "default"
    _insert_queue_item(session_queue_round_robin, queue_id, "user_a")
    _insert_queue_item(session_queue_round_robin, queue_id, "user_a")
    _insert_queue_item(session_queue_round_robin, queue_id, "user_b")
    _insert_queue_item(session_queue_round_robin, queue_id, "user_c")
    _insert_queue_item(session_queue_round_robin, queue_id, "user_c")
    _insert_queue_item(session_queue_round_robin, queue_id, "user_a")

    user_ids = _dequeue_user_ids(session_queue_round_robin, 6)
    assert user_ids == ["user_a", "user_b", "user_c", "user_a", "user_c", "user_a"]


def test_round_robin_single_user_behaves_like_fifo(session_queue_round_robin: SqliteSessionQueue) -> None:
    """Round-robin with only one user produces the same order as FIFO."""
    queue_id = "default"
    _insert_queue_item(session_queue_round_robin, queue_id, "user_a")
    _insert_queue_item(session_queue_round_robin, queue_id, "user_a")
    _insert_queue_item(session_queue_round_robin, queue_id, "user_a")

    user_ids = _dequeue_user_ids(session_queue_round_robin, 3)
    assert user_ids == ["user_a", "user_a", "user_a"]


def test_round_robin_handles_user_joining_mid_queue(session_queue_round_robin: SqliteSessionQueue) -> None:
    """Round-robin: a user who joins later is correctly interleaved."""
    queue_id = "default"
    _insert_queue_item(session_queue_round_robin, queue_id, "user_a")
    _insert_queue_item(session_queue_round_robin, queue_id, "user_a")
    _insert_queue_item(session_queue_round_robin, queue_id, "user_b")

    user_ids = _dequeue_user_ids(session_queue_round_robin, 3)
    # Round 1: A (oldest rank-1 item), B (rank-1 item)
    # Round 2: A (rank-2 item)
    assert user_ids == ["user_a", "user_b", "user_a"]


def test_round_robin_returns_none_when_empty(session_queue_round_robin: SqliteSessionQueue) -> None:
    """Round-robin: dequeue returns None when the queue is empty."""
    assert session_queue_round_robin.dequeue() is None


def test_round_robin_priority_within_user_respected(session_queue_round_robin: SqliteSessionQueue) -> None:
    """Round-robin: within a single user's items, higher priority is dequeued first."""
    queue_id = "default"
    # Insert low-priority item first, then high-priority for same user.
    _insert_queue_item(session_queue_round_robin, queue_id, "user_a", priority=0)
    _insert_queue_item(session_queue_round_robin, queue_id, "user_a", priority=10)
    _insert_queue_item(session_queue_round_robin, queue_id, "user_b", priority=0)

    # Round 1: user_a's best item (priority 10), user_b's only item.
    # Round 2: user_a's remaining item (priority 0).
    items = []
    for _ in range(3):
        item = session_queue_round_robin.dequeue()
        assert item is not None
        items.append((item.user_id, item.priority))

    assert items[0] == ("user_a", 10)
    assert items[1] == ("user_b", 0)
    assert items[2] == ("user_a", 0)


def _seed_completed_history(
    session_queue: SqliteSessionQueue,
    queue_id: str,
    user_id: str,
    count: int,
) -> None:
    """Insert `count` completed items (with started_at set) for a user, simulating retained history."""
    with session_queue._db.transaction() as cursor:
        for i in range(count):
            session_id = str(uuid.uuid4())
            batch_id = str(uuid.uuid4())
            cursor.execute(
                """--sql
                INSERT INTO session_queue
                    (queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination, retried_from_item_id, user_id, status, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?)
                """,
                (
                    queue_id,
                    _EMPTY_SESSION_JSON,
                    session_id,
                    batch_id,
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                    user_id,
                    # Monotonically increasing timestamps so MAX(started_at) is well-defined per user.
                    f"2026-01-01 {i // 3600 % 24:02d}:{i // 60 % 60:02d}:{i % 60:02d}",
                    f"2026-01-01 {i // 3600 % 24:02d}:{i // 60 % 60:02d}:{i % 60:02d}",
                ),
            )


def test_round_robin_dequeue_does_not_scan_full_history(session_queue_round_robin: SqliteSessionQueue) -> None:
    """Round-robin dequeue cost must scale with active users, not retained queue history.

    Regression guard for the scaling concern: the per-user "last served" lookup must be an
    indexed seek (MAX(started_at) WHERE user_id = ?) rather than a GROUP BY / scan over every
    historical started row. `max_queue_history` is unbounded by default, so a plan that scans
    the full history makes each dequeue O(total history) instead of O(active users).

    We seed a large completed history across several users plus a few pending items, then assert
    the dequeue query plan never scans the `session_queue` base table and resolves the
    last-served lookup via a seek on `idx_session_queue_user_started_at`.
    """
    queue_id = "default"
    for u in ("user_a", "user_b", "user_c"):
        _seed_completed_history(session_queue_round_robin, queue_id, u, count=500)
        _insert_queue_item(session_queue_round_robin, queue_id, u)

    with session_queue_round_robin._db.transaction() as cursor:
        plan_rows = cursor.execute("EXPLAIN QUERY PLAN " + ROUND_ROBIN_DEQUEUE_QUERY).fetchall()
    details = [row["detail"] for row in plan_rows]

    # No step may scan the session_queue base table — that is the full-history scan we are
    # eliminating. (CTE result scans like "SCAN uni" / "SCAN (subquery-N)" are fine; those are
    # one row per pending user.)
    offending = [d for d in details if d.startswith("SCAN session_queue")]
    assert not offending, f"dequeue plan scans full queue history: {offending}\nfull plan: {details}"

    # The last-served lookup must use the started_at index as a per-user seek.
    assert any(
        "idx_session_queue_user_started_at" in d and "user_id=?" in d for d in details
    ), f"last-served lookup is not an indexed seek; plan: {details}"

    # And the dequeue must still return the least-recently-served user (correctness under history).
    # user_a's history ends earliest only if seeded first; all three were seeded equal counts with
    # identical timestamps, so item_id ASC tie-breaks to the first-inserted pending item (user_a).
    item = session_queue_round_robin.dequeue()
    assert item is not None
    assert item.user_id == "user_a"


def test_round_robin_ignored_in_single_user_mode(mock_invoker: Invoker) -> None:
    """When multiuser=False, round_robin config is ignored and FIFO is used."""
    mock_invoker.services.configuration = InvokeAIAppConfig(
        use_memory_db=True,
        node_cache_size=0,
        multiuser=False,
        session_queue_mode="round_robin",
    )
    db = mock_invoker.services.board_records._db
    queue = SqliteSessionQueue(db=db)
    queue.start(mock_invoker)

    queue_id = "default"
    _insert_queue_item(queue, queue_id, "user_a")
    _insert_queue_item(queue, queue_id, "user_a")
    _insert_queue_item(queue, queue_id, "user_b")

    # FIFO order: user_a, user_a, user_b
    user_ids = _dequeue_user_ids(queue, 3)
    assert user_ids == ["user_a", "user_a", "user_b"]
