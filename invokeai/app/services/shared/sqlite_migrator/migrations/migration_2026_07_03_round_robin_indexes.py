"""Add indexes supporting round-robin dequeue.

The round-robin dequeue in multiuser mode relies on two access shapes:

1. A per-user "best pending item" selection (``status = 'pending'`` partitioned by
   ``user_id`` and ordered by ``priority DESC, item_id ASC``).
2. A per-candidate "last served" lookup (``MAX(started_at)`` for a single ``user_id``),
   evaluated as a correlated subquery once per user that has pending work.

With only the pre-existing single-column indexes on ``status``, ``priority``, and
``user_id``, the pending selection falls back to scanning the table and the last-served
lookup scans all retained history. Because completed/failed/canceled history is retained
(and ``max_queue_history`` defaults to unbounded), that cost would grow with total queue
history rather than with the number of pending items / active users.

``idx_session_queue_round_robin_pending`` lets the planner satisfy the pending selection
without touching historical rows. ``idx_session_queue_user_started_at`` turns each
``MAX(started_at) WHERE user_id = ?`` into an indexed seek (the planner's min/max
optimization reads the tail of the user's index range) rather than a scan, so the
last-served lookup costs ``O(log n)`` per active user instead of scanning history.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class RoundRobinIndexesCallback:
    """Add composite indexes matching the round-robin dequeue query shapes."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_queue';")
        if cursor.fetchone() is None:
            return

        # Pending-item selection: WHERE status = 'pending', PARTITION BY user_id,
        # ORDER BY priority DESC, item_id ASC.
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_session_queue_round_robin_pending
            ON session_queue (status, user_id, priority DESC, item_id ASC);
            """
        )

        # Last-served lookup: MAX(started_at) WHERE user_id = ?)
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_session_queue_user_started_at
            ON session_queue (user_id, started_at);
            """
        )


def build_migration() -> Migration:
    return Migration(
        id="2026_07_03_round_robin_indexes",
        # migration_27 added the user_id column to session_queue, which both indexes cover.
        depends_on="migration_27",
        callback=RoundRobinIndexesCallback(),
    )
