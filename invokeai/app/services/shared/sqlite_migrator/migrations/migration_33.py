"""Migration 33: Add indexes supporting round-robin dequeue.

The round-robin scheduler in multiuser mode runs two queries on every dequeue:

1. A per-user "last served" lookup (``MAX(started_at)`` grouped by ``user_id`` over rows
   with ``started_at IS NOT NULL``).
2. A per-user "best pending item" selection (``status = 'pending'`` partitioned by
   ``user_id`` and ordered by ``priority DESC, item_id ASC``).

With only the pre-existing single-column indexes on ``status``, ``priority``, and
``user_id``, both queries fall back to scanning the table and building temporary b-trees
for grouping/ordering. Because completed/failed/canceled history is retained (and
``max_queue_history`` defaults to unbounded), that cost grows with total queue history
rather than with the number of pending items. These covering indexes match the query
shapes so the planner can satisfy both without scanning historical rows or sorting in a
temp b-tree.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration33Callback:
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

        # Last-served lookup: WHERE started_at IS NOT NULL, GROUP BY user_id, MAX(started_at).
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_session_queue_user_started_at
            ON session_queue (user_id, started_at);
            """
        )


def build_migration_33() -> Migration:
    return Migration(
        from_version=32,
        to_version=33,
        callback=Migration33Callback(),
    )
