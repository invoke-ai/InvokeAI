"""Add bounded idempotency receipts for session-queue enqueue requests."""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddQueueEnqueueReceiptsCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS session_queue_enqueue_receipts (
                queue_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                requested INTEGER NOT NULL,
                enqueued INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                item_ids TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                acknowledged_at DATETIME,
                PRIMARY KEY (queue_id, user_id, idempotency_key),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_session_queue_enqueue_receipts_owner_ack
            ON session_queue_enqueue_receipts(user_id, acknowledged_at, byte_size);
            """
        )


def build_migration() -> Migration:
    return Migration(
        id="2026_09_04_add_queue_enqueue_receipts",
        depends_on="2026_07_01_add_session_queue_device",
        callback=AddQueueEnqueueReceiptsCallback(),
    )
