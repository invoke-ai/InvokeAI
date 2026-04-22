"""Migration 30: Add workflow-call relationship columns to session_queue."""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration30Callback:
    """Add durable parent/child workflow-call relationship columns to session_queue."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_queue';")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(session_queue);")
        columns = [row[1] for row in cursor.fetchall()]

        if "workflow_call_id" not in columns:
            cursor.execute("ALTER TABLE session_queue ADD COLUMN workflow_call_id TEXT;")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_queue_workflow_call_id ON session_queue(workflow_call_id);"
            )

        if "parent_item_id" not in columns:
            cursor.execute("ALTER TABLE session_queue ADD COLUMN parent_item_id INTEGER;")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_queue_parent_item_id ON session_queue(parent_item_id);"
            )

        if "parent_session_id" not in columns:
            cursor.execute("ALTER TABLE session_queue ADD COLUMN parent_session_id TEXT;")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_queue_parent_session_id ON session_queue(parent_session_id);"
            )

        if "root_item_id" not in columns:
            cursor.execute("ALTER TABLE session_queue ADD COLUMN root_item_id INTEGER;")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_queue_root_item_id ON session_queue(root_item_id);")

        if "workflow_call_depth" not in columns:
            cursor.execute("ALTER TABLE session_queue ADD COLUMN workflow_call_depth INTEGER;")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_queue_workflow_call_depth ON session_queue(workflow_call_depth);"
            )


def build_migration_30() -> Migration:
    return Migration(
        from_version=29,
        to_version=30,
        callback=Migration30Callback(),
    )
