import sqlite3


def v1(cursor: sqlite3.Cursor) -> None:
    """
    Migration for `session_queue` table v1
    https://github.com/invoke-ai/InvokeAI/pull/5148

    Adds the `workflow` column to the `session_queue` table.

    Workflows have been (correctly) made a property of a queue item, rather than individual nodes.
    This requires they be included in the session queue.
    """

    cursor.execute("PRAGMA table_info(session_queue)")
    columns = [column[1] for column in cursor.fetchall()]
    if "workflow" not in columns:
        cursor.execute(
            """--sql
            ALTER TABLE session_queue ADD COLUMN workflow TEXT;
            """
        )
