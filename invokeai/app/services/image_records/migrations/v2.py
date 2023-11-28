import sqlite3


def v2(cursor: sqlite3.Cursor) -> None:
    """
    Migration for `images` table v2
    https://github.com/invoke-ai/InvokeAI/pull/5148

    Adds the `has_workflow` column to the `images` table.

    Workflows associated with images are now only stored in the image file itself. This column
    indicates whether the image has a workflow embedded in it, so we don't need to read the image
    file to find out.
    """

    cursor.execute("PRAGMA table_info(images)")
    columns = [column[1] for column in cursor.fetchall()]
    if "has_workflow" not in columns:
        cursor.execute(
            """--sql
            ALTER TABLE images
            ADD COLUMN has_workflow BOOLEAN DEFAULT FALSE;
            """
        )
