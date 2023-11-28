import sqlite3


def v1(cursor: sqlite3.Cursor) -> None:
    """
    Migration for `images` table v1
    https://github.com/invoke-ai/InvokeAI/pull/4246

    Adds the `starred` column to the `images` table.
    """

    cursor.execute("PRAGMA table_info(images)")
    columns = [column[1] for column in cursor.fetchall()]
    if "starred" not in columns:
        cursor.execute(
            """--sql
            ALTER TABLE images
            ADD COLUMN starred BOOLEAN DEFAULT FALSE;
            """
        )
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_images_starred ON images(starred);
            """
        )
