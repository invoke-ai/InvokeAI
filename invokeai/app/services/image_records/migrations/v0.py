import sqlite3


def v0(cursor: sqlite3.Cursor) -> None:
    """
    Migration for `images` table v0
    https://github.com/invoke-ai/InvokeAI/pull/3443

    Adds the `images` table, indicies and triggers for the image_records service.
    """

    cursor.execute(
        """--sql
        CREATE TABLE IF NOT EXISTS images (
          image_name TEXT NOT NULL PRIMARY KEY,
          -- This is an enum in python, unrestricted string here for flexibility
          image_origin TEXT NOT NULL,
          -- This is an enum in python, unrestricted string here for flexibility
          image_category TEXT NOT NULL,
          width INTEGER NOT NULL,
          height INTEGER NOT NULL,
          session_id TEXT,
          node_id TEXT,
          metadata TEXT,
          is_intermediate BOOLEAN DEFAULT FALSE,
          created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
          -- Updated via trigger
          updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
          -- Soft delete, currently unused
          deleted_at DATETIME
        );
        """
    )
    cursor.execute(
        """--sql
        CREATE UNIQUE INDEX IF NOT EXISTS idx_images_image_name ON images(image_name);
        """
    )
    cursor.execute(
        """--sql
        CREATE INDEX IF NOT EXISTS idx_images_image_origin ON images(image_origin);
        """
    )
    cursor.execute(
        """--sql
        CREATE INDEX IF NOT EXISTS idx_images_image_category ON images(image_category);
        """
    )
    cursor.execute(
        """--sql
        CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at);
        """
    )
    cursor.execute(
        """--sql
        CREATE TRIGGER IF NOT EXISTS tg_images_updated_at
        AFTER
        UPDATE ON images FOR EACH ROW BEGIN
        UPDATE images
        SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
        WHERE image_name = old.image_name;
        END;
        """
    )
