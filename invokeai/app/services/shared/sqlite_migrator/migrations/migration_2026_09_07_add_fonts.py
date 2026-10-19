"""Add the account-scoped custom font catalog."""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddFontsCallback:
    """Create the font catalog and its ownership/source indexes."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS fonts (
                id TEXT NOT NULL PRIMARY KEY,
                owner_id TEXT,
                scope TEXT NOT NULL CHECK (scope IN ('private', 'shared')),
                source TEXT NOT NULL CHECK (source IN ('uploaded', 'directory')),
                filename TEXT NOT NULL,
                storage_path TEXT,
                source_path TEXT,
                family TEXT NOT NULL,
                label TEXT NOT NULL,
                style TEXT NOT NULL,
                weight INTEGER NOT NULL CHECK (weight >= 1 AND weight <= 1000),
                content_hash TEXT NOT NULL CHECK (length(content_hash) = 64),
                byte_size INTEGER NOT NULL CHECK (byte_size > 0),
                axes_json TEXT NOT NULL DEFAULT '[]',
                instances_json TEXT NOT NULL DEFAULT '[]',
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                FOREIGN KEY (owner_id) REFERENCES users(user_id) ON DELETE CASCADE,
                CHECK (
                    (source = 'directory' AND owner_id IS NULL AND scope = 'shared' AND storage_path IS NULL AND source_path IS NOT NULL)
                    OR
                    (source = 'uploaded' AND storage_path IS NOT NULL AND source_path IS NULL AND
                        ((scope = 'private' AND owner_id IS NOT NULL) OR (scope = 'shared' AND owner_id IS NULL)))
                )
            );
            """
        )
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_fonts_visible_private
            ON fonts(source, scope, owner_id, family COLLATE NOCASE);
            """
        )
        cursor.execute(
            """--sql
            CREATE UNIQUE INDEX IF NOT EXISTS idx_fonts_directory_source_path
            ON fonts(source_path) WHERE source = 'directory';
            """
        )
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_fonts_uploaded_hash
            ON fonts(source, scope, owner_id, content_hash);
            """
        )


def build_migration() -> Migration:
    return Migration(
        id="2026_09_07_add_fonts",
        depends_on="migration_27",
        callback=AddFontsCallback(),
    )
