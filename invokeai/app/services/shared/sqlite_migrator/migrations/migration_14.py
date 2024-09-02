import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration14Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_style_presets(cursor)

    def _create_style_presets(self, cursor: sqlite3.Cursor) -> None:
        """Create the table used to store style presets."""
        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS style_presets (
                id TEXT NOT NULL PRIMARY KEY,
                name TEXT NOT NULL,
                preset_data TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT "user",
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
            );
            """
        ]

        # Add trigger for `updated_at`.
        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS style_presets
            AFTER UPDATE
            ON style_presets FOR EACH ROW
            BEGIN
                UPDATE style_presets SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE id = old.id;
            END;
            """
        ]

        # Add indexes for searchable fields
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_style_presets_name ON style_presets(name);",
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)


def build_migration_14() -> Migration:
    """
    Build the migration from database version 13 to 14..

    This migration does the following:
    - Create the table used to store style presets.
    """
    migration_14 = Migration(
        from_version=13,
        to_version=14,
        callback=Migration14Callback(),
    )

    return migration_14
