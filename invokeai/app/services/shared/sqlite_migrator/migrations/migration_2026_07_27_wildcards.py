import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class WildcardsMigrationCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_wildcards_table(cursor)

    def _create_wildcards_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS wildcards (
                id TEXT NOT NULL PRIMARY KEY,
                name TEXT NOT NULL,
                values_json TEXT NOT NULL DEFAULT '[]',
                user_id TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
            );
            """
        )
        # A wildcard is referenced by name, so a user cannot own the same name twice. The index is
        # the authority: it makes a concurrent create fail on insert rather than in a check-then-act
        # race, and it is also the lookup path for listing a user's wildcards.
        cursor.execute(
            """--sql
            CREATE UNIQUE INDEX IF NOT EXISTS idx_wildcards_user_id_name ON wildcards(user_id, name);
            """
        )
        cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_wildcards_updated_at
            AFTER UPDATE ON wildcards FOR EACH ROW
            BEGIN
                UPDATE wildcards SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE id = old.id;
            END;
            """
        )


def build_migration() -> Migration:
    """
    Build the migration that adds per-user wildcard storage.

    Wildcards expand `__name__` in a prompt. They are user-owned content, so they live here rather
    than on disk, and a `WildcardManager` is built from these rows per request.
    """
    return Migration(
        id="2026_07_27_wildcards",
        depends_on="migration_9",
        callback=WildcardsMigrationCallback(),
    )
