import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_10_create_system_prompts import (
    DEFAULT_SYSTEM_PROMPTS,
    CreateSystemPromptsCallback,
    build_migration,
)


def _get_columns(cursor: sqlite3.Cursor, table_name: str) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table_name});")
    return {row[1] for row in cursor.fetchall()}


def test_creates_table_and_seeds_defaults() -> None:
    db = sqlite3.connect(":memory:")
    cursor = db.cursor()

    CreateSystemPromptsCallback()(cursor)

    assert _get_columns(cursor, "system_prompts") == {
        "id",
        "name",
        "content",
        "user_id",
        "is_public",
        "created_at",
        "updated_at",
    }
    cursor.execute("SELECT COUNT(*) FROM system_prompts WHERE user_id = 'system' AND is_public = TRUE;")
    assert cursor.fetchone()[0] == len(DEFAULT_SYSTEM_PROMPTS)

    db.close()


def test_backfills_multiuser_columns_on_a_pre_multiuser_table() -> None:
    # The migration id was bumped precisely so this path is reachable on dev DBs created from an
    # earlier revision of this branch, where the table exists without user_id/is_public.
    db = sqlite3.connect(":memory:")
    cursor = db.cursor()
    cursor.execute(
        """--sql
        CREATE TABLE system_prompts (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
        );
        """
    )
    cursor.execute("INSERT INTO system_prompts (id, name, content) VALUES ('kept', 'mine', 'body');")

    CreateSystemPromptsCallback()(cursor)

    assert {"user_id", "is_public"} <= _get_columns(cursor, "system_prompts")
    # The pre-existing row survives and picks up the column defaults.
    cursor.execute("SELECT user_id, is_public, content FROM system_prompts WHERE id = 'kept';")
    assert cursor.fetchone() == ("system", 0, "body")

    db.close()


def test_is_idempotent_and_preserves_user_edits() -> None:
    # Re-running must not duplicate the defaults or clobber edits to them: the fixed UUIDs make
    # every seed row a primary-key conflict that INSERT OR IGNORE skips.
    db = sqlite3.connect(":memory:")
    cursor = db.cursor()

    CreateSystemPromptsCallback()(cursor)
    default_id = DEFAULT_SYSTEM_PROMPTS[0][0]
    cursor.execute("UPDATE system_prompts SET content = 'edited by user' WHERE id = ?;", (default_id,))

    CreateSystemPromptsCallback()(cursor)

    cursor.execute("SELECT COUNT(*) FROM system_prompts;")
    assert cursor.fetchone()[0] == len(DEFAULT_SYSTEM_PROMPTS)
    cursor.execute("SELECT content FROM system_prompts WHERE id = ?;", (default_id,))
    assert cursor.fetchone()[0] == "edited by user"

    db.close()


def test_migration_id_matches_its_module_name() -> None:
    # The loader rejects a mismatch, but assert it here so a rename is a test failure rather than
    # a startup failure.
    assert build_migration().id == "2026_07_10_create_system_prompts"
