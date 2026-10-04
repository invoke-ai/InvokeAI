import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_27_wildcards import (
    WildcardsMigrationCallback,
    build_migration,
)


def _migrated_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys = ON;")
    # Created by migration_27, which this migration depends on.
    db.execute("CREATE TABLE users (user_id TEXT NOT NULL PRIMARY KEY);")

    WildcardsMigrationCallback()(db.cursor())

    return db


def _insert_wildcard(db: sqlite3.Connection, wildcard_id: str, name: str, user_id: str) -> None:
    db.execute(
        "INSERT INTO wildcards (id, name, values_json, user_id) VALUES (?, ?, '[]', ?);",
        (wildcard_id, name, user_id),
    )


def test_creates_the_wildcards_table_with_a_unique_name_per_owner() -> None:
    db = _migrated_db()
    db.execute("INSERT INTO users VALUES ('u1');")
    db.execute("INSERT INTO users VALUES ('u2');")

    _insert_wildcard(db, "w1", "colors", "u1")
    # The same name is free for a different owner.
    _insert_wildcard(db, "w2", "colors", "u2")

    try:
        _insert_wildcard(db, "w3", "colors", "u1")
        raise AssertionError("expected a unique constraint failure")
    except sqlite3.IntegrityError as e:
        assert "UNIQUE constraint failed" in str(e)

    db.close()


def test_wildcards_cascade_when_their_owner_is_deleted() -> None:
    db = _migrated_db()
    db.execute("INSERT INTO users VALUES ('u1');")
    _insert_wildcard(db, "w1", "colors", "u1")

    db.execute("DELETE FROM users WHERE user_id = 'u1';")

    assert db.execute("SELECT COUNT(*) FROM wildcards;").fetchone()[0] == 0

    db.close()


def test_an_unknown_owner_is_rejected() -> None:
    db = _migrated_db()

    try:
        _insert_wildcard(db, "w1", "colors", "ghost")
        raise AssertionError("expected a foreign key failure")
    except sqlite3.IntegrityError as e:
        assert "FOREIGN KEY constraint failed" in str(e)

    db.close()


def test_the_updated_at_trigger_fires_on_update() -> None:
    db = _migrated_db()
    db.execute("INSERT INTO users VALUES ('u1');")
    db.execute(
        "INSERT INTO wildcards (id, name, values_json, user_id, updated_at)"
        " VALUES ('w1', 'colors', '[]', 'u1', '2000-01-01 00:00:00.000');"
    )

    db.execute("UPDATE wildcards SET name = 'shades' WHERE id = 'w1';")

    assert db.execute("SELECT updated_at FROM wildcards WHERE id = 'w1';").fetchone()[0] != "2000-01-01 00:00:00.000"

    db.close()


def test_migration_is_idempotent() -> None:
    db = _migrated_db()

    WildcardsMigrationCallback()(db.cursor())

    assert db.execute("SELECT COUNT(*) FROM wildcards;").fetchone()[0] == 0

    db.close()


def test_build_migration_declares_stable_id_and_dependency() -> None:
    migration = build_migration()

    assert migration.id == "2026_07_27_wildcards"
    # migration_27 creates the users table the foreign key points at.
    assert migration.depends_on == "migration_27"
    assert migration.from_version is None
    assert migration.to_version is None
