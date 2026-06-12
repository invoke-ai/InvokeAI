"""Tests for migration 32: Repair model_relationships foreign keys."""

import sqlite3

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_32 import (
    Migration32Callback,
    build_migration_32,
)


def _create_models_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE models (
            id TEXT NOT NULL PRIMARY KEY,
            config TEXT NOT NULL
        );
        """
    )


def _create_broken_relationships_table(conn: sqlite3.Connection) -> None:
    """Recreates the broken state left by migration 22: FKs reference the dropped models_old table."""
    conn.execute(
        """
        CREATE TABLE model_relationships (
            model_key_1 TEXT NOT NULL,
            model_key_2 TEXT NOT NULL,
            created_at TEXT DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            PRIMARY KEY (model_key_1, model_key_2),
            FOREIGN KEY (model_key_1) REFERENCES "models_old"(id) ON DELETE CASCADE,
            FOREIGN KEY (model_key_2) REFERENCES "models_old"(id) ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        "CREATE INDEX keyx_model_relationships_model_key_2 ON model_relationships(model_key_2);"
    )


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _create_models_table(conn)
    _create_broken_relationships_table(conn)
    conn.execute("INSERT INTO models (id, config) VALUES ('a', '{}'), ('b', '{}'), ('c', '{}')")
    return conn


class TestMigration32:
    def test_repoints_foreign_keys_to_models(self, db: sqlite3.Connection):
        """After migration, the foreign keys reference models, not models_old."""
        Migration32Callback()(db.cursor())
        db.commit()

        sql = db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_relationships'"
        ).fetchone()[0]
        assert "models_old" not in sql
        assert "REFERENCES models(id)" in sql

    def test_preserves_valid_links(self, db: sqlite3.Connection):
        """Links between existing models are preserved."""
        db.execute("INSERT INTO model_relationships (model_key_1, model_key_2) VALUES ('a', 'b')")
        db.commit()

        Migration32Callback()(db.cursor())
        db.commit()

        rows = db.execute(
            "SELECT model_key_1, model_key_2 FROM model_relationships ORDER BY model_key_1"
        ).fetchall()
        assert rows == [("a", "b")]

    def test_drops_orphaned_links(self, db: sqlite3.Connection):
        """Links referencing missing models are dropped so the restored FKs are satisfiable."""
        db.execute("INSERT INTO model_relationships (model_key_1, model_key_2) VALUES ('a', 'b')")
        db.execute("INSERT INTO model_relationships (model_key_1, model_key_2) VALUES ('a', 'gone')")
        db.commit()

        Migration32Callback()(db.cursor())
        db.commit()

        rows = db.execute("SELECT model_key_1, model_key_2 FROM model_relationships").fetchall()
        assert rows == [("a", "b")]

    def test_cascade_works_after_repair(self, db: sqlite3.Connection):
        """ON DELETE CASCADE against models works once the FKs are repaired."""
        db.execute("INSERT INTO model_relationships (model_key_1, model_key_2) VALUES ('a', 'b')")
        db.commit()

        Migration32Callback()(db.cursor())
        db.commit()

        db.execute("PRAGMA foreign_keys = ON;")
        db.execute("DELETE FROM models WHERE id = 'a'")
        db.commit()

        rows = db.execute("SELECT * FROM model_relationships").fetchall()
        assert rows == []

    def test_index_recreated(self, db: sqlite3.Connection):
        """The lookup index on model_key_2 is recreated on the rebuilt table."""
        Migration32Callback()(db.cursor())
        db.commit()

        idx = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='keyx_model_relationships_model_key_2'"
        ).fetchone()
        assert idx is not None

    def test_idempotent_when_already_correct(self, db: sqlite3.Connection):
        """Running on an already-correct table is a no-op (no rebuild)."""
        Migration32Callback()(db.cursor())
        db.commit()
        # Second run should detect the correct FKs and do nothing.
        Migration32Callback()(db.cursor())
        db.commit()

        sql = db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_relationships'"
        ).fetchone()[0]
        assert "REFERENCES models(id)" in sql

    def test_no_relationships_table_is_noop(self):
        """If the table doesn't exist, migration is a no-op."""
        conn = sqlite3.connect(":memory:")
        Migration32Callback()(conn.cursor())  # should not raise

    def test_build_migration_32_version_numbers(self):
        migration = build_migration_32()
        assert migration.from_version == 31
        assert migration.to_version == 32
