"""Tests for migration 28: Add image_subfolder column to images table (Point 4)."""

import sqlite3

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_28 import (
    Migration28Callback,
    build_migration_28,
)


@pytest.fixture
def db() -> sqlite3.Connection:
    """In-memory SQLite database with a minimal images table mimicking pre-migration schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE images (
            image_name TEXT NOT NULL PRIMARY KEY,
            image_origin TEXT NOT NULL,
            image_category TEXT NOT NULL,
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0,
            session_id TEXT,
            node_id TEXT,
            metadata TEXT,
            is_intermediate BOOLEAN DEFAULT FALSE,
            created_at DATETIME NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%f', 'NOW')),
            updated_at DATETIME NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%f', 'NOW')),
            deleted_at DATETIME,
            starred BOOLEAN NOT NULL DEFAULT FALSE,
            has_workflow BOOLEAN NOT NULL DEFAULT FALSE
        );
        """
    )
    return conn


class TestMigration28:
    def test_adds_image_subfolder_column(self, db: sqlite3.Connection):
        """Migration adds image_subfolder column to existing images table."""
        callback = Migration28Callback()
        cursor = db.cursor()
        callback(cursor)

        cursor.execute("PRAGMA table_info(images);")
        columns = {row[1] for row in cursor.fetchall()}
        assert "image_subfolder" in columns

    def test_existing_rows_get_empty_string_default(self, db: sqlite3.Connection):
        """Pre-existing image rows should get image_subfolder = '' after migration."""
        db.execute(
            "INSERT INTO images (image_name, image_origin, image_category, width, height, has_workflow) "
            "VALUES ('old_image.png', 'internal', 'general', 512, 512, 0)"
        )
        db.commit()

        callback = Migration28Callback()
        callback(db.cursor())
        db.commit()

        row = db.execute("SELECT image_subfolder FROM images WHERE image_name = 'old_image.png'").fetchone()
        assert row is not None
        assert row[0] == ""

    def test_idempotent_migration(self, db: sqlite3.Connection):
        """Running migration twice should not fail (column already exists)."""
        callback = Migration28Callback()
        cursor = db.cursor()
        callback(cursor)
        # Running again should be safe
        callback(cursor)

        cursor.execute("PRAGMA table_info(images);")
        columns = [row[1] for row in cursor.fetchall()]
        assert columns.count("image_subfolder") == 1

    def test_no_images_table_is_noop(self):
        """If images table doesn't exist, migration is a no-op."""
        conn = sqlite3.connect(":memory:")
        callback = Migration28Callback()
        cursor = conn.cursor()
        # Should not raise
        callback(cursor)

    def test_build_migration_28_version_numbers(self):
        """build_migration_28 returns correct version range."""
        migration = build_migration_28()
        assert migration.from_version == 27
        assert migration.to_version == 28
