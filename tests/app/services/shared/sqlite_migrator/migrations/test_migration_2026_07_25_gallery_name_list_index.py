"""Tests for the gallery name-list covering-index migration."""

import sqlite3
from logging import Logger

import pytest

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_25_gallery_name_list_index import (
    GalleryNameListIndexCallback,
    build_migration,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration, MigrationError
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import SqliteMigrator


def _create_images_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE images (
            image_name TEXT PRIMARY KEY,
            image_category TEXT NOT NULL,
            is_intermediate BOOLEAN NOT NULL,
            starred BOOLEAN NOT NULL,
            created_at DATETIME NOT NULL
        );
        """
    )


def test_creates_expected_covering_index_and_is_idempotent() -> None:
    conn = sqlite3.connect(":memory:")
    _create_images_table(conn)

    GalleryNameListIndexCallback()(conn.cursor())
    GalleryNameListIndexCallback()(conn.cursor())

    columns = [
        (row[2], row[3]) for row in conn.execute("PRAGMA index_xinfo(idx_images_gallery_names)").fetchall() if row[5]
    ]
    assert columns == [
        ("image_category", 0),
        ("is_intermediate", 0),
        ("starred", 1),
        ("created_at", 1),
        ("image_name", 0),
    ]


def test_missing_images_table_fails_loud() -> None:
    conn = sqlite3.connect(":memory:")

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        GalleryNameListIndexCallback()(conn.cursor())


def test_failed_migration_is_not_recorded_as_applied() -> None:
    db = SqliteDatabase(db_path=None, logger=Logger("test"))
    migrator = SqliteMigrator(db)
    migrator.register_migration(Migration(from_version=0, to_version=1, callback=lambda cursor: None))
    migrator.register_migration(build_migration())

    with pytest.raises(MigrationError, match="no such table"):
        migrator.run_migrations()

    rows = db._conn.execute("SELECT migration_id FROM applied_migrations ORDER BY migration_id").fetchall()
    assert [row[0] for row in rows] == ["migration_1"]


def test_builder_metadata() -> None:
    migration = build_migration()
    assert migration.id == "2026_07_25_gallery_name_list_index"
    assert migration.depends_on == "migration_1"
    assert migration.from_version is None
    assert migration.to_version is None
