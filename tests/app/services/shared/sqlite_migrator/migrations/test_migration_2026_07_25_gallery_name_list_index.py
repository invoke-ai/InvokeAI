import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_25_gallery_name_list_index import (
    GalleryNameListIndexCallback,
    build_migration,
)


def _get_indexes(cursor: sqlite3.Cursor) -> set[str]:
    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'index';")
    return {row[0] for row in cursor.fetchall()}


def _make_images_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        """--sql
        CREATE TABLE images (
            image_name TEXT NOT NULL PRIMARY KEY,
            image_category TEXT NOT NULL,
            is_intermediate BOOLEAN DEFAULT FALSE,
            starred BOOLEAN DEFAULT FALSE,
            created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
        );
        """
    )


def test_adds_gallery_name_list_index() -> None:
    db = sqlite3.connect(":memory:")
    cursor = db.cursor()
    _make_images_table(cursor)

    GalleryNameListIndexCallback()(cursor)

    assert "idx_images_gallery_names" in _get_indexes(cursor)

    cursor.execute("PRAGMA index_info(idx_images_gallery_names);")
    columns = [row[2] for row in cursor.fetchall()]
    assert columns == ["image_category", "is_intermediate", "starred", "created_at", "image_name"]

    db.close()


def test_migration_is_idempotent_and_tolerates_missing_images_table() -> None:
    db = sqlite3.connect(":memory:")
    cursor = db.cursor()

    # No images table yet: the callback must be a no-op, not an error.
    GalleryNameListIndexCallback()(cursor)
    assert "idx_images_gallery_names" not in _get_indexes(cursor)

    _make_images_table(cursor)
    GalleryNameListIndexCallback()(cursor)
    GalleryNameListIndexCallback()(cursor)

    assert "idx_images_gallery_names" in _get_indexes(cursor)

    db.close()


def test_build_migration_declares_stable_id_and_dependency() -> None:
    migration = build_migration()

    assert migration.id == "2026_07_25_gallery_name_list_index"
    assert migration.depends_on == "migration_1"
    assert migration.from_version is None
    assert migration.to_version is None
