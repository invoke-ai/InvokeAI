"""Add a covering index for the gallery name-list query.

The image half of ``SqliteGalleryService.list_item_names()`` filters by
``image_category`` and ``is_intermediate`` and orders by ``starred`` and
``created_at``. This index keeps that common path covering while leaving query
shapes that need other columns on their existing access paths.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class GalleryNameListIndexCallback:
    """Add a covering composite index matching the image name-list query shape."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_images_gallery_names
            ON images (image_category, is_intermediate, starred DESC, created_at DESC, image_name);
            """
        )


def build_migration() -> Migration:
    return Migration(
        id="2026_07_25_gallery_name_list_index",
        depends_on="migration_1",
        callback=GalleryNameListIndexCallback(),
    )
