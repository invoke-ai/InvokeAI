"""Add a covering index for the gallery name-list query.

``GET /api/v1/images/names`` (``SqliteImageRecordStorage.get_image_names``) fetches the
full, ordered name list for the virtualized gallery. The client always filters by
``image_category`` and ``is_intermediate`` and orders by ``starred DESC, created_at
DESC``, but no existing index supports that shape: ``idx_images_starred`` and
``idx_images_created_at`` are single-column, so the planner scans one of them and then
sorts every matching row in a temp B-tree on each request.

``idx_images_gallery_names`` matches the query shape end to end: equality columns first
(``image_category``, ``is_intermediate``), then the ORDER BY columns (``starred DESC,
created_at DESC``), then ``image_name`` so the scan is covering (no row lookups). With
the anti-join form of the board filter in ``get_image_names``, the planner returns rows
directly in index order and the temp B-tree sort disappears.

This index only pays off together with that anti-join query shape; a plain composite
index without the covering columns measured 2.5-4.5x *slower* than baseline at large
gallery sizes because it turns a sequential scan into random row lookups.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class GalleryNameListIndexCallback:
    """Add a covering composite index matching the gallery name-list query shape."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images';")
        if cursor.fetchone() is None:
            return

        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_images_gallery_names
            ON images (image_category, is_intermediate, starred DESC, created_at DESC, image_name);
            """
        )


def build_migration() -> Migration:
    return Migration(
        id="2026_07_25_gallery_name_list_index",
        # migration_1 created the images table and (conditionally) the starred column;
        # every column this index touches exists once migration_1 has run.
        depends_on="migration_1",
        callback=GalleryNameListIndexCallback(),
    )
