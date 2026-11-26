"""Add the missing `ON DELETE CASCADE` to `image_subfolder_move_items.image_name`.

`migration_33` originally declared `image_name TEXT NOT NULL REFERENCES images(image_name)` with
no `ON DELETE` action, i.e. `NO ACTION`. Move items are inserted when Image Storage Maintenance
moves images into subfolders, and nothing ever deletes them. Foreign keys are enforced on the app
connection (`SqliteDatabase.__init__`), so every image that has ever been moved could not be
deleted: deleting it, or clearing intermediates, failed with `FOREIGN KEY constraint failed`.

`migration_33`'s DDL now carries the cascade, which fixes freshly-created databases. This
migration rebuilds the table on databases that already have the cascade-less version. SQLite
cannot `ALTER` a foreign key, so the table is recreated and the rows copied across.

Trade-off: deleting an image while a move job is mid-flight (file already relocated,
`images.image_subfolder` not yet updated) used to fail; now it succeeds, the cascade drops the
item row, and the relocated file is left on disk unreferenced. That narrow leak is preferable to
images that can never be deleted.

This is a back-port of Invoke v7's migration of the same ID, without v7's table-creation step
(which repairs a v7-only numbering collision). The ID must match v7's: v7 refuses a database that
records an applied migration ID it does not know, so a distinct ID would block upgrading to v7.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

# Column list in table order, used to build the copying INSERT/SELECT.
_ITEMS_COLUMNS = [
    "job_id",
    "image_name",
    "old_subfolder",
    "new_subfolder",
    "is_intermediate",
    "old_path",
    "new_path",
    "old_thumbnail_path",
    "new_thumbnail_path",
    "state",
    "error_message",
]

# Identical to migration_33's definition, which now also carries the cascade — keep the two in sync.
_ITEMS_TABLE_BODY = """
    job_id INTEGER NOT NULL REFERENCES image_subfolder_move_jobs(id),
    image_name TEXT NOT NULL REFERENCES images(image_name) ON DELETE CASCADE,
    old_subfolder TEXT NOT NULL,
    new_subfolder TEXT NOT NULL,
    is_intermediate BOOLEAN NOT NULL DEFAULT FALSE,
    old_path TEXT,
    new_path TEXT,
    old_thumbnail_path TEXT,
    new_thumbnail_path TEXT,
    state TEXT NOT NULL CHECK (
        state IN ('planned', 'moved', 'committed', 'error')
    ),
    error_message TEXT,
    PRIMARY KEY (job_id, image_name)
"""


class CascadeImageSubfolderMoveItemsCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        if self._items_fk_is_missing_cascade(cursor):
            self._rebuild_items_table(cursor)

    def _create_items_indexes(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_image_subfolder_move_items_job_state
            ON image_subfolder_move_items(job_id, state);
            """
        )
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_image_subfolder_move_items_image_name
            ON image_subfolder_move_items(image_name);
            """
        )

    def _items_fk_is_missing_cascade(self, cursor: sqlite3.Cursor) -> bool:
        """True if the `images` foreign key exists but does not cascade deletes."""
        cursor.execute("PRAGMA foreign_key_list(image_subfolder_move_items);")
        # Row shape: (id, seq, table, from, to, on_update, on_delete, match)
        return any(row[2] == "images" and row[6] != "CASCADE" for row in cursor.fetchall())

    def _rebuild_items_table(self, cursor: sqlite3.Cursor) -> None:
        """Recreate the table with the cascading FK and copy the rows across.

        Both joins are required: the new table's foreign keys to `images` and
        `image_subfolder_move_jobs` are checked on the copying INSERT. Rows orphaned while foreign
        keys were off (e.g. by external maintenance scripts) would otherwise abort the migration,
        and since nothing would be recorded, every later startup would fail the same way. Dropping
        them is safe: an item whose image is gone can never commit anyway.
        `DROP TABLE` is safe because nothing references `image_subfolder_move_items` as a parent.
        """
        columns = ", ".join(_ITEMS_COLUMNS)
        qualified = ", ".join(f"items.{column}" for column in _ITEMS_COLUMNS)
        cursor.execute("DROP TABLE IF EXISTS image_subfolder_move_items_new;")
        cursor.execute(f"CREATE TABLE image_subfolder_move_items_new ({_ITEMS_TABLE_BODY});")
        cursor.execute(
            f"""--sql
            INSERT INTO image_subfolder_move_items_new ({columns})
            SELECT {qualified}
            FROM image_subfolder_move_items AS items
            JOIN images ON images.image_name = items.image_name
            JOIN image_subfolder_move_jobs AS jobs ON jobs.id = items.job_id;
            """
        )
        cursor.execute("DROP TABLE image_subfolder_move_items;")
        cursor.execute("ALTER TABLE image_subfolder_move_items_new RENAME TO image_subfolder_move_items;")
        # The old table's indexes were dropped along with it; recreate them on the new one.
        self._create_items_indexes(cursor)


def build_migration() -> Migration:
    """Add `ON DELETE CASCADE` to `image_subfolder_move_items.image_name` on existing databases."""
    return Migration(
        id="2026_08_08_repair_image_subfolder_move_tables",
        depends_on="migration_33",
        callback=CascadeImageSubfolderMoveItemsCallback(),
    )
