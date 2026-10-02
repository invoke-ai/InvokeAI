"""Repair `image_subfolder_move_items` on databases that need it, for two independent reasons.

**1. The missing `ON DELETE CASCADE`.**

`migration_34` originally declared `image_name TEXT NOT NULL REFERENCES images(image_name)` with
no `ON DELETE` action, i.e. `NO ACTION`. Move items are an audit trail: they are inserted when a
subfolder move runs and updated to `committed`, but nothing ever deletes them. Foreign keys are
enforced on the app connection (`SqliteDatabase.__init__`), so a `NO ACTION` FK pointing at a row
that is never cleaned up permanently blocks `DELETE FROM images` for every image that has ever
been through a move. Any install running a non-`flat` `image_subfolder_strategy` could not delete
those images, and `invokeai-db-maintenance --operation clean` could not clean them either.

`migration_34`'s DDL now carries the cascade, which fixes freshly-created databases. This
migration rebuilds the table on databases that already have the cascade-less version. SQLite
cannot `ALTER` a foreign key, so the table must be recreated and the rows copied across.

One case does get worse, and it is the lesser evil rather than a clean win. Deleting an image
while a move job is mid-flight — file already relocated, `images.image_subfolder` not yet
updated — used to fail with the same IntegrityError; now it succeeds, the cascade drops the
item row, and the relocated file is left on disk with nothing referencing it. Image deletion is
not gated on an active job (`_has_active_job_for_image` is consulted only when planning). That
narrow leak is preferable to images that can never be deleted at all, but it is the reason not
to describe move items as a pure audit trail: the cascade does delete them.

**2. The burned `migration_34` slot.**

A pre-rename build of the image-map feature (commit `b3f46181d`) shipped the image index tables
as a numeric `migration_34`, colliding with this fork's `migration_34`. A database opened by that
build records `migration_34` as applied, so the migrator skips the real one forever and the move
tables are never created — the app then fails at startup in `ImageMoveService.startup_recovery()`
with `no such table: image_subfolder_move_jobs`.

This is the same failure mode, and the same remedy, as
`migration_2026_07_30_repair_projects_table`: a dated id cannot collide, so this migration runs
exactly once on every database and creates whatever is missing. The `CREATE TABLE` statements
below are copies of `migration_34`'s with the cascade added — keep the two in sync.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

_CREATE_JOBS_TABLE = """--sql
CREATE TABLE IF NOT EXISTS image_subfolder_move_jobs (
    id INTEGER PRIMARY KEY,
    state TEXT NOT NULL CHECK (
        state IN ('planned', 'moving', 'moved', 'committed', 'error')
    ),
    created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
    updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
    error_message TEXT
);
"""

# Column list in table order, used to build both the CREATE and the copying INSERT/SELECT.
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

# Identical to migration_34's definition, except `image_name` carries ON DELETE CASCADE.
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


class RepairImageSubfolderMoveTablesCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_missing_tables(cursor)
        if self._items_fk_is_missing_cascade(cursor):
            self._rebuild_items_table(cursor)

    def _create_missing_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create the move tables if the burned `migration_34` slot meant they never were."""
        cursor.execute(_CREATE_JOBS_TABLE)
        cursor.execute(f"CREATE TABLE IF NOT EXISTS image_subfolder_move_items ({_ITEMS_TABLE_BODY});")
        self._create_items_indexes(cursor)
        cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_image_subfolder_move_jobs_updated_at
            AFTER UPDATE
            ON image_subfolder_move_jobs FOR EACH ROW
            BEGIN
                UPDATE image_subfolder_move_jobs
                SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE id = old.id;
            END;
            """
        )

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

        SQLite cannot alter a foreign key in place. Foreign keys are enabled on this connection,
        which constrains the copy in two ways:

        - Both joins are required, not merely tidy. The new table declares foreign keys to
          `images` *and* `image_subfolder_move_jobs`, and both are checked on the copying
          INSERT. Deleting a parent used to be possible with foreign keys off —
          `gallery_maintenance.py` did exactly that before it started enabling the pragma — so
          orphaned item rows can exist. An orphan that reached the INSERT would abort the
          migration inside the migrator's transaction, and since nothing would then be
          recorded, every subsequent startup would fail the same way with no path to recovery.
          Dropping the rows is safe: `commit_database_updates` already treats an item whose
          image is gone as a hard failure, so these rows only block a job that can never
          commit. (`migration_2026_08_06_add_project_boards` quarantines instead of dropping,
          because there the rows are user-authored documents rather than an audit trail.)
        - `DROP TABLE` is safe: nothing references `image_subfolder_move_items` as a parent, so
          the implicit row removal cannot violate another table's constraint.
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
    """Build the migration that repairs `image_subfolder_move_items`:
    - Adds the missing `ON DELETE CASCADE` on `image_name`.
    - Creates the move tables on databases where the burned `migration_34` slot skipped them.
    """
    return Migration(
        id="2026_08_08_repair_image_subfolder_move_tables",
        # migration_34 created these tables; this repairs its output. The dated id means this
        # migration runs even on databases that wrongly recorded migration_34 as applied.
        depends_on="migration_34",
        callback=RepairImageSubfolderMoveTablesCallback(),
    )
