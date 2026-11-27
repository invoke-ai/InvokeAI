import sqlite3

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_08_08_repair_image_subfolder_move_tables import (
    RepairImageSubfolderMoveTablesCallback,
    build_migration,
)

# migration_34's original definition, with the NO ACTION foreign key this migration repairs.
_LEGACY_ITEMS_TABLE = """
CREATE TABLE image_subfolder_move_items (
    job_id INTEGER NOT NULL REFERENCES image_subfolder_move_jobs(id),
    image_name TEXT NOT NULL REFERENCES images(image_name),
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
);
"""

_LEGACY_JOBS_TABLE = """
CREATE TABLE image_subfolder_move_jobs (
    id INTEGER PRIMARY KEY,
    state TEXT NOT NULL CHECK (
        state IN ('planned', 'moving', 'moved', 'committed', 'error')
    ),
    created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
    updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
    error_message TEXT
);
"""


def _make_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys = ON;")
    db.execute("CREATE TABLE images (image_name TEXT NOT NULL PRIMARY KEY);")
    return db


def _make_legacy_db() -> sqlite3.Connection:
    """A database as migration_34 originally left it: move tables present, no cascade."""
    db = _make_db()
    db.execute(_LEGACY_JOBS_TABLE)
    db.execute(_LEGACY_ITEMS_TABLE)
    db.execute("CREATE INDEX idx_image_subfolder_move_items_job_state ON image_subfolder_move_items(job_id, state);")
    db.execute("CREATE INDEX idx_image_subfolder_move_items_image_name ON image_subfolder_move_items(image_name);")
    return db


def _add_move_item(db: sqlite3.Connection, image_name: str, job_id: int = 1) -> None:
    db.execute("INSERT OR IGNORE INTO image_subfolder_move_jobs (id, state) VALUES (?, 'committed');", (job_id,))
    db.execute(
        "INSERT INTO image_subfolder_move_items (job_id, image_name, old_subfolder, new_subfolder, state)"
        " VALUES (?, ?, '', 'a1', 'committed');",
        (job_id, image_name),
    )


def _on_delete_action(db: sqlite3.Connection) -> str:
    for row in db.execute("PRAGMA foreign_key_list(image_subfolder_move_items);").fetchall():
        if row[2] == "images":
            return str(row[6])
    raise AssertionError("no foreign key to images")


def test_legacy_schema_blocks_image_deletion() -> None:
    """The defect this migration exists to fix, pinned so the repair cannot be declared unnecessary."""
    db = _make_legacy_db()
    db.execute("INSERT INTO images VALUES ('moved.png');")
    _add_move_item(db, "moved.png")

    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM images WHERE image_name = 'moved.png';")


def test_repairs_missing_cascade_and_preserves_rows() -> None:
    db = _make_legacy_db()
    db.execute("INSERT INTO images VALUES ('moved.png');")
    _add_move_item(db, "moved.png")

    RepairImageSubfolderMoveTablesCallback()(db.cursor())

    assert _on_delete_action(db) == "CASCADE"
    assert db.execute(
        "SELECT job_id, image_name, new_subfolder, state FROM image_subfolder_move_items;"
    ).fetchall() == [(1, "moved.png", "a1", "committed")]

    # The deletion that used to fail now succeeds and takes the audit row with it.
    db.execute("DELETE FROM images WHERE image_name = 'moved.png';")
    assert db.execute("SELECT COUNT(*) FROM image_subfolder_move_items;").fetchone() == (0,)


def test_rebuild_drops_rows_orphaned_while_foreign_keys_were_off() -> None:
    """gallery_maintenance.py used to delete images with FKs off, which could orphan move items.

    Those rows cannot be carried into a table that declares a cascading FK.
    """
    db = _make_legacy_db()
    db.execute("INSERT INTO images VALUES ('kept.png');")
    _add_move_item(db, "kept.png")
    db.execute("INSERT INTO images VALUES ('vanished.png');")
    _add_move_item(db, "vanished.png")
    # `PRAGMA foreign_keys` is a no-op inside a transaction, and the inserts above opened one
    # implicitly, so commit before toggling it off.
    db.commit()
    db.execute("PRAGMA foreign_keys = OFF;")
    db.execute("DELETE FROM images WHERE image_name = 'vanished.png';")
    db.commit()
    db.execute("PRAGMA foreign_keys = ON;")

    RepairImageSubfolderMoveTablesCallback()(db.cursor())

    assert db.execute("SELECT image_name FROM image_subfolder_move_items;").fetchall() == [("kept.png",)]


def test_rebuild_drops_rows_orphaned_against_the_jobs_table() -> None:
    """The copy must guard BOTH foreign keys, not just the one to images.

    An item row whose job is missing would fail the new table's FK on insert, aborting the
    migration inside the migrator's transaction. Nothing would be recorded, so every later
    startup would fail identically with no route to recovery.
    """
    db = _make_legacy_db()
    db.execute("INSERT INTO images VALUES ('kept.png'), ('orphan.png');")
    _add_move_item(db, "kept.png", job_id=1)
    _add_move_item(db, "orphan.png", job_id=2)
    db.commit()
    db.execute("PRAGMA foreign_keys = OFF;")
    db.execute("DELETE FROM image_subfolder_move_jobs WHERE id = 2;")
    db.commit()
    db.execute("PRAGMA foreign_keys = ON;")

    RepairImageSubfolderMoveTablesCallback()(db.cursor())

    assert db.execute("SELECT image_name FROM image_subfolder_move_items;").fetchall() == [("kept.png",)]


def test_recreates_indexes_after_rebuild() -> None:
    db = _make_legacy_db()

    RepairImageSubfolderMoveTablesCallback()(db.cursor())

    names = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='image_subfolder_move_items';"
        ).fetchall()
    }
    assert "idx_image_subfolder_move_items_job_state" in names
    assert "idx_image_subfolder_move_items_image_name" in names
    # The scratch table must not survive the rebuild.
    assert db.execute("SELECT name FROM sqlite_master WHERE name='image_subfolder_move_items_new';").fetchone() is None


def test_creates_tables_when_the_burned_migration_34_slot_skipped_them() -> None:
    """A database from the pre-rename image-index build recorded migration_34 but has no move tables."""
    db = _make_db()

    RepairImageSubfolderMoveTablesCallback()(db.cursor())

    db.execute("INSERT INTO images VALUES ('i1.png');")
    _add_move_item(db, "i1.png")
    assert _on_delete_action(db) == "CASCADE"
    assert db.execute("SELECT name FROM sqlite_master WHERE type='trigger';").fetchone() is not None


def test_is_idempotent() -> None:
    db = _make_legacy_db()
    db.execute("INSERT INTO images VALUES ('moved.png');")
    _add_move_item(db, "moved.png")

    RepairImageSubfolderMoveTablesCallback()(db.cursor())
    RepairImageSubfolderMoveTablesCallback()(db.cursor())

    assert _on_delete_action(db) == "CASCADE"
    assert db.execute("SELECT COUNT(*) FROM image_subfolder_move_items;").fetchone() == (1,)


def test_already_repaired_database_is_not_rebuilt() -> None:
    db = _make_db()
    RepairImageSubfolderMoveTablesCallback()(db.cursor())
    assert RepairImageSubfolderMoveTablesCallback()._items_fk_is_missing_cascade(db.cursor()) is False

    # Leave a rowid gap: two rows inserted, the first deleted, so the survivor sits at rowid 2.
    # A rebuild would copy it into a fresh table and renumber it to 1, so this discriminates
    # between short-circuiting and rebuilding — which a single-row table would not.
    db.execute("INSERT INTO images VALUES ('gone.png'), ('kept.png');")
    _add_move_item(db, "gone.png")
    _add_move_item(db, "kept.png")
    db.execute("DELETE FROM images WHERE image_name = 'gone.png';")
    assert db.execute("SELECT rowid FROM image_subfolder_move_items;").fetchall() == [(2,)]

    RepairImageSubfolderMoveTablesCallback()(db.cursor())

    assert db.execute("SELECT rowid FROM image_subfolder_move_items;").fetchall() == [(2,)]


def test_migration_metadata() -> None:
    migration = build_migration()
    assert migration.id == "2026_08_08_repair_image_subfolder_move_tables"
    assert migration.depends_on == "migration_34"
