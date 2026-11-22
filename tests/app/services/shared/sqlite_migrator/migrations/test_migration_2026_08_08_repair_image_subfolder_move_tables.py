import sqlite3

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_08_08_repair_image_subfolder_move_tables import (
    CascadeImageSubfolderMoveItemsCallback,
    build_migration,
)

# migration_33's original definition, with the NO ACTION foreign key this migration repairs.
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

_JOBS_TABLE = """
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


def _make_legacy_db() -> sqlite3.Connection:
    """A database as migration_33 originally left it: move tables present, no cascade."""
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys = ON;")
    db.execute("CREATE TABLE images (image_name TEXT NOT NULL PRIMARY KEY);")
    db.execute(_JOBS_TABLE)
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
    """The defect this migration exists to fix (#9504, #9782)."""
    db = _make_legacy_db()
    db.execute("INSERT INTO images VALUES ('moved.png');")
    _add_move_item(db, "moved.png")

    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM images WHERE image_name = 'moved.png';")


def test_repairs_missing_cascade_and_preserves_rows() -> None:
    db = _make_legacy_db()
    db.execute("INSERT INTO images VALUES ('moved.png');")
    _add_move_item(db, "moved.png")

    CascadeImageSubfolderMoveItemsCallback()(db.cursor())

    assert _on_delete_action(db) == "CASCADE"
    assert db.execute(
        "SELECT job_id, image_name, new_subfolder, state FROM image_subfolder_move_items;"
    ).fetchall() == [(1, "moved.png", "a1", "committed")]

    db.execute("DELETE FROM images WHERE image_name = 'moved.png';")
    assert db.execute("SELECT COUNT(*) FROM image_subfolder_move_items;").fetchone() == (0,)


@pytest.mark.parametrize("orphan_parent", ["images", "jobs"])
def test_rebuild_drops_orphaned_rows(orphan_parent: str) -> None:
    """Rows orphaned while foreign keys were off must not abort the copy into the new table."""
    db = _make_legacy_db()
    db.execute("INSERT INTO images VALUES ('kept.png'), ('orphan.png');")
    _add_move_item(db, "kept.png", job_id=1)
    _add_move_item(db, "orphan.png", job_id=2)
    # `PRAGMA foreign_keys` is a no-op inside a transaction; commit the implicit one first.
    db.commit()
    db.execute("PRAGMA foreign_keys = OFF;")
    if orphan_parent == "images":
        db.execute("DELETE FROM images WHERE image_name = 'orphan.png';")
    else:
        db.execute("DELETE FROM image_subfolder_move_jobs WHERE id = 2;")
    db.commit()
    db.execute("PRAGMA foreign_keys = ON;")

    CascadeImageSubfolderMoveItemsCallback()(db.cursor())

    assert db.execute("SELECT image_name FROM image_subfolder_move_items;").fetchall() == [("kept.png",)]


def test_recreates_indexes_after_rebuild() -> None:
    db = _make_legacy_db()

    CascadeImageSubfolderMoveItemsCallback()(db.cursor())

    names = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='image_subfolder_move_items';"
        ).fetchall()
    }
    assert "idx_image_subfolder_move_items_job_state" in names
    assert "idx_image_subfolder_move_items_image_name" in names
    assert db.execute("SELECT name FROM sqlite_master WHERE name='image_subfolder_move_items_new';").fetchone() is None


def test_already_cascading_table_is_not_rebuilt() -> None:
    db = _make_legacy_db()
    CascadeImageSubfolderMoveItemsCallback()(db.cursor())

    # Leave a rowid gap so a rebuild (which renumbers) is distinguishable from a no-op.
    db.execute("INSERT INTO images VALUES ('gone.png'), ('kept.png');")
    _add_move_item(db, "gone.png")
    _add_move_item(db, "kept.png")
    db.execute("DELETE FROM images WHERE image_name = 'gone.png';")
    assert db.execute("SELECT rowid FROM image_subfolder_move_items;").fetchall() == [(2,)]

    CascadeImageSubfolderMoveItemsCallback()(db.cursor())

    assert db.execute("SELECT rowid FROM image_subfolder_move_items;").fetchall() == [(2,)]


def test_migration_metadata() -> None:
    migration = build_migration()
    assert migration.id == "2026_08_08_repair_image_subfolder_move_tables"
    assert migration.depends_on == "migration_33"
