import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_08_03_add_image_index_tables import (
    AddImageIndexTablesCallback,
    build_migration,
)


def _make_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys = ON;")
    # Referenced by the new tables' FKs: images (migration_1) and users (migration_27).
    db.execute("CREATE TABLE images (image_name TEXT NOT NULL PRIMARY KEY);")
    db.execute("CREATE TABLE users (user_id TEXT NOT NULL PRIMARY KEY);")
    return db


def test_creates_the_image_index_tables() -> None:
    db = _make_db()

    AddImageIndexTablesCallback()(db.cursor())

    db.execute("INSERT INTO images VALUES ('i1');")
    db.execute("INSERT INTO users VALUES ('u1');")
    db.execute("INSERT INTO image_embeddings (image_name, model_id, dim, embedding) VALUES ('i1', 'm1', 2, x'00');")
    db.execute(
        "INSERT INTO image_projections (user_id, model_id, scope_hash, params, point_count, image_names, coords)"
        " VALUES ('u1', 'm1', 'h', '{}', 1, '[]', x'00');"
    )

    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_image_embeddings_model_id';")
    assert cursor.fetchone() is not None
    cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='tg_image_projections_updated_at';")
    assert cursor.fetchone() is not None

    # FK cascade: deleting an image removes its embeddings.
    db.execute("DELETE FROM images WHERE image_name = 'i1';")
    cursor.execute("SELECT COUNT(*) FROM image_embeddings;")
    assert cursor.fetchone() == (0,)
    db.close()


def test_is_idempotent_on_a_database_that_already_has_the_tables() -> None:
    """Databases that acquired these tables from a pre-rename numeric build of this feature must
    no-op rather than fail, and existing rows must survive."""
    db = _make_db()
    AddImageIndexTablesCallback()(db.cursor())
    db.execute("INSERT INTO images VALUES ('i1');")
    db.execute("INSERT INTO image_embeddings (image_name, model_id, dim, embedding) VALUES ('i1', 'm1', 2, x'00');")

    AddImageIndexTablesCallback()(db.cursor())

    cursor = db.cursor()
    cursor.execute("SELECT image_name FROM image_embeddings;")
    assert cursor.fetchall() == [("i1",)]
    db.close()


def test_migration_has_a_dated_id_so_it_cannot_collide_with_an_upstream_number() -> None:
    migration = build_migration()
    assert migration.id == "2026_08_03_add_image_index_tables"
    assert migration.depends_on == "migration_33"
    assert migration.from_version is None
    assert migration.to_version is None
