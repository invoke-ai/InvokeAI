import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_08_03_add_image_index_tables import (
    AddImageIndexTablesCallback,
)
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_12_add_video_embeddings import (
    AddVideoEmbeddingsCallback,
)


def _make_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys = ON;")
    # Referenced by FKs: images/users (older migrations) and videos (2026_07_01_add_videos_tables).
    db.execute("CREATE TABLE images (image_name TEXT NOT NULL PRIMARY KEY);")
    db.execute("CREATE TABLE users (user_id TEXT NOT NULL PRIMARY KEY);")
    db.execute("CREATE TABLE videos (video_name TEXT NOT NULL PRIMARY KEY);")
    AddImageIndexTablesCallback()(db.cursor())
    return db


def test_creates_the_video_embeddings_table_with_its_cascade() -> None:
    db = _make_db()

    AddVideoEmbeddingsCallback()(db.cursor())

    db.execute("INSERT INTO videos VALUES ('v1');")
    db.execute("INSERT INTO video_embeddings (video_name, model_id, dim, embedding) VALUES ('v1', 'm1', 2, x'00');")

    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_video_embeddings_model_id';")
    assert cursor.fetchone() is not None

    # An embedding must not outlive the video it describes.
    db.execute("DELETE FROM videos WHERE video_name = 'v1';")
    cursor.execute("SELECT COUNT(*) FROM video_embeddings;")
    assert cursor.fetchone() == (0,)
    db.close()


def test_an_embedding_for_a_missing_video_is_refused() -> None:
    db = _make_db()
    AddVideoEmbeddingsCallback()(db.cursor())

    try:
        db.execute(
            "INSERT INTO video_embeddings (video_name, model_id, dim, embedding) VALUES ('gone.mp4', 'm1', 2, x'00');"
        )
        raise AssertionError("the foreign key should have refused this row")
    except sqlite3.IntegrityError:
        pass
    db.close()


def test_existing_projections_survive_and_read_as_all_images() -> None:
    # The upgrade must not invalidate cached projections: a NULL item_kinds means what every
    # projection meant before videos were indexable.
    db = _make_db()
    db.execute("INSERT INTO users VALUES ('u1');")
    db.execute(
        "INSERT INTO image_projections (user_id, model_id, scope_hash, params, point_count, image_names, coords)"
        " VALUES ('u1', 'm1', 'h', '{}', 1, '[\"a.png\"]', x'00');"
    )

    AddVideoEmbeddingsCallback()(db.cursor())

    cursor = db.cursor()
    cursor.execute("SELECT image_names, item_kinds FROM image_projections WHERE user_id = 'u1';")
    assert cursor.fetchone() == ('["a.png"]', None)
    db.close()


def test_is_idempotent_on_a_database_that_already_migrated() -> None:
    db = _make_db()
    AddVideoEmbeddingsCallback()(db.cursor())
    db.execute("INSERT INTO videos VALUES ('v1');")
    db.execute("INSERT INTO video_embeddings (video_name, model_id, dim, embedding) VALUES ('v1', 'm1', 2, x'00');")

    # The ALTER TABLE would raise "duplicate column name" without the guard, which would fail
    # the whole startup migration run.
    AddVideoEmbeddingsCallback()(db.cursor())

    cursor = db.cursor()
    cursor.execute("SELECT video_name FROM video_embeddings;")
    assert cursor.fetchall() == [("v1",)]
    db.close()


def test_migration_refuses_to_run_before_the_videos_table_exists() -> None:
    """The foreign key's target is created by an older migration, and SQLite resolves targets
    lazily — so a reordering would fail obscurely at the first embedding rather than here."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE users (user_id TEXT NOT NULL PRIMARY KEY);")
    db.execute("CREATE TABLE images (image_name TEXT NOT NULL PRIMARY KEY);")
    AddImageIndexTablesCallback()(db.cursor())

    try:
        AddVideoEmbeddingsCallback()(db.cursor())
        raise AssertionError("the migration should have refused to run")
    except RuntimeError as error:
        assert "videos table does not exist" in str(error)
    db.close()
