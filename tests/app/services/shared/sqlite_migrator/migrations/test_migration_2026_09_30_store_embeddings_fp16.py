import logging
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from invokeai.app.services.image_index.image_index_common import blob_to_embedding, embedding_to_blob
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.migration_loader import (
    MigrationBuildContext,
    build_migrations,
    discover_migration_builders,
)
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_30_store_embeddings_fp16 import (
    build_migration,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration, MigrationError
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import SqliteMigrator

MIGRATION_ID = "2026_09_30_store_embeddings_fp16"
DEPENDENCY_ID = "2026_09_12_add_video_embeddings"


class _FakeConfig:
    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path
        self.models_path = root_path / "models"
        self.convert_cache_path = self.models_path / ".cache"
        self.legacy_conf_path = root_path / "models.yaml"
        self.legacy_conf_dir = root_path


def _vector(seed: int) -> np.ndarray:
    values = np.random.default_rng(seed).standard_normal(8).astype(np.float32)
    return values / np.linalg.norm(values)


def _create_schema(connection: sqlite3.Connection, *, image_encoding: bool = False) -> None:
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute("CREATE TABLE images (image_name TEXT PRIMARY KEY);")
    connection.execute("CREATE TABLE videos (video_name TEXT PRIMARY KEY);")
    image_encoding_sql = (
        ", encoding TEXT NOT NULL DEFAULT 'float32' CHECK (encoding IN ('float32', 'float16'))"
        if image_encoding
        else ""
    )
    connection.execute(
        f"""CREATE TABLE image_embeddings (
            image_name TEXT NOT NULL,
            model_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
                embedding BLOB NOT NULL{image_encoding_sql},
                created_at DATETIME NOT NULL DEFAULT 'image-created',
                PRIMARY KEY (image_name, model_id),
                FOREIGN KEY (image_name) REFERENCES images(image_name) ON DELETE CASCADE
            );"""
    )
    connection.execute(
        """CREATE TABLE video_embeddings (
            video_name TEXT NOT NULL,
            model_id TEXT NOT NULL,
            dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at DATETIME NOT NULL DEFAULT 'video-created',
            PRIMARY KEY (video_name, model_id),
            FOREIGN KEY (video_name) REFERENCES videos(video_name) ON DELETE CASCADE
        );"""
    )
    connection.execute("CREATE INDEX idx_image_embeddings_model_id ON image_embeddings(model_id);")
    connection.execute("CREATE INDEX idx_video_embeddings_model_id ON video_embeddings(model_id);")
    connection.commit()


def _make_db(tmp_path: Path) -> SqliteDatabase:
    db = SqliteDatabase(db_path=tmp_path / "embeddings.db", logger=logging.getLogger(__name__))
    _create_schema(db._conn)
    return db


def _migrator(db: SqliteDatabase) -> SqliteMigrator:
    migrator = SqliteMigrator(db)
    migrator.register_migration(Migration(id=DEPENDENCY_ID, callback=lambda cursor: None))
    migrator.register_migration(build_migration(logger=logging.getLogger(__name__)))
    return migrator


def _insert(
    connection: sqlite3.Connection,
    table: str,
    record_name: str,
    model_id: str,
    vector_blob: bytes,
    *,
    dim: int = 8,
    encoding: str | None = None,
    created_at: str = "2026-09-01 12:00:00.000",
) -> None:
    parent_table = "images" if table == "image_embeddings" else "videos"
    parent_column = "image_name" if table == "image_embeddings" else "video_name"
    connection.execute(f"INSERT OR IGNORE INTO {parent_table} ({parent_column}) VALUES (?);", (record_name,))
    if encoding is None:
        connection.execute(
            f"INSERT INTO {table} ({parent_column}, model_id, dim, embedding, created_at) VALUES (?, ?, ?, ?, ?);",
            (record_name, model_id, dim, vector_blob, created_at),
        )
    else:
        connection.execute(
            f"""INSERT INTO {table} ({parent_column}, model_id, dim, embedding, created_at, encoding)
                VALUES (?, ?, ?, ?, ?, ?);""",
            (record_name, model_id, dim, vector_blob, created_at, encoding),
        )


def _has_encoding(connection: sqlite3.Connection, table: str) -> bool:
    return any(row[1] == "encoding" for row in connection.execute(f"PRAGMA table_info({table});"))


def test_migration_is_discovered_with_stable_id_and_dependency(tmp_path: Path) -> None:
    context = MigrationBuildContext(
        app_config=_FakeConfig(tmp_path),
        logger=logging.getLogger(__name__),
        image_files=object(),  # type: ignore[arg-type]
    )
    migrations = build_migrations(context)
    migration = next(migration for migration in migrations if migration.id == MIGRATION_ID)
    module_names = {builder.module_name for builder in discover_migration_builders()}

    assert migration.depends_on == DEPENDENCY_ID
    assert "invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_30_store_embeddings_fp16" in (
        module_names
    )


def test_actual_migrator_converts_both_tables_and_preserves_metadata_and_relations(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    connection = db._conn
    image_vector = _vector(1)
    video_vector = _vector(2)
    image_blob = image_vector.astype(np.float32).tobytes()
    video_blob = video_vector.astype(np.float32).tobytes()
    _insert(connection, "image_embeddings", "image.png", "model-a", image_blob, created_at="image-time")
    _insert(connection, "video_embeddings", "video.mp4", "model-b", video_blob, created_at="video-time")
    image_indexes_before = connection.execute("PRAGMA index_list(image_embeddings);").fetchall()
    video_indexes_before = connection.execute("PRAGMA index_list(video_embeddings);").fetchall()
    image_fks_before = connection.execute("PRAGMA foreign_key_list(image_embeddings);").fetchall()
    video_fks_before = connection.execute("PRAGMA foreign_key_list(video_embeddings);").fetchall()
    connection.commit()

    _migrator(db).run_migrations()

    rows = {
        "image_embeddings": connection.execute(
            "SELECT image_name, model_id, dim, embedding, created_at, encoding FROM image_embeddings;"
        ).fetchall(),
        "video_embeddings": connection.execute(
            "SELECT video_name, model_id, dim, embedding, created_at, encoding FROM video_embeddings;"
        ).fetchall(),
    }
    assert rows["image_embeddings"][0][0:3] == ("image.png", "model-a", 8)
    assert rows["image_embeddings"][0][4:] == ("image-time", "float16")
    assert rows["video_embeddings"][0][0:3] == ("video.mp4", "model-b", 8)
    assert rows["video_embeddings"][0][4:] == ("video-time", "float16")
    assert len(rows["image_embeddings"][0][3]) == 8 * 2
    assert len(rows["video_embeddings"][0][3]) == 8 * 2
    assert np.allclose(blob_to_embedding(rows["image_embeddings"][0][3], 8, "float16"), image_vector, atol=1e-3)
    assert np.allclose(blob_to_embedding(rows["video_embeddings"][0][3], 8, "float16"), video_vector, atol=1e-3)
    assert connection.execute("PRAGMA index_list(image_embeddings);").fetchall() == image_indexes_before
    assert connection.execute("PRAGMA index_list(video_embeddings);").fetchall() == video_indexes_before
    assert connection.execute("PRAGMA foreign_key_list(image_embeddings);").fetchall() == image_fks_before
    assert connection.execute("PRAGMA foreign_key_list(video_embeddings);").fetchall() == video_fks_before
    for table in ("image_embeddings", "video_embeddings"):
        encoding_column = next(
            row for row in connection.execute(f"PRAGMA table_info({table});") if row[1] == "encoding"
        )
        assert encoding_column[2] == "TEXT"
        assert encoding_column[3] == 1
        assert encoding_column[4] == "'float32'"
    assert connection.execute(
        "SELECT migration_id FROM applied_migrations WHERE migration_id = ?;", (MIGRATION_ID,)
    ).fetchone()
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE image_embeddings SET encoding = 'float64';")
    connection.rollback()
    connection.execute("DELETE FROM images WHERE image_name = 'image.png';")
    connection.execute("DELETE FROM videos WHERE video_name = 'video.mp4';")
    assert tuple(connection.execute("SELECT COUNT(*) FROM image_embeddings;").fetchone()) == (0,)
    assert tuple(connection.execute("SELECT COUNT(*) FROM video_embeddings;").fetchone()) == (0,)
    db._conn.close()


def test_empty_and_mixed_rows_are_safe_and_callback_is_idempotent(tmp_path: Path) -> None:
    db = SqliteDatabase(db_path=tmp_path / "mixed.db", logger=logging.getLogger(__name__))
    _create_schema(db._conn, image_encoding=True)
    connection = db._conn
    preserved_blob = embedding_to_blob(_vector(4))
    _insert(connection, "image_embeddings", "already-fp16.png", "model-a", preserved_blob, encoding="float16")
    legacy_vector = _vector(5)
    _insert(
        connection,
        "image_embeddings",
        "legacy.png",
        "model-b",
        legacy_vector.astype(np.float32).tobytes(),
        encoding="float32",
    )
    connection.commit()

    _migrator(db).run_migrations()
    after_first_run = connection.execute(
        "SELECT image_name, embedding, encoding FROM image_embeddings ORDER BY image_name;"
    ).fetchall()
    build_migration(logger=logging.getLogger(__name__)).callback(connection.cursor())
    after_repeat = connection.execute(
        "SELECT image_name, embedding, encoding FROM image_embeddings ORDER BY image_name;"
    ).fetchall()

    assert after_repeat == after_first_run
    assert after_repeat[0][1] == preserved_blob
    assert after_repeat[0][2] == "float16"
    assert after_repeat[1][2] == "float16"
    assert tuple(connection.execute("SELECT COUNT(*) FROM video_embeddings;").fetchone()) == (0,)
    db._conn.rollback()
    db._conn.close()


def test_keyset_batches_convert_more_than_two_batches_across_models(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    connection = db._conn
    row_count = 514
    for index in range(257):
        vector = _vector(index + 20)
        for model_id in ("model-a", "model-b"):
            _insert(
                connection,
                "image_embeddings",
                f"image-{index:03d}.png",
                model_id,
                vector.astype(np.float32).tobytes(),
            )
    connection.commit()
    statements: list[str] = []
    connection.set_trace_callback(statements.append)

    _migrator(db).run_migrations()

    assert tuple(
        connection.execute("SELECT COUNT(*) FROM image_embeddings WHERE encoding = 'float16';").fetchone()
    ) == (row_count,)
    assert [
        tuple(row)
        for row in connection.execute(
            "SELECT model_id, COUNT(*) FROM image_embeddings GROUP BY model_id ORDER BY model_id;"
        )
    ] == [("model-a", 257), ("model-b", 257)]
    selects = [
        statement.lower()
        for statement in statements
        if "from image_embeddings" in statement.lower() and "order by rowid" in statement.lower()
    ]
    assert len(selects) >= 3
    assert all("limit 256" in statement for statement in selects)
    assert all("offset" not in statement for statement in selects)
    db._conn.close()


def test_migration_logs_progress_and_row_totals(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    connection = db._conn
    vector_blob = _vector(30).astype(np.float32).tobytes()
    names = [(f"image-{index:05d}.png",) for index in range(10_001)]
    connection.executemany("INSERT INTO images (image_name) VALUES (?);", names)
    connection.executemany(
        "INSERT INTO image_embeddings (image_name, model_id, dim, embedding) VALUES (?, 'model-a', 8, ?);",
        ((name[0], vector_blob) for name in names),
    )
    connection.commit()

    with caplog.at_level(logging.INFO, logger=__name__):
        _migrator(db).run_migrations()

    messages = [record.getMessage() for record in caplog.records if record.name == __name__]
    assert "Starting image and video embedding fp16 migration" in messages
    assert messages.count("Converted 10000 image and video embeddings to fp16") == 1
    assert any(
        "Completed image and video embedding fp16 migration: image rows=10001, video rows=0, total rows=10001, "
        "elapsed=" in message
        for message in messages
    )
    db._conn.close()


def test_late_bad_video_rolls_back_schema_and_prior_updates_then_retries(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    connection = db._conn
    image_blob = _vector(6).astype(np.float32).tobytes()
    video_blob = _vector(7).astype(np.float32).tobytes()
    for index in range(257):
        _insert(connection, "image_embeddings", f"good-{index:03d}.png", "model-a", image_blob)
    _insert(connection, "video_embeddings", "first.mp4", "model-a", video_blob)
    _insert(connection, "video_embeddings", "broken.mp4", "model-b", b"bad-vector", dim=8)
    connection.commit()
    migrator = _migrator(db)

    with pytest.raises(MigrationError, match=r"video_embeddings.*broken\.mp4.*model-b"):
        migrator.run_migrations()

    assert not _has_encoding(connection, "image_embeddings")
    assert not _has_encoding(connection, "video_embeddings")
    assert tuple(
        connection.execute("SELECT COUNT(*) FROM image_embeddings WHERE length(embedding) = 32;").fetchone()
    ) == (257,)
    assert tuple(
        connection.execute("SELECT embedding FROM video_embeddings WHERE video_name = 'first.mp4';").fetchone()
    ) == (video_blob,)
    assert (
        connection.execute(
            "SELECT migration_id FROM applied_migrations WHERE migration_id = ?;", (MIGRATION_ID,)
        ).fetchone()
        is None
    )

    connection.execute(
        "UPDATE video_embeddings SET embedding = ? WHERE video_name = 'broken.mp4' AND model_id = 'model-b';",
        (_vector(8).astype(np.float32).tobytes(),),
    )
    connection.commit()
    migrator.run_migrations()

    assert _has_encoding(connection, "image_embeddings")
    assert _has_encoding(connection, "video_embeddings")
    assert tuple(
        connection.execute("SELECT COUNT(*) FROM image_embeddings WHERE encoding = 'float16';").fetchone()
    ) == (257,)
    assert tuple(
        connection.execute("SELECT COUNT(*) FROM video_embeddings WHERE encoding = 'float16';").fetchone()
    ) == (2,)
    assert connection.execute(
        "SELECT migration_id FROM applied_migrations WHERE migration_id = ?;", (MIGRATION_ID,)
    ).fetchone()
    db._conn.close()
