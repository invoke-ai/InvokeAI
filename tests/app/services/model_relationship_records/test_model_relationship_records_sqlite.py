"""Tests for SQLite model relationship storage contracts."""

import logging
import sqlite3

import pytest

from invokeai.app.services.model_relationship_records.model_relationship_records_sqlite import (
    SqliteModelRelationshipRecordStorage,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


@pytest.fixture
def storage() -> SqliteModelRelationshipRecordStorage:
    db = SqliteDatabase(db_path=None, logger=logging.getLogger(__name__))
    with db.transaction() as cursor:
        cursor.execute("CREATE TABLE models (id TEXT PRIMARY KEY NOT NULL)")
        cursor.execute(
            """
            CREATE TABLE model_relationships (
                model_key_1 TEXT NOT NULL,
                model_key_2 TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (model_key_1, model_key_2),
                FOREIGN KEY (model_key_1) REFERENCES models(id) ON DELETE CASCADE,
                FOREIGN KEY (model_key_2) REFERENCES models(id) ON DELETE CASCADE
            )
            """
        )
        cursor.executemany("INSERT INTO models (id) VALUES (?)", [("model-a",), ("model-b",)])
    return SqliteModelRelationshipRecordStorage(db)


def test_add_relationship_is_bidirectional_and_duplicate_is_idempotent(
    storage: SqliteModelRelationshipRecordStorage,
) -> None:
    storage.add_model_relationship("model-b", "model-a")

    assert storage.get_related_model_keys("model-a") == ["model-b"]
    assert storage.get_related_model_keys("model-b") == ["model-a"]

    storage.add_model_relationship("model-a", "model-b")
    assert storage.get_related_model_keys("model-a") == ["model-b"]
    assert storage.get_related_model_keys("model-b") == ["model-a"]


def test_add_relationship_rejects_missing_model_key(storage: SqliteModelRelationshipRecordStorage) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        storage.add_model_relationship("model-a", "missing-model")


def test_remove_missing_relationship_is_idempotent(storage: SqliteModelRelationshipRecordStorage) -> None:
    storage.remove_model_relationship("model-a", "model-b")
    assert storage.get_related_model_keys("model-a") == []


def test_delete_model_cascades_relationship(storage: SqliteModelRelationshipRecordStorage) -> None:
    storage.add_model_relationship("model-a", "model-b")
    db = storage._db
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM models WHERE id = ?", ("model-a",))

    assert storage.get_related_model_keys("model-b") == []
