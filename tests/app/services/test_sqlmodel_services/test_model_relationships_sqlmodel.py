"""Tests for SqlModelModelRelationshipRecordStorage."""

import json

import pytest

from invokeai.app.services.model_relationship_records.model_relationship_records_sqlmodel import (
    SqlModelModelRelationshipRecordStorage,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


def _add_model(db: SqliteDatabase, key: str, name: str = "test") -> None:
    """Helper to insert a model record for FK constraints using raw SQL (avoids generated column issues)."""
    config = json.dumps(
        {
            "key": key,
            "name": name,
            "base": "sd-1",
            "type": "main",
            "format": "diffusers",
            "path": f"/models/{key}",
            "hash": "abc123",
            "source": "/src",
            "source_type": "path",
            "file_size": 1024,
        }
    )
    with db.transaction() as cursor:
        cursor.execute("INSERT INTO models (id, config) VALUES (?, ?)", (key, config))


@pytest.fixture
def storage(db: SqliteDatabase) -> SqlModelModelRelationshipRecordStorage:
    return SqlModelModelRelationshipRecordStorage(db=db)


@pytest.fixture
def models(db: SqliteDatabase) -> tuple[str, str, str]:
    keys = ("model_a", "model_b", "model_c")
    for k in keys:
        _add_model(db, k, name=k)
    return keys


def test_add_and_get_relationship(storage: SqlModelModelRelationshipRecordStorage, models: tuple):
    a, b, _ = models
    storage.add_model_relationship(a, b)
    related = storage.get_related_model_keys(a)
    assert b in related


def test_bidirectional(storage: SqlModelModelRelationshipRecordStorage, models: tuple):
    a, b, _ = models
    storage.add_model_relationship(a, b)
    assert a in storage.get_related_model_keys(b)
    assert b in storage.get_related_model_keys(a)


def test_self_relationship_raises(storage: SqlModelModelRelationshipRecordStorage, models: tuple):
    a, _, _ = models
    with pytest.raises(ValueError, match="Cannot relate a model to itself"):
        storage.add_model_relationship(a, a)


def test_remove_relationship(storage: SqlModelModelRelationshipRecordStorage, models: tuple):
    a, b, _ = models
    storage.add_model_relationship(a, b)
    storage.remove_model_relationship(a, b)
    assert b not in storage.get_related_model_keys(a)


def test_duplicate_add_is_idempotent(storage: SqlModelModelRelationshipRecordStorage, models: tuple):
    a, b, _ = models
    storage.add_model_relationship(a, b)
    storage.add_model_relationship(a, b)  # should not raise
    related = storage.get_related_model_keys(a)
    assert related.count(b) == 1


def test_get_related_batch(storage: SqlModelModelRelationshipRecordStorage, models: tuple):
    a, b, c = models
    storage.add_model_relationship(a, b)
    storage.add_model_relationship(a, c)
    related = storage.get_related_model_keys_batch([a])
    assert set(related) == {b, c}


def test_no_relationships(storage: SqlModelModelRelationshipRecordStorage, models: tuple):
    a, _, _ = models
    assert storage.get_related_model_keys(a) == []
