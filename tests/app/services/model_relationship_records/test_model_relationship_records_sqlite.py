"""Tests for SQLite model relationship storage contracts."""

import logging
import sqlite3
from pathlib import Path

import pytest

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import ModelRecordServiceSQL
from invokeai.app.services.model_relationship_records.model_relationship_records_sqlite import (
    SqliteModelRelationshipRecordStorage,
)
from invokeai.backend.model_manager.configs.textual_inversion import TI_File_SD1_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelSourceType, ModelType
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def storage(tmp_path: Path) -> SqliteModelRelationshipRecordStorage:
    logger = logging.getLogger(__name__)
    config = InvokeAIAppConfig()
    config._root = tmp_path
    db = create_mock_sqlite_database(config, logger)
    records = ModelRecordServiceSQL(db, logger)
    for key in ("model-a", "model-b"):
        records.add_model(
            TI_File_SD1_Config(
                key=key,
                source=f"test/{key}",
                source_type=ModelSourceType.Path,
                path=f"/tmp/{key}.safetensors",
                file_size=1,
                name=key,
                base=BaseModelType.StableDiffusion1,
                type=ModelType.TextualInversion,
                format=ModelFormat.EmbeddingFile,
                hash=f"hash-{key}",
            )
        )
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


def test_delete_model_cascades_relationship_in_migrated_schema(storage: SqliteModelRelationshipRecordStorage) -> None:
    storage.add_model_relationship("model-a", "model-b")
    with storage._db.transaction() as cursor:
        cursor.execute("DELETE FROM models WHERE id = ?", ("model-a",))

    assert storage.get_related_model_keys("model-b") == []
