"""Model relationships on every database backend."""

import pytest

from invokeai.app.services.model_records import ModelRecordServiceSQL
from invokeai.app.services.model_relationship_records.model_relationship_records_default import (
    ModelRelationshipRecordStorage,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import ForeignKeyViolation
from invokeai.backend.model_manager.configs.textual_inversion import TI_File_SD1_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelSourceType, ModelType
from invokeai.backend.util.logging import InvokeAILogger


@pytest.fixture
def models(database: Database) -> ModelRecordServiceSQL:
    store = ModelRecordServiceSQL(database, InvokeAILogger.get_logger())
    for key in ("a", "b", "c", "d"):
        store.add_model(
            TI_File_SD1_Config(
                key=key,
                path=f"/tmp/{key}.bin",
                name=key,
                hash="ABC123",
                file_size=1024,
                source="test/source/",
                source_type=ModelSourceType.Path,
                base=BaseModelType.StableDiffusion1,
                type=ModelType.TextualInversion,
                format=ModelFormat.EmbeddingFile,
            )
        )
    return store


@pytest.fixture
def relationships(database: Database, models: ModelRecordServiceSQL) -> ModelRelationshipRecordStorage:
    return ModelRelationshipRecordStorage(database)


def test_a_relationship_is_seen_from_both_models_in_order(relationships: ModelRelationshipRecordStorage) -> None:
    relationships.add_model_relationship("d", "c")
    relationships.add_model_relationship("c", "a")
    relationships.add_model_relationship("b", "c")

    # A pair is stored with its smaller key first, so c's relatives come from both sides: d after it, a and b before.
    assert relationships.get_related_model_keys("c") == ["a", "b", "d"]
    assert relationships.get_related_model_keys("a") == ["c"]
    assert relationships.get_related_model_keys("d") == ["c"]


def test_relating_again_from_either_side_stores_the_relationship_once(
    relationships: ModelRelationshipRecordStorage,
) -> None:
    relationships.add_model_relationship("a", "c")
    relationships.add_model_relationship("c", "a")
    relationships.add_model_relationship("a", "c")

    # Stored once, so one removal ends it.
    relationships.remove_model_relationship("c", "a")

    assert relationships.get_related_model_keys("a") == []


def test_removing_a_relationship_leaves_the_others_and_again_is_harmless(
    relationships: ModelRelationshipRecordStorage,
) -> None:
    relationships.add_model_relationship("a", "b")
    relationships.add_model_relationship("a", "c")

    relationships.remove_model_relationship("b", "a")
    relationships.remove_model_relationship("b", "a")

    assert relationships.get_related_model_keys("a") == ["c"]
    assert relationships.get_related_model_keys("b") == []


def test_a_model_is_related_neither_to_itself_nor_to_one_that_does_not_exist(
    relationships: ModelRelationshipRecordStorage,
) -> None:
    with pytest.raises(ValueError):
        relationships.add_model_relationship("a", "a")
    # Only a relationship that exists already is skipped, not a missing model.
    with pytest.raises(ForeignKeyViolation):
        relationships.add_model_relationship("a", "unknown")


def test_deleting_a_model_ends_its_relationships(
    relationships: ModelRelationshipRecordStorage, models: ModelRecordServiceSQL
) -> None:
    relationships.add_model_relationship("a", "b")
    relationships.add_model_relationship("b", "c")
    relationships.add_model_relationship("a", "d")

    models.del_model("b")

    assert relationships.get_related_model_keys("a") == ["d"]
    assert relationships.get_related_model_keys("c") == []
