"""
Test the refactored model config classes.
"""

from hashlib import sha256
from typing import Any, Optional

import pytest
from pydantic import ValidationError

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import (
    DuplicateModelException,
    ModelRecordServiceBase,
    ModelRecordServiceSQL,
    UnknownModelException,
)
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges
from invokeai.backend.model_manager.config import (
    BaseModelType,
    MainDiffusersConfig,
    ModelFormat,
    ModelSourceType,
    ModelType,
    TextualInversionFileConfig,
    VaeDiffusersConfig,
)
from invokeai.backend.util.logging import InvokeAILogger
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def store(
    datadir: Any,
) -> ModelRecordServiceSQL:
    config = InvokeAIAppConfig(root=datadir)
    logger = InvokeAILogger.get_logger(config=config)
    db = create_mock_sqlite_database(config, logger)
    return ModelRecordServiceSQL(db)


def example_ti_config(key: Optional[str] = None) -> TextualInversionFileConfig:
    config = TextualInversionFileConfig(
        source="test/source/",
        source_type=ModelSourceType.Path,
        path="/tmp/pokemon.bin",
        name="old name",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.TextualInversion,
        format=ModelFormat.EmbeddingFile,
        hash="ABC123",
    )
    if key is not None:
        config.key = key
    return config


def test_type(store: ModelRecordServiceBase):
    config = example_ti_config("key1")
    store.add_model(config)
    config1 = store.get_model("key1")
    assert isinstance(config1, TextualInversionFileConfig)


def test_raises_on_violating_uniqueness(store: ModelRecordServiceBase):
    # Models have a uniqueness constraint by their name, base and type
    config1 = example_ti_config("key1")
    config2 = config1.model_copy(deep=True)
    config2.key = "key2"
    store.add_model(config1)
    with pytest.raises(DuplicateModelException):
        store.add_model(config1)
    with pytest.raises(DuplicateModelException):
        store.add_model(config2)


def test_model_records_updates_model(store: ModelRecordServiceBase):
    config = example_ti_config("key1")
    store.add_model(config)
    config = store.get_model("key1")
    assert config.name == "old name"
    new_name = "new name"
    changes = ModelRecordChanges(name=new_name)
    store.update_model(config.key, changes)
    new_config = store.get_model("key1")
    assert new_config.name == new_name


def test_model_records_rejects_invalid_changes(store: ModelRecordServiceBase):
    config = example_ti_config("key1")
    store.add_model(config)
    config = store.get_model("key1")
    # upcast_attention is an invalid field for TIs
    changes = ModelRecordChanges(upcast_attention=True)
    with pytest.raises(ValidationError):
        store.update_model(config.key, changes)


def test_unknown_key(store: ModelRecordServiceBase):
    config = example_ti_config("key1")
    store.add_model(config)
    with pytest.raises(UnknownModelException):
        store.update_model("unknown_key", ModelRecordChanges())


def test_delete(store: ModelRecordServiceBase):
    config = example_ti_config("key1")
    store.add_model(config)
    config = store.get_model("key1")
    store.del_model("key1")
    with pytest.raises(UnknownModelException):
        config = store.get_model("key1")


def test_exists(store: ModelRecordServiceBase):
    config = example_ti_config("key1")
    store.add_model(config)
    assert store.exists("key1")
    assert not store.exists("key2")


def test_filter(store: ModelRecordServiceBase):
    config1 = MainDiffusersConfig(
        key="config1",
        path="/tmp/config1",
        name="config1",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        hash="CONFIG1HASH",
        source="test/source",
        source_type=ModelSourceType.Path,
    )
    config2 = MainDiffusersConfig(
        key="config2",
        path="/tmp/config2",
        name="config2",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        hash="CONFIG2HASH",
        source="test/source",
        source_type=ModelSourceType.Path,
    )
    config3 = VaeDiffusersConfig(
        key="config3",
        path="/tmp/config3",
        name="config3",
        base=BaseModelType("sd-2"),
        type=ModelType.Vae,
        hash="CONFIG3HASH",
        source="test/source",
        source_type=ModelSourceType.Path,
    )
    for c in config1, config2, config3:
        store.add_model(c)
    matches = store.search_by_attr(model_type=ModelType.Main)
    assert len(matches) == 2
    assert matches[0].name in {"config1", "config2"}

    matches = store.search_by_attr(model_type=ModelType.Vae)
    assert len(matches) == 1
    assert matches[0].name == "config3"
    assert matches[0].key == "config3"
    assert isinstance(matches[0].type, ModelType)  # This tests that we get proper enums back

    matches = store.search_by_hash("CONFIG1HASH")
    assert len(matches) == 1
    assert matches[0].hash == "CONFIG1HASH"

    matches = store.all_models()
    assert len(matches) == 3


def test_unique(store: ModelRecordServiceBase):
    config1 = MainDiffusersConfig(
        path="/tmp/config1",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        name="nonuniquename",
        hash="CONFIG1HASH",
        source="test/source/",
        source_type=ModelSourceType.Path,
    )
    config2 = MainDiffusersConfig(
        path="/tmp/config2",
        base=BaseModelType("sd-2"),
        type=ModelType.Main,
        name="nonuniquename",
        hash="CONFIG1HASH",
        source="test/source/",
        source_type=ModelSourceType.Path,
    )
    config3 = VaeDiffusersConfig(
        path="/tmp/config3",
        base=BaseModelType("sd-2"),
        type=ModelType.Vae,
        name="nonuniquename",
        hash="CONFIG1HASH",
        source="test/source/",
        source_type=ModelSourceType.Path,
    )
    config4 = MainDiffusersConfig(
        path="/tmp/config4",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        name="nonuniquename",
        hash="CONFIG1HASH",
        source="test/source/",
        source_type=ModelSourceType.Path,
    )
    # config1, config2 and config3 are compatible because they have unique combos
    # of name, type and base
    for c in config1, config2, config3:
        c.key = sha256(c.path.encode("utf-8")).hexdigest()
        store.add_model(c)

    # config4 clashes with config1 and should raise an integrity error
    with pytest.raises(DuplicateModelException):
        config4.key = sha256(config4.path.encode("utf-8")).hexdigest()
        store.add_model(config4)


def test_filter_2(store: ModelRecordServiceBase):
    config1 = MainDiffusersConfig(
        path="/tmp/config1",
        name="config1",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        hash="CONFIG1HASH",
        source="test/source/",
        source_type=ModelSourceType.Path,
    )
    config2 = MainDiffusersConfig(
        path="/tmp/config2",
        name="config2",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        hash="CONFIG2HASH",
        source="test/source/",
        source_type=ModelSourceType.Path,
    )
    config3 = MainDiffusersConfig(
        path="/tmp/config3",
        name="dup_name1",
        base=BaseModelType("sd-2"),
        type=ModelType.Main,
        hash="CONFIG3HASH",
        source="test/source/",
        source_type=ModelSourceType.Path,
    )
    config4 = MainDiffusersConfig(
        path="/tmp/config4",
        name="dup_name1",
        base=BaseModelType("sdxl"),
        type=ModelType.Main,
        hash="CONFIG3HASH",
        source="test/source/",
        source_type=ModelSourceType.Path,
    )
    config5 = VaeDiffusersConfig(
        path="/tmp/config5",
        name="dup_name1",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Vae,
        hash="CONFIG3HASH",
        source="test/source/",
        source_type=ModelSourceType.Path,
    )
    for c in config1, config2, config3, config4, config5:
        store.add_model(c)

    matches = store.search_by_attr(
        model_type=ModelType.Main,
        model_name="dup_name1",
    )
    assert len(matches) == 2

    matches = store.search_by_attr(
        base_model=BaseModelType.StableDiffusion1,
        model_type=ModelType.Main,
    )
    assert len(matches) == 2

    matches = store.search_by_attr(
        base_model=BaseModelType.StableDiffusion1,
        model_type=ModelType.Vae,
        model_name="dup_name1",
    )
    assert len(matches) == 1
