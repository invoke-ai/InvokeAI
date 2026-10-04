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
    ModelRecordOrderBy,
    ModelRecordServiceBase,
    ModelRecordServiceSQL,
    UnknownModelException,
)
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.backend.model_manager.configs.controlnet import ControlAdapterDefaultSettings
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_SDXL_Config
from invokeai.backend.model_manager.configs.main import (
    Main_Diffusers_SD1_Config,
    Main_Diffusers_SD2_Config,
    Main_Diffusers_SDXL_Config,
    MainModelDefaultSettings,
)
from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_Qwen3Encoder_Config
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Diffusers_Config
from invokeai.backend.model_manager.configs.textual_inversion import TI_File_SD1_Config
from invokeai.backend.model_manager.configs.vae import VAE_Diffusers_SD1_Config
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelSourceType,
    ModelType,
    ModelVariantType,
    Qwen3VariantType,
    SchedulerPredictionType,
)
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def store(
    datadir: Any,
) -> ModelRecordServiceSQL:
    config = InvokeAIAppConfig()
    config._root = datadir
    logger = InvokeAILogger.get_logger(config=config)
    db = create_mock_sqlite_database(config, logger)
    return ModelRecordServiceSQL(db, logger)


def example_ti_config(key: Optional[str] = None) -> TI_File_SD1_Config:
    config = TI_File_SD1_Config(
        source="test/source/",
        source_type=ModelSourceType.Path,
        path="/tmp/pokemon.bin",
        file_size=1024,
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
    assert isinstance(config1, TI_File_SD1_Config)


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


def test_model_records_updates_model_class(store: ModelRecordServiceBase):
    config = example_ti_config("key1")
    store.add_model(config)
    changes = ModelRecordChanges(
        type=ModelType.LoRA,
        format=ModelFormat.LyCORIS,
        base=BaseModelType.StableDiffusionXL,
    )
    new_config = store.update_model(config.key, changes, allow_class_change=True)
    assert isinstance(new_config, LoRA_LyCORIS_SDXL_Config)


def test_update_changing_type_drops_stale_format_and_variant(store: ModelRecordServiceBase):
    """When the type changes, format/variant from the old class must not block validation of the new class.

    Regression test for https://github.com/invoke-ai/InvokeAI/issues/9090: switching a misidentified
    Qwen3 encoder to TextLLM previously failed because the old `format=qwen3_encoder` and `variant`
    fields were carried over and no discriminator under `type=text_llm` matched.
    """
    config = Qwen3Encoder_Qwen3Encoder_Config(
        source="test/source/",
        source_type=ModelSourceType.Path,
        path="/tmp/Qwen2.5-1.5B-Instruct",
        file_size=1024,
        name="Qwen2.5-1.5B-Instruct",
        hash="ABC123",
        variant=Qwen3VariantType.Qwen3_4B,
    )
    config.key = "key1"
    store.add_model(config)

    changes = ModelRecordChanges(type=ModelType.TextLLM)
    new_config = store.update_model(config.key, changes, allow_class_change=True)
    assert isinstance(new_config, TextLLM_Diffusers_Config)


@pytest.mark.parametrize("allow_class_change", [False, True])
def test_rejected_update_preserves_the_complete_stored_record(
    store: ModelRecordServiceBase, allow_class_change: bool
) -> None:
    original = example_ti_config("key1")
    store.add_model(original)
    before = store.get_model(original.key)
    # A classification edit may combine valid metadata changes with an incompatible model format.
    changes = ModelRecordChanges(name="edited name", path="/tmp/edited.bin", format=ModelFormat.Diffusers)

    with pytest.raises(ValidationError):
        store.update_model(original.key, changes, allow_class_change=allow_class_change)

    after = store.get_model(original.key)
    assert type(after) is type(before)
    assert after.model_dump() == before.model_dump()


def test_rejected_class_change_preserves_the_original_record(store: ModelRecordServiceBase) -> None:
    original = example_ti_config("key1")
    store.add_model(original)
    before = store.get_model(original.key)
    # Correcting a model's classification can leave an explicitly incompatible format in the submitted edit.
    changes = ModelRecordChanges(type=ModelType.LoRA, format=ModelFormat.EmbeddingFile, name="edited name")

    with pytest.raises(ValidationError):
        store.update_model(original.key, changes, allow_class_change=True)

    after = store.get_model(original.key)
    assert isinstance(after, TI_File_SD1_Config)
    assert after.model_dump() == before.model_dump()


@pytest.mark.parametrize("field", ["type", "base"])
def test_model_record_changes_reject_invalid_classification_values(field: str) -> None:
    with pytest.raises(ValidationError):
        ModelRecordChanges.model_validate({field: "not-a-model-classification"})


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
    config1 = Main_Diffusers_SD1_Config(
        key="config1",
        path="/tmp/config1",
        name="config1",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        hash="CONFIG1HASH",
        file_size=1001,
        source="test/source",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    config2 = Main_Diffusers_SD1_Config(
        key="config2",
        path="/tmp/config2",
        name="config2",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        hash="CONFIG2HASH",
        file_size=1002,
        source="test/source",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    config3 = VAE_Diffusers_SD1_Config(
        key="config3",
        path="/tmp/config3",
        name="config3",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.VAE,
        hash="CONFIG3HASH",
        file_size=1003,
        source="test/source",
        source_type=ModelSourceType.Path,
        repo_variant=ModelRepoVariant.Default,
    )
    for c in config1, config2, config3:
        store.add_model(c)
    matches = store.search_by_attr(model_type=ModelType.Main)
    assert len(matches) == 2
    assert matches[0].name in {"config1", "config2"}

    matches = store.search_by_attr(model_type=ModelType.VAE)
    assert len(matches) == 1
    assert matches[0].name == "config3"
    assert matches[0].key == "config3"
    assert isinstance(matches[0].type, ModelType)  # This tests that we get proper enums back

    matches = store.search_by_hash("CONFIG1HASH")
    assert len(matches) == 1
    assert matches[0].hash == "CONFIG1HASH"

    matches = store.all_models()
    assert len(matches) == 3


def test_unique_by_path(store: ModelRecordServiceBase):
    config1 = Main_Diffusers_SD1_Config(
        path="/tmp/config1",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        name="nonuniquename",
        hash="CONFIG1HASH",
        file_size=1004,
        source="test/source/",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    config2 = Main_Diffusers_SD2_Config(
        path="/tmp/config2",
        base=BaseModelType.StableDiffusion2,
        type=ModelType.Main,
        name="nonuniquename",
        hash="CONFIG1HASH",
        file_size=1005,
        source="test/source/",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    config3 = VAE_Diffusers_SD1_Config(
        path="/tmp/config3",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.VAE,
        name="nonuniquename",
        hash="CONFIG1HASH",
        file_size=1006,
        source="test/source/",
        source_type=ModelSourceType.Path,
        repo_variant=ModelRepoVariant.Default,
    )
    config4 = Main_Diffusers_SD1_Config(
        path="/tmp/config1",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        name="nonuniquename",
        hash="CONFIG1HASH",
        file_size=1007,
        source="test/source/",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    # config1, config2 and config3 are compatible because they have unique paths
    # of name, type and base
    for c in config1, config2, config3:
        c.key = sha256(c.path.encode("utf-8")).hexdigest()
        store.add_model(c)

    # config4 clashes with config1 (same path) and should raise an integrity error
    with pytest.raises(DuplicateModelException):
        config4.key = sha256(config4.path.encode("utf-8")).hexdigest()
        store.add_model(config4)


def test_filter_2(store: ModelRecordServiceBase):
    config1 = Main_Diffusers_SD1_Config(
        path="/tmp/config1",
        name="config1",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        hash="CONFIG1HASH",
        file_size=1008,
        source="test/source/",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    config2 = Main_Diffusers_SD1_Config(
        path="/tmp/config2",
        name="config2",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        hash="CONFIG2HASH",
        file_size=1009,
        source="test/source/",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    config3 = Main_Diffusers_SD2_Config(
        path="/tmp/config3",
        name="dup_name1",
        base=BaseModelType.StableDiffusion2,
        type=ModelType.Main,
        hash="CONFIG3HASH",
        file_size=1010,
        source="test/source/",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    config4 = Main_Diffusers_SDXL_Config(
        path="/tmp/config4",
        name="dup_name1",
        base=BaseModelType.StableDiffusionXL,
        type=ModelType.Main,
        hash="CONFIG3HASH",
        file_size=1011,
        source="test/source/",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    config5 = VAE_Diffusers_SD1_Config(
        path="/tmp/config5",
        name="dup_name1",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.VAE,
        hash="CONFIG3HASH",
        file_size=1012,
        source="test/source/",
        source_type=ModelSourceType.Path,
        repo_variant=ModelRepoVariant.Default,
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
        model_type=ModelType.VAE,
        model_name="dup_name1",
    )
    assert len(matches) == 1


def test_search_by_attr_sorting(store: ModelRecordServiceSQL):
    config1 = Main_Diffusers_SD1_Config(
        path="/tmp/config1",
        name="alpha",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        hash="CONFIG1HASH",
        file_size=1000,
        source="test/source/",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    config2 = Main_Diffusers_SD2_Config(
        path="/tmp/config2",
        name="beta",
        base=BaseModelType.StableDiffusion2,
        type=ModelType.Main,
        hash="CONFIG2HASH",
        file_size=2000,
        source="test/source/",
        source_type=ModelSourceType.Path,
        variant=ModelVariantType.Normal,
        prediction_type=SchedulerPredictionType.Epsilon,
        repo_variant=ModelRepoVariant.Default,
    )
    config3 = VAE_Diffusers_SD1_Config(
        path="/tmp/config3",
        name="gamma",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.VAE,
        hash="CONFIG3HASH",
        file_size=500,
        source="test/source/",
        source_type=ModelSourceType.Path,
        repo_variant=ModelRepoVariant.Default,
    )
    for c in config1, config2, config3:
        store.add_model(c)

    # Test sorting by Name Ascending
    matches = store.search_by_attr(order_by=ModelRecordOrderBy.Name, direction=SQLiteDirection.Ascending)
    assert len(matches) == 3
    assert matches[0].name == "alpha"
    assert matches[1].name == "beta"
    assert matches[2].name == "gamma"

    # Test sorting by Name Descending
    matches = store.search_by_attr(order_by=ModelRecordOrderBy.Name, direction=SQLiteDirection.Descending)
    assert matches[0].name == "gamma"
    assert matches[1].name == "beta"
    assert matches[2].name == "alpha"

    # Test sorting by Size Ascending
    matches = store.search_by_attr(order_by=ModelRecordOrderBy.Size, direction=SQLiteDirection.Ascending)
    assert matches[0].name == "gamma"  # 500
    assert matches[1].name == "alpha"  # 1000
    assert matches[2].name == "beta"  # 2000

    # Test sorting by Size Descending
    matches = store.search_by_attr(order_by=ModelRecordOrderBy.Size, direction=SQLiteDirection.Descending)
    assert matches[0].name == "beta"  # 2000
    assert matches[1].name == "alpha"  # 1000
    assert matches[2].name == "gamma"  # 500


def test_model_record_changes():
    # This test guards against some unexpected behaviours from pydantic's union evaluation. See #6035
    changes = ModelRecordChanges.model_validate({"default_settings": {"preprocessor": "value"}})
    assert isinstance(changes.default_settings, ControlAdapterDefaultSettings)

    changes = ModelRecordChanges.model_validate({"default_settings": {"vae": "value"}})
    assert isinstance(changes.default_settings, MainModelDefaultSettings)
