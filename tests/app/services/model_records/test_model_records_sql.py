"""
Test the refactored model config classes.
"""

import json
from hashlib import sha256
from typing import Any, Optional

import pytest
from pydantic import ValidationError
from sqlalchemy import insert

from invokeai.app.services.model_records import (
    DuplicateModelException,
    ModelRecordOrderBy,
    ModelRecordServiceBase,
    ModelRecordServiceSQL,
    UnknownModelException,
)
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries.models import ModelQueries
from invokeai.app.services.shared.database.schema.models import models
from invokeai.app.services.shared.pagination import SQLiteDirection
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
from tests.fixtures.races import when_called


@pytest.fixture
def store(database: Database) -> ModelRecordServiceSQL:
    return ModelRecordServiceSQL(database, InvokeAILogger.get_logger())


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
    # The key and the path are unique: config1 again repeats both, config2 its path.
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


def test_model_records_rejects_invalid_attr_changes(store: ModelRecordServiceBase):
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
    # config1, config2 and config3 can be installed together: their paths differ, though not their names
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


def _embedding(key: str, *, name: str = "embedding") -> TI_File_SD1_Config:
    return TI_File_SD1_Config(
        key=key,
        source="test/source/",
        source_type=ModelSourceType.Path,
        path=f"/tmp/{key}.bin",
        file_size=1024,
        name=name,
        base=BaseModelType.StableDiffusion1,
        type=ModelType.TextualInversion,
        format=ModelFormat.EmbeddingFile,
        hash="ABC123",
    )


def test_a_duplicate_is_reported_by_what_it_repeats(store: ModelRecordServiceBase) -> None:
    store.add_model(_embedding("key1"))

    with pytest.raises(DuplicateModelException, match="with path"):
        store.add_model(_embedding("key2").model_copy(update={"path": "/tmp/key1.bin"}))
    with pytest.raises(DuplicateModelException, match="with key"):
        store.add_model(_embedding("key1").model_copy(update={"path": "/tmp/elsewhere.bin"}))


def test_models_that_tie_are_listed_by_key_and_names_ignore_case(store: ModelRecordServiceBase) -> None:
    # Added against key order, so that only the tie-break lists equal names by key. By bytes "Beta" sorts before
    # "alpha", so only a case-insensitive order lists it last.
    for key, name in (("k4", "Beta"), ("k3", "alpha"), ("k2", "ALPHA"), ("k1", "Alpha")):
        store.add_model(_embedding(key, name=name))

    def keys(order_by: ModelRecordOrderBy, direction: SQLiteDirection) -> list[str]:
        return [model.key for model in store.search_by_attr(order_by=order_by, direction=direction)]

    # The embeddings share their type, base and format, so the default order goes by their names too.
    for order_by in (ModelRecordOrderBy.Name, ModelRecordOrderBy.Default):
        assert keys(order_by, SQLiteDirection.Ascending) == ["k1", "k2", "k3", "k4"]
        assert keys(order_by, SQLiteDirection.Descending) == ["k4", "k3", "k2", "k1"]


_MAIN = {
    "variant": ModelVariantType.Normal,
    "prediction_type": SchedulerPredictionType.Epsilon,
    "repo_variant": ModelRepoVariant.Default,
}
# Models that differ on every ordered attribute: key -> (class, fields of the class, name, path, size, added, modified).
# Timestamps are written as stored, so that no test waits for the clock.
_ORDERED_MODELS: dict[str, tuple[type, dict[str, Any], str, str, int, str, str]] = {
    "a": (
        Main_Diffusers_SDXL_Config,
        _MAIN,
        "delta",
        "/m/2",
        3000,
        "2000-01-03 00:00:00.000",
        "2000-02-02 00:00:00.000",
    ),
    "b": (
        VAE_Diffusers_SD1_Config,
        {"repo_variant": ModelRepoVariant.Default},
        "Charlie",
        "/m/4",
        1000,
        "2000-01-01 00:00:00.000",
        "2000-02-04 00:00:00.000",
    ),
    "c": (TI_File_SD1_Config, {}, "bravo", "/m/1", 4000, "2000-01-02 00:00:00.000", "2000-02-03 00:00:00.000"),
    "d": (LoRA_LyCORIS_SDXL_Config, {}, "Alpha", "/m/3", 2000, "2000-01-04 00:00:00.000", "2000-02-01 00:00:00.000"),
    "e": (Main_Diffusers_SD1_Config, _MAIN, "echo", "/m/5", 5000, "2000-01-05 00:00:00.000", "2000-02-05 00:00:00.000"),
}
# Each order's listing, ascending; descending is its reverse, ties included.
_ASCENDING = {
    ModelRecordOrderBy.Default: "cdeab",  # by type, then base: the two main models
    ModelRecordOrderBy.Type: "cdaeb",  # the two main models tie, by key
    ModelRecordOrderBy.Base: "bcead",
    ModelRecordOrderBy.Name: "dcbae",  # ignoring case; by bytes it would be "dbcae"
    ModelRecordOrderBy.Format: "abecd",
    ModelRecordOrderBy.Size: "bdace",
    ModelRecordOrderBy.DateAdded: "bcade",
    ModelRecordOrderBy.DateModified: "dacbe",
    ModelRecordOrderBy.Path: "cadbe",
}


@pytest.fixture
def ordered_store(database: Database, store: ModelRecordServiceSQL) -> ModelRecordServiceSQL:
    with database.begin(write=True) as conn:
        for key, (cls, fields, name, path, size, added, modified) in _ORDERED_MODELS.items():
            config = cls(
                key=key,
                name=name,
                path=path,
                file_size=size,
                hash=f"hash-{key}",
                source="test/source/",
                source_type=ModelSourceType.Path,
                **fields,
            )
            conn.execute(
                insert(models).values(id=key, config=config.model_dump_json(), created_at=added, updated_at=modified)
            )
    return store


def _listed(store: ModelRecordServiceBase, order_by: ModelRecordOrderBy, direction: SQLiteDirection) -> str:
    return "".join(model.key for model in store.search_by_attr(order_by=order_by, direction=direction))


def test_each_order_has_its_own_listing() -> None:
    # Else a test below could not tell one order's column from another's.
    assert len(set(_ASCENDING.values())) == len(_ASCENDING) == len(ModelRecordOrderBy)


@pytest.mark.parametrize("order_by", list(ModelRecordOrderBy))
def test_each_order_sorts_by_its_own_column(ordered_store: ModelRecordServiceSQL, order_by: ModelRecordOrderBy) -> None:
    assert _listed(ordered_store, order_by, SQLiteDirection.Ascending) == _ASCENDING[order_by]
    assert _listed(ordered_store, order_by, SQLiteDirection.Descending) == _ASCENDING[order_by][::-1]


def test_an_update_makes_a_model_the_last_modified(ordered_store: ModelRecordServiceSQL) -> None:
    ordered_store.update_model("c", ModelRecordChanges(description="edited"))

    assert _listed(ordered_store, ModelRecordOrderBy.DateModified, SQLiteDirection.Ascending) == "dabec"
    assert _listed(ordered_store, ModelRecordOrderBy.DateAdded, SQLiteDirection.Ascending) == "bcade"


def test_the_format_filter_applies(ordered_store: ModelRecordServiceSQL) -> None:
    diffusers = ordered_store.search_by_attr(model_format=ModelFormat.Diffusers)

    assert [model.key for model in diffusers] == ["e", "a", "b"]


def test_a_replaced_model_is_stored_and_an_unknown_one_is_not_found(ordered_store: ModelRecordServiceSQL) -> None:
    replaced = ordered_store.get_model("c").model_copy(update={"name": "renamed"})

    ordered_store.replace_model("c", replaced)

    assert ordered_store.get_model("c").name == "renamed"
    with pytest.raises(ValueError):
        ordered_store.replace_model("c", ordered_store.get_model("a"))
    with pytest.raises(UnknownModelException):
        ordered_store.replace_model("ghost", replaced.model_copy(update={"key": "ghost", "path": "/m/ghost"}))
    with pytest.raises(UnknownModelException):
        ordered_store.del_model("ghost")


def test_concurrent_updates_of_a_model_both_apply(
    store: ModelRecordServiceBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.add_model(example_ti_config("key1"))

    def second() -> object:
        return store.update_model("key1", ModelRecordChanges(description="second"))

    # The second update starts while the first holds the model's row: it must wait, then build on the first.
    ended = when_called(monkeypatch, ModelQueries, "lock", second)
    store.update_model("key1", ModelRecordChanges(name="first"))

    assert ended() == []
    updated = store.get_model("key1")
    assert (updated.name, updated.description) == ("first", "second")


def test_model_paths_include_records_that_no_longer_validate(store: ModelRecordServiceBase, database: Database) -> None:
    store.add_model(_embedding("key1"))
    # A record of a model type this version does not know, as after going back to an older version.
    unknown = _embedding("key2").model_dump(mode="json") | {"type": "from_the_future"}
    database.queries.models.insert("key2", json.dumps(unknown))

    assert sorted(store.get_model_paths()) == ["/tmp/key1.bin", "/tmp/key2.bin"]
    assert [model.key for model in store.search_by_attr()] == ["key1"]


def test_the_oldest_model_of_a_file_comes_first(store: ModelRecordServiceBase, database: Database) -> None:
    # The same file under two paths, the older record with the larger key: the route that recalls a model by its hash
    # takes the first.
    with database.begin(write=True) as conn:
        for key, added in (("k1", "2000-01-02 00:00:00.000"), ("k2", "2000-01-01 00:00:00.000")):
            config = _embedding(key).model_dump_json()
            conn.execute(insert(models).values(id=key, config=config, created_at=added, updated_at=added))

    assert [model.key for model in store.search_by_hash("ABC123")] == ["k2", "k1"]
