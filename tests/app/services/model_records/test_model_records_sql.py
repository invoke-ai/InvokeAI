"""
Test the refactored model config classes.
"""

from hashlib import sha256

import pytest

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import (
    DuplicateModelException,
    ModelRecordServiceBase,
    ModelRecordServiceSQL,
    UnknownModelException,
)
from invokeai.app.services.shared.sqlite import SqliteDatabase
from invokeai.backend.model_manager.config import (
    BaseModelType,
    MainDiffusersConfig,
    ModelType,
    TextualInversionConfig,
    VaeDiffusersConfig,
)
from invokeai.backend.util.logging import InvokeAILogger


@pytest.fixture
def store(datadir) -> ModelRecordServiceBase:
    config = InvokeAIAppConfig(root=datadir)
    logger = InvokeAILogger.get_logger(config=config)
    db = SqliteDatabase(config, logger)
    return ModelRecordServiceSQL(db)


def example_config() -> TextualInversionConfig:
    return TextualInversionConfig(
        path="/tmp/pokemon.bin",
        name="old name",
        base=BaseModelType("sd-1"),
        type=ModelType("embedding"),
        format="embedding_file",
        original_hash="ABC123",
    )


def test_add(store: ModelRecordServiceBase):
    raw = dict(
        path="/tmp/foo.ckpt",
        name="model1",
        base=BaseModelType("sd-1"),
        type="main",
        config="/tmp/foo.yaml",
        variant="normal",
        format="checkpoint",
        original_hash="111222333444",
    )
    store.add_model("key1", raw)
    config1 = store.get_model("key1")
    assert config1 is not None
    assert config1.base == BaseModelType("sd-1")
    assert config1.name == "model1"
    assert config1.original_hash == "111222333444"
    assert config1.current_hash is None


def test_dup(store: ModelRecordServiceBase):
    config = example_config()
    store.add_model("key1", example_config())
    with pytest.raises(DuplicateModelException):
        store.add_model("key1", config)
    with pytest.raises(DuplicateModelException):
        store.add_model("key2", config)


def test_update(store: ModelRecordServiceBase):
    config = example_config()
    store.add_model("key1", config)
    config = store.get_model("key1")
    assert config.name == "old name"

    config.name = "new name"
    store.update_model("key1", config)
    new_config = store.get_model("key1")
    assert new_config.name == "new name"

def test_rename(store: ModelRecordServiceBase):
    config = example_config()
    store.add_model("key1", config)
    config = store.get_model("key1")
    assert config.name == "old name"

    store.rename_model("key1", "new name")
    new_config = store.get_model("key1")
    assert new_config.name == "new name"


def test_unknown_key(store: ModelRecordServiceBase):
    config = example_config()
    store.add_model("key1", config)
    with pytest.raises(UnknownModelException):
        store.update_model("unknown_key", config)


def test_delete(store: ModelRecordServiceBase):
    config = example_config()
    store.add_model("key1", config)
    config = store.get_model("key1")
    store.del_model("key1")
    with pytest.raises(UnknownModelException):
        config = store.get_model("key1")


def test_exists(store: ModelRecordServiceBase):
    config = example_config()
    store.add_model("key1", config)
    assert store.exists("key1")
    assert not store.exists("key2")


def test_filter(store: ModelRecordServiceBase):
    config1 = MainDiffusersConfig(
        path="/tmp/config1",
        name="config1",
        base=BaseModelType("sd-1"),
        type=ModelType("main"),
        original_hash="CONFIG1HASH",
    )
    config2 = MainDiffusersConfig(
        path="/tmp/config2",
        name="config2",
        base=BaseModelType("sd-1"),
        type=ModelType("main"),
        original_hash="CONFIG2HASH",
    )
    config3 = VaeDiffusersConfig(
        path="/tmp/config3",
        name="config3",
        base=BaseModelType("sd-2"),
        type=ModelType("vae"),
        original_hash="CONFIG3HASH",
    )
    for c in config1, config2, config3:
        store.add_model(sha256(c.name.encode("utf-8")).hexdigest(), c)
    matches = store.search_by_attr(model_type=ModelType("main"))
    assert len(matches) == 2
    assert matches[0].name in {"config1", "config2"}

    matches = store.search_by_attr(model_type=ModelType("vae"))
    assert len(matches) == 1
    assert matches[0].name == "config3"
    assert matches[0].key == sha256("config3".encode("utf-8")).hexdigest()
    assert isinstance(matches[0].type, ModelType)  # This tests that we get proper enums back

    matches = store.search_by_attr(model_type=BaseModelType("sd-2"))

    matches = store.search_by_hash("CONFIG1HASH")
    assert len(matches) == 1
    assert matches[0].original_hash == "CONFIG1HASH"

    matches = store.all_models()
    assert len(matches) == 3


def test_filter_2(store: ModelRecordServiceBase):
    config1 = MainDiffusersConfig(
        path="/tmp/config1",
        name="config1",
        base=BaseModelType("sd-1"),
        type=ModelType("main"),
        original_hash="CONFIG1HASH",
    )
    config2 = MainDiffusersConfig(
        path="/tmp/config2",
        name="config2",
        base=BaseModelType("sd-1"),
        type=ModelType("main"),
        original_hash="CONFIG2HASH",
    )
    config3 = MainDiffusersConfig(
        path="/tmp/config3",
        name="dup_name1",
        base=BaseModelType("sd-2"),
        type=ModelType("main"),
        original_hash="CONFIG3HASH",
    )
    config4 = MainDiffusersConfig(
        path="/tmp/config4",
        name="dup_name1",
        base=BaseModelType("sd-2"),
        type=ModelType("main"),
        original_hash="CONFIG3HASH",
    )
    config5 = VaeDiffusersConfig(
        path="/tmp/config5",
        name="dup_name1",
        base=BaseModelType("sd-1"),
        type=ModelType("vae"),
        original_hash="CONFIG3HASH",
    )
    for c in config1, config2, config3, config4, config5:
        store.add_model(sha256(c.path.encode("utf-8")).hexdigest(), c)

    matches = store.search_by_attr(
        model_type=ModelType("main"),
        model_name="dup_name1",
    )
    assert len(matches) == 2

    matches = store.search_by_attr(
        base_model=BaseModelType("sd-1"),
        model_type=ModelType("main"),
    )
    assert len(matches) == 2

    matches = store.search_by_attr(
        base_model=BaseModelType("sd-1"),
        model_type=ModelType("vae"),
        model_name="dup_name1",
    )
    assert len(matches) == 1
