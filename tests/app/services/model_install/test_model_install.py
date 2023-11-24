"""
Test the model installer
"""

import pytest
from pathlib import Path
from pydantic import ValidationError
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.model_manager.config import ModelType, BaseModelType
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import ModelRecordServiceSQL, ModelRecordServiceBase
from invokeai.app.services.shared.sqlite import SqliteDatabase
from invokeai.app.services.model_install import ModelInstallService, ModelInstallServiceBase

@pytest.fixture
def test_file(datadir: Path) -> Path:
    return datadir / "test_embedding.safetensors"


@pytest.fixture
def app_config(datadir: Path) -> InvokeAIAppConfig:
    return InvokeAIAppConfig(
        root=datadir / "root",
        models_dir=datadir / "root/models",
    )


@pytest.fixture
def store(app_config: InvokeAIAppConfig) -> ModelRecordServiceBase:
    database = SqliteDatabase(app_config, InvokeAILogger.get_logger(config=app_config))
    store: ModelRecordServiceBase = ModelRecordServiceSQL(database)
    return store


@pytest.fixture
def installer(app_config: InvokeAIAppConfig,
              store: ModelRecordServiceBase) -> ModelInstallServiceBase:
    return ModelInstallService(app_config=app_config,
                               record_store=store
                               )


def test_registration(installer: ModelInstallServiceBase, test_file: Path) -> None:
    store = installer.record_store
    matches = store.search_by_attr(model_name="test_embedding")
    assert len(matches) == 0
    key = installer.register_path(test_file)
    assert key is not None
    assert len(key) == 32

def test_registration_meta(installer: ModelInstallServiceBase, test_file: Path) -> None:
    store = installer.record_store
    key = installer.register_path(test_file)
    model_record = store.get_model(key)
    assert model_record is not None
    assert model_record.name == "test_embedding"
    assert model_record.type == ModelType.TextualInversion
    assert Path(model_record.path) == test_file
    assert model_record.base == BaseModelType('sd-1')
    assert model_record.description is not None
    assert model_record.source is not None
    assert Path(model_record.source) == test_file

def test_registration_meta_override_fail(installer: ModelInstallServiceBase, test_file: Path) -> None:
    key = None
    with pytest.raises(ValidationError):
        key = installer.register_path(test_file, {"name": "banana_sushi", "type": ModelType("lora")})
    assert key is None

def test_registration_meta_override_succeed(installer: ModelInstallServiceBase, test_file: Path) -> None:
    store = installer.record_store
    key = installer.register_path(test_file,
                                  {
                                      "name": "banana_sushi",
                                      "source": "fake/repo_id",
                                      "current_hash": "New Hash"
                                  }
                                  )
    model_record = store.get_model(key)
    assert model_record.name == "banana_sushi"
    assert model_record.source == "fake/repo_id"
    assert model_record.current_hash == "New Hash"

def test_install(installer: ModelInstallServiceBase, test_file: Path, app_config: InvokeAIAppConfig) -> None:
    store = installer.record_store
    key = installer.install_path(test_file)
    model_record = store.get_model(key)
    assert model_record.path == "sd-1/embedding/test_embedding.safetensors"
    assert model_record.source == test_file.as_posix()
