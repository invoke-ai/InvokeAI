"""
Test model metadata fetching and storage.
"""
import pytest
import datetime

from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, ValidationError

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.app.services.model_records import ModelRecordServiceBase, ModelRecordServiceSQL, UnknownModelException
from invokeai.backend.model_manager.config import (
    BaseModelType,
    MainCheckpointConfig,
    MainDiffusersConfig,
    ModelType,
    TextualInversionConfig,
    VaeDiffusersConfig,
)
from invokeai.backend.model_manager.metadata import (
    ModelMetadataStore,
    AnyModelRepoMetadata,
    CommercialUsage,
    LicenseRestrictions,
    HuggingFaceMetadata,
    CivitaiMetadata,
)
from tests.fixtures.sqlite_database import create_mock_sqlite_database

@pytest.fixture
def app_config(datadir: Path) -> InvokeAIAppConfig:
    return InvokeAIAppConfig(
        root=datadir / "root",
        models_dir=datadir / "root/models",
    )

@pytest.fixture
def record_store(app_config: InvokeAIAppConfig) -> ModelRecordServiceSQL:
    logger = InvokeAILogger.get_logger(config=app_config)
    db = create_mock_sqlite_database(app_config, logger)
    store = ModelRecordServiceSQL(db)
    # add two config records to the database
    raw1 = {
        "path": "/tmp/foo2.ckpt",
        "name": "test2",
        "base": BaseModelType("sd-2"),
        "type": ModelType("vae"),
        "original_hash":"111222333444",
        "source": "stabilityai/sdxl-vae",
    }
    raw2 = {
        "path": "/tmp/foo1.ckpt",
        "name": "model1",
        "base": BaseModelType("sd-1"),
        "type": "main",
        "config": "/tmp/foo.yaml",
        "variant": "normal",
        "format": "checkpoint",
        "original_hash": "111222333444",
        "source": "https://civitai.com/models/206883/split",
    }
    store.add_model('test_config_1', raw1)
    store.add_model('test_config_2', raw2)
    return store

@pytest.fixture
def metadata_store(record_store: ModelRecordServiceSQL) -> ModelMetadataStore:
    db = record_store._db   # to ensure we are sharing the same database
    return ModelMetadataStore(db)

def test_metadata_store_put_get(metadata_store: ModelMetadataStore) -> None:
    input_metadata = HuggingFaceMetadata(name="sdxl-vae",
                                         author="stabilityai",
                                         tags={"text-to-image","diffusers"},
                                         id="stabilityai/sdxl-vae",
                                         tag_dict={"license":"other"},
                                         last_modified=datetime.datetime.now(),
                                         )
    metadata_store.add_metadata('test_config_1',input_metadata)
    output_metadata = metadata_store.get_metadata('test_config_1')
    assert input_metadata == output_metadata

