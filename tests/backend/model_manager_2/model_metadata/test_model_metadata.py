"""
Test model metadata fetching and storage.
"""
import datetime
from pathlib import Path

import pytest
import requests
from pydantic.networks import HttpUrl
from requests.sessions import Session
from requests_testadapter import TestAdapter

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import ModelRecordServiceSQL, UnknownModelException
from invokeai.backend.model_manager.config import (
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.model_manager.metadata import (
    CivitaiMetadata,
    CivitaiMetadataFetch,
    CommercialUsage,
    HuggingFaceMetadata,
    HuggingFaceMetadataFetch,
    ModelMetadataStore,
)
from invokeai.backend.util.logging import InvokeAILogger
from tests.backend.model_manager_2.model_metadata.metadata_examples import (
    RepoCivitaiModelMetadata1,
    RepoCivitaiVersionMetadata1,
    RepoHFMetadata1,
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
    # add three simple config records to the database
    raw1 = {
        "path": "/tmp/foo1",
        "format": ModelFormat("diffusers"),
        "name": "test2",
        "base": BaseModelType("sd-2"),
        "type": ModelType("vae"),
        "original_hash": "111222333444",
        "source": "stabilityai/sdxl-vae",
    }
    raw2 = {
        "path": "/tmp/foo2.ckpt",
        "name": "model1",
        "format": ModelFormat("checkpoint"),
        "base": BaseModelType("sd-1"),
        "type": "main",
        "config": "/tmp/foo.yaml",
        "variant": "normal",
        "original_hash": "111222333444",
        "source": "https://civitai.com/models/206883/split",
    }
    raw3 = {
        "path": "/tmp/foo3",
        "format": ModelFormat("diffusers"),
        "name": "test3",
        "base": BaseModelType("sdxl"),
        "type": ModelType("main"),
        "original_hash": "111222333444",
        "source": "author3/model3",
    }
    store.add_model("test_config_1", raw1)
    store.add_model("test_config_2", raw2)
    store.add_model("test_config_3", raw3)
    return store


@pytest.fixture
def session() -> Session:
    sess = requests.Session()
    sess.mount(
        "https://huggingface.co/api/models/stabilityai/sdxl-turbo",
        TestAdapter(
            RepoHFMetadata1,
            headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": len(RepoHFMetadata1)},
        ),
    )
    sess.mount(
        "https://civitai.com/api/v1/model-versions/242807",
        TestAdapter(
            RepoCivitaiVersionMetadata1,
            headers={
                "Content-Length": len(RepoCivitaiVersionMetadata1),
            },
        ),
    )
    sess.mount(
        "https://civitai.com/api/v1/models/215485",
        TestAdapter(
            RepoCivitaiModelMetadata1,
            headers={
                "Content-Length": len(RepoCivitaiModelMetadata1),
            },
        ),
    )
    return sess


@pytest.fixture
def metadata_store(record_store: ModelRecordServiceSQL) -> ModelMetadataStore:
    db = record_store._db  # to ensure we are sharing the same database
    return ModelMetadataStore(db)


def test_metadata_store_put_get(metadata_store: ModelMetadataStore) -> None:
    input_metadata = HuggingFaceMetadata(
        name="sdxl-vae",
        author="stabilityai",
        tags={"text-to-image", "diffusers"},
        id="stabilityai/sdxl-vae",
        tag_dict={"license": "other"},
        last_modified=datetime.datetime.now(),
    )
    metadata_store.add_metadata("test_config_1", input_metadata)
    output_metadata = metadata_store.get_metadata("test_config_1")
    assert input_metadata == output_metadata
    with pytest.raises(UnknownModelException):
        metadata_store.add_metadata("unknown_key", input_metadata)


def test_metadata_store_update(metadata_store: ModelMetadataStore) -> None:
    input_metadata = HuggingFaceMetadata(
        name="sdxl-vae",
        author="stabilityai",
        tags={"text-to-image", "diffusers"},
        id="stabilityai/sdxl-vae",
        tag_dict={"license": "other"},
        last_modified=datetime.datetime.now(),
    )
    metadata_store.add_metadata("test_config_1", input_metadata)
    input_metadata.name = "new-name"
    metadata_store.update_metadata("test_config_1", input_metadata)
    output_metadata = metadata_store.get_metadata("test_config_1")
    assert output_metadata.name == "new-name"
    assert input_metadata == output_metadata


def test_metadata_search(metadata_store: ModelMetadataStore) -> None:
    metadata1 = HuggingFaceMetadata(
        name="sdxl-vae",
        author="stabilityai",
        tags={"text-to-image", "diffusers"},
        id="stabilityai/sdxl-vae",
        tag_dict={"license": "other"},
        last_modified=datetime.datetime.now(),
    )
    metadata2 = HuggingFaceMetadata(
        name="model2",
        author="stabilityai",
        tags={"text-to-image", "diffusers", "community-contributed"},
        id="author2/model2",
        tag_dict={"license": "other"},
        last_modified=datetime.datetime.now(),
    )
    metadata3 = HuggingFaceMetadata(
        name="model3",
        author="author3",
        tags={"text-to-image", "checkpoint", "community-contributed"},
        id="author3/model3",
        tag_dict={"license": "other"},
        last_modified=datetime.datetime.now(),
    )
    metadata_store.add_metadata("test_config_1", metadata1)
    metadata_store.add_metadata("test_config_2", metadata2)
    metadata_store.add_metadata("test_config_3", metadata3)

    matches = metadata_store.search_by_author("stabilityai")
    assert len(matches) == 2
    assert "test_config_1" in matches
    assert "test_config_2" in matches
    matches = metadata_store.search_by_author("Sherlock Holmes")
    assert not matches

    matches = metadata_store.search_by_name("model3")
    assert len(matches) == 1
    assert "test_config_3" in matches

    matches = metadata_store.search_by_tag({"text-to-image"})
    assert len(matches) == 3

    matches = metadata_store.search_by_tag({"text-to-image", "diffusers"})
    assert len(matches) == 2
    assert "test_config_1" in matches
    assert "test_config_2" in matches

    matches = metadata_store.search_by_tag({"checkpoint", "community-contributed"})
    assert len(matches) == 1
    assert "test_config_3" in matches

    # does the tag table update correctly?
    matches = metadata_store.search_by_tag({"checkpoint", "licensed-for-commercial-use"})
    assert not matches
    metadata3.tags.add("licensed-for-commercial-use")
    metadata_store.update_metadata("test_config_3", metadata3)
    matches = metadata_store.search_by_tag({"checkpoint", "licensed-for-commercial-use"})
    assert len(matches) == 1


def test_metadata_civitai_fetch(session: Session) -> None:
    fetcher = CivitaiMetadataFetch(session)
    metadata = fetcher.from_url(HttpUrl("https://civitai.com/models/215485/SDXL-turbo"))
    assert isinstance(metadata, CivitaiMetadata)
    assert metadata.id == 215485
    assert metadata.author == "test_author"  # note that this is not the same as the original from Civitai
    assert metadata.allow_commercial_use  # changed to make sure we are reading locally not remotely
    assert metadata.restrictions.AllowCommercialUse == CommercialUsage("RentCivit")
    assert metadata.version_id == 242807
    assert metadata.tags == {"tool", "turbo", "sdxl turbo"}


def test_metadata_hf_fetch(session: Session) -> None:
    fetcher = HuggingFaceMetadataFetch(session)
    metadata = fetcher.from_url(HttpUrl("https://huggingface.co/stabilityai/sdxl-turbo"))
    assert isinstance(metadata, HuggingFaceMetadata)
    assert metadata.author == "test_author"  # this is not the same as the original
    assert metadata.files
    assert metadata.tags == {
        "diffusers",
        "onnx",
        "safetensors",
        "text-to-image",
        "license:other",
        "has_space",
        "diffusers:StableDiffusionXLPipeline",
        "region:us",
    }
