"""
Test model metadata fetching and storage.
"""
import datetime
from pathlib import Path

import pytest
from pydantic.networks import HttpUrl
from requests.sessions import Session

from invokeai.app.services.model_metadata import ModelMetadataStoreBase
from invokeai.backend.model_manager.config import ModelRepoVariant
from invokeai.backend.model_manager.metadata import (
    CivitaiMetadata,
    CivitaiMetadataFetch,
    CommercialUsage,
    HuggingFaceMetadata,
    HuggingFaceMetadataFetch,
    UnknownMetadataException,
)
from invokeai.backend.model_manager.util import select_hf_files
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403


def test_metadata_store_put_get(mm2_metadata_store: ModelMetadataStoreBase) -> None:
    tags = {"text-to-image", "diffusers"}
    input_metadata = HuggingFaceMetadata(
        name="sdxl-vae",
        author="stabilityai",
        tags=tags,
        id="stabilityai/sdxl-vae",
        tag_dict={"license": "other"},
        last_modified=datetime.datetime.now(),
    )
    mm2_metadata_store.add_metadata("test_config_1", input_metadata)
    output_metadata = mm2_metadata_store.get_metadata("test_config_1")
    assert input_metadata == output_metadata
    with pytest.raises(UnknownMetadataException):
        mm2_metadata_store.add_metadata("unknown_key", input_metadata)
    assert mm2_metadata_store.list_tags() == tags


def test_metadata_store_update(mm2_metadata_store: ModelMetadataStoreBase) -> None:
    input_metadata = HuggingFaceMetadata(
        name="sdxl-vae",
        author="stabilityai",
        tags={"text-to-image", "diffusers"},
        id="stabilityai/sdxl-vae",
        tag_dict={"license": "other"},
        last_modified=datetime.datetime.now(),
    )
    mm2_metadata_store.add_metadata("test_config_1", input_metadata)
    input_metadata.name = "new-name"
    mm2_metadata_store.update_metadata("test_config_1", input_metadata)
    output_metadata = mm2_metadata_store.get_metadata("test_config_1")
    assert output_metadata.name == "new-name"
    assert input_metadata == output_metadata


def test_metadata_search(mm2_metadata_store: ModelMetadataStoreBase) -> None:
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
    mm2_metadata_store.add_metadata("test_config_1", metadata1)
    mm2_metadata_store.add_metadata("test_config_2", metadata2)
    mm2_metadata_store.add_metadata("test_config_3", metadata3)

    matches = mm2_metadata_store.search_by_author("stabilityai")
    assert len(matches) == 2
    assert "test_config_1" in matches
    assert "test_config_2" in matches
    matches = mm2_metadata_store.search_by_author("Sherlock Holmes")
    assert not matches

    matches = mm2_metadata_store.search_by_name("model3")
    assert len(matches) == 1
    assert "test_config_3" in matches

    matches = mm2_metadata_store.search_by_tag({"text-to-image"})
    assert len(matches) == 3

    matches = mm2_metadata_store.search_by_tag({"text-to-image", "diffusers"})
    assert len(matches) == 2
    assert "test_config_1" in matches
    assert "test_config_2" in matches

    matches = mm2_metadata_store.search_by_tag({"checkpoint", "community-contributed"})
    assert len(matches) == 1
    assert "test_config_3" in matches

    # does the tag table update correctly?
    matches = mm2_metadata_store.search_by_tag({"checkpoint", "licensed-for-commercial-use"})
    assert not matches
    assert mm2_metadata_store.list_tags() == {"text-to-image", "diffusers", "community-contributed", "checkpoint"}
    metadata3.tags.add("licensed-for-commercial-use")
    mm2_metadata_store.update_metadata("test_config_3", metadata3)
    assert mm2_metadata_store.list_tags() == {
        "text-to-image",
        "diffusers",
        "community-contributed",
        "checkpoint",
        "licensed-for-commercial-use",
    }
    matches = mm2_metadata_store.search_by_tag({"checkpoint", "licensed-for-commercial-use"})
    assert len(matches) == 1


def test_metadata_civitai_fetch(mm2_session: Session) -> None:
    fetcher = CivitaiMetadataFetch(mm2_session)
    metadata = fetcher.from_url(HttpUrl("https://civitai.com/models/215485/SDXL-turbo"))
    assert isinstance(metadata, CivitaiMetadata)
    assert metadata.id == 215485
    assert metadata.author == "test_author"  # note that this is not the same as the original from Civitai
    assert metadata.allow_commercial_use  # changed to make sure we are reading locally not remotely
    assert CommercialUsage("RentCivit") in metadata.restrictions.AllowCommercialUse
    assert metadata.version_id == 242807
    assert metadata.tags == {"tool", "turbo", "sdxl turbo"}


def test_metadata_hf_fetch(mm2_session: Session) -> None:
    fetcher = HuggingFaceMetadataFetch(mm2_session)
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


def test_metadata_hf_filter(mm2_session: Session) -> None:
    metadata = HuggingFaceMetadataFetch(mm2_session).from_url(HttpUrl("https://huggingface.co/stabilityai/sdxl-turbo"))
    assert isinstance(metadata, HuggingFaceMetadata)
    files = [x.path for x in metadata.files]
    fp16_files = select_hf_files.filter_files(files, variant=ModelRepoVariant("fp16"))
    assert Path("sdxl-turbo/text_encoder/model.fp16.safetensors") in fp16_files
    assert Path("sdxl-turbo/text_encoder/model.safetensors") not in fp16_files

    fp32_files = select_hf_files.filter_files(files, variant=ModelRepoVariant("fp32"))
    assert Path("sdxl-turbo/text_encoder/model.safetensors") in fp32_files
    assert Path("sdxl-turbo/text_encoder/model.16.safetensors") not in fp32_files

    onnx_files = select_hf_files.filter_files(files, variant=ModelRepoVariant("onnx"))
    assert Path("sdxl-turbo/text_encoder/model.onnx") in onnx_files
    assert Path("sdxl-turbo/text_encoder/model.safetensors") not in onnx_files

    default_files = select_hf_files.filter_files(files)
    assert Path("sdxl-turbo/text_encoder/model.safetensors") in default_files
    assert Path("sdxl-turbo/text_encoder/model.16.safetensors") not in default_files

    openvino_files = select_hf_files.filter_files(files, variant=ModelRepoVariant("openvino"))
    print(openvino_files)
    assert len(openvino_files) == 0

    flax_files = select_hf_files.filter_files(files, variant=ModelRepoVariant("flax"))
    print(flax_files)
    assert not flax_files

    metadata = HuggingFaceMetadataFetch(mm2_session).from_url(
        HttpUrl("https://huggingface.co/stabilityai/sdxl-turbo-nofp16")
    )
    assert isinstance(metadata, HuggingFaceMetadata)
    files = [x.path for x in metadata.files]
    filtered_files = select_hf_files.filter_files(files, variant=ModelRepoVariant("fp16"))
    assert (
        Path("sdxl-turbo-nofp16/text_encoder/model.safetensors") in filtered_files
    )  # confirm that default is returned
    assert Path("sdxl-turbo-nofp16/text_encoder/model.16.safetensors") not in filtered_files


def test_metadata_hf_urls(mm2_session: Session) -> None:
    metadata = HuggingFaceMetadataFetch(mm2_session).from_url(HttpUrl("https://huggingface.co/stabilityai/sdxl-turbo"))
    assert isinstance(metadata, HuggingFaceMetadata)
