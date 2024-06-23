from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderTiny

from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.model_manager import ModelManagerServiceBase
from invokeai.app.services.shared.invocation_context import InvocationContext, build_invocation_context
from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403


@pytest.fixture()
def mock_context(
    mock_services: InvocationServices,
    mm2_model_manager: ModelManagerServiceBase,
) -> InvocationContext:
    mock_services.model_manager = mm2_model_manager
    return build_invocation_context(
        services=mock_services,
        data=None,  # type: ignore
        is_canceled=None,  # type: ignore
    )


def test_download_and_cache(mock_context: InvocationContext, mm2_root_dir: Path) -> None:
    downloaded_path = mock_context.models.download_and_cache_model(
        "https://www.test.foo/download/test_embedding.safetensors"
    )
    assert downloaded_path.is_file()
    assert downloaded_path.exists()
    assert downloaded_path.name == "test_embedding.safetensors"
    assert downloaded_path.parent.parent == mm2_root_dir / "models/.download_cache"

    downloaded_path_2 = mock_context.models.download_and_cache_model(
        "https://www.test.foo/download/test_embedding.safetensors"
    )
    assert downloaded_path == downloaded_path_2


def test_load_from_path(mock_context: InvocationContext, embedding_file: Path) -> None:
    downloaded_path = mock_context.models.download_and_cache_model(
        "https://www.test.foo/download/test_embedding.safetensors"
    )
    loaded_model_1 = mock_context.models.load_local_model(downloaded_path)
    assert isinstance(loaded_model_1, LoadedModelWithoutConfig)

    loaded_model_2 = mock_context.models.load_local_model(downloaded_path)
    assert isinstance(loaded_model_2, LoadedModelWithoutConfig)
    assert loaded_model_1.model is loaded_model_2.model

    loaded_model_3 = mock_context.models.load_local_model(embedding_file)
    assert isinstance(loaded_model_3, LoadedModelWithoutConfig)
    assert loaded_model_1.model is not loaded_model_3.model
    assert isinstance(loaded_model_1.model, dict)
    assert isinstance(loaded_model_3.model, dict)
    assert torch.equal(loaded_model_1.model["emb_params"], loaded_model_3.model["emb_params"])


@pytest.mark.skip(reason="This requires a test model to load")
def test_load_from_dir(mock_context: InvocationContext, vae_directory: Path) -> None:
    loaded_model = mock_context.models.load_local_model(vae_directory)
    assert isinstance(loaded_model, LoadedModelWithoutConfig)
    assert isinstance(loaded_model.model, AutoencoderTiny)


def test_download_and_load(mock_context: InvocationContext) -> None:
    loaded_model_1 = mock_context.models.load_remote_model("https://www.test.foo/download/test_embedding.safetensors")
    assert isinstance(loaded_model_1, LoadedModelWithoutConfig)

    loaded_model_2 = mock_context.models.load_remote_model("https://www.test.foo/download/test_embedding.safetensors")
    assert isinstance(loaded_model_2, LoadedModelWithoutConfig)
    assert loaded_model_1.model is loaded_model_2.model  # should be cached copy


def test_download_diffusers(mock_context: InvocationContext) -> None:
    model_path = mock_context.models.download_and_cache_model("stabilityai/sdxl-turbo")
    assert (model_path / "model_index.json").exists()
    assert (model_path / "vae").is_dir()


def test_download_diffusers_subfolder(mock_context: InvocationContext) -> None:
    model_path = mock_context.models.download_and_cache_model("stabilityai/sdxl-turbo::vae")
    assert model_path.is_dir()
    assert (model_path / "diffusion_pytorch_model.fp16.safetensors").exists() or (
        model_path / "diffusion_pytorch_model.safetensors"
    ).exists()
