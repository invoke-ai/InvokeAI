from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.model_manager import ModelManagerServiceBase
from invokeai.app.services.shared.invocation_context import (
    InvocationContext,
    InvocationContextData,
    build_invocation_context,
)
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities
from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache.model_cache import CacheClearResult
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403


@pytest.fixture()
def mock_context(
    mock_services: InvocationServices,
    mm2_model_manager: ModelManagerServiceBase,
) -> InvocationContext:
    mock_services.model_manager = mm2_model_manager
    return build_invocation_context(
        services=mock_services,
        data=InvocationContextData(queue_item=None, invocation=None, source_invocation_id=None),  # type: ignore
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


def test_download_and_load(mock_context: InvocationContext) -> None:
    loaded_model_1 = mock_context.models.load_remote_model("https://www.test.foo/download/test_embedding.safetensors")
    assert isinstance(loaded_model_1, LoadedModelWithoutConfig)

    loaded_model_2 = mock_context.models.load_remote_model("https://www.test.foo/download/test_embedding.safetensors")
    assert isinstance(loaded_model_2, LoadedModelWithoutConfig)
    assert loaded_model_1.model is loaded_model_2.model  # should be cached copy


@pytest.mark.parametrize("path_kind", ["file", "directory"])
def test_load_from_path_refuses_before_loader_when_host_headroom_is_insufficient(
    mock_context: InvocationContext,
    mm2_model_manager: ModelManagerServiceBase,
    tmp_path: Path,
    monkeypatch,
    path_kind: str,
) -> None:
    if path_kind == "file":
        model_path = tmp_path / "model.safetensors"
        model_path.write_bytes(b"weights")
        expected_size = model_path.stat().st_size
    else:
        model_path = tmp_path / "nested" / "model"
        model_path.mkdir(parents=True)
        (model_path / "weights").mkdir()
        (model_path / "weights" / "part.safetensors").write_bytes(b"weights")
        expected_size = (model_path / "weights" / "part.safetensors").stat().st_size
    loader = MagicMock(return_value={"weights": torch.ones(1)})
    make_room = MagicMock(
        return_value=CacheClearResult(
            models_cleared=0,
            bytes_freed=0,
            vram_release_refused=False,
            ram_headroom_satisfied=False,
        )
    )
    monkeypatch.setattr(mm2_model_manager.load.ram_cache, "make_room", make_room)

    with pytest.raises(MemoryError, match="Insufficient host memory"):
        mock_context.models.load_local_model(model_path, loader)

    loader.assert_not_called()
    make_room.assert_called_once_with(expected_size)


def test_load_from_path_cache_hit_does_not_admit(
    mock_context: InvocationContext, mm2_model_manager: ModelManagerServiceBase, tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"weights")
    loader = MagicMock(return_value={"weights": torch.ones(1)})
    first = mock_context.models.load_local_model(model_path, loader)

    make_room = MagicMock(side_effect=AssertionError("cache hit must not admit"))
    monkeypatch.setattr(mm2_model_manager.load.ram_cache, "make_room", make_room)
    second = mock_context.models.load_local_model(model_path, loader)

    assert first.model is second.model
    make_room.assert_not_called()


def test_external_model_load_raises(
    mock_context: InvocationContext, mm2_model_manager: ModelManagerServiceBase
) -> None:
    config = ExternalApiModelConfig(
        key="external_test",
        name="External Test",
        provider_id="openai",
        provider_model_id="gpt-image-1",
        capabilities=ExternalModelCapabilities(modes=["txt2img"]),
    )
    mm2_model_manager.store.add_model(config)

    model_field = ModelIdentifierField.from_config(config)

    with pytest.raises(ValueError, match="External API models"):
        mock_context.models.load(model_field)

    with pytest.raises(ValueError, match="External API models"):
        mock_context.models.load_by_attrs(name=config.name, base=config.base, type=config.type)


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
