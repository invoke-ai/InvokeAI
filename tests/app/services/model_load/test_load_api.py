import gc
from pathlib import Path

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


def test_a_local_model_keyed_under_a_model_is_dropped_with_it(
    mock_context: InvocationContext, embedding_file: Path
) -> None:
    """A part of an installed model read from its file (the Anima IP-Adapter's bundled LoRA) must not outlive a
    re-identify or delete of that model, which drop its cached entries by model key."""
    model_key = "aaaaaaaa-1111-2222-3333-444444444444"

    def load(cache_key=None):
        # Only the model is kept: a live wrapper holds its entry in a first-use window, which a drop defers.
        model = mock_context.models.load_local_model(embedding_file, cache_key=cache_key).model
        gc.collect()
        return model

    part = load(f"{model_key}:part")
    assert load(f"{model_key}:part") is part
    assert load() is not part, "the path key is separate"

    assert mock_context._services.model_manager.load.drop_cached_models([model_key]) == 1
    assert load(f"{model_key}:part") is not part


def test_download_and_load(mock_context: InvocationContext) -> None:
    loaded_model_1 = mock_context.models.load_remote_model("https://www.test.foo/download/test_embedding.safetensors")
    assert isinstance(loaded_model_1, LoadedModelWithoutConfig)

    loaded_model_2 = mock_context.models.load_remote_model("https://www.test.foo/download/test_embedding.safetensors")
    assert isinstance(loaded_model_2, LoadedModelWithoutConfig)
    assert loaded_model_1.model is loaded_model_2.model  # should be cached copy


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
