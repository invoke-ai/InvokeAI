from pathlib import Path

import pytest
import torch

from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.model_manager import ModelManagerServiceBase
from invokeai.app.services.shared.invocation_context import InvocationContext, build_invocation_context
from invokeai.backend.model_manager.load.load_base import LoadedModel
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
        cancel_event=None,  # type: ignore
    )


def test_download_and_cache(mock_context: InvocationContext, mm2_root_dir: Path) -> None:
    downloaded_path = mock_context.models.download_and_cache_ckpt(
        "https://www.test.foo/download/test_embedding.safetensors"
    )
    assert downloaded_path.is_file()
    assert downloaded_path.exists()
    assert downloaded_path.name == "test_embedding.safetensors"
    assert downloaded_path.parent.parent == mm2_root_dir / "models/.download_cache"

    downloaded_path_2 = mock_context.models.download_and_cache_ckpt(
        "https://www.test.foo/download/test_embedding.safetensors"
    )
    assert downloaded_path == downloaded_path_2


def test_load_from_path(mock_context: InvocationContext, embedding_file: Path) -> None:
    downloaded_path = mock_context.models.download_and_cache_ckpt(
        "https://www.test.foo/download/test_embedding.safetensors"
    )
    loaded_model_1 = mock_context.models.load_ckpt_from_path(downloaded_path)
    assert isinstance(loaded_model_1, LoadedModel)

    loaded_model_2 = mock_context.models.load_ckpt_from_path(downloaded_path)
    assert isinstance(loaded_model_2, LoadedModel)
    assert loaded_model_1.model is loaded_model_2.model

    loaded_model_3 = mock_context.models.load_ckpt_from_path(embedding_file)
    assert isinstance(loaded_model_3, LoadedModel)
    assert loaded_model_1.model is not loaded_model_3.model
    assert isinstance(loaded_model_1.model, dict)
    assert isinstance(loaded_model_3.model, dict)
    assert torch.equal(loaded_model_1.model["emb_params"], loaded_model_3.model["emb_params"])


def test_download_and_load(mock_context: InvocationContext) -> None:
    loaded_model_1 = mock_context.models.load_ckpt_from_url("https://www.test.foo/download/test_embedding.safetensors")
    assert isinstance(loaded_model_1, LoadedModel)

    loaded_model_2 = mock_context.models.load_ckpt_from_url("https://www.test.foo/download/test_embedding.safetensors")
    assert isinstance(loaded_model_2, LoadedModel)
    assert loaded_model_1.model is loaded_model_2.model  # should be cached copy
