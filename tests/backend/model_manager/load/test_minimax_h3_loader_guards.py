"""Boundary tests for the MiniMax H3 diffusers loader's components-only guards.

A slim ("components-only") install carries tokenizer/processor/VAEs plus bare config JSONs, but no
transformer or text-encoder weights - those come from single-file installs selected in the model
loader node. Requesting the missing submodels must fail with an actionable message instead of a
cryptic missing-shard error from inside diffusers/transformers.
"""

from pathlib import Path

import pytest
import torch

from invokeai.backend.model_manager.configs.main import Main_Diffusers_MiniMaxH3_Config
from invokeai.backend.model_manager.load.model_loaders.minimax_h3 import MiniMaxH3DiffusersModel
from invokeai.backend.model_manager.taxonomy import SubModelType


@pytest.fixture
def slim_model_dir(tmp_path: Path) -> Path:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text("{}")
    # No text_encoder directory at all - the slim source does not download one.
    return tmp_path


@pytest.fixture
def loader(monkeypatch) -> MiniMaxH3DiffusersModel:
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.minimax_h3.TorchDevice.choose_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.minimax_h3.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.float32,
    )
    return object.__new__(MiniMaxH3DiffusersModel)


@pytest.mark.parametrize("submodel_type", [SubModelType.Transformer, SubModelType.TextEncoder])
def test_components_only_install_raises_actionable_error(
    loader: MiniMaxH3DiffusersModel, slim_model_dir: Path, submodel_type: SubModelType
) -> None:
    config = Main_Diffusers_MiniMaxH3_Config.model_construct(path=str(slim_model_dir), components_only=True)

    with pytest.raises(ValueError, match="components-only"):
        loader._load_model(config, submodel_type)


def test_transformer_with_weight_shards_passes_guard(
    loader: MiniMaxH3DiffusersModel, slim_model_dir: Path, monkeypatch
) -> None:
    (slim_model_dir / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"")

    import invokeai.backend.minimax_h3 as minimax_h3_module

    sentinel = object()
    monkeypatch.setattr(
        minimax_h3_module.MiniMaxH3Transformer3DModel,
        "from_pretrained",
        classmethod(lambda _cls, *_args, **_kwargs: sentinel),
    )

    config = Main_Diffusers_MiniMaxH3_Config.model_construct(path=str(slim_model_dir))
    assert loader._load_model(config, SubModelType.Transformer) is sentinel
