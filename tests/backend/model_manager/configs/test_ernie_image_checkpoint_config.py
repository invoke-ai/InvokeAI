"""Identification of single-file ERNIE-Image transformers.

Comfy-Org ships ERNIE-Image as one safetensors file carrying the transformer alone, under the same
keys as the diffusers checkpoint. The fingerprint has to be specific enough not to claim Anima,
which also has an `x_embedder`, and the base and Turbo releases are indistinguishable on disk — the
architecture picks Turbo's defaults from the model name, so identification must leave that intact.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_ErnieImage_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat


def _ernie_image_tensors() -> dict[str, torch.Tensor]:
    """The released key layout at toy width: patch projection, text projection, model-level adaLN."""
    return {
        "x_embedder.proj.weight": torch.zeros(8, 4, 1, 1),
        "x_embedder.proj.bias": torch.zeros(8),
        "text_proj.weight": torch.zeros(8, 6),
        "adaLN_modulation.1.weight": torch.zeros(48, 8),
        "adaLN_modulation.1.bias": torch.zeros(48),
        "final_norm.linear.weight": torch.zeros(16, 8),
        "final_norm.linear.bias": torch.zeros(16),
        "final_linear.weight": torch.zeros(4, 8),
        "final_linear.bias": torch.zeros(4),
        "time_embedding.linear_1.weight": torch.zeros(8, 8),
        "layers.0.self_attention.to_q.weight": torch.zeros(8, 8),
    }


def _write(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    save_file(tensors, str(path))
    return path


def test_a_single_file_transformer_is_identified(tmp_path: Path) -> None:
    result = ModelConfigFactory.from_model_on_disk(
        _write(tmp_path / "ernie-image.safetensors", _ernie_image_tensors()), allow_unknown=False
    )

    assert isinstance(result.config, Main_Checkpoint_ErnieImage_Config), result.details
    assert result.config.base is BaseModelType.ErnieImage
    assert result.config.format is ModelFormat.Checkpoint


def test_the_turbo_release_keeps_its_own_defaults(tmp_path: Path) -> None:
    """Turbo and the base model share every key, so the name is the only signal — and it is the
    difference between 8 steps without CFG and 50 steps at CFG 4."""
    base = ModelConfigFactory.from_model_on_disk(
        _write(tmp_path / "ernie-image.safetensors", _ernie_image_tensors()), allow_unknown=False
    ).config
    turbo = ModelConfigFactory.from_model_on_disk(
        _write(tmp_path / "ernie-image-turbo.safetensors", _ernie_image_tensors()), allow_unknown=False
    ).config

    assert base is not None and turbo is not None
    assert (base.default_settings.steps, base.default_settings.cfg_scale) == (50, 4.0)
    assert (turbo.default_settings.steps, turbo.default_settings.cfg_scale) == (8, 1.0)


def test_an_anima_transformer_is_not_claimed(tmp_path: Path) -> None:
    """Anima carries an `x_embedder` too; its `llm_adapter` is what tells them apart."""
    tensors = {
        "x_embedder.proj.weight": torch.zeros(8, 4, 1, 1),
        "llm_adapter.proj.weight": torch.zeros(8, 8),
        "blocks.0.attn.to_q.weight": torch.zeros(8, 8),
        "t_embedder.linear_1.weight": torch.zeros(8, 8),
        "final_layer.linear.weight": torch.zeros(4, 8),
    }

    result = ModelConfigFactory.from_model_on_disk(_write(tmp_path / "anima.safetensors", tensors), allow_unknown=True)

    assert not isinstance(result.config, Main_Checkpoint_ErnieImage_Config)


@pytest.mark.parametrize(
    "dropped",
    [
        "text_proj.weight",
        "x_embedder.proj.weight",
        "adaLN_modulation.1.weight",
        "final_norm.linear.weight",
    ],
)
def test_a_file_missing_part_of_the_fingerprint_is_refused(tmp_path: Path, dropped: str) -> None:
    tensors = {key: value for key, value in _ernie_image_tensors().items() if key != dropped}

    result = ModelConfigFactory.from_model_on_disk(
        _write(tmp_path / "partial.safetensors", tensors), allow_unknown=True
    )

    assert not isinstance(result.config, Main_Checkpoint_ErnieImage_Config)
