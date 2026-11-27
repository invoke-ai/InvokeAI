"""Install-time refusal of SD 1.x / 2.x / SDXL GGUFs, made through ``ModelConfigFactory``.

No SD loader reads GGUF. These run at the default ``allow_unknown``, because that decides what the user sees: a
refusal with its reason, not a checkpoint or ControlNet record that fails at load, nor an Unknown model. Fixtures
are real GGUF files of a few kilobytes, read by the same GGUF loader identification uses.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from invokeai.backend.model_manager.configs.controlnet import (
    ControlNet_Checkpoint_SD1_Config,
    ControlNet_Checkpoint_SDXL_Config,
)
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from tests.fixtures.quantized_payloads import write_gguf

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test-model",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

_SD1_CONTEXT_KEY = "input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
_SDXL_CONTEXT_KEY = "input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"


def _identify(path: Path) -> Any:
    return ModelConfigFactory.from_model_on_disk(path, dict(_OVERRIDE_FIELDS), allow_unknown=True)


def _assert_refused_as_sd_gguf(result: Any) -> None:
    """Exactly one refusal: the installer shows the first, and the order of the candidate configs is not fixed."""
    assert result.config is None
    assert len(result.invalid_matches) == 1
    assert "not supported in GGUF format" in str(result.invalid_matches[0])


@pytest.mark.parametrize(
    ("context_key", "context_width"),
    [(_SD1_CONTEXT_KEY, 768), (_SDXL_CONTEXT_KEY, 2048)],
    ids=["sd1", "sdxl"],
)
def test_a_stable_diffusion_cpp_pipeline_gguf_is_refused_with_its_reason(
    tmp_path: Path, context_key: str, context_width: int
) -> None:
    """stable-diffusion.cpp's layout: the whole pipeline under its original names, quantized, with no metadata."""
    path = tmp_path / "stable-diffusion-Q8_0.gguf"
    write_gguf(
        path,
        {
            "model.diffusion_model.input_blocks.0.0.weight": torch.ones(8, 4, 3, 3, dtype=torch.float16),
            "first_stage_model.encoder.conv_in.weight": torch.ones(8, 3, 3, 3, dtype=torch.float16),
        },
        q8_0={
            f"model.diffusion_model.{context_key}": torch.randn(8, context_width),
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": torch.randn(8, 768),
        },
        architecture="",
    )

    _assert_refused_as_sd_gguf(_identify(path))


def test_a_bare_unet_gguf_is_refused_rather_than_installed_as_a_controlnet(tmp_path: Path) -> None:
    """ComfyUI-GGUF's layout: the UNet alone, its `model.diffusion_model.` prefix stripped. The SD 1.x UNet's
    `input_blocks.*` names are also a ControlNet's, which used to claim it."""
    path = tmp_path / "sd15-unet-Q8_0.gguf"
    write_gguf(
        path,
        {
            "input_blocks.0.0.weight": torch.ones(8, 4, 3, 3, dtype=torch.float16),
            "output_blocks.0.0.out_layers.3.weight": torch.ones(8, 8, 3, 3, dtype=torch.float16),
        },
        q8_0={_SD1_CONTEXT_KEY: torch.randn(8, 768)},
        architecture="sd1",
    )

    _assert_refused_as_sd_gguf(_identify(path))


def _sd1_controlnet_in_checkpoint_names(path: Path) -> None:
    write_gguf(
        path,
        {
            "input_blocks.0.0.weight": torch.ones(8, 4, 3, 3, dtype=torch.float16),
            "input_hint_block.0.weight": torch.ones(8, 3, 3, 3, dtype=torch.float16),
            "zero_convs.0.0.weight": torch.ones(8, 8, 1, 1, dtype=torch.float16),
        },
        q8_0={_SD1_CONTEXT_KEY: torch.randn(8, 768)},
        architecture="",
    )


def _sdxl_controlnet_in_diffusers_names(path: Path) -> None:
    write_gguf(
        path,
        {
            "controlnet_cond_embedding.conv_in.weight": torch.ones(16, 3, 3, 3, dtype=torch.float16),
            "controlnet_mid_block.bias": torch.ones(1280, dtype=torch.float16),
            "down_blocks.0.downsamplers.0.conv.weight": torch.ones(8, 8, 3, 3, dtype=torch.float16),
        },
        q8_0={"down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight": torch.randn(8, 2048)},
        architecture="",
    )


@pytest.mark.parametrize(
    ("write", "expected_config"),
    [
        (_sd1_controlnet_in_checkpoint_names, ControlNet_Checkpoint_SD1_Config),
        (_sdxl_controlnet_in_diffusers_names, ControlNet_Checkpoint_SDXL_Config),
    ],
    ids=["sd1-checkpoint-names", "sdxl-diffusers-names"],
)
def test_a_controlnet_gguf_still_identifies_as_a_controlnet(
    tmp_path: Path, write: Callable[[Path], None], expected_config: type
) -> None:
    """A ControlNet copies the UNet's encoder half under the same names; without the decoder half it is no UNet,
    so neither the main-model refusal nor the ControlNet config's GGUF guard applies to it."""
    path = tmp_path / "controlnet-Q8_0.gguf"
    write(path)

    result = _identify(path)

    assert isinstance(result.config, expected_config)
    assert not result.invalid_matches
