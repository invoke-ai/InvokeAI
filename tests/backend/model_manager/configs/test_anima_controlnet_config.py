"""Tests for Anima ControlNet-LLLite config probing.

Anima LLLite adapters (v2 named-key format) are identified by the presence of both the shared
conditioning trunk (`lllite_conditioning1.*`) and per-module weights (`lllite_dit_blocks_*`).
SDXL ControlNet-LLLite models (`lllite_unet_*`) and Z-Image Control adapters
(`control_layers.*` etc.) must not match.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.controlnet import (
    ControlNet_Checkpoint_Anima_Config,
    _has_anima_lllite_keys,
)
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

# Opt-in test against the real adapter weights (anima-lllite-inpainting-v2.safetensors).
_REAL_WEIGHTS_ENV_VAR = "ANIMA_LLLITE_WEIGHTS_PATH"
REAL_WEIGHTS_PATH = Path(os.environ[_REAL_WEIGHTS_ENV_VAR]) if _REAL_WEIGHTS_ENV_VAR in os.environ else None

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/anima-lllite.safetensors",
    "file_size": 1000,
    "name": "anima-lllite",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

ANIMA_LLLITE_KEYS = [
    "lllite_conditioning1.conv1.weight",
    "lllite_conditioning1.conv1.bias",
    "lllite_conditioning1.proj.weight",
    "lllite_conditioning1.out_norm.weight",
    "lllite_dit_blocks_0_self_attn_q_proj.down.weight",
    "lllite_dit_blocks_0_self_attn_q_proj.depth_embed",
    "lllite_dit_blocks_27_mlp_layer1.up.weight",
]

SDXL_LLLITE_KEYS = [
    "lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.cond_emb.weight",
    "lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.down.0.weight",
    "lllite_unet_middle_block_1_transformer_blocks_0_attn1_to_q.up.0.weight",
    "lllite_unet_output_blocks_0_1_transformer_blocks_0_attn1_to_q.mid.0.weight",
]

Z_IMAGE_CONTROL_KEYS = [
    "control_layers.0.attention.qkv.weight",
    "control_layers.1.attention.out.weight",
    "control_all_x_embedder.2-1.weight",
    "control_noise_refiner.0.attention.qkv.weight",
]


def _make_state_dict(keys: list[str]) -> dict[str, object]:
    return dict.fromkeys(keys)


def _make_mod(state_dict: dict[str, object]) -> MagicMock:
    mod = MagicMock()
    mod.load_state_dict.return_value = state_dict
    return mod


class TestHasAnimaLLLiteKeys:
    """Tests for the _has_anima_lllite_keys heuristic used during model identification."""

    def test_anima_lllite_keys(self):
        assert _has_anima_lllite_keys(_make_state_dict(ANIMA_LLLITE_KEYS)) is True

    def test_trunk_only_does_not_match(self):
        sd = _make_state_dict([k for k in ANIMA_LLLITE_KEYS if k.startswith("lllite_conditioning1.")])
        assert _has_anima_lllite_keys(sd) is False

    def test_modules_only_does_not_match(self):
        sd = _make_state_dict([k for k in ANIMA_LLLITE_KEYS if k.startswith("lllite_dit_blocks_")])
        assert _has_anima_lllite_keys(sd) is False

    def test_sdxl_lllite_does_not_match(self):
        assert _has_anima_lllite_keys(_make_state_dict(SDXL_LLLITE_KEYS)) is False

    def test_z_image_control_does_not_match(self):
        assert _has_anima_lllite_keys(_make_state_dict(Z_IMAGE_CONTROL_KEYS)) is False

    def test_empty_state_dict(self):
        assert _has_anima_lllite_keys({}) is False


class TestAnimaControlNetConfigProbe:
    """Tests for ControlNet_Checkpoint_Anima_Config.from_model_on_disk."""

    def test_matches_anima_lllite(self):
        mod = _make_mod(_make_state_dict(ANIMA_LLLITE_KEYS))

        config = ControlNet_Checkpoint_Anima_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

        assert config.base is BaseModelType.Anima
        assert config.type is ModelType.ControlNet
        assert config.format is ModelFormat.Checkpoint

    @pytest.mark.parametrize(
        "keys",
        [SDXL_LLLITE_KEYS, Z_IMAGE_CONTROL_KEYS, []],
        ids=["sdxl_lllite", "z_image_control", "empty"],
    )
    def test_rejects_non_anima_lllite(self, keys: list[str]):
        mod = _make_mod(_make_state_dict(keys))

        with pytest.raises(NotAMatchError, match="does not look like an Anima ControlNet-LLLite"):
            ControlNet_Checkpoint_Anima_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


@pytest.mark.skipif(
    REAL_WEIGHTS_PATH is None or not REAL_WEIGHTS_PATH.is_file(),
    reason=f"set {_REAL_WEIGHTS_ENV_VAR} to the real LLLite weights file to run",
)
def test_real_file_classifies_as_anima_controlnet():
    """The real adapter file must classify uniquely as an Anima ControlNet via the full factory."""
    from invokeai.backend.model_manager.configs.factory import ModelConfigFactory

    assert REAL_WEIGHTS_PATH is not None
    result = ModelConfigFactory.from_model_on_disk(REAL_WEIGHTS_PATH, allow_unknown=False)

    assert result.config is not None
    assert isinstance(result.config, ControlNet_Checkpoint_Anima_Config)
    assert result.match_count == 1
