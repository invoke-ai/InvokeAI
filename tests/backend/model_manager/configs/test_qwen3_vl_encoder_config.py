"""Tests for single-file Qwen3-VL text-encoder identification (used by Krea-2).

A single-file Qwen3-VL encoder is distinguished from the text-only ``Qwen3Encoder`` (Z-Image /
FLUX.2 Klein) by the presence of the Qwen3-VL **visual tower** (``visual.*`` / ``model.visual.*``).
Both have a Qwen3 text decoder (``model.layers.*``), so the visual tower is the deciding signal.
"""

from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
    Qwen3VLEncoder_Checkpoint_Config,
    _is_qwen3_vl_encoder_state_dict,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/qwen3vl.safetensors",
    "file_size": 1000,
    "name": "qwen3vl-encoder",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


class TestIsQwen3VLEncoderStateDict:
    def test_text_decoder_plus_visual_tower_matches(self) -> None:
        # ComfyUI single-file layout (implicit LM prefix): model.layers.* + model.visual.*
        sd = {
            "model.layers.0.self_attn.q_proj.weight": object(),
            "model.visual.blocks.0.attn.qkv.weight": object(),
        }
        assert _is_qwen3_vl_encoder_state_dict(sd) is True

    def test_explicit_language_model_prefix_matches(self) -> None:
        # Alternative single-file layout (explicit LM prefix): model.language_model.layers.* + model.visual.*
        sd = {
            "model.language_model.layers.0.self_attn.q_proj.weight": object(),
            "model.visual.blocks.0.attn.qkv.weight": object(),
        }
        assert _is_qwen3_vl_encoder_state_dict(sd) is True

    def test_text_only_decoder_does_not_match(self) -> None:
        # Z-Image / FLUX.2 Klein Qwen3 text encoder: text decoder but NO visual tower.
        sd = {
            "model.layers.0.self_attn.q_proj.weight": object(),
            "model.layers.0.mlp.down_proj.weight": object(),
            "model.norm.weight": object(),
        }
        assert _is_qwen3_vl_encoder_state_dict(sd) is False

    def test_visual_tower_only_does_not_match(self) -> None:
        sd = {"model.visual.blocks.0.attn.qkv.weight": object()}
        assert _is_qwen3_vl_encoder_state_dict(sd) is False

    def test_ignores_non_string_keys(self) -> None:
        sd: dict = {0: object(), 1: object()}
        assert _is_qwen3_vl_encoder_state_dict(sd) is False


class TestQwen3VLEncoderCheckpointConfig:
    def _make_mock_mod(self, state_dict: dict) -> MagicMock:
        mod = MagicMock()
        mod.load_state_dict.return_value = state_dict
        return mod

    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_for_override_fields")
    def test_matches_vl_single_file(self, _rfo, _rif) -> None:
        mod = self._make_mock_mod(
            {
                "model.layers.0.self_attn.q_proj.weight": object(),
                "model.visual.blocks.0.attn.qkv.weight": object(),
            }
        )
        config = Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.type == ModelType.Qwen3VLEncoder
        assert config.base == BaseModelType.Any
        assert config.format == ModelFormat.Checkpoint

    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.qwen3_vl_encoder.raise_for_override_fields")
    def test_rejects_text_only_encoder(self, _rfo, _rif) -> None:
        mod = self._make_mock_mod(
            {
                "model.layers.0.self_attn.q_proj.weight": object(),
                "model.norm.weight": object(),
            }
        )
        with pytest.raises(NotAMatchError):
            Qwen3VLEncoder_Checkpoint_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
