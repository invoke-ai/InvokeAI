"""Tests for identifying UNet-only SDXL LoRAs (e.g. self-attention "slider" LoRAs).

Some SDXL LoRAs only patch the UNet and contain no cross-attention (`attn2`) or
text-encoder (`lora_te*`) keys. `lora_token_vector_length()` reads the base's
context dimension from exactly those keys, so for such LoRAs it returns `None`
and the base can't be inferred that way.

`_state_dict_looks_like_sdxl_unet_lora()` recovers SDXL identification from the
UNet block structure alone: SDXL's UNet has a deep transformer stack (up to 10
transformer blocks) in its lower-resolution attention blocks, so
`transformer_blocks` indices reach >= 2, whereas SD1.x/SD2.x only ever have a
single transformer block (index 0) per attention.

Real-world example: https://civitai.com/models/1105685 ("Dramatic Lighting
Slider"), a Kohya-format LoRA whose 840 keys are all `lora_unet_..._attn1_...`.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import (
    LoRA_LyCORIS_SD1_Config,
    LoRA_LyCORIS_SD2_Config,
    LoRA_LyCORIS_SDXL_Config,
    _state_dict_looks_like_sdxl_unet_lora,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/slider.safetensors",
    "file_size": 1000,
    "name": "test-slider",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _sdxl_self_attention_only_keys() -> dict[str, None]:
    """A minimal SDXL UNet self-attention-only (slider) state dict.

    Mirrors the real "Dramatic Lighting Slider" LoRA: only `attn1` keys, targeting
    the deep transformer stack in `down_blocks_2` / `mid_block` / `up_blocks_0`
    (transformer_blocks index up to 9). Kohya diffusers naming, values unused by
    the key-only heuristics under test.
    """
    sd: dict[str, None] = {}
    for tb in range(10):  # SDXL has 10 transformer blocks in its deep attention blocks
        for group in ("down_blocks_2_attentions_1", "mid_block_attentions_0", "up_blocks_0_attentions_2"):
            for proj in ("to_q", "to_k", "to_v", "to_out_0"):
                base = f"lora_unet_{group}_transformer_blocks_{tb}_attn1_{proj}"
                sd[f"{base}.lora_down.weight"] = None
                sd[f"{base}.lora_up.weight"] = None
                sd[f"{base}.alpha"] = None
    return sd


class TestSdxlUnetLoraDetection:
    """Unit tests for the key-only structural heuristic."""

    def test_self_attention_only_slider(self):
        assert _state_dict_looks_like_sdxl_unet_lora(_sdxl_self_attention_only_keys()) is True

    def test_diffusers_dot_format(self):
        sd = {"unet.down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_k.lora.down.weight": None}
        assert _state_dict_looks_like_sdxl_unet_lora(sd) is True

    def test_sd15_shallow_transformer_stack_not_matched(self):
        # SD1.x/SD2.x only ever have transformer_blocks_0 per attention.
        sd = {
            "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight": None,
            "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_v.lora_down.weight": None,
        }
        assert _state_dict_looks_like_sdxl_unet_lora(sd) is False

    def test_qwen_image_lora_not_matched(self):
        # Qwen uses `transformer_blocks.N` with high N but no UNet `attentions` grouping.
        sd = {
            "lora_unet_transformer_blocks_20_attn_to_k.lora_down.weight": None,
            "transformer.transformer_blocks.39.attn.to_q.lora_A.weight": None,
        }
        assert _state_dict_looks_like_sdxl_unet_lora(sd) is False

    def test_flux_lora_not_matched(self):
        sd = {"lora_unet_double_blocks_5_img_attn_proj.lora_down.weight": None}
        assert _state_dict_looks_like_sdxl_unet_lora(sd) is False

    def test_empty_state_dict(self):
        assert _state_dict_looks_like_sdxl_unet_lora({}) is False

    def test_non_string_keys_ignored(self):
        # GGUF-style integer keys must not crash the heuristic.
        assert _state_dict_looks_like_sdxl_unet_lora({0: None, 1: None}) is False


class TestSdxlSliderLoraIdentification:
    """End-to-end identification through the LyCORIS config classes."""

    def _make_mock_mod(self, state_dict: dict[str, None]) -> MagicMock:
        mod = MagicMock()
        mod.path = Path(_REQUIRED_FIELDS["path"])
        mod.load_state_dict.return_value = state_dict
        mod.metadata.return_value = {}
        return mod

    @patch("invokeai.backend.model_manager.configs.lora._get_flux_lora_format", return_value=None)
    @patch("invokeai.backend.model_manager.configs.lora.has_cosmos_dit_peft_keys", return_value=False)
    @patch("invokeai.backend.model_manager.configs.lora.has_cosmos_dit_kohya_keys", return_value=False)
    @patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
    def test_sdxl_config_matches(self, _rif, _hck, _hcp, _flux):
        mod = self._make_mock_mod(_sdxl_self_attention_only_keys())
        config = LoRA_LyCORIS_SDXL_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.base is BaseModelType.StableDiffusionXL

    @pytest.mark.parametrize("cls", [LoRA_LyCORIS_SD1_Config, LoRA_LyCORIS_SD2_Config])
    @patch("invokeai.backend.model_manager.configs.lora._get_flux_lora_format", return_value=None)
    @patch("invokeai.backend.model_manager.configs.lora.has_cosmos_dit_peft_keys", return_value=False)
    @patch("invokeai.backend.model_manager.configs.lora.has_cosmos_dit_kohya_keys", return_value=False)
    @patch("invokeai.backend.model_manager.configs.lora.raise_if_not_file")
    def test_sd1_sd2_configs_reject(self, _rif, _hck, _hcp, _flux, cls):
        mod = self._make_mock_mod(_sdxl_self_attention_only_keys())
        with pytest.raises(NotAMatchError):
            cls.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
