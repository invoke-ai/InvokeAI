"""Tests for Qwen Image LoRA conversion utilities."""

import pytest
import torch

from invokeai.backend.patches.lora_conversions.qwen_image_lora_constants import (
    QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.qwen_image_lora_conversion_utils import (
    _convert_kohya_key,
    is_state_dict_likely_kohya_qwen_image,
    lora_model_from_qwen_image_state_dict,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict

# ---------------------------------------------------------------------------
# Minimal Kohya-format state dict (LoKR)
# ---------------------------------------------------------------------------
_KOHYA_LOKR_KEYS: dict[str, list[int]] = {
    "lora_unet_transformer_blocks_0_attn_to_k.lokr_w1": [16, 64],
    "lora_unet_transformer_blocks_0_attn_to_k.lokr_w2": [64, 16],
    "lora_unet_transformer_blocks_1_attn_to_q.lokr_w1": [16, 64],
    "lora_unet_transformer_blocks_1_attn_to_q.lokr_w2": [64, 16],
}

# Diffusers-format keys (without the lora_unet_ prefix)
_DIFFUSERS_KEYS: dict[str, list[int]] = {
    "transformer_blocks.0.attn.to_k.lora_down.weight": [16, 64],
    "transformer_blocks.0.attn.to_k.lora_up.weight": [64, 16],
}


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


def test_is_state_dict_likely_kohya_qwen_image_true():
    """is_state_dict_likely_kohya_qwen_image returns True for Kohya keys."""
    state_dict = keys_to_mock_state_dict(_KOHYA_LOKR_KEYS)
    assert is_state_dict_likely_kohya_qwen_image(state_dict)


def test_is_state_dict_likely_kohya_qwen_image_false_diffusers():
    """is_state_dict_likely_kohya_qwen_image returns False for Diffusers-format keys."""
    state_dict = keys_to_mock_state_dict(_DIFFUSERS_KEYS)
    assert not is_state_dict_likely_kohya_qwen_image(state_dict)


def test_is_state_dict_likely_kohya_qwen_image_false_empty():
    """is_state_dict_likely_kohya_qwen_image returns False for an empty state dict."""
    assert not is_state_dict_likely_kohya_qwen_image({})


# ---------------------------------------------------------------------------
# Key conversion tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kohya_layer,expected_path",
    [
        ("lora_unet_transformer_blocks_0_attn_to_k", "transformer_blocks.0.attn.to_k"),
        ("lora_unet_transformer_blocks_3_attn_to_q", "transformer_blocks.3.attn.to_q"),
        ("lora_unet_transformer_blocks_0_attn_to_v", "transformer_blocks.0.attn.to_v"),
        ("lora_unet_transformer_blocks_0_attn_add_k_proj", "transformer_blocks.0.attn.add_k_proj"),
        ("lora_unet_transformer_blocks_0_attn_to_out_0", "transformer_blocks.0.attn.to_out.0"),
        ("lora_unet_transformer_blocks_0_img_mlp_net_0_proj", "transformer_blocks.0.img_mlp.net.0.proj"),
    ],
)
def test_convert_kohya_key_known_modules(kohya_layer: str, expected_path: str):
    """_convert_kohya_key correctly maps known Kohya layer names to model paths."""
    result = _convert_kohya_key(kohya_layer)
    assert result == expected_path


def test_convert_kohya_key_unknown_submodule_returns_none():
    """_convert_kohya_key returns None for an unrecognized sub-module."""
    result = _convert_kohya_key("lora_unet_transformer_blocks_0_completely_unknown_layer")
    assert result is None


def test_convert_kohya_key_wrong_prefix_returns_none():
    """_convert_kohya_key returns None when the key does not match the expected prefix pattern."""
    result = _convert_kohya_key("some_other_prefix_0_attn_to_k")
    assert result is None


# ---------------------------------------------------------------------------
# Full conversion tests
# ---------------------------------------------------------------------------


def test_lora_model_from_qwen_image_state_dict_kohya_lokr():
    """Kohya LoKR state dict converts to a ModelPatchRaw with the expected layer keys."""
    state_dict = keys_to_mock_state_dict(_KOHYA_LOKR_KEYS)
    model = lora_model_from_qwen_image_state_dict(state_dict)

    expected_keys = {
        f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k",
        f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.1.attn.to_q",
    }
    assert set(model.layers.keys()) == expected_keys


def test_lora_model_from_qwen_image_state_dict_kohya_unknown_submodule_skipped():
    """Unknown Kohya sub-module keys are silently skipped and do not appear in the output."""
    state_dict = {
        "lora_unet_transformer_blocks_0_attn_to_k.lokr_w1": torch.empty(16, 64),
        "lora_unet_transformer_blocks_0_attn_to_k.lokr_w2": torch.empty(64, 16),
        # This sub-module is not in _KOHYA_MODULE_MAP so should be skipped
        "lora_unet_transformer_blocks_0_unknown_submodule.lokr_w1": torch.empty(16, 64),
        "lora_unet_transformer_blocks_0_unknown_submodule.lokr_w2": torch.empty(64, 16),
    }
    model = lora_model_from_qwen_image_state_dict(state_dict)

    # Only the known layer should be present
    assert len(model.layers) == 1
    assert f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k" in model.layers
    # Unknown layer must not appear
    unknown_keys = [k for k in model.layers if "unknown_submodule" in k]
    assert unknown_keys == [], f"Unexpected layers in output: {unknown_keys}"


def test_lora_model_from_qwen_image_state_dict_diffusers():
    """Diffusers-format state dict converts to a ModelPatchRaw with the expected layer key."""
    state_dict = keys_to_mock_state_dict(_DIFFUSERS_KEYS)
    model = lora_model_from_qwen_image_state_dict(state_dict)

    expected_key = f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k"
    assert expected_key in model.layers


def test_lora_model_from_qwen_image_state_dict_diffusers_with_transformer_prefix():
    """Diffusers-format state dict with an extra 'transformer.' prefix is handled correctly."""
    state_dict = {
        "transformer.transformer_blocks.0.attn.to_k.lora_down.weight": torch.empty(16, 64),
        "transformer.transformer_blocks.0.attn.to_k.lora_up.weight": torch.empty(64, 16),
    }
    model = lora_model_from_qwen_image_state_dict(state_dict)

    expected_key = f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k"
    assert expected_key in model.layers
