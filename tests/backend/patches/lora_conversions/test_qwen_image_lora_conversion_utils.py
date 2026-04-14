"""Tests for Qwen Image LoRA conversion utilities."""

import torch

from invokeai.backend.patches.lora_conversions.qwen_image_lora_constants import (
    QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.qwen_image_lora_conversion_utils import (
    _convert_kohya_key,
    is_state_dict_likely_kohya_qwen_image,
    lora_model_from_qwen_image_state_dict,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.qwen_image_lora_diffusers_format import (
    state_dict_keys as diffusers_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.qwen_image_lora_kohya_format import (
    state_dict_keys as kohya_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict

# ---- Format detection tests ----


def test_is_kohya_format_true():
    """Kohya-format state dict is correctly identified."""
    state_dict = keys_to_mock_state_dict(kohya_state_dict_keys)
    assert is_state_dict_likely_kohya_qwen_image(state_dict)


def test_is_kohya_format_false_diffusers():
    """Diffusers-format state dict is not identified as Kohya."""
    state_dict = keys_to_mock_state_dict(diffusers_state_dict_keys)
    assert not is_state_dict_likely_kohya_qwen_image(state_dict)


def test_is_kohya_format_false_empty():
    """Empty state dict is not identified as Kohya."""
    assert not is_state_dict_likely_kohya_qwen_image({})


# ---- Kohya key conversion tests ----


def test_convert_kohya_key_attention():
    """Kohya attention projection keys convert correctly."""
    assert _convert_kohya_key("lora_unet_transformer_blocks_0_attn_to_k") == "transformer_blocks.0.attn.to_k"
    assert _convert_kohya_key("lora_unet_transformer_blocks_5_attn_to_q") == "transformer_blocks.5.attn.to_q"
    assert _convert_kohya_key("lora_unet_transformer_blocks_0_attn_to_v") == "transformer_blocks.0.attn.to_v"
    assert _convert_kohya_key("lora_unet_transformer_blocks_0_attn_to_out_0") == "transformer_blocks.0.attn.to_out.0"
    assert (
        _convert_kohya_key("lora_unet_transformer_blocks_0_attn_add_k_proj") == "transformer_blocks.0.attn.add_k_proj"
    )


def test_convert_kohya_key_mlp():
    """Kohya MLP keys convert correctly."""
    assert (
        _convert_kohya_key("lora_unet_transformer_blocks_0_img_mlp_net_0_proj")
        == "transformer_blocks.0.img_mlp.net.0.proj"
    )
    assert _convert_kohya_key("lora_unet_transformer_blocks_0_txt_mlp_net_2") == "transformer_blocks.0.txt_mlp.net.2"


def test_convert_kohya_key_unknown_returns_none():
    """Unknown Kohya sub-module returns None."""
    assert _convert_kohya_key("lora_unet_transformer_blocks_0_unknown_projection") is None


def test_convert_kohya_key_non_matching_returns_none():
    """Key that doesn't match the regex returns None."""
    assert _convert_kohya_key("some_random_key") is None


# ---- Full model conversion tests ----


def test_kohya_conversion_produces_correct_layer_keys():
    """Kohya state dict converts to ModelPatchRaw with correct prefixed layer keys."""
    state_dict = keys_to_mock_state_dict(kohya_state_dict_keys)
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    # Build expected keys: convert each Kohya layer name to model path, add prefix
    expected_keys: set[str] = set()
    for k in kohya_state_dict_keys:
        layer_name = k.split(".", 1)[0]  # e.g. lora_unet_transformer_blocks_0_attn_to_k
        model_path = _convert_kohya_key(layer_name)
        if model_path is not None:
            expected_keys.add(f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}{model_path}")

    assert set(model.layers.keys()) == expected_keys
    assert len(model.layers) > 0


def test_diffusers_conversion_produces_correct_layer_keys():
    """Diffusers state dict converts to ModelPatchRaw with correct prefixed layer keys."""
    state_dict = keys_to_mock_state_dict(diffusers_state_dict_keys)
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    expected_keys = {
        f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k",
        f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_q",
        f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.1.attn.to_k",
    }

    assert set(model.layers.keys()) == expected_keys


def test_diffusers_with_transformer_prefix_strips_it():
    """Diffusers keys with 'transformer.' prefix get it stripped."""
    state_dict = {
        "transformer.transformer_blocks.0.attn.to_k.lora_down.weight": torch.empty(64, 3072),
        "transformer.transformer_blocks.0.attn.to_k.lora_up.weight": torch.empty(3072, 64),
    }
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    expected_key = f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k"
    assert expected_key in model.layers


# ---- Unknown key handling tests ----


def test_kohya_unknown_submodule_is_silently_skipped():
    """Unknown Kohya sub-modules are skipped, producing no layers for them."""
    state_dict = {
        # Known key — should produce a layer
        "lora_unet_transformer_blocks_0_attn_to_k.lokr_w1": torch.empty(3072, 16),
        "lora_unet_transformer_blocks_0_attn_to_k.lokr_w2": torch.empty(16, 3072),
        # Unknown key — should be skipped
        "lora_unet_transformer_blocks_0_unknown_projection.lokr_w1": torch.empty(3072, 16),
        "lora_unet_transformer_blocks_0_unknown_projection.lokr_w2": torch.empty(16, 3072),
    }
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    # Only the known key should produce a layer
    assert len(model.layers) == 1
    expected_key = f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k"
    assert expected_key in model.layers


def test_kohya_all_unknown_submodules_produces_empty_model():
    """State dict with only unknown Kohya sub-modules produces an empty ModelPatchRaw."""
    state_dict = {
        "lora_unet_transformer_blocks_0_totally_unknown.lokr_w1": torch.empty(3072, 16),
        "lora_unet_transformer_blocks_0_totally_unknown.lokr_w2": torch.empty(16, 3072),
    }
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    assert len(model.layers) == 0
