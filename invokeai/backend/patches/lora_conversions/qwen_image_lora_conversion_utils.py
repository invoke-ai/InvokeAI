"""Qwen Image LoRA conversion utilities.

Qwen Image uses QwenImageTransformer2DModel architecture.
Supports multiple LoRA formats:
- Diffusers/PEFT: transformer_blocks.0.attn.to_k.lora_down.weight
- With prefix: transformer.transformer_blocks.0.attn.to_k.lora_down.weight
- Kohya: lora_unet_transformer_blocks_0_attn_to_k.lora_down.weight (underscores instead of dots)
"""

import re
from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.qwen_image_lora_constants import (
    QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# Regex for Kohya-format Qwen Image LoRA keys.
# Example: lora_unet_transformer_blocks_0_attn_to_k
# Groups: (block_idx, sub_module_with_underscores)
_KOHYA_KEY_REGEX = re.compile(r"lora_unet_transformer_blocks_(\d+)_(.*)")

# Mapping from Kohya underscore-separated sub-module names to dot-separated model paths.
# The Kohya format uses underscores everywhere, but some underscores are part of the
# module name (e.g., add_k_proj, to_out). We match the longest prefix first.
_KOHYA_MODULE_MAP: list[tuple[str, str]] = [
    # Attention projections
    ("attn_add_k_proj", "attn.add_k_proj"),
    ("attn_add_q_proj", "attn.add_q_proj"),
    ("attn_add_v_proj", "attn.add_v_proj"),
    ("attn_to_add_out", "attn.to_add_out"),
    ("attn_to_out_0", "attn.to_out.0"),
    ("attn_to_k", "attn.to_k"),
    ("attn_to_q", "attn.to_q"),
    ("attn_to_v", "attn.to_v"),
    # Image stream MLP and modulation
    ("img_mlp_net_0_proj", "img_mlp.net.0.proj"),
    ("img_mlp_net_2", "img_mlp.net.2"),
    ("img_mod_1", "img_mod.1"),
    # Text stream MLP and modulation
    ("txt_mlp_net_0_proj", "txt_mlp.net.0.proj"),
    ("txt_mlp_net_2", "txt_mlp.net.2"),
    ("txt_mod_1", "txt_mod.1"),
]


def is_state_dict_likely_kohya_qwen_image(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Check if the state dict uses Kohya-format Qwen Image LoRA keys."""
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if not str_keys:
        return False
    # Check if any key matches the Kohya pattern
    return any(k.startswith("lora_unet_transformer_blocks_") for k in str_keys)


def _convert_kohya_key(kohya_layer: str) -> str | None:
    """Convert a Kohya-format layer name to a dot-separated model module path.

    Example: lora_unet_transformer_blocks_0_attn_to_k -> transformer_blocks.0.attn.to_k
    """
    m = _KOHYA_KEY_REGEX.match(kohya_layer)
    if not m:
        return None

    block_idx = m.group(1)
    sub_module = m.group(2)

    for kohya_name, model_path in _KOHYA_MODULE_MAP:
        if sub_module == kohya_name:
            return f"transformer_blocks.{block_idx}.{model_path}"

    # Fallback: unknown sub-module, return None so caller can warn/skip
    return None


def lora_model_from_qwen_image_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None = None
) -> ModelPatchRaw:
    """Convert a Qwen Image LoRA state dict to a ModelPatchRaw.

    Handles three key formats:
    - Diffusers/PEFT: transformer_blocks.0.attn.to_k.lora_down.weight
    - With prefix: transformer.transformer_blocks.0.attn.to_k.lora_down.weight
    - Kohya: lora_unet_transformer_blocks_0_attn_to_k.lora_down.weight
    """
    is_kohya = is_state_dict_likely_kohya_qwen_image(state_dict)

    if is_kohya:
        return _convert_kohya_format(state_dict, alpha)
    else:
        return _convert_diffusers_format(state_dict, alpha)


def _convert_kohya_format(state_dict: Dict[str, torch.Tensor], alpha: float | None) -> ModelPatchRaw:
    """Convert Kohya-format state dict. Keys are like lora_unet_transformer_blocks_0_attn_to_k.lokr_w1"""
    layers: dict[str, BaseLayerPatch] = {}

    # Group by layer (split at first dot: layer_name.param_name)
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        layer_name, param_name = key.split(".", 1)
        if layer_name not in grouped:
            grouped[layer_name] = {}
        grouped[layer_name][param_name] = value

    for kohya_layer, layer_dict in grouped.items():
        model_path = _convert_kohya_key(kohya_layer)
        if model_path is None:
            continue  # Skip unrecognized layers

        layer = any_lora_layer_from_state_dict(layer_dict)
        final_key = f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}{model_path}"
        layers[final_key] = layer

    return ModelPatchRaw(layers=layers)


def _convert_diffusers_format(state_dict: Dict[str, torch.Tensor], alpha: float | None) -> ModelPatchRaw:
    """Convert Diffusers/PEFT format state dict."""
    layers: dict[str, BaseLayerPatch] = {}

    # Some LoRAs use a "transformer." prefix on keys
    strip_prefixes = ["transformer."]

    grouped = _group_by_layer(state_dict)

    for layer_key, layer_dict in grouped.items():
        values = _normalize_lora_keys(layer_dict, alpha)
        layer = any_lora_layer_from_state_dict(values)
        clean_key = layer_key
        for prefix in strip_prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
                break
        final_key = f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}{clean_key}"
        layers[final_key] = layer

    return ModelPatchRaw(layers=layers)


def _normalize_lora_keys(layer_dict: dict[str, torch.Tensor], alpha: float | None) -> dict[str, torch.Tensor]:
    """Normalize LoRA key names to internal format."""
    if "lora_A.weight" in layer_dict:
        values: dict[str, torch.Tensor] = {
            "lora_down.weight": layer_dict["lora_A.weight"],
            "lora_up.weight": layer_dict["lora_B.weight"],
        }
        if alpha is not None:
            values["alpha"] = torch.tensor(alpha)
        return values
    elif "lora_down.weight" in layer_dict:
        return layer_dict
    else:
        return layer_dict


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Group state dict keys by layer path."""
    layer_dict: dict[str, dict[str, torch.Tensor]] = {}

    known_suffixes = [
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_down.weight",
        ".lora_up.weight",
        ".dora_scale",
        ".alpha",
    ]

    for key in state_dict:
        if not isinstance(key, str):
            continue

        layer_name = None
        key_name = None
        for suffix in known_suffixes:
            if key.endswith(suffix):
                layer_name = key[: -len(suffix)]
                key_name = suffix[1:]
                break

        if layer_name is None:
            parts = key.rsplit(".", maxsplit=2)
            layer_name = parts[0]
            key_name = ".".join(parts[1:])

        if layer_name not in layer_dict:
            layer_dict[layer_name] = {}
        layer_dict[layer_name][key_name] = state_dict[key]

    return layer_dict
