"""Qwen Image Edit LoRA conversion utilities.

Qwen Image Edit uses QwenImageTransformer2DModel architecture.
LoRAs follow the standard format with lora_down.weight/lora_up.weight/alpha keys.
"""

from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.qwen_image_edit_lora_constants import (
    QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


def lora_model_from_qwen_image_edit_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None = None
) -> ModelPatchRaw:
    """Convert a Qwen Image Edit LoRA state dict to a ModelPatchRaw.

    The Lightning LoRA keys are in the format:
        transformer_blocks.0.attn.to_k.lora_down.weight
        transformer_blocks.0.attn.to_k.lora_up.weight
        transformer_blocks.0.attn.to_k.alpha

    These are already the correct module paths for QwenImageTransformer2DModel.
    """
    layers: dict[str, BaseLayerPatch] = {}

    # Some LoRAs use a "transformer." prefix on keys (e.g. "transformer.transformer_blocks.0.attn.to_k")
    # while the model's module paths start at "transformer_blocks.0.attn.to_k". Strip it if present.
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
