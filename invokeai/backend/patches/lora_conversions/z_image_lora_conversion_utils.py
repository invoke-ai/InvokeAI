"""Z-Image LoRA conversion utilities.

Z-Image uses S3-DiT transformer architecture with Qwen3 text encoder.
LoRAs for Z-Image typically follow the diffusers PEFT format or Kohya format.
"""

import re
from typing import Any, Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import (
    Z_IMAGE_LORA_QWEN3_PREFIX,
    Z_IMAGE_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# Regex for Kohya-format Z-Image transformer keys.
# Example keys:
#   lora_unet__layers_0_attention_to_k.alpha
#   lora_unet__layers_0_attention_to_k.lora_down.weight
#   lora_unet__context_refiner_0_feed_forward_w1.lora_up.weight
#   lora_unet__noise_refiner_1_attention_to_v.lora_down.weight
Z_IMAGE_KOHYA_TRANSFORMER_KEY_REGEX = (
    r"lora_unet__(layers|context_refiner|noise_refiner)_(\d+)_(attention|feed_forward)_(to_k|to_q|to_v|w1|w2|w3)"
)


def is_state_dict_likely_z_image_kohya_lora(state_dict: dict[str | int, Any]) -> bool:
    """Checks if the provided state dict is likely a Z-Image LoRA in Kohya format.

    Kohya Z-Image LoRAs have keys like:
    - lora_unet__layers_0_attention_to_k.lora_down.weight
    - lora_unet__context_refiner_0_feed_forward_w1.alpha
    - lora_unet__noise_refiner_1_attention_to_v.lora_up.weight
    """
    return any(
        isinstance(k, str) and re.match(Z_IMAGE_KOHYA_TRANSFORMER_KEY_REGEX, k.split(".")[0]) for k in state_dict.keys()
    )


def is_state_dict_likely_z_image_lora(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely a Z-Image LoRA.

    Z-Image LoRAs can have keys for transformer and/or Qwen3 text encoder.
    They may use various prefixes depending on the training framework.
    """
    if is_state_dict_likely_z_image_kohya_lora(state_dict):
        return True

    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]

    # Check for Z-Image transformer keys (S3-DiT architecture)
    # Various training frameworks use different prefixes
    has_transformer_keys = any(
        k.startswith(
            (
                "transformer.",
                "base_model.model.transformer.",
                "diffusion_model.",
            )
        )
        for k in str_keys
    )

    # Check for Qwen3 text encoder keys
    has_qwen3_keys = any(k.startswith(("text_encoder.", "base_model.model.text_encoder.")) for k in str_keys)

    return has_transformer_keys or has_qwen3_keys


def lora_model_from_z_image_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None = None
) -> ModelPatchRaw:
    """Convert a Z-Image LoRA state dict to a ModelPatchRaw.

    Z-Image LoRAs can contain layers for:
    - Transformer (S3-DiT architecture)
    - Qwen3 text encoder

    Z-Image LoRAs may use various key prefixes depending on how they were trained:
    - "transformer." or "base_model.model.transformer." for diffusers PEFT format
    - "diffusion_model." for some training frameworks
    - "text_encoder." or "base_model.model.text_encoder." for Qwen3 encoder
    - "lora_unet__" for Kohya format (underscores instead of dots)

    Args:
        state_dict: The LoRA state dict
        alpha: The alpha value for LoRA scaling. If None, uses rank as alpha.

    Returns:
        A ModelPatchRaw containing the LoRA layers
    """
    # If Kohya format, convert keys first then process normally
    if is_state_dict_likely_z_image_kohya_lora(state_dict):
        state_dict = _convert_z_image_kohya_state_dict(state_dict)

    layers: dict[str, BaseLayerPatch] = {}

    # Group keys by layer
    grouped_state_dict = _group_by_layer(state_dict)

    for layer_key, layer_dict in grouped_state_dict.items():
        # Convert PEFT format keys to internal format
        values = _get_lora_layer_values(layer_dict, alpha)

        # Determine the appropriate prefix based on the layer type and clean up the key
        clean_key = layer_key

        # Handle various transformer prefixes
        transformer_prefixes = [
            "base_model.model.transformer.diffusion_model.",
            "base_model.model.transformer.",
            "transformer.diffusion_model.",
            "transformer.",
            "diffusion_model.",
        ]

        # Handle text encoder prefixes
        text_encoder_prefixes = [
            "base_model.model.text_encoder.",
            "text_encoder.",
        ]

        is_text_encoder = False

        # Check and strip text encoder prefixes first
        for prefix in text_encoder_prefixes:
            if layer_key.startswith(prefix):
                clean_key = layer_key[len(prefix) :]
                is_text_encoder = True
                break

        # If not text encoder, check transformer prefixes
        if not is_text_encoder:
            for prefix in transformer_prefixes:
                if layer_key.startswith(prefix):
                    clean_key = layer_key[len(prefix) :]
                    break

        # Apply the appropriate internal prefix
        if is_text_encoder:
            final_key = f"{Z_IMAGE_LORA_QWEN3_PREFIX}{clean_key}"
        else:
            final_key = f"{Z_IMAGE_LORA_TRANSFORMER_PREFIX}{clean_key}"

        layer = any_lora_layer_from_state_dict(values)
        layers[final_key] = layer

    return ModelPatchRaw(layers=layers)


def _convert_z_image_kohya_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Converts a Kohya-format Z-Image LoRA state dict to diffusion_model dot-notation.

    Example key conversions:
    - lora_unet__layers_0_attention_to_k.lora_down.weight -> diffusion_model.layers.0.attention.to_k.lora_down.weight
    - lora_unet__context_refiner_0_feed_forward_w1.alpha -> diffusion_model.context_refiner.0.feed_forward.w1.alpha
    - lora_unet__noise_refiner_1_attention_to_v.lora_up.weight -> diffusion_model.noise_refiner.1.attention.to_v.lora_up.weight
    """
    converted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key.startswith("lora_unet__"):
            converted[key] = value
            continue

        # Split into layer name and param suffix (e.g. "lora_down.weight", "alpha")
        layer_name, _, param_suffix = key.partition(".")

        # Strip lora_unet__ prefix
        remainder = layer_name[len("lora_unet__") :]

        # Convert Kohya underscore format to dot-notation using the known structure
        match = re.match(
            r"(layers|context_refiner|noise_refiner)_(\d+)_(attention|feed_forward)_(to_k|to_q|to_v|w1|w2|w3)$",
            remainder,
        )
        if match:
            block, idx, submodule, param = match.groups()
            new_layer = f"diffusion_model.{block}.{idx}.{submodule}.{param}"
        else:
            # Fallback: keep original key for unrecognized patterns
            converted[key] = value
            continue

        new_key = f"{new_layer}.{param_suffix}" if param_suffix else new_layer
        converted[new_key] = value

    return converted


def _get_lora_layer_values(layer_dict: dict[str, torch.Tensor], alpha: float | None) -> dict[str, torch.Tensor]:
    """Convert layer dict keys from PEFT format to internal format."""
    if "lora_A.weight" in layer_dict:
        # PEFT format: lora_A.weight, lora_B.weight
        values = {
            "lora_down.weight": layer_dict["lora_A.weight"],
            "lora_up.weight": layer_dict["lora_B.weight"],
        }
        if alpha is not None:
            values["alpha"] = torch.tensor(alpha)
        return values
    elif "lora_down.weight" in layer_dict:
        # Already in internal format
        return layer_dict
    else:
        # Unknown format, return as-is
        return layer_dict


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Groups the keys in the state dict by layer.

    Z-Image LoRAs have keys like:
    - diffusion_model.layers.17.attention.to_k.alpha
    - diffusion_model.layers.17.attention.to_k.dora_scale
    - diffusion_model.layers.17.attention.to_k.lora_down.weight
    - diffusion_model.layers.17.attention.to_k.lora_up.weight

    We need to group these by the full layer path (e.g., diffusion_model.layers.17.attention.to_k)
    and extract the suffix (alpha, dora_scale, lora_down.weight, lora_up.weight).
    """
    layer_dict: dict[str, dict[str, torch.Tensor]] = {}

    # Known suffixes that indicate the end of a layer name
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

        # Try to find a known suffix
        layer_name = None
        key_name = None
        for suffix in known_suffixes:
            if key.endswith(suffix):
                layer_name = key[: -len(suffix)]
                key_name = suffix[1:]  # Remove leading dot
                break

        if layer_name is None:
            # Fallback to original logic for unknown formats
            parts = key.rsplit(".", maxsplit=2)
            layer_name = parts[0]
            key_name = ".".join(parts[1:])

        if layer_name not in layer_dict:
            layer_dict[layer_name] = {}
        layer_dict[layer_name][key_name] = state_dict[key]

    return layer_dict
