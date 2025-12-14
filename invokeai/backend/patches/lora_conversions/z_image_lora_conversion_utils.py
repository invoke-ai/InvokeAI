"""Z-Image LoRA conversion utilities.

Z-Image uses S3-DiT transformer architecture with Qwen3 text encoder.
LoRAs for Z-Image typically follow the diffusers PEFT format.
"""

from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import (
    Z_IMAGE_LORA_QWEN3_PREFIX,
    Z_IMAGE_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


def is_state_dict_likely_z_image_lora(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely a Z-Image LoRA.

    Z-Image LoRAs can have keys for transformer and/or Qwen3 text encoder.
    They may use various prefixes depending on the training framework.
    """
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

    Args:
        state_dict: The LoRA state dict
        alpha: The alpha value for LoRA scaling. If None, uses rank as alpha.

    Returns:
        A ModelPatchRaw containing the LoRA layers
    """
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
    """Groups the keys in the state dict by layer."""
    layer_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key in state_dict:
        if not isinstance(key, str):
            continue
        # Split the 'lora_A.weight' or 'lora_B.weight' suffix from the layer name.
        parts = key.rsplit(".", maxsplit=2)
        layer_name = parts[0]
        key_name = ".".join(parts[1:])
        if layer_name not in layer_dict:
            layer_dict[layer_name] = {}
        layer_dict[layer_name][key_name] = state_dict[key]
    return layer_dict
