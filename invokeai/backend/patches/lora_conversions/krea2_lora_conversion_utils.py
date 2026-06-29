"""Krea-2 LoRA conversion utilities.

Krea-2 uses a single-stream MMDiT (``Krea2Transformer2DModel``) with a Qwen3-VL text encoder.
Published LoRAs (e.g. krea/Krea-2-LoRA-*) are diffusers PEFT format: keys like
``transformer.<module>.lora_A.weight`` / ``lora_B.weight``. The distinctive Krea-2 module is the
``text_fusion`` stage, which we use to disambiguate from Qwen-Image / Z-Image LoRAs (which otherwise
share the ``transformer.transformer_blocks.`` prefix).
"""

from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import (
    KREA2_LORA_QWEN3VL_PREFIX,
    KREA2_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# Module-name fragments unique to the Krea-2 transformer (text-fusion stage + timestep modulation proj).
KREA2_TRANSFORMER_SIGNATURE_KEYS = ("text_fusion", "time_mod_proj")


def is_state_dict_likely_krea2_lora(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely a Krea-2 LoRA.

    Requires the distinctive Krea-2 ``text_fusion`` / ``time_mod_proj`` modules so it does not
    false-match Qwen-Image or Z-Image LoRAs that also carry ``transformer.transformer_blocks.`` keys.
    """
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    has_krea2_module = any(any(sig in k for sig in KREA2_TRANSFORMER_SIGNATURE_KEYS) for k in str_keys)
    has_lora_suffix = any(
        k.endswith((".lora_A.weight", ".lora_B.weight", ".lora_down.weight", ".lora_up.weight")) for k in str_keys
    )
    return has_krea2_module and has_lora_suffix


def lora_model_from_krea2_state_dict(state_dict: Dict[str, torch.Tensor], alpha: float | None = None) -> ModelPatchRaw:
    """Convert a Krea-2 LoRA state dict (diffusers PEFT) to a ModelPatchRaw.

    Handles transformer layers and (if present) Qwen3-VL text encoder layers. ``alpha=None`` is treated
    as ``alpha=rank`` internally (the common diffusers default).
    """
    layers: dict[str, BaseLayerPatch] = {}
    grouped_state_dict = _group_by_layer(state_dict)

    transformer_prefixes = (
        "base_model.model.transformer.",
        "transformer.",
        "diffusion_model.",
    )
    text_encoder_prefixes = (
        "base_model.model.text_encoder.",
        "text_encoder.",
    )

    for layer_key, layer_dict in grouped_state_dict.items():
        values = _get_lora_layer_values(layer_dict, alpha)

        is_text_encoder = False
        clean_key = layer_key
        for prefix in text_encoder_prefixes:
            if layer_key.startswith(prefix):
                clean_key = layer_key[len(prefix) :]
                is_text_encoder = True
                break
        if not is_text_encoder:
            for prefix in transformer_prefixes:
                if layer_key.startswith(prefix):
                    clean_key = layer_key[len(prefix) :]
                    break

        if is_text_encoder:
            final_key = f"{KREA2_LORA_QWEN3VL_PREFIX}{clean_key}"
        else:
            final_key = f"{KREA2_LORA_TRANSFORMER_PREFIX}{clean_key}"

        layers[final_key] = any_lora_layer_from_state_dict(values)

    return ModelPatchRaw(layers=layers)


def _get_lora_layer_values(layer_dict: dict[str, torch.Tensor], alpha: float | None) -> dict[str, torch.Tensor]:
    """Convert PEFT (lora_A/lora_B) layer values to internal (lora_down/lora_up) format."""
    if "lora_A.weight" in layer_dict:
        values = {
            "lora_down.weight": layer_dict["lora_A.weight"],
            "lora_up.weight": layer_dict["lora_B.weight"],
        }
        if alpha is not None:
            values["alpha"] = torch.tensor(alpha)
        return values
    return layer_dict


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Groups state dict keys by layer path, splitting off the LoRA weight suffix."""
    known_suffixes = [
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_down.weight",
        ".lora_up.weight",
        ".dora_scale",
        ".alpha",
    ]
    layer_dict: dict[str, dict[str, torch.Tensor]] = {}
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
        layer_dict.setdefault(layer_name, {})[key_name] = state_dict[key]
    return layer_dict
