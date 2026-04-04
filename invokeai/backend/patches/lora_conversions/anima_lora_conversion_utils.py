"""Anima LoRA conversion utilities.

Anima uses a Cosmos Predict2 DiT transformer architecture.
LoRAs for Anima typically follow the Kohya-style format with underscore-separated keys
(e.g., lora_unet_blocks_0_cross_attn_k_proj) that map to model parameter paths
(e.g., blocks.0.cross_attn.k_proj).

Some Anima LoRAs also target the Qwen3 text encoder with lora_te_ prefix keys
(e.g., lora_te_layers_0_self_attn_q_proj -> layers.0.self_attn.q_proj).
"""

import re
from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.anima_lora_constants import (
    ANIMA_LORA_QWEN3_PREFIX,
    ANIMA_LORA_TRANSFORMER_PREFIX,
    has_cosmos_dit_kohya_keys,
    has_cosmos_dit_peft_keys,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)


def is_state_dict_likely_anima_lora(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely an Anima LoRA.

    Anima LoRAs use Kohya-style naming with lora_unet_ prefix and underscore-separated
    model key paths targeting Cosmos DiT blocks.  Detection requires Cosmos DiT-specific
    subcomponent names (cross_attn, self_attn, mlp, adaln_modulation) to avoid
    false-positives on other architectures that also use ``blocks`` in their paths.
    """
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]

    if has_cosmos_dit_kohya_keys(str_keys):
        return True

    return has_cosmos_dit_peft_keys(str_keys)


# Mapping from Kohya underscore-style substrings to model parameter names.
# Order matters: longer/more specific patterns should come first to avoid partial matches.
_KOHYA_UNET_KEY_REPLACEMENTS = [
    ("adaln_modulation_cross_attn_", "adaln_modulation_cross_attn."),
    ("adaln_modulation_self_attn_", "adaln_modulation_self_attn."),
    ("adaln_modulation_mlp_", "adaln_modulation_mlp."),
    ("cross_attn_k_proj", "cross_attn.k_proj"),
    ("cross_attn_q_proj", "cross_attn.q_proj"),
    ("cross_attn_v_proj", "cross_attn.v_proj"),
    ("cross_attn_output_proj", "cross_attn.output_proj"),
    ("cross_attn_o_proj", "cross_attn.o_proj"),
    ("self_attn_k_proj", "self_attn.k_proj"),
    ("self_attn_q_proj", "self_attn.q_proj"),
    ("self_attn_v_proj", "self_attn.v_proj"),
    ("self_attn_output_proj", "self_attn.output_proj"),
    ("self_attn_o_proj", "self_attn.o_proj"),
    ("mlp_layer1", "mlp.layer1"),
    ("mlp_layer2", "mlp.layer2"),
]

# Mapping for Qwen3 text encoder Kohya keys.
_KOHYA_TE_KEY_REPLACEMENTS = [
    ("self_attn_k_proj", "self_attn.k_proj"),
    ("self_attn_q_proj", "self_attn.q_proj"),
    ("self_attn_v_proj", "self_attn.v_proj"),
    ("self_attn_o_proj", "self_attn.o_proj"),
    ("mlp_down_proj", "mlp.down_proj"),
    ("mlp_gate_proj", "mlp.gate_proj"),
    ("mlp_up_proj", "mlp.up_proj"),
]


def _convert_kohya_unet_key(kohya_layer_name: str) -> str:
    """Convert a Kohya-style LoRA layer name to a model parameter path.

    Example: lora_unet_blocks_0_cross_attn_k_proj -> blocks.0.cross_attn.k_proj
    Example: lora_unet_llm_adapter_blocks_0_cross_attn_k_proj -> llm_adapter.blocks.0.cross_attn.k_proj
    """
    key = kohya_layer_name
    if key.startswith("lora_unet_"):
        key = key[len("lora_unet_") :]

    # Handle llm_adapter prefix: strip it, run the standard block conversion, then re-add with dot
    llm_adapter_prefix = ""
    if key.startswith("llm_adapter_"):
        key = key[len("llm_adapter_") :]
        llm_adapter_prefix = "llm_adapter."

    # Convert blocks_N_ to blocks.N.
    key = re.sub(r"^blocks_(\d+)_", r"blocks.\1.", key)

    # Apply known replacements for subcomponent names
    for old, new in _KOHYA_UNET_KEY_REPLACEMENTS:
        if old in key:
            key = key.replace(old, new, 1)
            break

    return llm_adapter_prefix + key


def _convert_kohya_te_key(kohya_layer_name: str) -> str:
    """Convert a Kohya-style text encoder LoRA layer name to a model parameter path.

    The Qwen3 text encoder is loaded as Qwen3ForCausalLM which wraps the base model
    under a `model.` prefix, so the final path must include it.

    Example: lora_te_layers_0_self_attn_q_proj -> model.layers.0.self_attn.q_proj
    """
    key = kohya_layer_name
    if key.startswith("lora_te_"):
        key = key[len("lora_te_") :]

    # Convert layers_N_ to layers.N.
    key = re.sub(r"^layers_(\d+)_", r"layers.\1.", key)

    # Apply known replacements
    for old, new in _KOHYA_TE_KEY_REPLACEMENTS:
        if old in key:
            key = key.replace(old, new, 1)
            break

    # Qwen3ForCausalLM wraps the base Qwen3Model under `model.`
    key = f"model.{key}"

    return key


def _make_layer_patch(layer_dict: dict[str, torch.Tensor]) -> BaseLayerPatch:
    """Create a layer patch from a layer dict, handling DoRA+LoKR edge case.

    Some Anima LoRAs combine DoRA (dora_scale) with LoKR (lokr_w1/lokr_w2) weights.
    The shared any_lora_layer_from_state_dict checks dora_scale first and expects
    lora_up/lora_down keys, which don't exist in LoKR layers. We strip dora_scale
    from LoKR layers so they fall through to the LoKR handler instead.
    """
    has_lokr = "lokr_w1" in layer_dict or "lokr_w1_a" in layer_dict
    has_dora = "dora_scale" in layer_dict
    if has_lokr and has_dora:
        layer_dict = {k: v for k, v in layer_dict.items() if k != "dora_scale"}
        logger.warning("Stripped dora_scale from LoKR layer (DoRA+LoKR combination not supported, using LoKR only)")
    return any_lora_layer_from_state_dict(layer_dict)


# Known suffixes for Kohya format
_KOHYA_KNOWN_SUFFIXES = [
    ".lora_A.weight",
    ".lora_B.weight",
    ".lora_down.weight",
    ".lora_up.weight",
    ".dora_scale",
    ".alpha",
]

# Additional suffixes for PEFT/LoKR format
_PEFT_EXTRA_SUFFIXES = [
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
]


def _group_keys_by_layer(
    state_dict: Dict[str, torch.Tensor],
    extra_suffixes: list[str] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Group state dict keys by layer name based on known suffixes.

    Args:
        state_dict: The LoRA state dict to group.
        extra_suffixes: Additional suffixes to recognize beyond the base Kohya set.

    Returns:
        Dict mapping layer names to their component tensors.
    """
    layer_dict: dict[str, dict[str, torch.Tensor]] = {}

    known_suffixes = list(_KOHYA_KNOWN_SUFFIXES)
    if extra_suffixes:
        known_suffixes.extend(extra_suffixes)

    for key in state_dict:
        if not isinstance(key, str):
            continue

        layer_name = None
        key_name = None
        for suffix in known_suffixes:
            if key.endswith(suffix):
                layer_name = key[: -len(suffix)]
                key_name = suffix[1:]  # Remove leading dot
                break

        if layer_name is None:
            parts = key.rsplit(".", maxsplit=2)
            layer_name = parts[0]
            key_name = ".".join(parts[1:])

        if layer_name not in layer_dict:
            layer_dict[layer_name] = {}
        layer_dict[layer_name][key_name] = state_dict[key]

    return layer_dict


def _get_lora_layer_values(layer_dict: dict[str, torch.Tensor], alpha: float | None) -> dict[str, torch.Tensor]:
    """Convert layer dict keys from PEFT format to internal format."""
    if "lora_A.weight" in layer_dict:
        values = {
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


def lora_model_from_anima_state_dict(state_dict: Dict[str, torch.Tensor], alpha: float | None = None) -> ModelPatchRaw:
    """Convert an Anima LoRA state dict to a ModelPatchRaw.

    Supports both Kohya-style keys (lora_unet_blocks_0_...) and diffusers PEFT format.
    Also supports text encoder LoRA keys (lora_te_layers_0_...) targeting the Qwen3 encoder.

    Args:
        state_dict: The LoRA state dict
        alpha: The alpha value for LoRA scaling. If None, uses rank as alpha.

    Returns:
        A ModelPatchRaw containing the LoRA layers
    """
    layers: dict[str, BaseLayerPatch] = {}

    # Detect format
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    is_kohya = any(k.startswith(("lora_unet_", "lora_te_")) for k in str_keys)

    if is_kohya:
        # Kohya format: group by layer name (everything before .lora_down/.lora_up/.alpha)
        grouped = _group_keys_by_layer(state_dict)
        for kohya_layer_name, layer_dict in grouped.items():
            if kohya_layer_name.startswith("lora_te_"):
                model_key = _convert_kohya_te_key(kohya_layer_name)
                final_key = f"{ANIMA_LORA_QWEN3_PREFIX}{model_key}"
            else:
                model_key = _convert_kohya_unet_key(kohya_layer_name)
                final_key = f"{ANIMA_LORA_TRANSFORMER_PREFIX}{model_key}"
            layer = _make_layer_patch(layer_dict)
            layers[final_key] = layer
    else:
        # Diffusers PEFT format
        grouped = _group_keys_by_layer(state_dict, extra_suffixes=_PEFT_EXTRA_SUFFIXES)
        for layer_key, layer_dict in grouped.items():
            values = _get_lora_layer_values(layer_dict, alpha)
            clean_key = layer_key

            # Check for text encoder prefixes
            text_encoder_prefixes = [
                "base_model.model.text_encoder.",
                "text_encoder.",
            ]

            is_text_encoder = False
            for prefix in text_encoder_prefixes:
                if layer_key.startswith(prefix):
                    clean_key = layer_key[len(prefix) :]
                    is_text_encoder = True
                    break

            # If not text encoder, check transformer prefixes
            if not is_text_encoder:
                for prefix in [
                    "base_model.model.transformer.",
                    "transformer.",
                    "diffusion_model.",
                ]:
                    if layer_key.startswith(prefix):
                        clean_key = layer_key[len(prefix) :]
                        break

            if is_text_encoder:
                final_key = f"{ANIMA_LORA_QWEN3_PREFIX}{clean_key}"
            else:
                final_key = f"{ANIMA_LORA_TRANSFORMER_PREFIX}{clean_key}"

            layer = _make_layer_patch(values)
            layers[final_key] = layer

    return ModelPatchRaw(layers=layers)
