"""Qwen Image LoRA conversion utilities.

Qwen Image uses QwenImageTransformer2DModel architecture, with a Qwen2.5-VL text encoder.
Supports multiple LoRA formats:
- Diffusers/PEFT: transformer_blocks.0.attn.to_k.lora_down.weight
- With prefix: transformer.transformer_blocks.0.attn.to_k.lora_down.weight (also diffusion_model./base_model.model.)
- Kohya: lora_unet_transformer_blocks_0_attn_to_k.lora_down.weight (underscores instead of dots)
- LyCORIS: lycoris_transformer_blocks_0_attn_to_k.lokr_w1 (also ai-toolkit's lora_transformer_transformer_blocks_0_...)

Text-encoder layers are routed to the Qwen2.5-VL encoder under their own prefix:
- Diffusers/PEFT: text_encoder.model.language_model.layers.0.mlp.down_proj.lora_A.weight (also base_model.model.,
  and OneTrainer's ComfyUI export text_encoders.qwen25_7b.transformer.)
- Kohya: lora_te_model_language_model_layers_0_mlp_down_proj.lora_down.weight
"""

import re
from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.dora_magnitude_keys import (
    DORA_MAGNITUDE_SUFFIX_TO_VALUE_KEY,
    DORA_MAGNITUDE_VALUE_KEYS,
)
from invokeai.backend.patches.lora_conversions.kohya_key_utils import (
    INDEX_PLACEHOLDER,
    ParsingTree,
    insert_periods_into_kohya_key,
    kohya_module_path_is_leaf,
)
from invokeai.backend.patches.lora_conversions.qwen_image_lora_constants import (
    QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX,
    QWEN_IMAGE_LORA_QWEN_VL_PREFIX,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.util import InvokeAILogger

_KOHYA_TEXT_ENCODER_PREFIX = "lora_te_"
# `text_encoders.qwen25_7b.transformer.` is OneTrainer's ComfyUI export (`QwenModel.lora_text_encoders`), written beside
# `diffusion_model.` transformer keys.
_DIFFUSERS_TEXT_ENCODER_PREFIXES = (
    "base_model.model.text_encoder.",
    "text_encoder.",
    "text_encoders.qwen25_7b.transformer.",
)

_MLP: ParsingTree = {"gate_proj": {}, "up_proj": {}, "down_proj": {}}

# Linear modules of Qwen2_5_VLForConditionalGeneration (transformers >= 4.52 layout). Kohya keys flatten the dots to
# underscores, and some module names contain underscores themselves, so the tree is what tells them apart.
_KOHYA_TEXT_ENCODER_TREE: ParsingTree = {
    "model": {
        "language_model": {
            "layers": {
                INDEX_PLACEHOLDER: {
                    "self_attn": {"q_proj": {}, "k_proj": {}, "v_proj": {}, "o_proj": {}},
                    "mlp": _MLP,
                }
            }
        },
        "visual": {
            "blocks": {INDEX_PLACEHOLDER: {"attn": {"qkv": {}, "proj": {}}, "mlp": _MLP}},
            "merger": {"mlp": {"0": {}, "2": {}}},
        },
    },
    "lm_head": {},
}

# Prefixes of the flattened (Kohya-style) spellings: Kohya's, LyCORIS' own, and ai-toolkit's.
_KOHYA_PREFIXES = (
    "lora_unet_transformer_blocks_",
    "lycoris_transformer_blocks_",
    "lora_transformer_transformer_blocks_",
)

# Regex for Kohya-format Qwen Image LoRA keys.
# Example: lora_unet_transformer_blocks_0_attn_to_k
# Groups: (block_idx, sub_module_with_underscores)
_KOHYA_KEY_REGEX = re.compile(r"(?:lora_unet_|lycoris_|lora_transformer_)transformer_blocks_(\d+)_(.*)")

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
    return any(k.startswith(_KOHYA_PREFIXES) for k in str_keys)


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


def _convert_kohya_text_encoder_key(kohya_layer: str) -> str | None:
    """Convert a `lora_te_*` layer name to a dotted Qwen2.5-VL module path, or None when it names no known Linear.

    Example: lora_te_model_language_model_layers_0_mlp_down_proj -> model.language_model.layers.0.mlp.down_proj
    """
    try:
        module_path = insert_periods_into_kohya_key(
            kohya_layer[len(_KOHYA_TEXT_ENCODER_PREFIX) :], _KOHYA_TEXT_ENCODER_TREE
        )
    except ValueError:
        return None
    # A prefix of a real path (e.g. ...layers.0.mlp) parses cleanly without naming a module.
    return module_path if kohya_module_path_is_leaf(module_path, _KOHYA_TEXT_ENCODER_TREE) else None


def _warn_unmapped_layers(count: int) -> None:
    if count:
        InvokeAILogger.get_logger(__name__).warning(
            f"Skipped {count} Qwen Image LoRA layer(s) whose key does not name a known transformer or "
            "text-encoder module."
        )


def lora_model_from_qwen_image_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None = None
) -> ModelPatchRaw:
    """Convert a Qwen Image LoRA state dict to a ModelPatchRaw.

    Handles three key formats (text-encoder layers are routed to the Qwen2.5-VL prefix in each):
    - Diffusers/PEFT: transformer_blocks.0.attn.to_k.lora_down.weight
    - With prefix: transformer.transformer_blocks.0.attn.to_k.lora_down.weight
      (also diffusion_model., base_model.model.)
    - Kohya: lora_unet_transformer_blocks_0_attn_to_k.lora_down.weight
      (also LyCORIS' lycoris_transformer_blocks_ and ai-toolkit's lora_transformer_transformer_blocks_)
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
        param_name = DORA_MAGNITUDE_SUFFIX_TO_VALUE_KEY.get(f".{param_name}", param_name)
        if layer_name not in grouped:
            grouped[layer_name] = {}
        grouped[layer_name][param_name] = value

    unmapped = 0
    for kohya_layer, layer_dict in grouped.items():
        if kohya_layer.startswith(_KOHYA_TEXT_ENCODER_PREFIX):
            model_path = _convert_kohya_text_encoder_key(kohya_layer)
            prefix = QWEN_IMAGE_LORA_QWEN_VL_PREFIX
        else:
            model_path = _convert_kohya_key(kohya_layer)
            prefix = QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX
        if model_path is None:
            unmapped += 1
            continue

        layers[f"{prefix}{model_path}"] = any_lora_layer_from_state_dict(_fold_lycoris_scalar(layer_dict))

    _warn_unmapped_layers(unmapped)
    return ModelPatchRaw(layers=layers)


def _convert_diffusers_format(state_dict: Dict[str, torch.Tensor], alpha: float | None) -> ModelPatchRaw:
    """Convert Diffusers/PEFT format state dict."""
    layers: dict[str, BaseLayerPatch] = {}

    # The diffusers (`transformer.`), ComfyUI (`diffusion_model.`) and raw PEFT (`base_model.model.`) key prefixes
    strip_prefixes = ["transformer.", "diffusion_model.", "base_model.model."]

    grouped = _group_by_layer(state_dict)

    unmapped = 0
    for layer_key, layer_dict in grouped.items():
        values = _normalize_lora_keys(_fold_lycoris_scalar(layer_dict), alpha)
        # Text-encoder layers first: `base_model.model.` is also in the transformer strip list.
        text_encoder_prefix = next((p for p in _DIFFUSERS_TEXT_ENCODER_PREFIXES if layer_key.startswith(p)), None)
        if text_encoder_prefix is not None:
            module_path = layer_key[len(text_encoder_prefix) :]
            # Only Linears of the current transformers layout can take a patch: the patcher would otherwise reach
            # layers it cannot patch (`patch_embed` is a Conv3d) or crash on them.
            if not kohya_module_path_is_leaf(module_path, _KOHYA_TEXT_ENCODER_TREE):
                unmapped += 1
                continue
            layers[f"{QWEN_IMAGE_LORA_QWEN_VL_PREFIX}{module_path}"] = any_lora_layer_from_state_dict(values)
            continue
        layer = any_lora_layer_from_state_dict(values)
        clean_key = layer_key
        for prefix in strip_prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
                break
        final_key = f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}{clean_key}"
        layers[final_key] = layer

    _warn_unmapped_layers(unmapped)
    return ModelPatchRaw(layers=layers)


def _fold_lycoris_scalar(layer_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold LyCORIS' learned `scalar` into the layer's weights.

    LyCORIS multiplies a LoKr, LoHa or LoRA/LoCon layer's whole update by `scalar` (trained from zero, so it is
    usually far from 1), and the patch layers ignore it. The update is linear in the LoKr's `w1`, the LoHa's `w1_a`
    and the LoRA's up matrix, so scaling that factor gives the same update. Runs before key normalization, which
    drops keys it does not know.
    """
    if "scalar" not in layer_dict:
        return layer_dict
    folded = dict(layer_dict)
    scalar = folded.pop("scalar")
    for key in ("lokr_w1", "lokr_w1_a", "hada_w1_a", "lora_up.weight", "lora_B.weight"):
        if key in folded:
            folded[key] = folded[key] * scalar.to(folded[key].dtype)
            break
    return folded


def _normalize_lora_keys(layer_dict: dict[str, torch.Tensor], alpha: float | None) -> dict[str, torch.Tensor]:
    """Normalize LoRA key names to internal format."""
    if "lora_A.weight" in layer_dict:
        values: dict[str, torch.Tensor] = {
            "lora_down.weight": layer_dict["lora_A.weight"],
            "lora_up.weight": layer_dict["lora_B.weight"],
        }
        if "alpha" in layer_dict:
            values["alpha"] = layer_dict["alpha"]
        elif alpha is not None:
            values["alpha"] = torch.tensor(alpha)
        # Dropping a DoRA magnitude would apply the adapter as a plain LoRA.
        for magnitude_key in DORA_MAGNITUDE_VALUE_KEYS:
            if magnitude_key in layer_dict:
                values[magnitude_key] = layer_dict[magnitude_key]
        return values
    elif "lora_down.weight" in layer_dict:
        return layer_dict
    else:
        return layer_dict


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Group state dict keys by layer path."""
    layer_dict: dict[str, dict[str, torch.Tensor]] = {}

    known_suffixes = {
        ".lora_A.weight": "lora_A.weight",
        ".lora_B.weight": "lora_B.weight",
        ".lora_down.weight": "lora_down.weight",
        ".lora_up.weight": "lora_up.weight",
        ".alpha": "alpha",
        ".scalar": "scalar",
        **DORA_MAGNITUDE_SUFFIX_TO_VALUE_KEY,
        # LoKr and LoHa parameters
        **{
            suffix: suffix[1:]
            for suffix in (
                ".lokr_w1",
                ".lokr_w1_a",
                ".lokr_w1_b",
                ".lokr_w2",
                ".lokr_w2_a",
                ".lokr_w2_b",
                ".lokr_t2",
                ".hada_w1_a",
                ".hada_w1_b",
                ".hada_w2_a",
                ".hada_w2_b",
                ".hada_t1",
                ".hada_t2",
            )
        },
    }

    for key in state_dict:
        if not isinstance(key, str):
            continue

        layer_name = None
        key_name = None
        for suffix, value_key in known_suffixes.items():
            if key.endswith(suffix):
                layer_name = key[: -len(suffix)]
                key_name = value_key
                break

        if layer_name is None:
            parts = key.rsplit(".", maxsplit=2)
            layer_name = parts[0]
            key_name = ".".join(parts[1:])

        if layer_name not in layer_dict:
            layer_dict[layer_name] = {}
        layer_dict[layer_name][key_name] = state_dict[key]

    return layer_dict
