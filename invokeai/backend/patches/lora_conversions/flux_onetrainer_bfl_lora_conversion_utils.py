"""Utilities for detecting and converting FLUX LoRAs in OneTrainer BFL format.

This format is produced by newer versions of OneTrainer and uses BFL internal key names
(double_blocks, single_blocks, img_attn, etc.) with a 'transformer.' prefix and
InvokeAI-native LoRA suffixes (lora_down.weight, lora_up.weight, alpha).

Unlike the standard BFL PEFT format (which uses 'diffusion_model.' prefix and lora_A/lora_B),
this format also has split QKV projections:
  - double_blocks.{i}.img_attn.qkv.{0,1,2} (Q, K, V separate)
  - double_blocks.{i}.txt_attn.qkv.{0,1,2} (Q, K, V separate)
  - single_blocks.{i}.linear1.{0,1,2,3} (Q, K, V, MLP separate)

Example keys:
    transformer.double_blocks.0.img_attn.qkv.0.lora_down.weight
    transformer.double_blocks.0.img_attn.qkv.0.lora_up.weight
    transformer.double_blocks.0.img_attn.qkv.0.alpha
    transformer.single_blocks.0.linear1.3.lora_down.weight
    transformer.double_blocks.0.img_mlp.0.lora_down.weight
"""

import re
from typing import Any, Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch, Range
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

_TRANSFORMER_PREFIX = "transformer."

# Valid LoRA weight suffixes in this format.
_LORA_SUFFIXES = ("lora_down.weight", "lora_up.weight", "alpha")

# Regex to detect split QKV keys in double blocks: e.g. "double_blocks.0.img_attn.qkv.1"
_SPLIT_QKV_RE = re.compile(r"^(double_blocks\.\d+\.(img_attn|txt_attn)\.qkv)\.\d+$")

# Regex to detect split linear1 keys in single blocks: e.g. "single_blocks.0.linear1.2"
_SPLIT_LINEAR1_RE = re.compile(r"^(single_blocks\.\d+\.linear1)\.\d+$")


def is_state_dict_likely_in_flux_onetrainer_bfl_format(
    state_dict: dict[str | int, Any],
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Checks if the provided state dict is likely in the OneTrainer BFL FLUX LoRA format.

    This format uses BFL internal key names with 'transformer.' prefix and split QKV projections.
    """
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if not str_keys:
        return False

    # All keys must start with 'transformer.'
    if not all(k.startswith(_TRANSFORMER_PREFIX) for k in str_keys):
        return False

    # All keys must end with recognized LoRA suffixes.
    if not all(k.endswith(_LORA_SUFFIXES) for k in str_keys):
        return False

    # Must have BFL block structure (double_blocks or single_blocks) under transformer prefix.
    has_bfl_blocks = any(
        k.startswith("transformer.double_blocks.") or k.startswith("transformer.single_blocks.") for k in str_keys
    )
    if not has_bfl_blocks:
        return False

    # Must have split QKV pattern (qkv.0, qkv.1, qkv.2) to distinguish from other formats
    # that might use transformer. prefix in the future.
    has_split_qkv = any(".qkv.0." in k or ".qkv.1." in k or ".qkv.2." in k or ".linear1.0." in k for k in str_keys)
    if not has_split_qkv:
        return False

    return True


def _split_key(key: str) -> tuple[str, str]:
    """Split a key into (layer_name, weight_suffix).

    Handles:
    - 2-component suffixes ending with '.weight': e.g., 'lora_down.weight' → split at 2nd-to-last dot
    - 1-component suffixes: e.g., 'alpha' → split at last dot
    """
    if key.endswith(".weight"):
        parts = key.rsplit(".", maxsplit=2)
        return parts[0], f"{parts[1]}.{parts[2]}"
    else:
        parts = key.rsplit(".", maxsplit=1)
        return parts[0], parts[1]


def lora_model_from_flux_onetrainer_bfl_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:
    """Convert a OneTrainer BFL format FLUX LoRA state dict to a ModelPatchRaw.

    Strips the 'transformer.' prefix, groups by layer, and merges split QKV/linear1
    layers into MergedLayerPatch instances.
    """
    # Step 1: Strip prefix and group by layer name.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue

        # Strip 'transformer.' prefix.
        key = key[len(_TRANSFORMER_PREFIX) :]

        layer_name, suffix = _split_key(key)

        if layer_name not in grouped_state_dict:
            grouped_state_dict[layer_name] = {}
        grouped_state_dict[layer_name][suffix] = value

    # Step 2: Build LoRA layers, merging split QKV and linear1.
    layers: dict[str, BaseLayerPatch] = {}

    # Identify which layers need merging.
    merge_groups: dict[str, list[str]] = {}
    standalone_keys: list[str] = []

    for layer_key in grouped_state_dict:
        qkv_match = _SPLIT_QKV_RE.match(layer_key)
        linear1_match = _SPLIT_LINEAR1_RE.match(layer_key)

        if qkv_match:
            parent = qkv_match.group(1)
            if parent not in merge_groups:
                merge_groups[parent] = []
            merge_groups[parent].append(layer_key)
        elif linear1_match:
            parent = linear1_match.group(1)
            if parent not in merge_groups:
                merge_groups[parent] = []
            merge_groups[parent].append(layer_key)
        else:
            standalone_keys.append(layer_key)

    # Process standalone layers.
    for layer_key in standalone_keys:
        layer_sd = grouped_state_dict[layer_key]
        layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}{layer_key}"] = any_lora_layer_from_state_dict(layer_sd)

    # Process merged layers.
    for parent_key, sub_keys in merge_groups.items():
        # Sort by the numeric index at the end (e.g., qkv.0, qkv.1, qkv.2).
        sub_keys.sort(key=lambda k: int(k.rsplit(".", maxsplit=1)[1]))

        sub_layers: list[BaseLayerPatch] = []
        sub_ranges: list[Range] = []
        dim_0_offset = 0

        for sub_key in sub_keys:
            layer_sd = grouped_state_dict[sub_key]
            sub_layer = any_lora_layer_from_state_dict(layer_sd)

            # Determine the output dimension from the up weight shape.
            up_weight = layer_sd["lora_up.weight"]
            out_dim = up_weight.shape[0]

            sub_layers.append(sub_layer)
            sub_ranges.append(Range(dim_0_offset, dim_0_offset + out_dim))
            dim_0_offset += out_dim

        layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}{parent_key}"] = MergedLayerPatch(sub_layers, sub_ranges)

    return ModelPatchRaw(layers=layers)
