import re
from typing import Any, Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# A regex pattern that matches all of the transformer keys in the xlabs FLUX LoRA format.
# Example keys:
#   double_blocks.0.processor.qkv_lora1.down.weight
#   double_blocks.0.processor.qkv_lora1.up.weight
#   double_blocks.0.processor.proj_lora1.down.weight
#   double_blocks.0.processor.proj_lora1.up.weight
#   double_blocks.0.processor.qkv_lora2.down.weight
#   double_blocks.0.processor.proj_lora2.up.weight
FLUX_XLABS_KEY_REGEX = r"double_blocks\.(\d+)\.processor\.(qkv|proj)_lora([12])\.(down|up)\.weight"


def is_state_dict_likely_in_flux_xlabs_format(state_dict: dict[str | int, Any]) -> bool:
    """Checks if the provided state dict is likely in the xlabs FLUX LoRA format.

    The xlabs format is characterized by keys matching the pattern:
    double_blocks.{block_idx}.processor.{qkv|proj}_lora{1|2}.{down|up}.weight

    Where:
    - lora1 corresponds to the image attention stream (img_attn)
    - lora2 corresponds to the text attention stream (txt_attn)
    """
    if not state_dict:
        return False

    # Check that all keys match the xlabs pattern
    for key in state_dict.keys():
        if not isinstance(key, str):
            continue
        if not re.match(FLUX_XLABS_KEY_REGEX, key):
            return False

    # Ensure we have at least some valid keys
    return any(isinstance(k, str) and re.match(FLUX_XLABS_KEY_REGEX, k) for k in state_dict.keys())


def lora_model_from_flux_xlabs_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:
    """Converts an xlabs FLUX LoRA state dict to the InvokeAI ModelPatchRaw format.

    The xlabs format uses:
    - lora1 for image attention stream (img_attn)
    - lora2 for text attention stream (txt_attn)
    - qkv for query/key/value projection
    - proj for output projection

    Key mapping:
    - double_blocks.X.processor.qkv_lora1 -> double_blocks.X.img_attn.qkv
    - double_blocks.X.processor.proj_lora1 -> double_blocks.X.img_attn.proj
    - double_blocks.X.processor.qkv_lora2 -> double_blocks.X.txt_attn.qkv
    - double_blocks.X.processor.proj_lora2 -> double_blocks.X.txt_attn.proj
    """
    # Group keys by layer (without the .down.weight/.up.weight suffix)
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}

    for key, value in state_dict.items():
        match = re.match(FLUX_XLABS_KEY_REGEX, key)
        if not match:
            raise ValueError(f"Key '{key}' does not match the expected pattern for xlabs FLUX LoRA weights.")

        block_idx = match.group(1)
        component = match.group(2)  # qkv or proj
        lora_stream = match.group(3)  # 1 or 2
        direction = match.group(4)  # down or up

        # Map lora1 -> img_attn, lora2 -> txt_attn
        attn_type = "img_attn" if lora_stream == "1" else "txt_attn"

        # Create the InvokeAI-style layer key
        layer_key = f"double_blocks.{block_idx}.{attn_type}.{component}"

        if layer_key not in grouped_state_dict:
            grouped_state_dict[layer_key] = {}

        # Map down/up to lora_down/lora_up
        param_name = f"lora_{direction}.weight"
        grouped_state_dict[layer_key][param_name] = value

    # Create LoRA layers
    layers: dict[str, BaseLayerPatch] = {}
    for layer_key, layer_state_dict in grouped_state_dict.items():
        layers[FLUX_LORA_TRANSFORMER_PREFIX + layer_key] = any_lora_layer_from_state_dict(layer_state_dict)

    return ModelPatchRaw(layers=layers)
