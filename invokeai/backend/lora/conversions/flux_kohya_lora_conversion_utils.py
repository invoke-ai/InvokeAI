import re
from typing import Any, Dict, TypeVar

import torch

from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.lora.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.lora.lora_model_raw import LoRAModelRaw

# A regex pattern that matches all of the keys in the Kohya FLUX LoRA format.
# Example keys:
#   lora_unet_double_blocks_0_img_attn_proj.alpha
#   lora_unet_double_blocks_0_img_attn_proj.lora_down.weight
#   lora_unet_double_blocks_0_img_attn_proj.lora_up.weight
FLUX_KOHYA_KEY_REGEX = (
    r"lora_unet_(\w+_blocks)_(\d+)_(img_attn|img_mlp|img_mod|txt_attn|txt_mlp|txt_mod|linear1|linear2|modulation)_?(.*)"
)


def is_state_dict_likely_in_flux_kohya_format(state_dict: Dict[str, Any]) -> bool:
    """Checks if the provided state dict is likely in the Kohya FLUX LoRA format.

    This is intended to be a high-precision detector, but it is not guaranteed to have perfect precision. (A
    perfect-precision detector would require checking all keys against a whitelist and verifying tensor shapes.)
    """
    return all(re.match(FLUX_KOHYA_KEY_REGEX, k) for k in state_dict.keys())


def lora_model_from_flux_kohya_state_dict(state_dict: Dict[str, torch.Tensor]) -> LoRAModelRaw:
    # Group keys by layer.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        layer_name, param_name = key.split(".", 1)
        if layer_name not in grouped_state_dict:
            grouped_state_dict[layer_name] = {}
        grouped_state_dict[layer_name][param_name] = value

    # Convert the state dict to the InvokeAI format.
    grouped_state_dict = convert_flux_kohya_state_dict_to_invoke_format(grouped_state_dict)

    # Create LoRA layers.
    layers: dict[str, AnyLoRALayer] = {}
    for layer_key, layer_state_dict in grouped_state_dict.items():
        layers[layer_key] = any_lora_layer_from_state_dict(layer_state_dict)

    # Create and return the LoRAModelRaw.
    return LoRAModelRaw(layers=layers)


T = TypeVar("T")


def convert_flux_kohya_state_dict_to_invoke_format(state_dict: Dict[str, T]) -> Dict[str, T]:
    """Converts a state dict from the Kohya FLUX LoRA format to LoRA weight format used internally by InvokeAI.

    Example key conversions:
    "lora_unet_double_blocks_0_img_attn_proj" -> "double_blocks.0.img_attn.proj"
    "lora_unet_double_blocks_0_img_attn_proj" -> "double_blocks.0.img_attn.proj"
    "lora_unet_double_blocks_0_img_attn_proj" -> "double_blocks.0.img_attn.proj"
    "lora_unet_double_blocks_0_img_attn_qkv" -> "double_blocks.0.img_attn.qkv"
    "lora_unet_double_blocks_0_img_attn_qkv" -> "double_blocks.0.img.attn.qkv"
    "lora_unet_double_blocks_0_img_attn_qkv" -> "double_blocks.0.img.attn.qkv"
    """

    def replace_func(match: re.Match[str]) -> str:
        s = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
        if match.group(4):
            s += f".{match.group(4)}"
        return s

    converted_dict: dict[str, T] = {}
    for k, v in state_dict.items():
        match = re.match(FLUX_KOHYA_KEY_REGEX, k)
        if match:
            new_key = re.sub(FLUX_KOHYA_KEY_REGEX, replace_func, k)
            converted_dict[new_key] = v
        else:
            raise ValueError(f"Key '{k}' does not match the expected pattern for FLUX LoRA weights.")

    return converted_dict
