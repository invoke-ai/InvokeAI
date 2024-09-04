import re
from typing import Any, Dict

import torch

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
    for k in state_dict.keys():
        if not re.match(FLUX_KOHYA_KEY_REGEX, k):
            return False
    return True


def convert_flux_kohya_state_dict_to_invoke_format(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Converts a state dict from the Kohya FLUX LoRA format to LoRA weight format used internally by InvokeAI.

    Example key conversions:
    "lora_unet_double_blocks_0_img_attn_proj.alpha" -> "double_blocks.0.img_attn.proj.alpha
    "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight" -> "double_blocks.0.img_attn.proj.lora_down.weight"
    "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight" -> "double_blocks.0.img_attn.proj.lora_up.weight"
    "lora_unet_double_blocks_0_img_attn_qkv.alpha" -> "double_blocks.0.img_attn.qkv.alpha"
    "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight" -> "double_blocks.0.img.attn.qkv.lora_down.weight"
    "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight" -> "double_blocks.0.img.attn.qkv.lora_up.weight"

    """
    replacement = r"\1.\2.\3.\4"

    converted_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        match = re.match(FLUX_KOHYA_KEY_REGEX, k)
        if match:
            new_key = re.sub(FLUX_KOHYA_KEY_REGEX, replacement, k)
            converted_dict[new_key] = v
        else:
            raise ValueError(f"Key '{k}' does not match the expected pattern for FLUX LoRA weights.")

    return converted_dict
