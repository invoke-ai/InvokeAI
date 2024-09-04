import re
from typing import Dict

import torch


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
    pattern = r"lora_unet_(\w+_blocks)_(\d+)_(img_attn|img_mlp|img_mod|txt_attn|txt_mlp|txt_mod|linear1|linear2|modulation)_?(.*)"
    replacement = r"\1.\2.\3.\4"

    converted_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        match = re.match(pattern, k)
        if match:
            new_key = re.sub(pattern, replacement, k)
            converted_dict[new_key] = v
        else:
            raise ValueError(f"Key '{k}' does not match the expected pattern for FLUX LoRA weights.")

    return converted_dict
