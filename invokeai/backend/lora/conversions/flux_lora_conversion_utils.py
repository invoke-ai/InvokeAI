from typing import Dict

import torch


def convert_flux_kohya_state_dict_to_invoke_format(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Converts a state dict from the Kohya model to the InvokeAI model format.


    Example conversions:
    ```
    "lora_unet_double_blocks_0_img_attn_proj.alpha": "double_blocks.0.img_attn.proj.alpha
    "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": "double_blocks.0.img_attn.proj.lora_down.weight"
    "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": "double_blocks.0.img_attn.proj.lora_up.weight"
    "lora_unet_double_blocks_0_img_attn_qkv.alpha": "double_blocks.0.img_attn.qkv.alpha"
    "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": "double_blocks.0.img.attn.qkv.lora_down.weight"
    "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": "double_blocks.0.img.attn.qkv.lora_up.weight"
    ```

    """
    new_sd: dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        new_key = ""

        # Remove the lora_unet_ prefix.
        k = k.replace("lora_unet_", "")

        # Split at the underscores.
        parts = k.split("_")

        # Handle the block key (either "double_blocks" or "single_blocks")
        new_key += "_".join(parts[:2])

        # Handle the block index.
        new_key += "." + parts[2]

        remaining_key = "_".join(parts[3:])

        # Handle next module.
        for module_name in [
            "img_attn",
            "img_mlp",
            "img_mod",
            "txt_attn",
            "txt_mlp",
            "txt_mod",
            "linear1",
            "linear2",
            "modulation",
        ]:
            if remaining_key.startswith(module_name):
                new_key += "." + module_name
                remaining_key = remaining_key.replace(module_name, "")
                break

        # Handle the rest of the key.
        while len(remaining_key) > 0:
            next_chunk, remaining_key = remaining_key.split("_", 1)
            if next_chunk.startswith("."):
                new_key += next_chunk
            else:
                new_key += "." + next_chunk

        new_sd[new_key] = v

    return new_sd
