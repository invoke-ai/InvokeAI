import re
import torch

from typing import Any, Dict
from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.lora.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.lora.lora_model_raw import LoRAModelRaw


# A regex pattern that matches all of the keys in the Flux Dev/Canny LoRA format.
# Example keys:
#   guidance_in.in_layer.lora_B.bias
#   single_blocks.0.linear1.lora_A.weight
#   double_blocks.0.img_attn.norm.key_norm.scale
FLUX_STRUCTURAL_TRANSFORMER_KEY_REGEX = r"(final_layer|vector_in|txt_in|time_in|img_in|guidance_in|\w+_blocks)(\.(\d+))?\.(lora_(A|B)|(in|out)_layer|adaLN_modulation|img_attn|img_mlp|img_mod|txt_attn|txt_mlp|txt_mod|linear|linear1|linear2|modulation|norm)\.?(.*)"

def is_state_dict_likely_flux_control(state_dict: Dict[str, Any]) -> bool:
    """Checks if the provided state dict is likely in the FLUX Control LoRA format.

    This is intended to be a high-precision detector, but it is not guaranteed to have perfect precision. (A
    perfect-precision detector would require checking all keys against a whitelist and verifying tensor shapes.)
    """
    return all(
        re.match(FLUX_STRUCTURAL_TRANSFORMER_KEY_REGEX, k) or re.match(FLUX_STRUCTURAL_TRANSFORMER_KEY_REGEX, k)
        for k in state_dict.keys()
    )

def lora_model_from_flux_control_state_dict(state_dict: Dict[str, torch.Tensor]) -> LoRAModelRaw:
    converted_state_dict = _convert_lora_bfl_control(state_dict=state_dict)
    # Group keys by layer.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in converted_state_dict.items():
        key_props = key.split(".")
        # Got it loading using lora_down and lora_up but it didn't seem to match this lora's structure
        # Leaving this in since it doesn't hurt anything and may be better
        layer_prop_size = -2 if any(prop in key for prop in ["lora_down", "lora_up"]) else -1
        layer_name = ".".join(key_props[:layer_prop_size])
        param_name = ".".join(key_props[layer_prop_size:])
        if layer_name not in grouped_state_dict:
            grouped_state_dict[layer_name] = {}
        grouped_state_dict[layer_name][param_name] = value

    # Create LoRA layers.
    layers: dict[str, AnyLoRALayer] = {}
    for layer_key, layer_state_dict in grouped_state_dict.items():
        # Convert to a full layer diff
        layers[layer_key] = any_lora_layer_from_state_dict(state_dict=layer_state_dict)

    # Create and return the LoRAModelRaw.
    return LoRAModelRaw(layers=layers)


def _convert_lora_bfl_control(state_dict: dict[str, torch.Tensor])-> dict[str, torch.Tensor]:
    sd_out: dict[str, torch.Tensor] = {}
    for k in state_dict:
        if k.endswith(".scale"): # TODO: Fix these patches
            continue
        k_to = k.replace(".lora_B.bias", ".lora_B.diff_b")\
                .replace(".lora_A.weight", ".lora_A.diff")\
                .replace(".lora_B.weight", ".lora_B.diff")
        sd_out[k_to] = state_dict[k]

    # sd_out["img_in.reshape_weight"] = torch.tensor([state_dict["img_in.lora_B.weight"].shape[0], state_dict["img_in.lora_A.weight"].shape[1]])
    return sd_out
