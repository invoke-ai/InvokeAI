import re
import torch

from typing import Any, Dict
from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.lora.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.lora.lora_model_raw import LoRAModelRaw
from invokeai.backend.lora.conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.lora.layers.lora_layer import LoRALayer
from invokeai.backend.lora.layers.set_parameter_layer import SetParameterLayer


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
    # converted_state_dict = _convert_lora_bfl_control(state_dict=state_dict)
    # Group keys by layer.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        key_props = key.split(".")
        # Got it loading using lora_down and lora_up but it didn't seem to match this lora's structure
        # Leaving this in since it doesn't hurt anything and may be better
        layer_prop_size = -2 if any(prop in key for prop in ["lora_B", "lora_A"]) else -1
        layer_name = ".".join(key_props[:layer_prop_size])
        param_name = ".".join(key_props[layer_prop_size:])
        if layer_name not in grouped_state_dict:
            grouped_state_dict[layer_name] = {}
        grouped_state_dict[layer_name][param_name] = value

    # Create LoRA layers.
    layers: dict[str, AnyLoRALayer] = {}
    for layer_key, layer_state_dict in grouped_state_dict.items():
        # Convert to a full layer diff
        prefixed_key = f"{FLUX_LORA_TRANSFORMER_PREFIX}{layer_key}"
        if all(k in layer_state_dict for k in ["lora_A.weight", "lora_B.bias", "lora_B.weight"]):
            layers[prefixed_key] = LoRALayer(
                layer_state_dict["lora_B.weight"],
                None,
                layer_state_dict["lora_A.weight"],
                None,
                layer_state_dict["lora_B.bias"]
            )
        elif "scale" in layer_state_dict:
            layers[prefixed_key] = SetParameterLayer("scale", layer_state_dict["scale"])
        else:
            raise AssertionError(f"{layer_key} not expected")
    # Create and return the LoRAModelRaw.
    return LoRAModelRaw(layers=layers)

