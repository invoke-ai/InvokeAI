import re
from typing import Any, Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.flux_control_lora_layer import FluxControlLoRALayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.set_parameter_layer import SetParameterLayer
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# A regex pattern that matches all of the keys in the Flux Dev/Canny LoRA format.
# Example keys:
#   guidance_in.in_layer.lora_B.bias
#   single_blocks.0.linear1.lora_A.weight
#   double_blocks.0.img_attn.norm.key_norm.scale
FLUX_CONTROL_TRANSFORMER_KEY_REGEX = r"(\w+\.)+(lora_A\.weight|lora_B\.weight|lora_B\.bias|scale)"


def is_state_dict_likely_flux_control(state_dict: Dict[str, Any]) -> bool:
    """Checks if the provided state dict is likely in the FLUX Control LoRA format.

    This is intended to be a high-precision detector, but it is not guaranteed to have perfect precision. (A
    perfect-precision detector would require checking all keys against a whitelist and verifying tensor shapes.)
    """

    all_keys_match = all(re.match(FLUX_CONTROL_TRANSFORMER_KEY_REGEX, str(k)) for k in state_dict.keys())

    # Check the shape of the img_in weight, because this layer shape is modified by FLUX control LoRAs.
    lora_a_weight = state_dict.get("img_in.lora_A.weight", None)
    lora_b_bias = state_dict.get("img_in.lora_B.bias", None)
    lora_b_weight = state_dict.get("img_in.lora_B.weight", None)

    return (
        all_keys_match
        and lora_a_weight is not None
        and lora_b_bias is not None
        and lora_b_weight is not None
        and lora_a_weight.shape[1] == 128
        and lora_b_weight.shape[0] == 3072
        and lora_b_bias.shape[0] == 3072
    )


def lora_model_from_flux_control_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:
    # Group keys by layer.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        key_props = key.split(".")
        layer_prop_size = -2 if any(prop in key for prop in ["lora_B", "lora_A"]) else -1
        layer_name = ".".join(key_props[:layer_prop_size])
        param_name = ".".join(key_props[layer_prop_size:])
        if layer_name not in grouped_state_dict:
            grouped_state_dict[layer_name] = {}
        grouped_state_dict[layer_name][param_name] = value

    # Create LoRA layers.
    layers: dict[str, BaseLayerPatch] = {}
    for layer_key, layer_state_dict in grouped_state_dict.items():
        prefixed_key = f"{FLUX_LORA_TRANSFORMER_PREFIX}{layer_key}"
        if layer_key == "img_in":
            # img_in is a special case because it changes the shape of the original weight.
            layers[prefixed_key] = FluxControlLoRALayer(
                layer_state_dict["lora_B.weight"],
                None,
                layer_state_dict["lora_A.weight"],
                None,
                layer_state_dict["lora_B.bias"],
            )
        elif all(k in layer_state_dict for k in ["lora_A.weight", "lora_B.bias", "lora_B.weight"]):
            layers[prefixed_key] = LoRALayer(
                layer_state_dict["lora_B.weight"],
                None,
                layer_state_dict["lora_A.weight"],
                None,
                layer_state_dict["lora_B.bias"],
            )
        elif "scale" in layer_state_dict:
            layers[prefixed_key] = SetParameterLayer("scale", layer_state_dict["scale"])
        else:
            raise ValueError(f"{layer_key} not expected")

    return ModelPatchRaw(layers=layers)
