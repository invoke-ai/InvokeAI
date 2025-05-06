import re
from typing import Any, Dict, TypeVar

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_lora_constants import (
    FLUX_LORA_CLIP_PREFIX,
    FLUX_LORA_T5_PREFIX,
    FLUX_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# A regex pattern that matches all of the transformer keys in the Kohya FLUX LoRA format.
# Example keys:
#   lora_unet_double_blocks_0_img_attn_proj.alpha
#   lora_unet_double_blocks_0_img_attn_proj.lora_down.weight
#   lora_unet_double_blocks_0_img_attn_proj.lora_up.weight
FLUX_KOHYA_TRANSFORMER_KEY_REGEX = (
    r"lora_unet_(\w+_blocks)_(\d+)_(img_attn|img_mlp|img_mod|txt_attn|txt_mlp|txt_mod|linear1|linear2|modulation)_?(.*)"
)
# A regex pattern that matches all of the CLIP keys in the Kohya FLUX LoRA format.
# Example keys:
#   lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha
#   lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight
#   lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight
FLUX_KOHYA_CLIP_KEY_REGEX = r"lora_te1_text_model_encoder_layers_(\d+)_(mlp|self_attn)_(\w+)\.?.*"

# A regex pattern that matches all of the T5 keys in the Kohya FLUX LoRA format.
# Example keys:
#   lora_te2_encoder_block_0_layer_0_SelfAttention_k.alpha
#   lora_te2_encoder_block_0_layer_0_SelfAttention_k.dora_scale
#   lora_te2_encoder_block_0_layer_0_SelfAttention_k.lora_down.weight
#   lora_te2_encoder_block_0_layer_0_SelfAttention_k.lora_up.weight
FLUX_KOHYA_T5_KEY_REGEX = r"lora_te2_encoder_block_(\d+)_layer_(\d+)_(DenseReluDense|SelfAttention)_(\w+)_?(\w+)?\.?.*"


def is_state_dict_likely_in_flux_kohya_format(state_dict: Dict[str, Any]) -> bool:
    """Checks if the provided state dict is likely in the Kohya FLUX LoRA format.

    This is intended to be a high-precision detector, but it is not guaranteed to have perfect precision. (A
    perfect-precision detector would require checking all keys against a whitelist and verifying tensor shapes.)
    """
    return all(
        re.match(FLUX_KOHYA_TRANSFORMER_KEY_REGEX, k)
        or re.match(FLUX_KOHYA_CLIP_KEY_REGEX, k)
        or re.match(FLUX_KOHYA_T5_KEY_REGEX, k)
        for k in state_dict.keys()
    )


def lora_model_from_flux_kohya_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:
    # Group keys by layer.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        layer_name, param_name = key.split(".", 1)
        if layer_name not in grouped_state_dict:
            grouped_state_dict[layer_name] = {}
        grouped_state_dict[layer_name][param_name] = value

    # Split the grouped state dict into transformer, CLIP, and T5 state dicts.
    transformer_grouped_sd: dict[str, dict[str, torch.Tensor]] = {}
    clip_grouped_sd: dict[str, dict[str, torch.Tensor]] = {}
    t5_grouped_sd: dict[str, dict[str, torch.Tensor]] = {}
    for layer_name, layer_state_dict in grouped_state_dict.items():
        if layer_name.startswith("lora_unet"):
            transformer_grouped_sd[layer_name] = layer_state_dict
        elif layer_name.startswith("lora_te1"):
            clip_grouped_sd[layer_name] = layer_state_dict
        elif layer_name.startswith("lora_te2"):
            t5_grouped_sd[layer_name] = layer_state_dict
        else:
            raise ValueError(f"Layer '{layer_name}' does not match the expected pattern for FLUX LoRA weights.")

    # Convert the state dicts to the InvokeAI format.
    transformer_grouped_sd = _convert_flux_transformer_kohya_state_dict_to_invoke_format(transformer_grouped_sd)
    clip_grouped_sd = _convert_flux_clip_kohya_state_dict_to_invoke_format(clip_grouped_sd)
    t5_grouped_sd = _convert_flux_t5_kohya_state_dict_to_invoke_format(t5_grouped_sd)

    # Create LoRA layers.
    layers: dict[str, BaseLayerPatch] = {}
    for model_prefix, grouped_sd in [
        (FLUX_LORA_TRANSFORMER_PREFIX, transformer_grouped_sd),
        (FLUX_LORA_CLIP_PREFIX, clip_grouped_sd),
        (FLUX_LORA_T5_PREFIX, t5_grouped_sd),
    ]:
        for layer_key, layer_state_dict in grouped_sd.items():
            layers[model_prefix + layer_key] = any_lora_layer_from_state_dict(layer_state_dict)

    # Create and return the LoRAModelRaw.
    return ModelPatchRaw(layers=layers)


T = TypeVar("T")


def _convert_flux_clip_kohya_state_dict_to_invoke_format(state_dict: Dict[str, T]) -> Dict[str, T]:
    """Converts a CLIP LoRA state dict from the Kohya FLUX LoRA format to LoRA weight format used internally by
    InvokeAI.

    Example key conversions:

    "lora_te1_text_model_encoder_layers_0_mlp_fc1" -> "text_model.encoder.layers.0.mlp.fc1",
    "lora_te1_text_model_encoder_layers_0_self_attn_k_proj" -> "text_model.encoder.layers.0.self_attn.k_proj"
    """
    converted_sd: dict[str, T] = {}
    for k, v in state_dict.items():
        match = re.match(FLUX_KOHYA_CLIP_KEY_REGEX, k)
        if match:
            new_key = f"text_model.encoder.layers.{match.group(1)}.{match.group(2)}.{match.group(3)}"
            converted_sd[new_key] = v
        else:
            raise ValueError(f"Key '{k}' does not match the expected pattern for FLUX LoRA weights.")

    return converted_sd


def _convert_flux_transformer_kohya_state_dict_to_invoke_format(state_dict: Dict[str, T]) -> Dict[str, T]:
    """Converts a FLUX tranformer LoRA state dict from the Kohya FLUX LoRA format to LoRA weight format used internally
    by InvokeAI.

    Example key conversions:
    "lora_unet_double_blocks_0_img_attn_proj" -> "double_blocks.0.img_attn.proj"
    "lora_unet_double_blocks_0_img_attn_qkv" -> "double_blocks.0.img_attn.qkv"
    """

    def replace_func(match: re.Match[str]) -> str:
        s = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
        if match.group(4):
            s += f".{match.group(4)}"
        return s

    converted_dict: dict[str, T] = {}
    for k, v in state_dict.items():
        match = re.match(FLUX_KOHYA_TRANSFORMER_KEY_REGEX, k)
        if match:
            new_key = re.sub(FLUX_KOHYA_TRANSFORMER_KEY_REGEX, replace_func, k)
            converted_dict[new_key] = v
        else:
            raise ValueError(f"Key '{k}' does not match the expected pattern for FLUX LoRA weights.")

    return converted_dict


def _convert_flux_t5_kohya_state_dict_to_invoke_format(state_dict: Dict[str, T]) -> Dict[str, T]:
    """Converts a T5 LoRA state dict from the Kohya FLUX LoRA format to LoRA weight format used internally by
    InvokeAI.

    Example key conversions:

    "lora_te2_encoder_block_0_layer_0_SelfAttention_k" -> "encoder.block.0.layer.0.SelfAttention.k"
    "lora_te2_encoder_block_0_layer_1_DenseReluDense_wi_0" -> "encoder.block.0.layer.1.DenseReluDense.wi.0"
    """

    def replace_func(match: re.Match[str]) -> str:
        s = f"encoder.block.{match.group(1)}.layer.{match.group(2)}.{match.group(3)}.{match.group(4)}"
        if match.group(5):
            s += f".{match.group(5)}"
        return s

    converted_dict: dict[str, T] = {}
    for k, v in state_dict.items():
        match = re.match(FLUX_KOHYA_T5_KEY_REGEX, k)
        if match:
            new_key = re.sub(FLUX_KOHYA_T5_KEY_REGEX, replace_func, k)
            converted_dict[new_key] = v
        else:
            raise ValueError(f"Key '{k}' does not match the expected pattern for FLUX LoRA weights.")

    return converted_dict
