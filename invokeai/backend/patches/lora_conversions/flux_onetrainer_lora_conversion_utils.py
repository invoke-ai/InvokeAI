import re
from typing import Any, Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_diffusers_lora_conversion_utils import (
    lora_layers_from_flux_diffusers_grouped_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_kohya_lora_conversion_utils import (
    FLUX_KOHYA_CLIP_KEY_REGEX,
    FLUX_KOHYA_T5_KEY_REGEX,
    _convert_flux_clip_kohya_state_dict_to_invoke_format,
    _convert_flux_t5_kohya_state_dict_to_invoke_format,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import (
    FLUX_LORA_CLIP_PREFIX,
    FLUX_LORA_T5_PREFIX,
)
from invokeai.backend.patches.lora_conversions.kohya_key_utils import (
    INDEX_PLACEHOLDER,
    ParsingTree,
    insert_periods_into_kohya_key,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# A regex pattern that matches all of the transformer keys in the OneTrainer FLUX LoRA format.
# The OneTrainer format uses a mix of the Kohya and Diffusers formats:
#   - The base model keys are in Diffusers format.
#   - Periods are replaced with underscores, to match Kohya.
#   - The LoRA key suffixes (e.g. .alpha, .lora_down.weight, .lora_up.weight) match Kohya.
# Example keys:
# - "lora_transformer_single_transformer_blocks_0_attn_to_k.alpha"
# - "lora_transformer_single_transformer_blocks_0_attn_to_k.dora_scale"
# - "lora_transformer_single_transformer_blocks_0_attn_to_k.lora_down.weight"
# - "lora_transformer_single_transformer_blocks_0_attn_to_k.lora_up.weight"
FLUX_ONETRAINER_TRANSFORMER_KEY_REGEX = (
    r"lora_transformer_(single_transformer_blocks|transformer_blocks)_(\d+)_(\w+)\.(.*)"
)

# A regex pattern that matches the layers outside the transformer blocks that OneTrainer trains with a full layer
# filter. A fixed list: the single blocks have a `proj_out` of their own, and Qwen-Image, SD3 and others share some of
# these module names. Only the top-level `proj_out` (the last layer) is meant here, which is why the names carry no
# block index.
# Example keys:
# - "lora_transformer_x_embedder.lora_down.weight"
# - "lora_transformer_time_text_embed_timestep_embedder_linear_1.alpha"
FLUX_ONETRAINER_NON_BLOCK_KEY_REGEX = (
    r"lora_transformer_(x_embedder|context_embedder|norm_out_linear|proj_out"
    r"|time_text_embed_(timestep|text|guidance)_embedder_linear_[12])\..*"
)


def is_state_dict_likely_in_flux_onetrainer_format(state_dict: dict[str | int, Any]) -> bool:
    """Checks if the provided state dict is likely in the OneTrainer FLUX LoRA format.

    This is intended to be a high-precision detector, but it is not guaranteed to have perfect precision. (A
    perfect-precision detector would require checking all keys against a whitelist and verifying tensor shapes.)

    Note that OneTrainer matches the Kohya format for the CLIP and T5 models.

    The layers outside the blocks (`x_embedder`, `context_embedder`, ...) are accepted, but a state dict needs at least
    one block, CLIP or T5 key as well, and a single-block key: Qwen-Image and SD3 share those module names and the
    double-block names, but have no single blocks.
    """
    anchored = False
    has_non_block = False
    has_single_block = False
    for k in state_dict.keys():
        if not isinstance(k, str):
            continue
        if (
            re.match(FLUX_ONETRAINER_TRANSFORMER_KEY_REGEX, k)
            or re.match(FLUX_KOHYA_CLIP_KEY_REGEX, k)
            or re.match(FLUX_KOHYA_T5_KEY_REGEX, k)
        ):
            anchored = True
            has_single_block = has_single_block or k.startswith("lora_transformer_single_transformer_blocks_")
        elif re.fullmatch(FLUX_ONETRAINER_NON_BLOCK_KEY_REGEX, k):
            has_non_block = True
        else:
            return False
    # SD3 and Qwen-Image share FLUX's double-block names and the top-level names above, so those alone cannot tell them
    # apart. Only FLUX has single blocks, and OneTrainer's full layer filter always trains them.
    return anchored and (has_single_block or not has_non_block)


def lora_model_from_flux_onetrainer_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:  # type: ignore
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
        if layer_name.startswith("lora_transformer"):
            transformer_grouped_sd[layer_name] = layer_state_dict
        elif layer_name.startswith("lora_te1"):
            clip_grouped_sd[layer_name] = layer_state_dict
        elif layer_name.startswith("lora_te2"):
            t5_grouped_sd[layer_name] = layer_state_dict
        else:
            raise ValueError(f"Layer '{layer_name}' does not match the expected pattern for FLUX LoRA weights.")

    # Convert the state dicts to the InvokeAI format.
    clip_grouped_sd = _convert_flux_clip_kohya_state_dict_to_invoke_format(clip_grouped_sd)
    t5_grouped_sd = _convert_flux_t5_kohya_state_dict_to_invoke_format(t5_grouped_sd)

    # Create LoRA layers.
    layers: dict[str, BaseLayerPatch] = {}
    for model_prefix, grouped_sd in [
        # (FLUX_LORA_TRANSFORMER_PREFIX, transformer_grouped_sd),
        (FLUX_LORA_CLIP_PREFIX, clip_grouped_sd),
        (FLUX_LORA_T5_PREFIX, t5_grouped_sd),
    ]:
        for layer_key, layer_state_dict in grouped_sd.items():
            layers[model_prefix + layer_key] = any_lora_layer_from_state_dict(layer_state_dict)

    # Handle the transformer.
    transformer_layers = _convert_flux_transformer_onetrainer_state_dict_to_invoke_format(transformer_grouped_sd)
    layers.update(transformer_layers)

    # Create and return the LoRAModelRaw.
    return ModelPatchRaw(layers=layers)


# This parsing tree was generated by calling `generate_kohya_parsing_tree_from_keys()` on the keys in
# flux_lora_diffusers_format.py, plus the layers outside the blocks (FLUX_ONETRAINER_NON_BLOCK_KEY_REGEX).
flux_transformer_kohya_parsing_tree: ParsingTree = {
    "transformer": {
        "x_embedder": {},
        "context_embedder": {},
        "norm_out": {"linear": {}},
        "proj_out": {},
        "time_text_embed": {
            "timestep_embedder": {"linear_1": {}, "linear_2": {}},
            "text_embedder": {"linear_1": {}, "linear_2": {}},
            "guidance_embedder": {"linear_1": {}, "linear_2": {}},
        },
        "single_transformer_blocks": {
            INDEX_PLACEHOLDER: {
                "attn": {"to_k": {}, "to_q": {}, "to_v": {}},
                "norm": {"linear": {}},
                "proj_mlp": {},
                "proj_out": {},
            }
        },
        "transformer_blocks": {
            INDEX_PLACEHOLDER: {
                "attn": {
                    "add_k_proj": {},
                    "add_q_proj": {},
                    "add_v_proj": {},
                    "to_add_out": {},
                    "to_k": {},
                    "to_out": {INDEX_PLACEHOLDER: {}},
                    "to_q": {},
                    "to_v": {},
                },
                "ff": {"net": {INDEX_PLACEHOLDER: {"proj": {}}}},
                "ff_context": {"net": {INDEX_PLACEHOLDER: {"proj": {}}}},
                "norm1": {"linear": {}},
                "norm1_context": {"linear": {}},
            }
        },
    }
}


def _convert_flux_transformer_onetrainer_state_dict_to_invoke_format(
    state_dict: Dict[str, Dict[str, torch.Tensor]],
) -> dict[str, BaseLayerPatch]:
    """Converts a FLUX transformer LoRA state dict from the OneTrainer FLUX LoRA format to the LoRA weight format used
    internally by InvokeAI.
    """

    # Step 1: Convert the Kohya-style keys with underscores to classic keys with periods.
    # Example:
    # "lora_transformer_single_transformer_blocks_0_attn_to_k.lora_down.weight" -> "transformer.single_transformer_blocks.0.attn.to_k.lora_down.weight"
    lora_prefix = "lora_"
    lora_prefix_length = len(lora_prefix)
    kohya_state_dict: dict[str, Dict[str, torch.Tensor]] = {}
    for key in state_dict.keys():
        # Remove the "lora_" prefix.
        assert key.startswith(lora_prefix)
        new_key = key[lora_prefix_length:]

        # Add periods to the Kohya-style module keys.
        new_key = insert_periods_into_kohya_key(new_key, flux_transformer_kohya_parsing_tree)

        # Replace the old key with the new key.
        kohya_state_dict[new_key] = state_dict[key]

    # Step 2: Convert diffusers module names to the BFL module names.
    return lora_layers_from_flux_diffusers_grouped_state_dict(kohya_state_dict, alpha=None)
