import json
from collections import defaultdict
from typing import Any

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


def is_state_dict_likely_in_aitoolkit_format(state_dict: dict[str, Any], metadata: dict[str, Any] = None) -> bool:
    if metadata:
        software = json.loads(metadata.get("software", "{}"))
        return software.get("name") == "ai-toolkit"
    # metadata got lost somewhere
    return any("diffusion_model" == k.split(".", 1)[0] for k in state_dict.keys())


def lora_model_from_flux_aitoolkit_state_dict(state_dict: dict[str, torch.Tensor]) -> ModelPatchRaw:
    # Group keys by layer.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, value in state_dict.items():
        layer_name, param_name = key.split(".", 1)
        grouped_state_dict[layer_name][param_name] = value

    transformer_grouped_sd: dict[str, dict[str, torch.Tensor]] = {}

    for layer_name, layer_state_dict in grouped_state_dict.items():
        if layer_name.startswith("diffusion_model"):
            transformer_grouped_sd[layer_name] = layer_state_dict
        else:
            raise ValueError(f"Layer '{layer_name}' does not match the expected pattern for FLUX LoRA weights.")

    layers: dict[str, BaseLayerPatch] = {}
    for layer_key, layer_state_dict in transformer_grouped_sd.items():
        layers[FLUX_LORA_TRANSFORMER_PREFIX + layer_key] = any_lora_layer_from_state_dict(layer_state_dict)

    return ModelPatchRaw(layers=layers)
