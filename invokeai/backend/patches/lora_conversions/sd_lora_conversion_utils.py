from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


def lora_model_from_sd_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = _group_state(state_dict)

    layers: dict[str, BaseLayerPatch] = {}
    for layer_key, values in grouped_state_dict.items():
        layers[layer_key] = any_lora_layer_from_state_dict(values)

    return ModelPatchRaw(layers=layers)


def _group_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    state_dict_groupped: Dict[str, Dict[str, torch.Tensor]] = {}

    for key, value in state_dict.items():
        stem, leaf = key.split(".", 1)
        if stem not in state_dict_groupped:
            state_dict_groupped[stem] = {}
        state_dict_groupped[stem][leaf] = value

    return state_dict_groupped
