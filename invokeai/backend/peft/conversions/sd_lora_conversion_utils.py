from typing import Dict

import torch

from invokeai.backend.peft.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.peft.layers.utils import peft_layer_from_state_dict
from invokeai.backend.peft.lora import LoRAModelRaw


def lora_model_from_sd_state_dict(state_dict: Dict[str, torch.Tensor]) -> LoRAModelRaw:
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = _group_state(state_dict)

    layers: dict[str, AnyLoRALayer] = {}
    for layer_key, values in grouped_state_dict.items():
        layer = peft_layer_from_state_dict(layer_key, values)
        layers[layer_key] = layer

    return LoRAModelRaw(layers=layers)


def _group_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    state_dict_groupped: Dict[str, Dict[str, torch.Tensor]] = {}

    for key, value in state_dict.items():
        stem, leaf = key.split(".", 1)
        if stem not in state_dict_groupped:
            state_dict_groupped[stem] = {}
        state_dict_groupped[stem][leaf] = value

    return state_dict_groupped