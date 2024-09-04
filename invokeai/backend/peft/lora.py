# Copyright (c) 2024 The InvokeAI Development team
from typing import Dict, Optional

import torch
from typing_extensions import Self

from invokeai.backend.peft.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.peft.layers.full_layer import FullLayer
from invokeai.backend.peft.layers.ia3_layer import IA3Layer
from invokeai.backend.peft.layers.loha_layer import LoHALayer
from invokeai.backend.peft.layers.lokr_layer import LoKRLayer
from invokeai.backend.peft.layers.lora_layer import LoRALayer
from invokeai.backend.peft.layers.norm_layer import NormLayer
from invokeai.backend.raw_model import RawModel


class LoRAModelRaw(RawModel):  # (torch.nn.Module):
    def __init__(self, layers: Dict[str, AnyLoRALayer]):
        self.layers = layers

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        # TODO: try revert if exception?
        for _key, layer in self.layers.items():
            layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        model_size = 0
        for _, layer in self.layers.items():
            model_size += layer.calc_size()
        return model_size

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32

        grouped_state_dict: dict[str, dict[str, torch.Tensor]] = cls._group_state(state_dict)
        del state_dict  # Delete state_dict so that layers can be gc'd as they are processed.

        layers: dict[str, AnyLoRALayer] = {}
        for layer_key, values in grouped_state_dict.items():
            # Detect layers according to LyCORIS detection logic(`weight_list_det`)
            # https://github.com/KohakuBlueleaf/LyCORIS/tree/8ad8000efb79e2b879054da8c9356e6143591bad/lycoris/modules

            # lora and locon
            if "lora_up.weight" in values:
                layer: AnyLoRALayer = LoRALayer(layer_key, values)
            # loha
            elif "hada_w1_a" in values:
                layer = LoHALayer(layer_key, values)
            # lokr
            elif "lokr_w1" in values or "lokr_w1_a" in values:
                layer = LoKRLayer(layer_key, values)
            # diff
            elif "diff" in values:
                layer = FullLayer(layer_key, values)
            # ia3
            elif "on_input" in values:
                layer = IA3Layer(layer_key, values)
            # norms
            elif "w_norm" in values:
                layer = NormLayer(layer_key, values)
            else:
                raise ValueError(f"Unsupported lora format: {layer_key} - {list(values.keys())}")

            # Reduce memory consumption by removing references to layer values that have already been handled.
            grouped_state_dict[layer_key].clear()

            layer.to(device=device, dtype=dtype)
            layers[layer_key] = layer

        return cls(layers=layers)

    @staticmethod
    def _group_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        state_dict_groupped: Dict[str, Dict[str, torch.Tensor]] = {}

        for key, value in state_dict.items():
            stem, leaf = key.split(".", 1)
            if stem not in state_dict_groupped:
                state_dict_groupped[stem] = {}
            state_dict_groupped[stem][leaf] = value

        return state_dict_groupped
