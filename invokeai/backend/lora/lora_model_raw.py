# Copyright (c) 2024 The InvokeAI Development team
from typing import Mapping, Optional

import torch

from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.raw_model import RawModel


class LoRAModelRaw(RawModel):  # (torch.nn.Module):
    def __init__(self, layers: Mapping[str, AnyLoRALayer]):
        self.layers = layers

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        for _key, layer in self.layers.items():
            layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        model_size = 0
        for _, layer in self.layers.items():
            model_size += layer.calc_size()
        return model_size
