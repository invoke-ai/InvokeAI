# Copyright (c) 2024 The InvokeAI Development team
from typing import Mapping, Optional

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.raw_model import RawModel


class ModelPatchRaw(RawModel):
    def __init__(self, layers: Mapping[str, BaseLayerPatch]):
        self.layers = layers

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        for layer in self.layers.values():
            layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return sum(layer.calc_size() for layer in self.layers.values())
