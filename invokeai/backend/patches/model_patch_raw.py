# Copyright (c) 2024 The InvokeAI Development team
from typing import Iterable, Optional

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.raw_model import RawModel


class ModelPatchRaw(RawModel):
    def __init__(self, layers: Iterable[tuple[str, BaseLayerPatch]]):
        # HACK(ryand): Update all places a dict is passed in.
        if isinstance(layers, dict):
            self.layers = [(k, v) for k, v in layers.items()]
        else:
            self.layers = layers

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        for _, layer in self.layers:
            layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return sum(layer.calc_size() for _, layer in self.layers)
