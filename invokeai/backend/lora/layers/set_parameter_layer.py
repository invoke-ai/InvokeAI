from typing import Dict

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensor_size


class SetParameterLayer(LoRALayerBase):
    def __init__(self, param_name: str, weight: torch.Tensor):
        super().__init__(None, None)
        self.weight = weight
        self.param_name = param_name

    def rank(self) -> int | None:
        return None

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        return self.weight - orig_weight

    def get_parameters(self, orig_module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {self.param_name: self.get_weight(orig_module.get_parameter(self.param_name))}

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        self.weight = self.weight.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensor_size(self.weight)
