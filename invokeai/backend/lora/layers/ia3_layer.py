from typing import Dict, Optional

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class IA3Layer(LoRALayerBase):
    # weight: torch.Tensor
    # on_input: torch.Tensor

    def __init__(
        self,
        values: Dict[str, torch.Tensor],
    ):
        super().__init__(values)

        self.weight = values["weight"]
        self.on_input = values["on_input"]

        self.rank = None  # unscaled
        self.check_keys(values, {"weight", "on_input"})

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if not self.on_input:
            weight = weight.reshape(-1, 1)
        assert orig_weight is not None
        return orig_weight * weight

    def calc_size(self) -> int:
        model_size = super().calc_size()
        model_size += calc_tensors_size([self.weight, self.on_input])
        return model_size

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().to(device=device, dtype=dtype)

        self.weight = self.weight.to(device=device, dtype=dtype)
        self.on_input = self.on_input.to(device=device, dtype=dtype)
