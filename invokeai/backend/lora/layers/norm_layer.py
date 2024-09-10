from typing import Dict, Optional

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensor_size


class NormLayer(LoRALayerBase):
    # bias handled in LoRALayerBase(calc_size, to)
    # weight: torch.Tensor
    # bias: Optional[torch.Tensor]

    def __init__(
        self,
        values: Dict[str, torch.Tensor],
    ):
        super().__init__(values)

        self.weight = values["w_norm"]
        self.bias = values.get("b_norm", None)

        self.rank = None  # unscaled
        self.check_keys(values, {"w_norm", "b_norm"})

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        return self.weight

    def calc_size(self) -> int:
        model_size = super().calc_size()
        model_size += calc_tensor_size(self.weight)
        return model_size

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().to(device=device, dtype=dtype)

        self.weight = self.weight.to(device=device, dtype=dtype)
