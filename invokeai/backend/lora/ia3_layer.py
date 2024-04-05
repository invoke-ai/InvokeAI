from typing import Dict, Optional

import torch

from invokeai.backend.lora.lora_layer_base import LoRALayerBase


class IA3Layer(LoRALayerBase):
    # weight: torch.Tensor
    # on_input: torch.Tensor

    def __init__(
        self,
        layer_key: str,
        values: Dict[str, torch.Tensor],
    ):
        super().__init__(layer_key, values)

        self.weight = values["weight"]
        self.on_input = values["on_input"]

        self.rank = None  # unscaled

    def get_weight(self, orig_weight: Optional[torch.Tensor]) -> torch.Tensor:
        weight = self.weight
        if not self.on_input:
            weight = weight.reshape(-1, 1)
        assert orig_weight is not None
        return orig_weight * weight

    def calc_size(self) -> int:
        model_size = super().calc_size()
        model_size += self.weight.nelement() * self.weight.element_size()
        model_size += self.on_input.nelement() * self.on_input.element_size()
        return model_size

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().to(device=device, dtype=dtype)

        self.weight = self.weight.to(device=device, dtype=dtype)
        self.on_input = self.on_input.to(device=device, dtype=dtype)
