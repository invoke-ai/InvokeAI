from typing import Dict, Optional

import torch

from invokeai.backend.lora.lora_layer_base import LoRALayerBase


class FullLayer(LoRALayerBase):
    # weight: torch.Tensor

    def __init__(
        self,
        layer_key: str,
        values: Dict[str, torch.Tensor],
    ):
        super().__init__(layer_key, values)

        self.weight = values["diff"]

        if len(values.keys()) > 1:
            _keys = list(values.keys())
            _keys.remove("diff")
            raise NotImplementedError(f"Unexpected keys in lora diff layer: {_keys}")

        self.rank = None  # unscaled

    def get_weight(self, orig_weight: Optional[torch.Tensor]) -> torch.Tensor:
        return self.weight

    def calc_size(self) -> int:
        model_size = super().calc_size()
        model_size += self.weight.nelement() * self.weight.element_size()
        return model_size

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().to(device=device, dtype=dtype)

        self.weight = self.weight.to(device=device, dtype=dtype)
