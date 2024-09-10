from typing import Dict, Optional

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase


# TODO: find and debug lora/locon with bias
class LoRALayer(LoRALayerBase):
    # up: torch.Tensor
    # mid: Optional[torch.Tensor]
    # down: torch.Tensor

    def __init__(
        self,
        layer_key: str,
        values: Dict[str, torch.Tensor],
    ):
        super().__init__(layer_key, values)

        self.up = values["lora_up.weight"]
        self.down = values["lora_down.weight"]
        self.mid = values.get("lora_mid.weight", None)

        self.rank = self.down.shape[0]
        self.check_keys(
            values,
            {
                "lora_up.weight",
                "lora_down.weight",
                "lora_mid.weight",
            },
        )

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        if self.mid is not None:
            up = self.up.reshape(self.up.shape[0], self.up.shape[1])
            down = self.down.reshape(self.down.shape[0], self.down.shape[1])
            weight = torch.einsum("m n w h, i m, n j -> i j w h", self.mid, up, down)
        else:
            weight = self.up.reshape(self.up.shape[0], -1) @ self.down.reshape(self.down.shape[0], -1)

        return weight

    def calc_size(self) -> int:
        model_size = super().calc_size()
        for val in [self.up, self.mid, self.down]:
            if val is not None:
                model_size += val.nelement() * val.element_size()
        return model_size

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().to(device=device, dtype=dtype)

        self.up = self.up.to(device=device, dtype=dtype)
        self.down = self.down.to(device=device, dtype=dtype)

        if self.mid is not None:
            self.mid = self.mid.to(device=device, dtype=dtype)
