from typing import Dict, Optional

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase


class LoHALayer(LoRALayerBase):
    # w1_a: torch.Tensor
    # w1_b: torch.Tensor
    # w2_a: torch.Tensor
    # w2_b: torch.Tensor
    # t1: Optional[torch.Tensor] = None
    # t2: Optional[torch.Tensor] = None

    def __init__(self, layer_key: str, values: Dict[str, torch.Tensor]):
        super().__init__(layer_key, values)

        self.w1_a = values["hada_w1_a"]
        self.w1_b = values["hada_w1_b"]
        self.w2_a = values["hada_w2_a"]
        self.w2_b = values["hada_w2_b"]
        self.t1 = values.get("hada_t1", None)
        self.t2 = values.get("hada_t2", None)

        self.rank = self.w1_b.shape[0]
        self.check_keys(
            values,
            {
                "hada_w1_a",
                "hada_w1_b",
                "hada_w2_a",
                "hada_w2_b",
                "hada_t1",
                "hada_t2",
            },
        )

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        if self.t1 is None:
            weight: torch.Tensor = (self.w1_a @ self.w1_b) * (self.w2_a @ self.w2_b)

        else:
            rebuild1 = torch.einsum("i j k l, j r, i p -> p r k l", self.t1, self.w1_b, self.w1_a)
            rebuild2 = torch.einsum("i j k l, j r, i p -> p r k l", self.t2, self.w2_b, self.w2_a)
            weight = rebuild1 * rebuild2

        return weight

    def calc_size(self) -> int:
        model_size = super().calc_size()
        for val in [self.w1_a, self.w1_b, self.w2_a, self.w2_b, self.t1, self.t2]:
            if val is not None:
                model_size += val.nelement() * val.element_size()
        return model_size

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().to(device=device, dtype=dtype)

        self.w1_a = self.w1_a.to(device=device, dtype=dtype)
        self.w1_b = self.w1_b.to(device=device, dtype=dtype)
        if self.t1 is not None:
            self.t1 = self.t1.to(device=device, dtype=dtype)

        self.w2_a = self.w2_a.to(device=device, dtype=dtype)
        self.w2_b = self.w2_b.to(device=device, dtype=dtype)
        if self.t2 is not None:
            self.t2 = self.t2.to(device=device, dtype=dtype)
