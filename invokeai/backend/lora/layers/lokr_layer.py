from typing import Dict, Optional

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase


class LoKRLayer(LoRALayerBase):
    # w1: Optional[torch.Tensor] = None
    # w1_a: Optional[torch.Tensor] = None
    # w1_b: Optional[torch.Tensor] = None
    # w2: Optional[torch.Tensor] = None
    # w2_a: Optional[torch.Tensor] = None
    # w2_b: Optional[torch.Tensor] = None
    # t2: Optional[torch.Tensor] = None

    def __init__(
        self,
        layer_key: str,
        values: Dict[str, torch.Tensor],
    ):
        super().__init__(layer_key, values)

        self.w1 = values.get("lokr_w1", None)
        if self.w1 is None:
            self.w1_a = values["lokr_w1_a"]
            self.w1_b = values["lokr_w1_b"]
        else:
            self.w1_b = None
            self.w1_a = None

        self.w2 = values.get("lokr_w2", None)
        if self.w2 is None:
            self.w2_a = values["lokr_w2_a"]
            self.w2_b = values["lokr_w2_b"]
        else:
            self.w2_a = None
            self.w2_b = None

        self.t2 = values.get("lokr_t2", None)

        if self.w1_b is not None:
            self.rank = self.w1_b.shape[0]
        elif self.w2_b is not None:
            self.rank = self.w2_b.shape[0]
        else:
            self.rank = None  # unscaled

        self.check_keys(
            values,
            {
                "lokr_w1",
                "lokr_w1_a",
                "lokr_w1_b",
                "lokr_w2",
                "lokr_w2_a",
                "lokr_w2_b",
                "lokr_t2",
            },
        )

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        w1: Optional[torch.Tensor] = self.w1
        if w1 is None:
            assert self.w1_a is not None
            assert self.w1_b is not None
            w1 = self.w1_a @ self.w1_b

        w2 = self.w2
        if w2 is None:
            if self.t2 is None:
                assert self.w2_a is not None
                assert self.w2_b is not None
                w2 = self.w2_a @ self.w2_b
            else:
                w2 = torch.einsum("i j k l, i p, j r -> p r k l", self.t2, self.w2_a, self.w2_b)

        if len(w2.shape) == 4:
            w1 = w1.unsqueeze(2).unsqueeze(2)
        w2 = w2.contiguous()
        assert w1 is not None
        assert w2 is not None
        weight = torch.kron(w1, w2)

        return weight

    def calc_size(self) -> int:
        model_size = super().calc_size()
        for val in [self.w1, self.w1_a, self.w1_b, self.w2, self.w2_a, self.w2_b, self.t2]:
            if val is not None:
                model_size += val.nelement() * val.element_size()
        return model_size

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().to(device=device, dtype=dtype)

        if self.w1 is not None:
            self.w1 = self.w1.to(device=device, dtype=dtype)
        else:
            assert self.w1_a is not None
            assert self.w1_b is not None
            self.w1_a = self.w1_a.to(device=device, dtype=dtype)
            self.w1_b = self.w1_b.to(device=device, dtype=dtype)

        if self.w2 is not None:
            self.w2 = self.w2.to(device=device, dtype=dtype)
        else:
            assert self.w2_a is not None
            assert self.w2_b is not None
            self.w2_a = self.w2_a.to(device=device, dtype=dtype)
            self.w2_b = self.w2_b.to(device=device, dtype=dtype)

        if self.t2 is not None:
            self.t2 = self.t2.to(device=device, dtype=dtype)
