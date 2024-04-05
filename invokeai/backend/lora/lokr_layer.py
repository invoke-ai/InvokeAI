from typing import Dict, Optional

import torch

from invokeai.backend.lora.lora_layer_base import LoRALayerBase


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

        if "lokr_w1" in values:
            self.w1: Optional[torch.Tensor] = values["lokr_w1"]
            self.w1_a = None
            self.w1_b = None
        else:
            self.w1 = None
            self.w1_a = values["lokr_w1_a"]
            self.w1_b = values["lokr_w1_b"]

        if "lokr_w2" in values:
            self.w2: Optional[torch.Tensor] = values["lokr_w2"]
            self.w2_a = None
            self.w2_b = None
        else:
            self.w2 = None
            self.w2_a = values["lokr_w2_a"]
            self.w2_b = values["lokr_w2_b"]

        if "lokr_t2" in values:
            self.t2: Optional[torch.Tensor] = values["lokr_t2"]
        else:
            self.t2 = None

        if "lokr_w1_b" in values:
            self.rank = values["lokr_w1_b"].shape[0]
        elif "lokr_w2_b" in values:
            self.rank = values["lokr_w2_b"].shape[0]
        else:
            self.rank = None  # unscaled

    def get_weight(self, orig_weight: Optional[torch.Tensor]) -> torch.Tensor:
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

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
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
