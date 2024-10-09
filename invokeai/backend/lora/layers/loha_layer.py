from typing import Dict

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class LoHALayer(LoRALayerBase):
    """LoHA LyCoris layer.

    Example model for testing this layer type: https://civitai.com/models/27397/loha-renoir-the-dappled-light-style
    """

    def __init__(
        self,
        w1_a: torch.Tensor,
        w1_b: torch.Tensor,
        w2_a: torch.Tensor,
        w2_b: torch.Tensor,
        t1: torch.Tensor | None,
        t2: torch.Tensor | None,
        alpha: float | None,
        bias: torch.Tensor | None,
    ):
        super().__init__(alpha=alpha, bias=bias)
        self.w1_a = w1_a
        self.w1_b = w1_b
        self.w2_a = w2_a
        self.w2_b = w2_b
        self.t1 = t1
        self.t2 = t2
        assert (self.t1 is None) == (self.t2 is None)

    def rank(self) -> int | None:
        return self.w1_b.shape[0]

    @classmethod
    def from_state_dict_values(
        cls,
        values: Dict[str, torch.Tensor],
    ):
        alpha = cls._parse_alpha(values.get("alpha", None))
        bias = cls._parse_bias(
            values.get("bias_indices", None), values.get("bias_values", None), values.get("bias_size", None)
        )
        layer = cls(
            w1_a=values["hada_w1_a"],
            w1_b=values["hada_w1_b"],
            w2_a=values["hada_w2_a"],
            w2_b=values["hada_w2_b"],
            t1=values.get("hada_t1", None),
            t2=values.get("hada_t2", None),
            alpha=alpha,
            bias=bias,
        )

        cls.warn_on_unhandled_keys(
            values=values,
            handled_keys={
                # Default keys.
                "alpha",
                "bias_indices",
                "bias_values",
                "bias_size",
                # Layer-specific keys.
                "hada_w1_a",
                "hada_w1_b",
                "hada_w2_a",
                "hada_w2_b",
                "hada_t1",
                "hada_t2",
            },
        )

        return layer

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        if self.t1 is None:
            weight: torch.Tensor = (self.w1_a @ self.w1_b) * (self.w2_a @ self.w2_b)
        else:
            rebuild1 = torch.einsum("i j k l, j r, i p -> p r k l", self.t1, self.w1_b, self.w1_a)
            rebuild2 = torch.einsum("i j k l, j r, i p -> p r k l", self.t2, self.w2_b, self.w2_a)
            weight = rebuild1 * rebuild2

        return weight

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        self.w1_a = self.w1_a.to(device=device, dtype=dtype)
        self.w1_b = self.w1_b.to(device=device, dtype=dtype)
        self.w2_a = self.w2_a.to(device=device, dtype=dtype)
        self.w2_b = self.w2_b.to(device=device, dtype=dtype)
        self.t1 = self.t1.to(device=device, dtype=dtype) if self.t1 is not None else self.t1
        self.t2 = self.t2.to(device=device, dtype=dtype) if self.t2 is not None else self.t2

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensors_size([self.w1_a, self.w1_b, self.w2_a, self.w2_b, self.t1, self.t2])
