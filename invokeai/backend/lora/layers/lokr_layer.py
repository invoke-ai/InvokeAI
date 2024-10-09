from typing import Dict

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class LoKRLayer(LoRALayerBase):
    """LoKR LyCoris layer.

    Example model for testing this layer type: https://civitai.com/models/346747/lokrnekopara-allgirl-for-jru2
    """

    def __init__(
        self,
        w1: torch.Tensor | None,
        w1_a: torch.Tensor | None,
        w1_b: torch.Tensor | None,
        w2: torch.Tensor | None,
        w2_a: torch.Tensor | None,
        w2_b: torch.Tensor | None,
        t2: torch.Tensor | None,
        alpha: float | None,
        bias: torch.Tensor | None,
    ):
        super().__init__(alpha=alpha, bias=bias)
        self.w1 = w1
        self.w1_a = w1_a
        self.w1_b = w1_b
        self.w2 = w2
        self.w2_a = w2_a
        self.w2_b = w2_b
        self.t2 = t2

        # Validate parameters.
        assert (self.w1 is None) != (self.w1_a is None)
        assert (self.w1_a is None) == (self.w1_b is None)
        assert (self.w2 is None) != (self.w2_a is None)
        assert (self.w2_a is None) == (self.w2_b is None)

    def rank(self) -> int | None:
        if self.w1_b is not None:
            return self.w1_b.shape[0]
        elif self.w2_b is not None:
            return self.w2_b.shape[0]
        else:
            return None

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
            w1=values.get("lokr_w1", None),
            w1_a=values.get("lokr_w1_a", None),
            w1_b=values.get("lokr_w1_b", None),
            w2=values.get("lokr_w2", None),
            w2_a=values.get("lokr_w2_a", None),
            w2_b=values.get("lokr_w2_b", None),
            t2=values.get("lokr_t2", None),
            alpha=alpha,
            bias=bias,
        )

        cls.warn_on_unhandled_keys(
            values,
            {
                # Default keys.
                "alpha",
                "bias_indices",
                "bias_values",
                "bias_size",
                # Layer-specific keys.
                "lokr_w1",
                "lokr_w1_a",
                "lokr_w1_b",
                "lokr_w2",
                "lokr_w2_a",
                "lokr_w2_b",
                "lokr_t2",
            },
        )

        return layer

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        w1 = self.w1
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
        weight = torch.kron(w1, w2)
        return weight

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        self.w1 = self.w1.to(device=device, dtype=dtype) if self.w1 is not None else self.w1
        self.w1_a = self.w1_a.to(device=device, dtype=dtype) if self.w1_a is not None else self.w1_a
        self.w1_b = self.w1_b.to(device=device, dtype=dtype) if self.w1_b is not None else self.w1_b
        self.w2 = self.w2.to(device=device, dtype=dtype) if self.w2 is not None else self.w2
        self.w2_a = self.w2_a.to(device=device, dtype=dtype) if self.w2_a is not None else self.w2_a
        self.w2_b = self.w2_b.to(device=device, dtype=dtype) if self.w2_b is not None else self.w2_b
        self.t2 = self.t2.to(device=device, dtype=dtype) if self.t2 is not None else self.t2

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensors_size(
            [self.w1, self.w1_a, self.w1_b, self.w2, self.w2_a, self.w2_b, self.t2]
        )
