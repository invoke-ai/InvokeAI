from typing import Dict

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase


class LoKRLayer(LoRALayerBase):
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
        self.w1 = torch.nn.Parameter(w1) if w1 is not None else None
        self.w1_a = torch.nn.Parameter(w1_a) if w1_a is not None else None
        self.w1_b = torch.nn.Parameter(w1_b) if w1_b is not None else None
        self.w2 = torch.nn.Parameter(w2) if w2 is not None else None
        self.w2_a = torch.nn.Parameter(w2_a) if w2_a is not None else None
        self.w2_b = torch.nn.Parameter(w2_b) if w2_b is not None else None
        self.t2 = torch.nn.Parameter(t2) if t2 is not None else None

        # Validate parameters.
        assert (self.w1 is None) != (self.w1_a is None)
        assert (self.w1_a is None) == (self.w1_b is None)
        assert (self.w2 is None) != (self.w2_a is None)

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
