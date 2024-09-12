from typing import Dict

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase


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
        self.w1_a = torch.nn.Parameter(w1_a)
        self.w1_b = torch.nn.Parameter(w1_b)
        self.w2_a = torch.nn.Parameter(w2_a)
        self.w2_b = torch.nn.Parameter(w2_b)
        self.t1 = torch.nn.Parameter(t1) if t1 is not None else None
        self.t2 = torch.nn.Parameter(t2) if t2 is not None else None
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
