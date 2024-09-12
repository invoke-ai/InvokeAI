from typing import Dict

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase


class NormLayer(LoRALayerBase):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__(alpha=None, bias=bias)
        self.weight = torch.nn.Parameter(weight)

    @classmethod
    def from_state_dict_values(
        cls,
        values: Dict[str, torch.Tensor],
    ):
        layer = cls(weight=values["w_norm"], bias=values.get("b_norm", None))
        cls.warn_on_unhandled_keys(values, {"w_norm", "b_norm"})
        return layer

    def rank(self) -> int | None:
        return None

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        return self.weight
