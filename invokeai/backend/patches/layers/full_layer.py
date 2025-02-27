from typing import Dict, Optional

import torch

from invokeai.backend.patches.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensor_size


class FullLayer(LoRALayerBase):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__(alpha=None, bias=bias)
        self.weight = torch.nn.Parameter(weight)

    @classmethod
    def from_state_dict_values(
        cls,
        values: Dict[str, torch.Tensor],
    ):
        layer = cls(weight=values["diff"], bias=values.get("diff_b", None))
        cls.warn_on_unhandled_keys(values=values, handled_keys={"diff", "diff_b"})
        return layer

    def _rank(self) -> int | None:
        return None

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        return self.weight

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        self.weight = self.weight.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensor_size(self.weight)
