from typing import Dict, Optional

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensor_size


class SetWeightLayer(LoRALayerBase):
    # TODO: Just everything in this class 
    def __init__(self, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], scale: Optional[torch.Tensor]):
        super().__init__(alpha=None, bias=bias)
        self.weight = torch.nn.Parameter(weight) if weight is not None else None
        self.manual_scale = scale

    def scale(self):
        return self.manual_scale.float() if self.manual_scale is not None else super().scale()

    @classmethod
    def from_state_dict_values(
        cls,
        values: Dict[str, torch.Tensor],
    ):
        layer = cls(weight=values.get("set_weight", None), bias=values.get("set_bias", None), scale=values.get("set_scale", None))
        cls.warn_on_unhandled_keys(values=values, handled_keys={"set_weight", "set_bias", "set_scale"})
        return layer

    def rank(self) -> int | None:
        return None

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        return self.weight

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        if self.weight is not None:
            self.weight = self.weight.to(device=device, dtype=dtype)
        if self.manual_scale is not None:
            self.manual_scale = self.manual_scale.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensor_size(self.manual_scale)
