from typing import Dict, Optional

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class LoRALayer(LoRALayerBase):
    def __init__(
        self,
        up: torch.Tensor,
        mid: Optional[torch.Tensor],
        down: torch.Tensor,
        alpha: float | None,
        bias: Optional[torch.Tensor],
    ):
        super().__init__(alpha, bias)
        self.up = up
        self.mid = mid
        self.down = down

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
            up=values["lora_up.weight"],
            down=values["lora_down.weight"],
            mid=values.get("lora_mid.weight", None),
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
                "lora_up.weight",
                "lora_down.weight",
                "lora_mid.weight",
            },
        )

        return layer

    def rank(self) -> int:
        return self.down.shape[0]

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        if self.mid is not None:
            up = self.up.reshape(self.up.shape[0], self.up.shape[1])
            down = self.down.reshape(self.down.shape[0], self.down.shape[1])
            weight = torch.einsum("m n w h, i m, n j -> i j w h", self.mid, up, down)
        else:
            weight = self.up.reshape(self.up.shape[0], -1) @ self.down.reshape(self.down.shape[0], -1)

        return weight

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        self.up = self.up.to(device=device, dtype=dtype)
        if self.mid is not None:
            self.mid = self.mid.to(device=device, dtype=dtype)
        self.down = self.down.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensors_size([self.up, self.mid, self.down])
