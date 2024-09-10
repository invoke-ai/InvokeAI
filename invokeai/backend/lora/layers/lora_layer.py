from typing import Dict, Optional

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class LoRALayer(LoRALayerBase):
    def __init__(
        self,
        up: torch.Tensor,
        down: torch.Tensor,
        mid: Optional[torch.Tensor],
        alpha: float | None,
        bias: torch.Tensor | None,
    ):
        super().__init__(alpha=alpha, bias=bias)

        self.up = up
        self.down = down
        self.mid = mid

    @classmethod
    def from_state_dict_values(
        cls,
        values: Dict[str, torch.Tensor],
    ):
        alpha = cls._parse_alpha(values.get("alpha", None))
        bias = cls._parse_bias(
            values.get("bias_indices", None), values.get("bias_values", None), values.get("bias_size", None)
        )

        cls(
            up=values["lora_up.weight"],
            down=values["lora_down.weight"],
            mid=values.get("lora_mid.weight", None),
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
                "lora_up.weight",
                "lora_down.weight",
                "lora_mid.weight",
            },
        )

    @property
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

    def calc_size(self) -> int:
        model_size = super().calc_size()
        model_size += calc_tensors_size([self.up, self.mid, self.down])
        return model_size

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().to(device=device, dtype=dtype)

        self.up = self.up.to(device=device, dtype=dtype)
        self.down = self.down.to(device=device, dtype=dtype)

        if self.mid is not None:
            self.mid = self.mid.to(device=device, dtype=dtype)
