from typing import Dict, Optional

import torch

from invokeai.backend.patches.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class DoRALayer(LoRALayerBase):
    """A DoRA layer. As defined in https://arxiv.org/pdf/2402.09353."""

    def __init__(
        self,
        up: torch.Tensor,
        down: torch.Tensor,
        dora_scale: torch.Tensor,
        alpha: float | None,
        bias: Optional[torch.Tensor],
    ):
        super().__init__(alpha, bias)
        self.up = up
        self.down = down
        self.dora_scale = dora_scale

    @classmethod
    def from_state_dict_values(cls, values: Dict[str, torch.Tensor]):
        alpha = cls._parse_alpha(values.get("alpha", None))
        bias = cls._parse_bias(
            values.get("bias_indices", None), values.get("bias_values", None), values.get("bias_size", None)
        )

        layer = cls(
            up=values["lora_up.weight"],
            down=values["lora_down.weight"],
            dora_scale=values["dora_scale"],
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
                "dora_scale",
            },
        )

        return layer

    def _rank(self) -> int:
        return self.down.shape[0]

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        # Note: Variable names (e.g. delta_v) are based on the paper.
        delta_v = self.up.reshape(self.up.shape[0], -1) @ self.down.reshape(self.down.shape[0], -1)
        delta_v = delta_v.reshape(orig_weight.shape)

        # TODO(ryand): Should alpha be applied to delta_v here rather than the final diff?
        # TODO(ryand): I expect this to fail if the original weight is BnB Quantized. This class shouldn't have to worry
        # about that, but we should add a clear error message further up the stack.

        # At this point, out_weight is the unnormalized direction matrix.
        out_weight = orig_weight + delta_v

        # TODO(ryand): Simplify this logic.
        direction_norm = (
            out_weight.transpose(0, 1)
            .reshape(out_weight.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(out_weight.shape[1], *[1] * (out_weight.dim() - 1))
            .transpose(0, 1)
        )

        out_weight *= self.dora_scale / direction_norm

        return out_weight - orig_weight

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        self.up = self.up.to(device=device, dtype=dtype)
        self.down = self.down.to(device=device, dtype=dtype)
        self.dora_scale = self.dora_scale.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensors_size([self.up, self.down, self.dora_scale])
