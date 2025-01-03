from typing import Dict, Optional

import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.patches.layers.lora_layer_base import LoRALayerBase


class IA3Layer(LoRALayerBase):
    """IA3 Layer

    Example model for testing this layer type: https://civitai.com/models/123930/gwendolyn-tennyson-ben-10-ia3
    """

    def __init__(self, weight: torch.Tensor, on_input: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__(alpha=None, bias=bias)
        self.weight = weight
        self.on_input = on_input

    def _rank(self) -> int | None:
        return None

    @classmethod
    def from_state_dict_values(
        cls,
        values: Dict[str, torch.Tensor],
    ):
        bias = cls._parse_bias(
            values.get("bias_indices", None), values.get("bias_values", None), values.get("bias_size", None)
        )
        layer = cls(
            weight=values["weight"],
            on_input=values["on_input"],
            bias=bias,
        )
        cls.warn_on_unhandled_keys(
            values=values,
            handled_keys={
                # Default keys.
                "bias_indices",
                "bias_values",
                "bias_size",
                # Layer-specific keys.
                "weight",
                "on_input",
            },
        )
        return layer

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if not self.on_input:
            weight = weight.reshape(-1, 1)
        return cast_to_device(orig_weight, weight.device) * weight

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device, dtype)
        self.weight = self.weight.to(device, dtype)
        self.on_input = self.on_input.to(device, dtype)
