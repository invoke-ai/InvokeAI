from typing import Optional, Sequence

import torch

from invokeai.backend.lora.layers.lora_layer import LoRALayer
from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase


class ConcatenatedLoRALayer(LoRALayerBase):
    """A LoRA layer that is composed of multiple LoRA layers concatenated along a specified axis.

    This class was created to handle a special case with FLUX LoRA models. In the BFL FLUX model format, the attention
    Q, K, V matrices are concatenated along the first dimension. In the diffusers LoRA format, the Q, K, V matrices are
    stored as separate tensors. This class enables diffusers LoRA layers to be used in BFL FLUX models.
    """

    def __init__(self, lora_layers: Sequence[LoRALayer], offsets: Sequence[tuple[int, int]]):
        super().__init__(alpha=None, bias=None)

        self.lora_layers = lora_layers
        self.offsets = offsets

    def rank(self) -> int | None:
        return None

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        # TODO(ryand): Currently, we pass orig_weight=None to the sub-layers. If we want to support sub-layers that
        # require this value, we will need to implement chunking of the original weight tensor here.

        weight = torch.zeros_like(orig_weight)
        for lora_layer, offset in zip(self.lora_layers, self.offsets, strict=True):
            # Note that we must apply the sub-layer scales here.
            sub_weight = lora_layer.get_weight(None) * lora_layer.scale()
            weight[offset[0] : offset[0] + sub_weight.shape[0], offset[1] : offset[1] + sub_weight.shape[1]] = (
                sub_weight
            )
        return weight

    def get_bias(self, orig_bias: torch.Tensor) -> Optional[torch.Tensor]:
        # TODO(ryand): Currently, we pass orig_bias=None to the sub-layers. If we want to support sub-layers that
        # require this value, we will need to implement chunking of the original bias tensor here.

        layer_biases: list[torch.Tensor | None] = []
        # Note that we must apply the sub-layer scales here.
        for lora_layer in self.lora_layers:
            layer_bias = lora_layer.get_bias(None)
            if layer_bias is not None:
                layer_biases.append(layer_bias * lora_layer.scale())
            else:
                layer_biases.append(None)

        if all(bias is None for bias in layer_biases):
            return None

        bias = torch.zeros_like(orig_bias)
        for layer_bias, offset in zip(layer_biases, self.offsets, strict=True):
            if layer_bias is not None:
                bias[offset[0] : offset[0] + layer_bias.shape[0]] = layer_bias
        return bias

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        for lora_layer in self.lora_layers:
            lora_layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return super().calc_size() + sum(lora_layer.calc_size() for lora_layer in self.lora_layers)
