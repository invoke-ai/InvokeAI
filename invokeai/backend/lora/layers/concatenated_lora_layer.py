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

    def __init__(self, lora_layers: Sequence[LoRALayer], concat_axis: int = 0):
        super().__init__(alpha=None, bias=None)

        self.lora_layers = lora_layers
        self.concat_axis = concat_axis

    def rank(self) -> int | None:
        return None

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        # TODO(ryand): Currently, we pass orig_weight=None to the sub-layers. If we want to support sub-layers that
        # require this value, we will need to implement chunking of the original weight tensor here.
        # Note that we must apply the sub-layer scales here.
        layer_weights = [lora_layer.get_weight(None) * lora_layer.scale() for lora_layer in self.lora_layers]  # pyright: ignore[reportArgumentType]
        return torch.cat(layer_weights, dim=self.concat_axis)

    def get_bias(self, orig_bias: torch.Tensor) -> Optional[torch.Tensor]:
        # TODO(ryand): Currently, we pass orig_bias=None to the sub-layers. If we want to support sub-layers that
        # require this value, we will need to implement chunking of the original bias tensor here.
        # Note that we must apply the sub-layer scales here.
        layer_biases: list[torch.Tensor] = []
        for lora_layer in self.lora_layers:
            layer_bias = lora_layer.get_bias(None)
            if layer_bias is not None:
                layer_biases.append(layer_bias * lora_layer.scale())

        if len(layer_biases) == 0:
            return None

        assert len(layer_biases) == len(self.lora_layers)
        return torch.cat(layer_biases, dim=self.concat_axis)

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        for lora_layer in self.lora_layers:
            lora_layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return super().calc_size() + sum(lora_layer.calc_size() for lora_layer in self.lora_layers)
