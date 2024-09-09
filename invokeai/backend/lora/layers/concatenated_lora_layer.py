from typing import List, Optional

import torch

from invokeai.backend.lora.layers.lora_layer_base import LoRALayerBase


class ConcatenatedLoRALayer(LoRALayerBase):
    """A LoRA layer that is composed of multiple LoRA layers concatenated along a specified axis.

    This class was created to handle a special case with FLUX LoRA models. In the BFL FLUX model format, the attention
    Q, K, V matrices are concatenated along the first dimension. In the diffusers LoRA format, the Q, K, V matrices are
    stored as separate tensors. This class enables diffusers LoRA layers to be used in BFL FLUX models.
    """

    def __init__(self, layer_key: str, lora_layers: List[LoRALayerBase], concat_axis: int = 0):
        # Note: We pass values={} to the base class, because the values are handled by the individual LoRA layers.
        super().__init__(layer_key, values={})

        self._lora_layers = lora_layers
        self._concat_axis = concat_axis

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        # TODO(ryand): Currently, we pass orig_weight=None to the sub-layers. If we want to support sub-layers that
        # require this value, we will need to implement chunking of the original weight tensor here.
        layer_weights = [lora_layer.get_weight(None) for lora_layer in self._lora_layers]  # pyright: ignore[reportArgumentType]
        return torch.cat(layer_weights, dim=self._concat_axis)

    def get_bias(self, orig_bias: torch.Tensor) -> Optional[torch.Tensor]:
        # TODO(ryand): Currently, we pass orig_bias=None to the sub-layers. If we want to support sub-layers that
        # require this value, we will need to implement chunking of the original bias tensor here.
        layer_biases = [lora_layer.get_bias(None) for lora_layer in self._lora_layers]  # pyright: ignore[reportArgumentType]
        layer_bias_is_none = [layer_bias is None for layer_bias in layer_biases]
        if any(layer_bias_is_none):
            assert all(layer_bias_is_none)
            return None

        # Ignore the type error, because we have just verified that all layer biases are non-None.
        return torch.cat(layer_biases, dim=self._concat_axis)

    def calc_size(self) -> int:
        return sum(lora_layer.calc_size() for lora_layer in self._lora_layers)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        for lora_layer in self._lora_layers:
            lora_layer.to(device=device, dtype=dtype)
