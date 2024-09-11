import torch

from invokeai.backend.lora.layers.lora_layer import LoRALayer


class LoRALinearSidecarLayer(torch.nn.Module):
    def __init__(
        self,
        lora_layer: LoRALayer,
        weight: float,
    ):
        super().__init__()

        self._lora_layer = lora_layer
        self._weight = weight

    def to(self, device: torch.device, dtype: torch.dtype):
        self._lora_layer.to(device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.linear(x, self._lora_layer.down)
        if self._lora_layer.mid is not None:
            x = torch.nn.functional.linear(x, self._lora_layer.mid)
        x = torch.nn.functional.linear(x, self._lora_layer.up, bias=self._lora_layer.bias)
        scale = self._lora_layer.alpha / self._lora_layer.rank if self._lora_layer.alpha is not None else 1.0
        x *= self._weight * scale
        return x
