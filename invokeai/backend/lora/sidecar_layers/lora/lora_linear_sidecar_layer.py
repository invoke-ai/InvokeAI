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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.linear(x, self._lora_layer.down)
        if self._lora_layer.mid is not None:
            x = torch.nn.functional.linear(x, self._lora_layer.mid)
        x = torch.nn.functional.linear(x, self._lora_layer.up, bias=self._lora_layer.bias)
        x *= self._weight * self._lora_layer.scale()
        return x

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self._lora_layer.to(device=device, dtype=dtype)
        return self
