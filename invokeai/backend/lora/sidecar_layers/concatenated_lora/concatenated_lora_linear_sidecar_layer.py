import torch

from invokeai.backend.lora.layers.concatenated_lora_layer import ConcatenatedLoRALayer


class ConcatenatedLoRALinearSidecarLayer(torch.nn.Module):
    def __init__(
        self,
        concatenated_lora_layer: ConcatenatedLoRALayer,
        weight: float,
    ):
        super().__init__()

        self._concatenated_lora_layer = concatenated_lora_layer
        self._weight = weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_chunks: list[torch.Tensor] = []
        for lora_layer in self._concatenated_lora_layer.lora_layers:
            x_chunk = torch.nn.functional.linear(input, lora_layer.down)
            if lora_layer.mid is not None:
                x_chunk = torch.nn.functional.linear(x_chunk, lora_layer.mid)
            x_chunk = torch.nn.functional.linear(x_chunk, lora_layer.up, bias=lora_layer.bias)
            x_chunk *= self._weight * lora_layer.scale()
            x_chunks.append(x_chunk)

        # TODO(ryand): Generalize to support concat_axis != 0.
        assert self._concatenated_lora_layer.concat_axis == 0
        x = torch.cat(x_chunks, dim=-1)
        return x

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self._concatenated_lora_layer.to(device=device, dtype=dtype)
        return self
