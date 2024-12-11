from typing import TypeVar

import torch

T = TypeVar("T", torch.Tensor, None, torch.Tensor | None)

# Properties to preserve:
# - isinstance(m, torch.nn.Linear) should still work
# - patching the weights should still work if non-quantized


def cast_to_device(t: T, to_device: torch.device, non_blocking: bool = True) -> T:
    if t is None:
        return t

    if t.device.type != to_device.type:
        return t.to(to_device, non_blocking=non_blocking)
    return t


def inject_custom_layers_into_module(model: torch.nn.Module):
    def inject_custom_layers(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            module.__class__ = CustomLinear
        elif isinstance(module, torch.nn.Conv1d):
            module.__class__ = CustomConv1d
        elif isinstance(module, torch.nn.Conv2d):
            module.__class__ = CustomConv2d
        elif isinstance(module, torch.nn.GroupNorm):
            module.__class__ = CustomGroupNorm
        elif isinstance(module, torch.nn.Embedding):
            module.__class__ = CustomEmbedding

    model.apply(inject_custom_layers)


class CustomLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        bias = cast_to_device(self.bias, input.device)
        return torch.nn.functional.linear(input, weight, bias)


class CustomConv1d(torch.nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        bias = cast_to_device(self.bias, input.device)
        return self._conv_forward(input, weight, bias)


class CustomConv2d(torch.nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        bias = cast_to_device(self.bias, input.device)
        return self._conv_forward(input, weight, bias)


class CustomGroupNorm(torch.nn.GroupNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        bias = cast_to_device(self.bias, input.device)
        return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)


class CustomEmbedding(torch.nn.Embedding):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        return torch.nn.functional.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
