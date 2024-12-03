from typing import TypeVar

import torch

T = TypeVar("T", torch.Tensor, None)


def cast_to_device(t: T, to_device: torch.device, non_blocking: bool = True) -> T:
    if t is None:
        return t
    return t.to(to_device, non_blocking=non_blocking)


def inject_custom_layers_into_module(model: torch.nn.Module):
    def inject_custom_layers(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            module.__class__ = CustomLinear

    model.apply(inject_custom_layers)


class CustomLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        bias = cast_to_device(self.bias, input.device)
        return torch.nn.functional.linear(input, weight, bias)
