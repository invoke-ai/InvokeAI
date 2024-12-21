from typing import TypeVar

import torch

T = TypeVar("T", torch.Tensor, None, torch.Tensor | None)

# This file contains custom torch.nn.Module classes that support streaming of weights to the target device.
# Each class sub-classes the original module type that is is replacing, so the following properties are preserved:
# - isinstance(m, torch.nn.OrginalModule) should still work.
# - Patching the weights (e.g. for LoRA) should still work if non-quantized.


def cast_to_device(t: T, to_device: torch.device) -> T:
    if t is None:
        return t

    if t.device.type != to_device.type:
        return t.to(to_device)
    return t


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
