import copy
from typing import TypeVar

import bitsandbytes as bnb
import torch

from invokeai.backend.quantization.bnb_llm_int8 import InvokeLinear8bitLt
from invokeai.backend.quantization.bnb_nf4 import InvokeLinearNF4

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


class CustomInvokeLinear8bitLt(InvokeLinear8bitLt):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matmul_state = bnb.MatmulLtState()
        matmul_state.threshold = self.state.threshold
        matmul_state.has_fp16_weights = self.state.has_fp16_weights
        matmul_state.use_pool = self.state.use_pool
        matmul_state.is_training = self.training
        # The underlying InvokeInt8Params weight must already be quantized.
        assert self.weight.CB is not None
        matmul_state.CB = cast_to_device(self.weight.CB, x.device)
        matmul_state.SCB = cast_to_device(self.weight.SCB, x.device)

        # weights are cast automatically as Int8Params, but the bias has to be cast manually.
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        # NOTE(ryand): The second parameter should not be needed at all given our expected inference configuration, but
        # it's dtype field must be accessible, even though it's not used. We pass in self.weight even though it could be
        # on the wrong device.
        return bnb.matmul(x, self.weight, bias=cast_to_device(self.bias, x.device), state=matmul_state)


class CustomInvokeLinearNF4(InvokeLinearNF4):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bnb.nn.modules.fix_4bit_weight_quant_state_from_module(self)

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)

        # HACK(ryand): Casting self.weight to the device also casts the self.weight.quant_state in-place (i.e. it
        # does not follow the tensor semantics of returning a new copy when converting to a different device). This
        # means that quant_state elements that started on the CPU would be left on the GPU, which we don't want. To
        # avoid this side effect we make a shallow copy of the original state dict so that we can restore it. Fixing
        # this properly would require more invasive changes to the bitsandbytes library.

        # Make a shallow copy of the quant_state so that we can undo the in-place modification that occurs when casting
        # to a new device.
        old_quant_state = copy.copy(self.weight.quant_state)
        weight = cast_to_device(self.weight, x.device)
        self.weight.quant_state = old_quant_state

        bias = cast_to_device(self.bias, x.device)
        return bnb.matmul_4bit(x, weight.t(), bias=bias, quant_state=weight.quant_state).to(inp_dtype)
