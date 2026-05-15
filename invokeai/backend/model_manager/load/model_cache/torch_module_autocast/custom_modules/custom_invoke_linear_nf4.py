import copy

import bitsandbytes as bnb
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    autocast_linear_forward_sidecar_patches,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.patches.layers.param_shape_utils import get_param_shape
from invokeai.backend.quantization.bnb_nf4 import InvokeLinearNF4
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


class CustomInvokeLinearNF4(InvokeLinearNF4, CustomModuleMixin):
    def _cast_tensor_for_input(self, tensor: torch.Tensor | None, input: torch.Tensor) -> torch.Tensor | None:
        tensor = cast_to_device(tensor, input.device)
        if (
            tensor is not None
            and input.is_floating_point()
            and tensor.is_floating_point()
            and not isinstance(tensor, GGMLTensor)
            and tensor.dtype != input.dtype
        ):
            tensor = tensor.to(dtype=input.dtype)
        return tensor

    def _cast_weight_bias_for_input(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The NF4 weight is a Params4bit whose .shape reports the *packed-byte* layout, not the logical
        # (out_features, in_features) shape. We hand patches a meta-device tensor with the correct
        # logical shape so that shape-only patches (LoRA, LoHA, MergedLayerPatch over LoRA, ...) work.
        # Patches that read the original weight values (e.g. SetParameterLayer, DoRA) are not supported
        # on NF4-quantized modules.
        weight = torch.empty(get_param_shape(self.weight), device="meta")
        bias = self._cast_tensor_for_input(self.bias, input)
        return weight, bias

    def _autocast_forward_with_patches(self, x: torch.Tensor) -> torch.Tensor:
        return autocast_linear_forward_sidecar_patches(self, x, self._patches_and_weights)

    def _autocast_forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # avoid this side effect we make a shallow copy of the original quant_state so that we can restore it. Fixing
        # this properly would require more invasive changes to the bitsandbytes library.

        # Make a shallow copy of the quant_state so that we can undo the in-place modification that occurs when casting
        # to a new device.
        old_quant_state = copy.copy(self.weight.quant_state)
        weight = cast_to_device(self.weight, x.device)
        self.weight.quant_state = old_quant_state

        # For some reason, the quant_state.to(...) implementation fails to cast the quant_state.code field. We do this
        # manually here.
        weight.quant_state.code = cast_to_device(weight.quant_state.code, x.device)

        bias = cast_to_device(self.bias, x.device)
        return bnb.matmul_4bit(x, weight.t(), bias=bias, quant_state=weight.quant_state).to(inp_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            return self._autocast_forward_with_patches(x)
        elif self._device_autocasting_enabled:
            return self._autocast_forward(x)
        else:
            return super().forward(x)
