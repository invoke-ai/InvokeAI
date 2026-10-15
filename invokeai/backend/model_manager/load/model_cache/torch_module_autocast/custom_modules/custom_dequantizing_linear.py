import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    autocast_linear_forward_sidecar_patches,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.quantization.dequantizing_linear import DequantizingLinear
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear


class CustomDequantizingLinear(DequantizingLinear, CustomModuleMixin):
    """Sidecar patches (LoRA) and device autocast for a ``DequantizingLinear``, subclassed once per scheme below.

    The weight lives in quantized buffers and is dequantized per forward, so a patch cannot be written into
    it: patches ride as sidecars. The scheme's own forward runs unchanged, and a LoRA's low-rank residual is
    added in activation space at full compute precision -- the delta is never rounded into the quantized
    grid. Callers must pass ``force_sidecar_patching=True`` to the ``LayerPatcher`` when a model contains
    these modules (see ``requires_sidecar_patching``): the smart patcher's heuristics look at parameters,
    and these modules have none.

    Device autocast falls out of the base forward, which already moves the buffers to the input's device
    per call, so a partially loaded module left on the CPU streams its quantized payload on demand.
    """

    def _cast_tensor_for_input(self, tensor: torch.Tensor | None, input: torch.Tensor) -> torch.Tensor | None:
        tensor = cast_to_device(tensor, input.device)
        if (
            tensor is not None
            and input.is_floating_point()
            and tensor.is_floating_point()
            and tensor.dtype != input.dtype
        ):
            tensor = tensor.to(dtype=input.dtype)
        return tensor

    def _cast_weight_bias_for_input(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # A meta tensor lets patch types without an optimized sidecar path read the weight's shape but never its
        # quantized values; LoRA layers take the residual path and never get here. Patches are reshaped to this
        # shape, so it has to be the unpacked one, which the weight buffer's is not for nvfp4.
        weight = torch.empty((self.out_features, self.in_features), device="meta")
        bias = self._cast_tensor_for_input(self.bias, input)
        return weight, bias

    def _autocast_forward_with_patches(self, input: torch.Tensor) -> torch.Tensor:
        return autocast_linear_forward_sidecar_patches(self, input, self._patches_and_weights)

    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        return DequantizingLinear.forward(self, input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            return self._autocast_forward_with_patches(input)
        return self._autocast_forward(input)


class CustomInt8ConvrotLinear(CustomDequantizingLinear, Int8ConvrotLinear):
    pass


class CustomNVFP4Linear(CustomDequantizingLinear, NVFP4Linear):
    pass
