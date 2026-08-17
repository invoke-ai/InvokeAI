import torch

from invokeai.backend.minimax_h3.int8_convrot import Int8ConvrotLinear
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    autocast_linear_forward_sidecar_patches,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.patches.layers.param_shape_utils import get_param_shape


class CustomInt8ConvrotLinear(Int8ConvrotLinear, CustomModuleMixin):
    """Custom wrapper for ``Int8ConvrotLinear`` enabling sidecar patches (LoRA) and device autocast.

    The base module stores an int8 weight plus a per-output-channel scale as buffers and
    dequantizes per forward. Direct (in-place) weight patching is impossible on int8 storage,
    so LoRAs are applied as sidecar patches: the base module's own quantized forward runs
    unchanged, and the low-rank residual is added in activation space at full compute
    precision — the delta is never rounded into the int8 grid. Callers must pass
    ``force_sidecar_patching=True`` to the ``LayerPatcher`` when a model contains these
    modules (the smart patcher's device heuristics would otherwise try direct patching).

    Device autocast falls out of the base implementation: ``Int8ConvrotLinear.forward``
    already moves its buffers to the input's device per call, so a partially-loaded module
    left on the CPU streams its (int8, i.e. half-bf16-sized) weight on demand.
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
        # Substitute a meta tensor for the weight so patch types without an optimized sidecar
        # path can read its *shape* but never the quantized int8 values (same convention as
        # CustomInvokeLinear8bitLt). LoRA layers take the optimized residual path and never
        # get here.
        weight = torch.empty(get_param_shape(self.weight), device="meta")
        bias = self._cast_tensor_for_input(self.bias, input)
        return weight, bias

    def _autocast_forward_with_patches(self, input: torch.Tensor) -> torch.Tensor:
        return autocast_linear_forward_sidecar_patches(self, input, self._patches_and_weights)

    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        # The base forward dequantizes into the input's device and dtype per call, which is
        # already autocast-safe.
        return Int8ConvrotLinear.forward(self, input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            return self._autocast_forward_with_patches(input)
        return self._autocast_forward(input)
