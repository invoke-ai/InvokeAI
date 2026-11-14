import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    autocast_linear_forward_sidecar_patches,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.quantization.dequantizing_linear import DequantizingLinear
from invokeai.backend.quantization.fp8_rowwise import Fp8Linear
from invokeai.backend.quantization.int8_convrot import Int4ConvrotLinear, Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear


def _reads_weight_values(patch: BaseLayerPatch) -> bool:
    if isinstance(patch, SwappedHalvesLayerPatch):
        return _reads_weight_values(patch.layer)
    if isinstance(patch, MergedLayerPatch):
        return any(_reads_weight_values(layer) for layer in patch.lora_layers)
    # Only DoRA reads the original weight's values; every other layer applied to a Linear reads its shape.
    return isinstance(patch, DoRALayer)


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
        # Most patch types only read the weight's shape, so they get a meta tensor and no dequantization is paid
        # for them. Patches are reshaped to this shape, so it has to be the unpacked one, which the weight buffer's
        # is not for nvfp4 or int4. A patch that reads values (DoRA) gets the dequantized weight instead, built
        # for this call only: it is neither stored on the module nor cached.
        if any(_reads_weight_values(patch) for patch, _ in self._patches_and_weights):
            weight = self._dequantized_weight(input.device, input.dtype)
        else:
            weight = torch.empty((self.out_features, self.in_features), device="meta")
        bias = self._cast_tensor_for_input(self.bias, input)
        return weight, bias

    def _autocast_forward_with_patches(self, input: torch.Tensor) -> torch.Tensor:
        return autocast_linear_forward_sidecar_patches(self, input, self._patches_and_weights)

    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        # The scheme's own forward: the next one after this class in the subclass's MRO, e.g.
        # `Int8ConvrotLinear.forward`, which is where int8 compute decides between its kernels and the dequantized
        # path. Naming `DequantizingLinear.forward` here would skip that decision for every wrapped layer.
        return super().forward(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            return self._autocast_forward_with_patches(input)
        return self._autocast_forward(input)


class CustomInt8ConvrotLinear(CustomDequantizingLinear, Int8ConvrotLinear):
    pass


class CustomInt4ConvrotLinear(CustomDequantizingLinear, Int4ConvrotLinear):
    pass


class CustomNVFP4Linear(CustomDequantizingLinear, NVFP4Linear):
    pass


class CustomFp8Linear(CustomDequantizingLinear, Fp8Linear):
    # Wrapped for its sidecar patches only. Streaming these weights would let the cache hold back the margin its
    # reservation asks for, and on a 24 GB card that costs the Ideogram 4 fp8 pair its full residency: measured at
    # 1024x1024, 96.7% resident and 43 s per image instead of 100% and 34 s.
    streams_weights = False
