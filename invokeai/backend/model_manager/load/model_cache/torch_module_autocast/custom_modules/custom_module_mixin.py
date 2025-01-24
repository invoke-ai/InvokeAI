import copy

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.param_shape_utils import get_param_shape
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


class CustomModuleMixin:
    """A mixin class for custom modules that enables device autocasting of module parameters."""

    def __init__(self):
        self._device_autocasting_enabled = False
        self._patches_and_weights: list[tuple[BaseLayerPatch, float]] = []

    def set_device_autocasting_enabled(self, enabled: bool):
        """Pass True to enable autocasting of module parameters to the same device as the input tensor. Pass False to
        disable autocasting, which results in slightly faster execution speed when we know that device autocasting is
        not needed.
        """
        self._device_autocasting_enabled = enabled

    def is_device_autocasting_enabled(self) -> bool:
        """Check if device autocasting is enabled for the module."""
        return self._device_autocasting_enabled

    def add_patch(self, patch: BaseLayerPatch, patch_weight: float):
        """Add a patch to the module."""
        self._patches_and_weights.append((patch, patch_weight))

    def clear_patches(self):
        """Clear all patches from the module."""
        self._patches_and_weights = []

    def get_num_patches(self) -> int:
        """Get the number of patches in the module."""
        return len(self._patches_and_weights)

    def _aggregate_patch_parameters(
        self,
        patches_and_weights: list[tuple[BaseLayerPatch, float]],
        orig_params: dict[str, torch.Tensor],
        device: torch.device | None = None,
    ):
        """Helper function that aggregates the parameters from all patches into a single dict."""
        # HACK(ryand): If the original parameters are in a quantized format whose weights can't be accessed, we replace
        # them with dummy tensors on the 'meta' device. This allows patch layers to access the shapes of the original
        # parameters. But, of course, any sub-layers that need to access the actual values of the parameters will fail.
        for param_name in orig_params.keys():
            param = orig_params[param_name]
            if type(param) is torch.nn.Parameter and type(param.data) is torch.Tensor:
                pass
            elif type(param) is GGMLTensor:
                # Move to device and dequantize here. Doing it in the patch layer can result in redundant casts /
                # dequantizations.
                orig_params[param_name] = param.to(device=device).get_dequantized_tensor()
            else:
                orig_params[param_name] = torch.empty(get_param_shape(param), device="meta")

        params: dict[str, torch.Tensor] = {}

        for patch, patch_weight in patches_and_weights:
            if device is not None:
                # Shallow copy the patch so that we can cast it to the target device without modifying the original patch.
                patch = copy.copy(patch)
                patch.to(device)

            # TODO(ryand): `self` could be a quantized module. Depending on what the patch is doing with the original
            # parameters, this might fail or return incorrect results.
            layer_params = patch.get_parameters(orig_params, weight=patch_weight)

            for param_name, param_weight in layer_params.items():
                if param_name not in params:
                    params[param_name] = param_weight
                else:
                    params[param_name] += param_weight

        return params
