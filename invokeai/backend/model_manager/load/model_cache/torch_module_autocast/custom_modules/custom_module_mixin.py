import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch


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
        self, patches_and_weights: list[tuple[BaseLayerPatch, float]]
    ) -> dict[str, torch.Tensor]:
        """Helper function that aggregates the parameters from all patches into a single dict."""
        params: dict[str, torch.Tensor] = {}

        for patch, patch_weight in patches_and_weights:
            # TODO(ryand): `self` could be a quantized module. Depending on what the patch is doing with the original
            # parameters, this might fail or return incorrect results.
            layer_params = patch.get_parameters(dict(self.named_parameters(recurse=False)), weight=patch_weight)  # type: ignore

            for param_name, param_weight in layer_params.items():
                if param_name not in params:
                    params[param_name] = param_weight
                else:
                    params[param_name] += param_weight

        return params
