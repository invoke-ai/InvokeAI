import torch

from invokeai.backend.flux.modules.layers import RMSNorm
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.patches.layers.set_parameter_layer import SetParameterLayer


class CustomFluxRMSNorm(RMSNorm, CustomModuleMixin):
    def _autocast_forward_with_patches(self, x: torch.Tensor) -> torch.Tensor:
        # Currently, CustomFluxRMSNorm layers only support patching with a single SetParameterLayer.
        assert len(self._patches_and_weights) == 1
        patch, _patch_weight = self._patches_and_weights[0]
        assert isinstance(patch, SetParameterLayer)
        assert patch.param_name == "scale"

        scale = cast_to_device(patch.weight, x.device)

        # Apply the patch.
        # NOTE(ryand): Currently, we ignore the patch weight when running as a sidecar. It's not clear how this should
        # be handled.
        return torch.nn.functional.rms_norm(x, scale.shape, scale, eps=1e-6)

    def _autocast_forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = cast_to_device(self.scale, x.device)
        return torch.nn.functional.rms_norm(x, scale.shape, scale, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            return self._autocast_forward_with_patches(x)
        elif self._device_autocasting_enabled:
            return self._autocast_forward(x)
        else:
            return super().forward(x)
