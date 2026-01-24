import torch
import torch.nn.functional as F

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)


class CustomLayerNorm(torch.nn.LayerNorm, CustomModuleMixin):
    """Custom wrapper for torch.nn.LayerNorm that supports device autocasting for partial model loading."""

    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device) if self.weight is not None else None
        bias = cast_to_device(self.bias, input.device) if self.bias is not None else None
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            raise RuntimeError("LayerNorm layers do not support patches")

        if self._device_autocasting_enabled:
            return self._autocast_forward(input)
        else:
            return super().forward(input)
