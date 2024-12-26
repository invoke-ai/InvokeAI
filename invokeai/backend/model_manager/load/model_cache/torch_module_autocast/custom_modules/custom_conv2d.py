import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)


class CustomConv2d(torch.nn.Conv2d, CustomModuleMixin):
    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        bias = cast_to_device(self.bias, input.device)
        return self._conv_forward(input, weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._device_autocasting_enabled:
            return self._autocast_forward(input)
        else:
            return super().forward(input)
