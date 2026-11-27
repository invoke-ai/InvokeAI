import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.util.cast_to_device import cast_to_device


class CustomGroupNorm(torch.nn.GroupNorm, CustomModuleMixin):
    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        bias = cast_to_device(self.bias, input.device)
        return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            raise RuntimeError("GroupNorm layers do not support patches")

        if self._device_autocasting_enabled:
            return self._autocast_forward(input)
        else:
            return super().forward(input)
