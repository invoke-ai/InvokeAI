import torch
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)


class CustomDiffusersRMSNorm(DiffusersRMSNorm, CustomModuleMixin):
    """Custom wrapper for diffusers RMSNorm that supports device autocasting for partial model loading."""

    def _autocast_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, hidden_states.device) if self.weight is not None else None
        bias = cast_to_device(self.bias, hidden_states.device) if self.bias is not None else None

        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if weight is not None:
            # convert into half-precision if necessary
            if weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(weight.dtype)
            hidden_states = hidden_states * weight
            if bias is not None:
                hidden_states = hidden_states + bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            raise RuntimeError("DiffusersRMSNorm layers do not support patches")

        if self._device_autocasting_enabled:
            return self._autocast_forward(hidden_states)
        else:
            return super().forward(hidden_states)
