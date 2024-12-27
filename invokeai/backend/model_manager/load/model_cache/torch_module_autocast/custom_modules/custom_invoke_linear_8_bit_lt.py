import bitsandbytes as bnb
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    autocast_linear_forward_sidecar_patches,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.quantization.bnb_llm_int8 import InvokeLinear8bitLt


class CustomInvokeLinear8bitLt(InvokeLinear8bitLt, CustomModuleMixin):
    def _autocast_forward_with_patches(self, x: torch.Tensor) -> torch.Tensor:
        return autocast_linear_forward_sidecar_patches(self, x, self._patches_and_weights)

    def _autocast_forward(self, x: torch.Tensor) -> torch.Tensor:
        matmul_state = bnb.MatmulLtState()
        matmul_state.threshold = self.state.threshold
        matmul_state.has_fp16_weights = self.state.has_fp16_weights
        matmul_state.use_pool = self.state.use_pool
        matmul_state.is_training = self.training
        # The underlying InvokeInt8Params weight must already be quantized.
        assert self.weight.CB is not None
        matmul_state.CB = cast_to_device(self.weight.CB, x.device)
        matmul_state.SCB = cast_to_device(self.weight.SCB, x.device)

        # weights are cast automatically as Int8Params, but the bias has to be cast manually.
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        # NOTE(ryand): The second parameter should not be needed at all given our expected inference configuration, but
        # it's dtype field must be accessible, even though it's not used. We pass in self.weight even though it could be
        # on the wrong device.
        return bnb.matmul(x, self.weight, bias=cast_to_device(self.bias, x.device), state=matmul_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            return self._autocast_forward_with_patches(x)
        elif self._device_autocasting_enabled:
            return self._autocast_forward(x)
        else:
            return super().forward(x)
