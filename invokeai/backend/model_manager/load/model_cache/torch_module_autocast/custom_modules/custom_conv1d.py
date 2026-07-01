import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.utils import (
    add_nullable_tensors,
)


class CustomConv1d(torch.nn.Conv1d, CustomModuleMixin):
    def _autocast_forward_with_patches(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        bias = cast_to_device(self.bias, input.device)

        # Prepare the original parameters for the patch aggregation.
        orig_params = {"weight": weight, "bias": bias}
        # Filter out None values.
        orig_params = {k: v for k, v in orig_params.items() if v is not None}

        aggregated_param_residuals = self._aggregate_patch_parameters(
            patches_and_weights=self._patches_and_weights,
            orig_params=orig_params,
            device=input.device,
        )

        weight = add_nullable_tensors(weight, aggregated_param_residuals.get("weight", None))
        bias = add_nullable_tensors(bias, aggregated_param_residuals.get("bias", None))
        return self._conv_forward(input, weight, bias)

    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        bias = cast_to_device(self.bias, input.device)
        return self._conv_forward(input, weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            return self._autocast_forward_with_patches(input)
        elif self._device_autocasting_enabled:
            return self._autocast_forward(input)
        else:
            return super().forward(input)
