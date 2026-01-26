import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.utils import (
    add_nullable_tensors,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


class CustomConv2d(torch.nn.Conv2d, CustomModuleMixin):
    def _cast_tensor_for_input(self, tensor: torch.Tensor | None, input: torch.Tensor) -> torch.Tensor | None:
        tensor = cast_to_device(tensor, input.device)
        if (
            tensor is not None
            and input.is_floating_point()
            and tensor.is_floating_point()
            and not isinstance(tensor, GGMLTensor)
            and tensor.dtype != input.dtype
        ):
            tensor = tensor.to(dtype=input.dtype)
        return tensor

    def _autocast_forward_with_patches(self, input: torch.Tensor) -> torch.Tensor:
        weight = self._cast_tensor_for_input(self.weight, input)
        bias = self._cast_tensor_for_input(self.bias, input)

        # Prepare the original parameters for the patch aggregation.
        orig_params = {"weight": weight, "bias": bias}
        # Filter out None values.
        orig_params = {k: v for k, v in orig_params.items() if v is not None}

        aggregated_param_residuals = self._aggregate_patch_parameters(
            patches_and_weights=self._patches_and_weights,
            orig_params=orig_params,
            device=input.device,
        )

        residual_weight = self._cast_tensor_for_input(aggregated_param_residuals.get("weight", None), input)
        residual_bias = self._cast_tensor_for_input(aggregated_param_residuals.get("bias", None), input)
        weight = add_nullable_tensors(weight, residual_weight)
        bias = add_nullable_tensors(bias, residual_bias)
        return self._conv_forward(input, weight, bias)

    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self._cast_tensor_for_input(self.weight, input)
        bias = self._cast_tensor_for_input(self.bias, input)
        return self._conv_forward(input, weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            return self._autocast_forward_with_patches(input)
        elif self._device_autocasting_enabled:
            return self._autocast_forward(input)
        elif (
            input.is_floating_point()
            and (
                (
                    self.weight.is_floating_point()
                    and not isinstance(self.weight, GGMLTensor)
                    and self.weight.dtype != input.dtype
                )
                or (
                    self.bias is not None
                    and self.bias.is_floating_point()
                    and not isinstance(self.bias, GGMLTensor)
                    and self.bias.dtype != input.dtype
                )
            )
        ):
            weight = self._cast_tensor_for_input(self.weight, input)
            bias = self._cast_tensor_for_input(self.bias, input)
            return self._conv_forward(input, weight, bias)
        else:
            return super().forward(input)
