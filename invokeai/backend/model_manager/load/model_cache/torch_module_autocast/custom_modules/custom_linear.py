import copy

import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.flux_control_lora_layer import FluxControlLoRALayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.quantization.fp8_scaled import (
    FP8_DTYPE,
    dequantize_weight,
    device_supports_fp8_matmul,
    is_fp8_matmul_enabled,
    scaled_mm_linear,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor


def linear_lora_forward(input: torch.Tensor, lora_layer: LoRALayer, lora_weight: float) -> torch.Tensor:
    """An optimized implementation of the residual calculation for a sidecar linear LoRALayer."""
    # up matrix and down matrix have different ranks so we can't simply multiply them
    if lora_layer.up.shape[1] != lora_layer.down.shape[0]:
        x = torch.nn.functional.linear(input, lora_layer.get_weight(lora_weight), bias=lora_layer.bias)
        x *= lora_weight * lora_layer.scale()
        return x

    x = torch.nn.functional.linear(input, lora_layer.down)
    if lora_layer.mid is not None:
        x = torch.nn.functional.linear(x, lora_layer.mid)
    x = torch.nn.functional.linear(x, lora_layer.up, bias=lora_layer.bias)
    x *= lora_weight * lora_layer.scale()
    return x


def autocast_linear_forward_sidecar_patches(
    orig_module: torch.nn.Linear, input: torch.Tensor, patches_and_weights: list[tuple[BaseLayerPatch, float]]
) -> torch.Tensor:
    """A function that runs a linear layer (quantized or non-quantized) with sidecar patches for a linear layer.
    Compatible with both quantized and non-quantized Linear layers.
    """
    # First, apply the original linear layer.
    # NOTE: We slice the input to match the original weight shape in order to work with FluxControlLoRAs, which
    # change the linear layer's in_features.
    orig_input = input
    input = orig_input[..., : orig_module.in_features]
    output = orig_module._autocast_forward(input)

    # Then, apply layers for which we have optimized implementations.
    unprocessed_patches_and_weights: list[tuple[BaseLayerPatch, float]] = []
    for patch, patch_weight in patches_and_weights:
        # Shallow copy the patch so that we can cast it to the target device without modifying the original patch.
        patch = copy.copy(patch)
        patch.to(input.device)

        if isinstance(patch, FluxControlLoRALayer):
            # Note that we use the original input here, not the sliced input.
            output += linear_lora_forward(orig_input, patch, patch_weight)
        elif isinstance(patch, LoRALayer):
            output += linear_lora_forward(input, patch, patch_weight)
        else:
            unprocessed_patches_and_weights.append((patch, patch_weight))

    # Finally, apply any remaining patches.
    if len(unprocessed_patches_and_weights) > 0:
        weight, bias = orig_module._cast_weight_bias_for_input(input)
        # Prepare the original parameters for the patch aggregation.
        orig_params = {"weight": weight, "bias": bias}
        # Filter out None values.
        orig_params = {k: v for k, v in orig_params.items() if v is not None}

        aggregated_param_residuals = orig_module._aggregate_patch_parameters(
            unprocessed_patches_and_weights, orig_params=orig_params, device=input.device
        )
        residual_weight = orig_module._cast_tensor_for_input(aggregated_param_residuals["weight"], input)
        residual_bias = orig_module._cast_tensor_for_input(aggregated_param_residuals.get("bias", None), input)
        assert residual_weight is not None
        output += torch.nn.functional.linear(input, residual_weight, residual_bias)

    return output


class CustomLinear(torch.nn.Linear, CustomModuleMixin):
    def _cast_tensor_for_input(self, tensor: torch.Tensor | None, input: torch.Tensor) -> torch.Tensor | None:
        tensor = cast_to_device(tensor, input.device)
        if (
            tensor is not None
            and input.is_floating_point()
            and tensor.is_floating_point()
            and not isinstance(tensor, (GGMLTensor, SDNQTensor))
            and tensor.dtype != input.dtype
        ):
            tensor = tensor.to(dtype=input.dtype)
        return tensor

    def _cast_weight_bias_for_input(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = cast_to_device(self.weight, input.device)
        assert weight is not None
        if weight.dtype == FP8_DTYPE:
            # A scaled fp8 weight must never be cast with a plain `.to(dtype)`: that drops the
            # scale and yields a wrongly-scaled weight. `weight_scale` is None for fp8 weights that
            # carry no scale (e.g. produced by fp8_storage layerwise casting), where a plain cast
            # is correct.
            weight = dequantize_weight(weight, self._fp8_weight_scale(input.device), input.dtype)
        else:
            weight = self._cast_tensor_for_input(weight, input)
        bias = self._cast_tensor_for_input(self.bias, input)
        assert weight is not None
        return weight, bias

    def _fp8_weight_scale(self, device: torch.device) -> torch.Tensor | None:
        scale = getattr(self, "weight_scale", None)
        return cast_to_device(scale, device) if scale is not None else None

    def _can_use_fp8_matmul(self, input: torch.Tensor) -> bool:
        """Whether this forward may run on the fp8 tensor cores.

        Every condition is a hard requirement of `torch._scaled_mm` or an explicit instruction from
        the checkpoint producer; failing any of them falls back to the dequantized path rather than
        raising, because a mid-generation failure is far worse than losing the speedup.
        """
        if not is_fp8_matmul_enabled():
            return False
        if self.weight.dtype != FP8_DTYPE or not input.is_floating_point():
            return False
        if getattr(self, "_fp8_full_precision_matmul", False):
            # The producer measured this layer as unsafe to multiply in fp8.
            return False
        if not device_supports_fp8_matmul(input.device):
            return False
        if self.in_features % 16 != 0 or self.out_features % 16 != 0:
            return False
        # Partial loading may leave the weight on the CPU; _scaled_mm needs both operands co-located.
        return self.weight.device == input.device

    def _maybe_fp8_forward(self, input: torch.Tensor) -> torch.Tensor | None:
        """Run the fp8 matmul if this module and input allow it, else None so callers fall back."""
        if not self._can_use_fp8_matmul(input):
            return None
        return scaled_mm_linear(
            input,
            self.weight,
            self._fp8_weight_scale(input.device),
            cast_to_device(self.bias, input.device),
            input_scale=cast_to_device(getattr(self, "input_scale", None), input.device),
        )

    def _autocast_forward_with_patches(self, input: torch.Tensor) -> torch.Tensor:
        return autocast_linear_forward_sidecar_patches(self, input, self._patches_and_weights)

    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self._maybe_fp8_forward(input)
        if out is not None:
            return out
        weight, bias = self._cast_weight_bias_for_input(input)
        return torch.nn.functional.linear(input, weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            return self._autocast_forward_with_patches(input)
        # Checked before the autocasting branch on purpose: `apply_custom_layers_to_model` leaves
        # device autocasting *disabled* whenever the model is fully resident, so a check that only
        # lived in `_autocast_forward` would silently never run in the common case.
        fp8_out = self._maybe_fp8_forward(input)
        if fp8_out is not None:
            return fp8_out
        if self._device_autocasting_enabled:
            return self._autocast_forward(input)
        elif input.is_floating_point() and (
            (self.weight.is_floating_point() and self.weight.dtype != input.dtype)
            or (
                self.bias is not None
                and self.bias.is_floating_point()
                and not isinstance(self.bias, (GGMLTensor, SDNQTensor))
                and self.bias.dtype != input.dtype
            )
        ):
            weight, bias = self._cast_weight_bias_for_input(input)
            return torch.nn.functional.linear(input, weight, bias)
        else:
            return super().forward(input)
