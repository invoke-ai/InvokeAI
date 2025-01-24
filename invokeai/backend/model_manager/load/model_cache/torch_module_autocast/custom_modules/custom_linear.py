import copy

import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)
from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.flux_control_lora_layer import FluxControlLoRALayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer


def linear_lora_forward(input: torch.Tensor, lora_layer: LoRALayer, lora_weight: float) -> torch.Tensor:
    """An optimized implementation of the residual calculation for a sidecar linear LoRALayer."""
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
        # Prepare the original parameters for the patch aggregation.
        orig_params = {"weight": orig_module.weight, "bias": orig_module.bias}
        # Filter out None values.
        orig_params = {k: v for k, v in orig_params.items() if v is not None}

        aggregated_param_residuals = orig_module._aggregate_patch_parameters(
            unprocessed_patches_and_weights, orig_params=orig_params, device=input.device
        )
        output += torch.nn.functional.linear(
            input, aggregated_param_residuals["weight"], aggregated_param_residuals.get("bias", None)
        )

    return output


class CustomLinear(torch.nn.Linear, CustomModuleMixin):
    def _autocast_forward_with_patches(self, input: torch.Tensor) -> torch.Tensor:
        return autocast_linear_forward_sidecar_patches(self, input, self._patches_and_weights)

    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = cast_to_device(self.weight, input.device)
        bias = cast_to_device(self.bias, input.device)
        return torch.nn.functional.linear(input, weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(self._patches_and_weights) > 0:
            return self._autocast_forward_with_patches(input)
        elif self._device_autocasting_enabled:
            return self._autocast_forward(input)
        else:
            return super().forward(input)
