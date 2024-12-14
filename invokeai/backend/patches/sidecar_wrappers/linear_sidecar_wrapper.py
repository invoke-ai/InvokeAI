import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.patches.layers.flux_control_lora_layer import FluxControlLoRALayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.sidecar_wrappers.base_sidecar_wrapper import BaseSidecarWrapper


class LinearSidecarWrapper(BaseSidecarWrapper):
    def _lora_forward(self, input: torch.Tensor, lora_layer: LoRALayer, lora_weight: float) -> torch.Tensor:
        """An optimized implementation of the residual calculation for a Linear LoRALayer."""
        x = torch.nn.functional.linear(input, lora_layer.down)
        if lora_layer.mid is not None:
            x = torch.nn.functional.linear(x, lora_layer.mid)
        x = torch.nn.functional.linear(x, lora_layer.up, bias=lora_layer.bias)
        x *= lora_weight * lora_layer.scale()
        return x

    def _concatenated_lora_forward(
        self, input: torch.Tensor, concatenated_lora_layer: ConcatenatedLoRALayer, lora_weight: float
    ) -> torch.Tensor:
        """An optimized implementation of the residual calculation for a Linear ConcatenatedLoRALayer."""
        x_chunks: list[torch.Tensor] = []
        for lora_layer in concatenated_lora_layer.lora_layers:
            x_chunk = torch.nn.functional.linear(input, lora_layer.down)
            if lora_layer.mid is not None:
                x_chunk = torch.nn.functional.linear(x_chunk, lora_layer.mid)
            x_chunk = torch.nn.functional.linear(x_chunk, lora_layer.up, bias=lora_layer.bias)
            x_chunk *= lora_weight * lora_layer.scale()
            x_chunks.append(x_chunk)

        # TODO(ryand): Generalize to support concat_axis != 0.
        assert concatenated_lora_layer.concat_axis == 0
        x = torch.cat(x_chunks, dim=-1)
        return x

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # First, apply the original linear layer.
        # NOTE: We slice the input to match the original weight shape in order to work with FluxControlLoRAs, which
        # change the linear layer's in_features.
        orig_input = input
        input = orig_input[..., : self.orig_module.in_features]
        output = self.orig_module(input)

        # Then, apply layers for which we have optimized implementations.
        unprocessed_patches_and_weights: list[tuple[BaseLayerPatch, float]] = []
        for patch, patch_weight in self._patches_and_weights:
            if isinstance(patch, FluxControlLoRALayer):
                # Note that we use the original input here, not the sliced input.
                output += self._lora_forward(orig_input, patch, patch_weight)
            elif isinstance(patch, LoRALayer):
                output += self._lora_forward(input, patch, patch_weight)
            elif isinstance(patch, ConcatenatedLoRALayer):
                output += self._concatenated_lora_forward(input, patch, patch_weight)
            else:
                unprocessed_patches_and_weights.append((patch, patch_weight))

        # Finally, apply any remaining patches.
        if len(unprocessed_patches_and_weights) > 0:
            aggregated_param_residuals = self._aggregate_patch_parameters(unprocessed_patches_and_weights)
            output += torch.nn.functional.linear(
                input, aggregated_param_residuals["weight"], aggregated_param_residuals.get("bias", None)
            )

        return output
