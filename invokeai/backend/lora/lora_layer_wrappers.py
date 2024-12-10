import torch

from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer
from invokeai.backend.lora.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.lora.layers.lora_layer import LoRALayer


class LoRASidecarWrapper(torch.nn.Module):
    def __init__(self, orig_module: torch.nn.Module, lora_layers: list[AnyLoRALayer], lora_weights: list[float]):
        super().__init__()
        self._orig_module = orig_module
        self._lora_layers = lora_layers
        self._lora_weights = lora_weights

    @property
    def orig_module(self) -> torch.nn.Module:
        return self._orig_module

    def add_lora_layer(self, lora_layer: AnyLoRALayer, lora_weight: float):
        self._lora_layers.append(lora_layer)
        self._lora_weights.append(lora_weight)

    @torch.no_grad()
    def _get_lora_patched_parameters(
        self, orig_params: dict[str, torch.Tensor], lora_layers: list[AnyLoRALayer], lora_weights: list[float]
    ) -> dict[str, torch.Tensor]:
        params: dict[str, torch.Tensor] = {}
        for lora_layer, lora_weight in zip(lora_layers, lora_weights, strict=True):
            layer_params = lora_layer.get_parameters(self._orig_module)
            for param_name, param_weight in layer_params.items():
                if orig_params[param_name].shape != param_weight.shape:
                    param_weight = param_weight.reshape(orig_params[param_name].shape)

                if param_name not in params:
                    params[param_name] = param_weight * (lora_layer.scale() * lora_weight)
                else:
                    params[param_name] += param_weight * (lora_layer.scale() * lora_weight)

        return params


class LoRALinearWrapper(LoRASidecarWrapper):
    def _lora_linear_forward(self, input: torch.Tensor, lora_layer: LoRALayer, lora_weight: float) -> torch.Tensor:
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
        # Split the LoRA layers into those that have optimized implementations and those that don't.
        optimized_layer_types = (LoRALayer, ConcatenatedLoRALayer)
        optimized_layers = [
            (layer, weight)
            for layer, weight in zip(self._lora_layers, self._lora_weights, strict=True)
            if isinstance(layer, optimized_layer_types)
        ]
        non_optimized_layers = [
            (layer, weight)
            for layer, weight in zip(self._lora_layers, self._lora_weights, strict=True)
            if not isinstance(layer, optimized_layer_types)
        ]

        # First, calculate the residual for LoRA layers for which there is an optimized implementation.
        residual = None
        for lora_layer, lora_weight in optimized_layers:
            if isinstance(lora_layer, LoRALayer):
                added_residual = self._lora_linear_forward(input, lora_layer, lora_weight)
            elif isinstance(lora_layer, ConcatenatedLoRALayer):
                added_residual = self._concatenated_lora_forward(input, lora_layer, lora_weight)
            else:
                raise ValueError(f"Unsupported LoRA layer type: {type(lora_layer)}")

            if residual is None:
                residual = added_residual
            else:
                residual += added_residual

        # Next, calculate the residuals for the LoRA layers for which there is no optimized implementation.
        if non_optimized_layers:
            unoptimized_layers, unoptimized_weights = zip(*non_optimized_layers, strict=True)
            params = self._get_lora_patched_parameters(
                orig_params={"weight": self._orig_module.weight, "bias": self._orig_module.bias},
                lora_layers=unoptimized_layers,
                lora_weights=unoptimized_weights,
            )
            added_residual = torch.nn.functional.linear(input, params["weight"], params.get("bias", None))
            if residual is None:
                residual = added_residual
            else:
                residual += added_residual

        return self.orig_module(input) + residual


class LoRAConv1dWrapper(LoRASidecarWrapper):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        params = self._get_lora_patched_parameters(
            orig_params={"weight": self._orig_module.weight, "bias": self._orig_module.bias},
            lora_layers=self._lora_layers,
            lora_weights=self._lora_weights,
        )
        return self.orig_module(input) + torch.nn.functional.conv1d(input, params["weight"], params.get("bias", None))


class LoRAConv2dWrapper(LoRASidecarWrapper):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        params = self._get_lora_patched_parameters(
            orig_params={"weight": self._orig_module.weight, "bias": self._orig_module.bias},
            lora_layers=self._lora_layers,
            lora_weights=self._lora_weights,
        )
        return self.orig_module(input) + torch.nn.functional.conv2d(input, params["weight"], params.get("bias", None))
