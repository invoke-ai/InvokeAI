import torch

from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer


class LoRAModuleWrapper(torch.nn.Module):
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
    def _get_lora_patched_parameters(self, params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for lora_layer, lora_weight in zip(self._lora_layers, self._lora_weights, strict=True):
            layer_params = lora_layer.get_parameters(self._orig_module)
            for param_name, param_weight in layer_params.items():
                if params[param_name].shape != param_weight.shape:
                    param_weight = param_weight.reshape(params[param_name].shape)

                # NOTE: It is important that params[param_name] is not modified in-place, because we initialize it
                # with the original parameter - which we don't want to modify. In other words,
                # `out_params[param_name] += ...` would not work.
                params[param_name] = params[param_name] + param_weight * (lora_layer.scale() * lora_weight)

        return params


class LoRALinearWrapper(LoRAModuleWrapper):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        params = self._get_lora_patched_parameters(
            params={"weight": self._orig_module.weight, "bias": self._orig_module.bias}
        )
        return torch.nn.functional.linear(input, params["weight"], params["bias"])


class LoRAConv1dWrapper(LoRAModuleWrapper):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        params = self._get_lora_patched_parameters(
            params={"weight": self._orig_module.weight, "bias": self._orig_module.bias}
        )
        return torch.nn.functional.conv1d(input, params["weight"], params["bias"])


class LoRAConv2dWrapper(LoRAModuleWrapper):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        params = self._get_lora_patched_parameters(
            params={"weight": self._orig_module.weight, "bias": self._orig_module.bias}
        )
        return torch.nn.functional.conv2d(input, params["weight"], params["bias"])
