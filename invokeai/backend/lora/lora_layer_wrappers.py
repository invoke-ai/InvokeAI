import torch

from invokeai.backend.lora.layers.any_lora_layer import AnyLoRALayer


class LoRAModuleWrapper(torch.nn.Module):
    def __init__(self, orig_module: torch.nn.Module, lora_layers: list[AnyLoRALayer], lora_weights: list[float]):
        super().__init__()
        self._orig_module = orig_module
        self._lora_layers = lora_layers
        self._lora_weights = lora_weights

    @torch.no_grad()
    def _get_lora_patched_parameters(self) -> dict[str, torch.Tensor]:
        out_params: dict[str, torch.Tensor] = {}
        for lora_layer, lora_weight in zip(self._lora_layers, self._lora_weights, strict=True):
            layer_params = lora_layer.get_parameters(self._orig_module)
            for param_name, param_weight in layer_params.items():
                # If the parameter already exists in out_params, use that one. Otherwise, use original parameter.
                if param_name not in out_params:
                    out_params[param_name] = self._orig_module.get_parameter(param_name)

                if out_params[param_name].shape != param_weight.shape:
                    param_weight = param_weight.reshape(out_params[param_name].shape)

                out_params[param_name] += param_weight * (lora_layer.scale() * lora_weight)

        return out_params


class LoRALinearWrapper(LoRAModuleWrapper):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        params = self._get_lora_patched_parameters()
        return torch.nn.functional.linear(input, params["weight"], params.get("bias", None))


class LoRAConv1dWrapper(LoRAModuleWrapper):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        params = self._get_lora_patched_parameters()
        return torch.nn.functional.conv1d(input, params["weight"], params.get("bias", None))


class LoRAConv2dWrapper(LoRAModuleWrapper):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        params = self._get_lora_patched_parameters()
        return torch.nn.functional.conv2d(input, params["weight"], params.get("bias", None))
