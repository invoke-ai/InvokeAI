from typing import Dict, Optional

import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.patches.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class DoRALayer(LoRALayerBase):
    """A DoRA layer. As defined in https://arxiv.org/pdf/2402.09353."""

    def __init__(
        self,
        up: torch.Tensor,
        down: torch.Tensor,
        dora_scale: torch.Tensor,
        alpha: float | None,
        bias: Optional[torch.Tensor],
    ):
        super().__init__(alpha, bias)
        self.up = up
        self.down = down
        self.dora_scale = dora_scale

    @classmethod
    def from_state_dict_values(cls, values: Dict[str, torch.Tensor]):
        alpha = cls._parse_alpha(values.get("alpha", None))
        bias = cls._parse_bias(
            values.get("bias_indices", None), values.get("bias_values", None), values.get("bias_size", None)
        )

        layer = cls(
            up=values["lora_up.weight"],
            down=values["lora_down.weight"],
            dora_scale=values["dora_scale"],
            alpha=alpha,
            bias=bias,
        )

        cls.warn_on_unhandled_keys(
            values=values,
            handled_keys={
                # Default keys.
                "alpha",
                "bias_indices",
                "bias_values",
                "bias_size",
                # Layer-specific keys.
                "lora_up.weight",
                "lora_down.weight",
                "dora_scale",
            },
        )

        return layer

    def _rank(self) -> int:
        return self.down.shape[0]

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_weight = cast_to_device(orig_weight, self.up.device)

        # Note: Variable names (e.g. delta_v) are based on the paper.
        delta_v = self.up.reshape(self.up.shape[0], -1) @ self.down.reshape(self.down.shape[0], -1)
        delta_v = delta_v.reshape(orig_weight.shape)

        delta_v = delta_v * self.scale()

        # At this point, out_weight is the unnormalized direction matrix.
        out_weight = orig_weight + delta_v

        # TODO(ryand): Simplify this logic.
        direction_norm = (
            out_weight.transpose(0, 1)
            .reshape(out_weight.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(out_weight.shape[1], *[1] * (out_weight.dim() - 1))
            .transpose(0, 1)
        )

        out_weight *= self.dora_scale / direction_norm

        return out_weight - orig_weight

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().to(device=device, dtype=dtype)
        self.up = self.up.to(device=device, dtype=dtype)
        self.down = self.down.to(device=device, dtype=dtype)
        self.dora_scale = self.dora_scale.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensors_size([self.up, self.down, self.dora_scale])

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        if any(p.device.type == "meta" for p in orig_parameters.values()):
            # If any of the original parameters are on the 'meta' device, we assume this is because the base model is in
            # a quantization format that doesn't allow easy dequantization.
            raise RuntimeError(
                "The base model quantization format (likely bitsandbytes) is not compatible with DoRA patches."
            )

        scale = self.scale()
        params = {"weight": self.get_weight(orig_parameters["weight"]) * weight}
        bias = self.get_bias(orig_parameters.get("bias", None))
        if bias is not None:
            params["bias"] = bias * (weight * scale)

        # Reshape all params to match the original module's shape.
        for param_name, param_weight in params.items():
            orig_param = orig_parameters[param_name]
            if param_weight.shape != orig_param.shape:
                params[param_name] = param_weight.reshape(orig_param.shape)

        return params
