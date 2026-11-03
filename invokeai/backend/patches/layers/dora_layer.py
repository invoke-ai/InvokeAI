from typing import Dict, Optional

import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.patches.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class DoRALayer(LoRALayerBase):
    """A DoRA layer. As defined in https://arxiv.org/pdf/2402.09353.

    Two mutually-incompatible magnitude conventions exist in the wild, and they differ in *which axis* of the
    ``(out_features, in_features)`` weight the magnitude vector indexes:

    - **Input axis**: one entry per **input** column, i.e. the direction norm is taken across the output dim.
      Shape ``(1, in_features)``. This is LyCORIS' ``.dora_scale`` with ``wd_on_out=False``.
    - **Output axis**: one entry per **output** row, i.e. the direction norm is taken across the input dim
      (``torch.linalg.norm(w, dim=1)``). PEFT / Diffusers / ai-toolkit (``.lora_magnitude_vector.weight`` /
      ``.magnitude``, shape ``(out_features,)``), and the ``.dora_scale`` that LyCORIS writes by default
      (``wd_on_out=True``) and OneTrainer writes with "DoRA on output axis": shape ``(out_features, 1)``.

    Applying one convention's magnitude with the other's math silently produces wrong weights, so the orientation
    is carried explicitly. A ``.dora_scale`` can hold either, and its stored shape tells which (see
    ``_dora_scale_is_out_dim``).
    """

    def __init__(
        self,
        up: torch.Tensor,
        down: torch.Tensor,
        dora_scale: torch.Tensor,
        alpha: float | None,
        bias: Optional[torch.Tensor],
        magnitude_is_out_dim: bool = False,
    ):
        super().__init__(alpha, bias)
        self.up = up
        self.down = down
        self.dora_scale = dora_scale
        # False -> LyCORIS convention (magnitude indexes in_features). True -> PEFT/ai-toolkit convention
        # (magnitude indexes out_features).
        self.magnitude_is_out_dim = magnitude_is_out_dim

    @classmethod
    def from_state_dict_values(cls, values: Dict[str, torch.Tensor]):
        alpha = cls._parse_alpha(values.get("alpha", None))
        bias = cls._parse_bias(
            values.get("bias_indices", None), values.get("bias_values", None), values.get("bias_size", None)
        )

        up = values["lora_up.weight"]
        # ``dora_magnitude`` is the PEFT/ai-toolkit out-dim magnitude. ``dora_scale`` can index either axis.
        if "dora_magnitude" in values:
            dora_scale = values["dora_magnitude"]
            magnitude_is_out_dim = True
        else:
            dora_scale = values["dora_scale"]
            magnitude_is_out_dim = cls._dora_scale_is_out_dim(dora_scale, out_features=up.shape[0])

        layer = cls(
            up=up,
            down=values["lora_down.weight"],
            dora_scale=dora_scale,
            alpha=alpha,
            bias=bias,
            magnitude_is_out_dim=magnitude_is_out_dim,
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
                "dora_magnitude",
            },
        )

        return layer

    @staticmethod
    def _dora_scale_is_out_dim(dora_scale: torch.Tensor, out_features: int) -> bool:
        """Whether a ``.dora_scale`` holds one entry per output row.

        LyCORIS stores ``(out_features, 1)``, or ``(out_features, 1, 1, 1)`` for a conv layer, with
        ``wd_on_out=True`` (its default) and ``(1, in_features[, 1, 1])`` with ``wd_on_out=False``. The two
        shapes only coincide when ``out_features == in_features == 1``, where both conventions give the same
        weight. A 1-D scale has no axis to tell from, so it keeps the input-axis reading.
        """
        return (
            dora_scale.dim() >= 2
            and dora_scale.shape[0] == out_features
            and all(size == 1 for size in dora_scale.shape[1:])
        )

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

        if self.magnitude_is_out_dim:
            # PEFT / ai-toolkit: norm over the input dim, one entry per output row.
            trailing_dims = [1] * (out_weight.dim() - 1)
            direction_norm = (
                out_weight.reshape(out_weight.shape[0], -1).norm(dim=1).reshape(out_weight.shape[0], *trailing_dims)
            )
            dora_scale = self.dora_scale.reshape(out_weight.shape[0], *trailing_dims)
        else:
            # LyCORIS / kohya: norm over the output dim, one entry per input column.
            # TODO(ryand): Simplify this logic.
            direction_norm = (
                out_weight.transpose(0, 1)
                .reshape(out_weight.shape[1], -1)
                .norm(dim=1, keepdim=True)
                .reshape(out_weight.shape[1], *[1] * (out_weight.dim() - 1))
                .transpose(0, 1)
            )
            dora_scale = self.dora_scale

        out_weight *= dora_scale / direction_norm

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
