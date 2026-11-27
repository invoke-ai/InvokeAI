import copy
from typing import Dict, Optional

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch, patch_tensors, sub_patches
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.util.calc_tensor_size import calc_tensors_size
from invokeai.backend.util.cast_to_device import cast_to_device
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


class WeightDecomposedLayer(LoRALayerBase):
    """A weight-decomposed (DoRA) layer: the update ``delta`` and a magnitude ``m`` give

        W' = m * (W + delta) / ||W + delta||

    and the layer's residual is ``W' - W``. Subclasses supply ``delta`` (``_scaled_delta``); the decomposition, the
    axis the magnitude indexes, the float32 promotion and the application to a module are shared here.

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

    Unlike every other layer applied to a module, this one reads the values of the original weight, so it cannot be
    applied to a weight that has no readable values (see ``reads_weight_values``).
    """

    def __init__(
        self,
        dora_scale: torch.Tensor,
        alpha: float | None,
        bias: Optional[torch.Tensor],
        magnitude_is_out_dim: bool,
        direction_norm: torch.Tensor | None = None,
    ):
        super().__init__(alpha, bias)
        self.dora_scale = dora_scale
        # False -> LyCORIS convention (magnitude indexes in_features). True -> PEFT/ai-toolkit convention
        # (magnitude indexes out_features).
        self.magnitude_is_out_dim = magnitude_is_out_dim
        # Application-local shared norm for a split fused input-axis layer; never bound on a cached adapter.
        self.direction_norm = direction_norm

    @staticmethod
    def _dora_scale_is_out_dim(dora_scale: torch.Tensor, out_features: int, in_features: int | None = None) -> bool:
        """Whether a ``.dora_scale`` holds one entry per output row.

        LyCORIS stores ``(out_features, 1)``, or ``(out_features, 1, 1, 1)`` for a conv layer, with
        ``wd_on_out=True`` (its default) and ``(1, in_features[, 1, 1])`` with ``wd_on_out=False``. The two
        shapes only coincide when ``out_features == in_features == 1``, where both conventions give the same
        weight.

        LyCORIS never writes a 1-D scale, so a 1-D one comes from another writer or a squeezed file, and only its
        length can tell the axis, which needs ``in_features``: ``(out_features,)`` is an output-axis scale when the
        layer is not square, as PEFT's ``lora_magnitude_vector`` is. On a square layer both lengths are the same and
        nothing in the file says which axis it indexes; it keeps the input-axis reading it has always had, the LyCORIS
        ``dora_scale`` convention this key comes from. Without ``in_features`` every 1-D scale is read that way.
        """
        if dora_scale.dim() == 1:
            return in_features is not None and dora_scale.shape[0] == out_features != in_features
        return (
            dora_scale.dim() >= 2
            and dora_scale.shape[0] == out_features
            and all(size == 1 for size in dora_scale.shape[1:])
        )

    def _residual(self, orig_weight: torch.Tensor, delta_v: torch.Tensor, consume_delta: bool = False) -> torch.Tensor:
        """``W' - W`` for the original weight and the layer's scaled update, both promoted by the caller where a
        low-precision dtype would round away a small update. With ``consume_delta`` the direction is formed in
        ``delta_v``'s own storage, which saves a copy of the weight at the peak; the caller must hand over a tensor it
        made and no longer reads."""
        # At this point, out_weight is the unnormalized direction matrix.
        if consume_delta and torch.promote_types(delta_v.dtype, orig_weight.dtype) == delta_v.dtype:
            out_weight = delta_v.add_(orig_weight)
        else:
            out_weight = orig_weight + delta_v

        if self.direction_norm is not None:
            direction_norm = self.direction_norm
            dora_scale = self.dora_scale
        elif self.magnitude_is_out_dim:
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

        # A row whose direction is all zeros has no norm to divide by; LyCORIS adds its epsilon for the same reason.
        # Clamping leaves every other norm bit-identical and that row's result 0 rather than NaN, at any strength.
        out_weight *= dora_scale / direction_norm.clamp_min(torch.finfo(direction_norm.dtype).eps)

        return out_weight - orig_weight

    @staticmethod
    def _refuse_unreadable_weight(orig_parameters: dict[str, torch.Tensor]) -> None:
        if any(p.device.type == "meta" for p in orig_parameters.values()):
            # If any of the original parameters are on the 'meta' device, we assume this is because the base model is in
            # a quantization format that doesn't allow easy dequantization.
            raise RuntimeError(
                "The base model quantization format (likely bitsandbytes) is not compatible with DoRA patches."
            )

    @staticmethod
    def _reshape_to_orig(params: dict[str, torch.Tensor], orig_parameters: dict[str, torch.Tensor]):
        """Reshape all params to match the original module's shape."""
        for param_name, param_weight in params.items():
            orig_param = orig_parameters[param_name]
            if param_weight.shape != orig_param.shape:
                params[param_name] = param_weight.reshape(orig_param.shape)
        return params

    def tensors(self) -> tuple[torch.Tensor | None, ...]:
        """Every tensor the layer holds, for memory estimates."""
        raise NotImplementedError()


def reads_weight_values(patch: BaseLayerPatch) -> bool:
    """Whether applying `patch` reads the values of the original weight, rather than only its shape.

    Only a weight-decomposed layer does, so a module whose weight is stored quantized has to dequantize it (or refuse
    the patch) for exactly these. A wrapper patch reads them if one of its sub-patches does.
    """
    return isinstance(patch, WeightDecomposedLayer) or any(reads_weight_values(sub) for sub in sub_patches(patch))


class DoRALayer(WeightDecomposedLayer):
    """A DoRA layer. As defined in https://arxiv.org/pdf/2402.09353: the update is the low-rank ``up @ down``."""

    def __init__(
        self,
        up: torch.Tensor,
        down: torch.Tensor,
        dora_scale: torch.Tensor,
        alpha: float | None,
        bias: Optional[torch.Tensor],
        magnitude_is_out_dim: bool = False,
        direction_norm: torch.Tensor | None = None,
    ):
        super().__init__(
            dora_scale=dora_scale,
            alpha=alpha,
            bias=bias,
            magnitude_is_out_dim=magnitude_is_out_dim,
            direction_norm=direction_norm,
        )
        self.up = up
        self.down = down

    @classmethod
    def from_state_dict_values(cls, values: Dict[str, torch.Tensor]):
        alpha = cls._parse_alpha(values.get("alpha", None))
        bias = cls._parse_bias(
            values.get("bias_indices", None), values.get("bias_values", None), values.get("bias_size", None)
        )

        up = values["lora_up.weight"]
        dora_scale, magnitude_is_out_dim = _magnitude_from_values(
            values, out_features=up.shape[0], in_features=values["lora_down.weight"].shape[1]
        )

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

    def _rank(self) -> int:
        return self.down.shape[0]

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_weight = cast_to_device(orig_weight, self.up.device)

        # Form the direction and residual in float32 so low-precision rounding cannot erase a small update.
        up, down = self.up, self.down
        if orig_weight.dtype in _LOW_PRECISION_DTYPES or up.dtype in _LOW_PRECISION_DTYPES:
            orig_weight = orig_weight.float()
            up, down = up.float(), down.float()

        # Note: Variable names (e.g. delta_v) are based on the paper.
        delta_v = up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1)
        delta_v = delta_v.reshape(orig_weight.shape)

        delta_v = delta_v * self.scale()

        return self._residual(orig_weight, delta_v)

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        self._refuse_unreadable_weight(orig_parameters)

        scale = self.scale()
        params = {"weight": self.get_weight(orig_parameters["weight"]) * weight}
        bias = self.get_bias(orig_parameters.get("bias", None))
        if bias is not None:
            params["bias"] = bias * (weight * scale)

        return self._reshape_to_orig(params, orig_parameters)

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        if self.direction_norm is not None:
            # Keep the cached factors' precision; get_weight promotes them only while computing the residual.
            dtype = None
        super().to(device=device, dtype=dtype)
        self.up = self.up.to(device=device, dtype=dtype)
        self.down = self.down.to(device=device, dtype=dtype)
        self.dora_scale = self.dora_scale.to(device=device, dtype=dtype)
        if self.direction_norm is not None:
            # Accumulated in float32 across all fused rows; keep that precision with low-precision factors.
            self.direction_norm = self.direction_norm.to(device=device)

    def calc_size(self) -> int:
        return super().calc_size() + calc_tensors_size([self.up, self.down, self.dora_scale, self.direction_norm])

    def tensors(self) -> tuple[torch.Tensor | None, ...]:
        return (self.up, self.down, self.dora_scale, self.bias, self.direction_norm)


class FactorizedDoRALayer(WeightDecomposedLayer):
    """A LoKr or LoHa layer with a DoRA magnitude (LyCORIS ``dora_wd=True``: "DoKr" / "DoHa").

    The update is the wrapped patch's own weight residual (its ``ΔW``, scaled by ``alpha / rank``, in the weight's
    shape), decomposed like any other DoRA update. The wrapped patch is a ``LoKRLayer`` or ``LoHALayer`` and keeps the
    factors, alpha and bias, or a ``MergedLayerPatch`` of them for the row ranges of one split fused projection.
    """

    def __init__(self, layer: BaseLayerPatch, dora_scale: torch.Tensor, magnitude_is_out_dim: bool):
        # The magnitude stays float32 like the direction norm: the residual is the small difference of two nearly
        # equal magnitudes, and a low-precision one rounds that difference away.
        if dora_scale.dtype in _LOW_PRECISION_DTYPES:
            dora_scale = dora_scale.float()
        super().__init__(dora_scale=dora_scale, alpha=None, bias=None, magnitude_is_out_dim=magnitude_is_out_dim)
        self.layer = layer

    @classmethod
    def from_state_dict_values(cls, layer: LoKRLayer | LoHALayer, values: Dict[str, torch.Tensor]):
        """`layer`, built from `values` without their magnitude, with the magnitude in `values`.

        Raises ``ValueError`` for a magnitude that fits neither axis of the update's weight: it would otherwise fail
        in the middle of a generation, or be applied as an elementwise factor."""
        out_features, in_features = layer.out_features(), layer.in_features()
        dora_scale, magnitude_is_out_dim = _magnitude_from_values(values, out_features, in_features)
        if not _magnitude_fits(dora_scale, magnitude_is_out_dim, out_features, in_features):
            raise ValueError(
                f"A DoRA magnitude of shape {tuple(dora_scale.shape)} fits neither axis of the {out_features}x"
                f"{in_features} weight the {type(layer).__name__} updates: it needs one entry per output row, "
                f"({out_features},) or ({out_features}, 1), or per input column, ({in_features},) or (1, {in_features})."
            )
        return cls(layer, dora_scale=dora_scale, magnitude_is_out_dim=magnitude_is_out_dim)

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        if any(p.device.type == "meta" for p in orig_parameters.values()):
            # The weight has no readable values (bitsandbytes NF4 / LLM.int8), so the magnitude cannot be applied.
            # It applies as the plain inner LoKr/LoHa rather than failing the generation; a low-rank DoRA raises.
            _warn_magnitude_unreadable_once()
            return self.layer.get_parameters(orig_parameters, weight)

        layer = self.layer
        factors = _factor_tensors(layer)
        orig_parameters = {**orig_parameters, "weight": cast_to_device(orig_parameters["weight"], factors[0].device)}
        # The same float32 promotion as DoRALayer, applied to the factors before their product is formed.
        promoted = orig_parameters
        if orig_parameters["weight"].dtype in _LOW_PRECISION_DTYPES or any(
            t.dtype in _LOW_PRECISION_DTYPES for t in factors
        ):
            promoted = {**orig_parameters, "weight": orig_parameters["weight"].float()}
            layer = copy.copy(layer)
            layer.to(dtype=torch.float32)

        # The wrapped patch scales its update by its own alpha / rank.
        delta = layer.get_parameters(promoted, 1.0)
        # `delta` is the layer's own fresh tensor, which the residual forms its direction in.
        params = {"weight": self._residual(promoted["weight"], delta.pop("weight"), consume_delta=True) * weight}
        if "bias" in delta:
            params["bias"] = delta["bias"] * weight

        return self._reshape_to_orig(params, orig_parameters)

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.layer.to(device=device, dtype=dtype)
        self.dora_scale = self.dora_scale.to(
            device=device, dtype=torch.float32 if dtype in _LOW_PRECISION_DTYPES else dtype
        )

    def calc_size(self) -> int:
        return self.layer.calc_size() + calc_tensors_size([self.dora_scale])

    def tensors(self) -> tuple[torch.Tensor | None, ...]:
        return (*(tensor for _, _, tensor in patch_tensors(self.layer)), self.dora_scale)


_magnitude_unreadable_warned = False


def _warn_magnitude_unreadable_once() -> None:
    global _magnitude_unreadable_warned
    if not _magnitude_unreadable_warned:
        _magnitude_unreadable_warned = True
        logger.warning(
            "LoKr/LoHa layers with a DoRA magnitude apply as plain LoKr/LoHa on bitsandbytes-quantized weights: "
            "the magnitude needs the weight's values, which these formats don't expose."
        )


def _factor_tensors(patch: BaseLayerPatch) -> list[torch.Tensor]:
    """The factors of a patch and its sub-patches, not a sparse bias."""
    return [tensor for _, _, tensor in patch_tensors(patch) if not tensor.is_sparse]


def _magnitude_from_values(
    values: Dict[str, torch.Tensor], out_features: int, in_features: int
) -> tuple[torch.Tensor, bool]:
    """The magnitude of a layer's state dict values and whether it indexes the output axis. ``dora_magnitude`` is the
    PEFT/ai-toolkit output-axis magnitude; ``dora_scale`` can index either axis."""
    if "dora_magnitude" in values:
        return values["dora_magnitude"], True
    dora_scale = values["dora_scale"]
    return dora_scale, WeightDecomposedLayer._dora_scale_is_out_dim(dora_scale, out_features, in_features)


def _magnitude_fits(dora_scale: torch.Tensor, is_out_dim: bool, out_features: int, in_features: int) -> bool:
    """Whether a magnitude read on an axis has one entry per output row, or per input column, of the update's
    weight, and size 1 in every other dimension. `in_features` counts a conv's flattened input columns where the
    factors do not show the kernel, so a conv's per-channel ``(1, channels, 1, 1)`` need only divide it."""
    shape = tuple(dora_scale.shape)
    if not shape:
        return False
    if is_out_dim:
        return shape[0] == out_features and all(size == 1 for size in shape[1:])
    if len(shape) == 1:
        return shape[0] == in_features
    return shape[0] == 1 and in_features % shape[1] == 0 and all(size == 1 for size in shape[2:])
