from typing import Optional

import torch

import invokeai.backend.util.logging as logger
from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.param_shape_utils import get_param_shape
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class LoRALayerBase(BaseLayerPatch):
    """Base class for all LoRA-like patching layers."""

    # Note: It is tempting to make this a torch.nn.Module sub-class and make all tensors 'torch.nn.Parameter's. Then we
    # could inherit automatic .to(...) behavior for this class, its subclasses, and all sidecar layers that wrap a
    # LoRALayerBase. We would also be able to implement a single calc_size() method that could be inherited by all
    # subclasses. But, it turns out that the speed overhead of the default .to(...) implementation in torch.nn.Module is
    # noticeable, so for now we have opted not to use torch.nn.Module.

    def __init__(self, alpha: float | None, bias: torch.Tensor | None):
        self._alpha = alpha
        self.bias = bias

    @classmethod
    def _parse_bias(
        cls, bias_indices: torch.Tensor | None, bias_values: torch.Tensor | None, bias_size: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Helper function to parse a bias tensor from a state dict in LyCORIS format."""
        assert (bias_indices is None) == (bias_values is None) == (bias_size is None)

        bias = None
        if bias_indices is not None:
            bias = torch.sparse_coo_tensor(bias_indices, bias_values, tuple(bias_size))
        return bias

    @classmethod
    def _parse_alpha(
        cls,
        alpha: torch.Tensor | None,
    ) -> float | None:
        return alpha.item() if alpha is not None else None

    def _rank(self) -> int | None:
        """Return the rank of the LoRA-like layer. Or None if the layer does not have a rank. This value is used to
        calculate the scale.
        """
        raise NotImplementedError()

    def scale(self) -> float:
        rank = self._rank()
        if self._alpha is None or rank is None:
            return 1.0
        return self._alpha / rank

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def get_bias(self, orig_bias: torch.Tensor | None) -> Optional[torch.Tensor]:
        return self.bias

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        scale = self.scale()
        params = {"weight": self.get_weight(orig_parameters["weight"]) * (weight * scale)}
        bias = self.get_bias(orig_parameters.get("bias", None))
        if bias is not None:
            params["bias"] = bias * (weight * scale)

        # Reshape all params to match the original module's shape.
        for param_name, param_weight in params.items():
            orig_param = orig_parameters[param_name]
            if param_weight.shape != get_param_shape(orig_param):
                params[param_name] = param_weight.reshape(get_param_shape(orig_param))

        return params

    @classmethod
    def warn_on_unhandled_keys(cls, values: dict[str, torch.Tensor], handled_keys: set[str]):
        """Log a warning if values contains unhandled keys."""
        unknown_keys = set(values.keys()) - handled_keys
        if unknown_keys:
            logger.warning(
                f"Unexpected keys found in LoRA/LyCORIS layer, model might work incorrectly! Unexpected keys: {unknown_keys}"
            )

    def calc_size(self) -> int:
        return calc_tensors_size([self.bias])

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        if self.bias is not None:
            self.bias = self.bias.to(device=device, dtype=dtype)
