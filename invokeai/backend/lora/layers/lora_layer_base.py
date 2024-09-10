from typing import Dict, Optional, Set

import torch

import invokeai.backend.util.logging as logger
from invokeai.backend.util.calc_tensor_size import calc_tensors_size


class LoRALayerBase:
    """Base class for all LoRA-like patching layers."""

    def __init__(self, alpha: float | None, bias: torch.Tensor | None):
        self.alpha = alpha
        self.bias = bias

    @classmethod
    def _parse_bias(
        cls, bias_indices: torch.Tensor | None, bias_values: torch.Tensor | None, bias_size: torch.Tensor | None
    ) -> torch.Tensor | None:
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

    @property
    def rank(self) -> int | None:
        raise NotImplementedError()

    def get_weight(self, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def get_bias(self, orig_bias: torch.Tensor) -> Optional[torch.Tensor]:
        return self.bias

    def get_parameters(self, orig_module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        params = {"weight": self.get_weight(orig_module.weight)}
        bias = self.get_bias(orig_module.bias)
        if bias is not None:
            params["bias"] = bias
        return params

    def calc_size(self) -> int:
        return calc_tensors_size([self.bias])

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        if self.bias is not None:
            self.bias = self.bias.to(device=device, dtype=dtype)

    @classmethod
    def warn_on_unhandled_keys(cls, values: Dict[str, torch.Tensor], handled_keys: Set[str]):
        """Log a warning if values contains unhandled keys."""
        unknown_keys = set(values.keys()) - handled_keys
        if unknown_keys:
            logger.warning(
                f"Unexpected keys found in LoRA/LyCORIS layer, model might work incorrectly! Unexpected keys: {unknown_keys}"
            )
