from typing import Dict, Optional, Set

import torch

import invokeai.backend.util.logging as logger


class LoRALayerBase(torch.nn.Module):
    """Base class for all LoRA-like patching layers."""

    def __init__(self, alpha: float | None, bias: torch.Tensor | None):
        super().__init__()
        self._alpha = alpha
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

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

    def rank(self) -> int | None:
        raise NotImplementedError()

    def scale(self) -> float:
        if self._alpha is None or self.rank() is None:
            return 1.0
        return self._alpha / self.rank()

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

    @classmethod
    def warn_on_unhandled_keys(cls, values: Dict[str, torch.Tensor], handled_keys: Set[str]):
        """Log a warning if values contains unhandled keys."""
        unknown_keys = set(values.keys()) - handled_keys
        if unknown_keys:
            logger.warning(
                f"Unexpected keys found in LoRA/LyCORIS layer, model might work incorrectly! Unexpected keys: {unknown_keys}"
            )

    def calc_size(self) -> int:
        # HACK(ryand): Fix this issue with circular imports.
        from invokeai.backend.model_manager.load.model_util import calc_module_size

        return calc_module_size(self)
