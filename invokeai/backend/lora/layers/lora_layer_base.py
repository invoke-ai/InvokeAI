from typing import Dict, Optional, Set

import torch

import invokeai.backend.util.logging as logger


class LoRALayerBase:
    # rank: Optional[int]
    # alpha: Optional[float]
    # bias: Optional[torch.Tensor]
    # layer_key: str

    # @property
    # def scale(self):
    #    return self.alpha / self.rank if (self.alpha and self.rank) else 1.0

    def __init__(
        self,
        layer_key: str,
        values: Dict[str, torch.Tensor],
    ):
        if "alpha" in values:
            self.alpha = values["alpha"].item()
        else:
            self.alpha = None

        if "bias_indices" in values and "bias_values" in values and "bias_size" in values:
            self.bias: Optional[torch.Tensor] = torch.sparse_coo_tensor(
                values["bias_indices"],
                values["bias_values"],
                tuple(values["bias_size"]),
            )

        else:
            self.bias = None

        self.rank = None  # set in layer implementation
        self.layer_key = layer_key

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
        model_size = 0
        for val in [self.bias]:
            if val is not None:
                model_size += val.nelement() * val.element_size()
        return model_size

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        if self.bias is not None:
            self.bias = self.bias.to(device=device, dtype=dtype)

    def check_keys(self, values: Dict[str, torch.Tensor], known_keys: Set[str]):
        """Log a warning if values contains unhandled keys."""
        # {"alpha", "bias_indices", "bias_values", "bias_size"} are hard-coded, because they are handled by
        # `LoRALayerBase`. Sub-classes should provide the known_keys that they handled.
        all_known_keys = known_keys | {"alpha", "bias_indices", "bias_values", "bias_size"}
        unknown_keys = set(values.keys()) - all_known_keys
        if unknown_keys:
            logger.warning(
                f"Unexpected keys found in LoRA/LyCORIS layer, model might work incorrectly! Keys: {unknown_keys}"
            )
