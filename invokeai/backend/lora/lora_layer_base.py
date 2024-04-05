from typing import Dict, Optional

import torch


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

    def get_weight(self, orig_weight: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    def calc_size(self) -> int:
        model_size = 0
        for val in [self.bias]:
            if val is not None:
                model_size += val.nelement() * val.element_size()
        return model_size

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if self.bias is not None:
            self.bias = self.bias.to(device=device, dtype=dtype)
