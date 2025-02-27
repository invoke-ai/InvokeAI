import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.util.calc_tensor_size import calc_tensor_size


class SetParameterLayer(BaseLayerPatch):
    """A layer that sets a single parameter to a new target value.
    (The diff between the target value and current value is calculated internally.)
    """

    def __init__(self, param_name: str, weight: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.param_name = param_name

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        # Note: We intentionally ignore the weight parameter here. This matches the behavior in the official FLUX
        # Control LoRA implementation.
        diff = self.weight - orig_parameters[self.param_name]
        return {self.param_name: diff}

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.weight = self.weight.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return calc_tensor_size(self.weight)
