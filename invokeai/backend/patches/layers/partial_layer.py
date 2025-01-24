from dataclasses import dataclass

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.lora_layer_base import LoRALayerBase
from invokeai.backend.patches.layers.param_shape_utils import get_param_shape
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


@dataclass
class Range:
    start: int
    end: int


class PartialLayer(BaseLayerPatch):
    """A layer patch that only modifies a sub-range of the weights in the original layer.

    This class was created to handle a special case with FLUX LoRA models. In the BFL FLUX model format, the attention
    Q, K, V matrices are concatenated along the first dimension. In the diffusers LoRA format, the Q, K, V matrices are
    stored as separate tensors. This class enables diffusers LoRA layers to be used in BFL FLUX models.
    """

    def __init__(self, lora_layer: LoRALayerBase, range: tuple[Range, Range]):
        super().__init__()

        self.lora_layer = lora_layer
        # self.range[i] gives the range to be modified in the original layer for the i'th dimension.
        self.range = range

    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        # HACK(ryand): If the original parameters are in a quantized format that can't be sliced, we replace them with
        # dummy tensors on the 'meta' device. This allows sub-layers to access the shapes of the sliced parameters. But,
        # of course, any sub-layers that need to access the actual values of the parameters will fail.
        for param_name in orig_parameters.keys():
            param = orig_parameters[param_name]
            if type(param) is torch.nn.Parameter and type(param.data) is torch.Tensor:
                pass
            elif type(param) is GGMLTensor:
                pass
            else:
                orig_parameters[param_name] = torch.empty(get_param_shape(param), device="meta")

        # Slice the original parameters to the specified range.
        sliced_parameters: dict[str, torch.Tensor] = {}
        for param_name, param_weight in orig_parameters.items():
            if param_name == "weight":
                sliced_parameters[param_name] = param_weight[
                    self.range[0].start : self.range[0].end, self.range[1].start : self.range[1].end
                ]
            elif param_name == "bias":
                sliced_parameters[param_name] = param_weight[self.range[0].start : self.range[0].end]
            else:
                raise ValueError(f"Unexpected parameter name: {param_name}")

        # Apply the LoRA layer to the sliced parameters.
        params = self.lora_layer.get_parameters(sliced_parameters, weight)

        # Expand the parameters diffs to match the original parameter shape.
        out_params: dict[str, torch.Tensor] = {}
        for param_name, param_weight in params.items():
            orig_param = orig_parameters[param_name]
            out_params[param_name] = torch.zeros(
                get_param_shape(orig_param), dtype=param_weight.dtype, device=param_weight.device
            )

            if param_name == "weight":
                out_params[param_name][
                    self.range[0].start : self.range[0].end, self.range[1].start : self.range[1].end
                ] = param_weight
            elif param_name == "bias":
                out_params[param_name][self.range[0].start : self.range[0].end] = param_weight
            else:
                raise ValueError(f"Unexpected parameter name: {param_name}")

        return out_params

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.lora_layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return self.lora_layer.calc_size()
