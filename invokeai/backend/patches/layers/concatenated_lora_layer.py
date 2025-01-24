from dataclasses import dataclass
from typing import Sequence

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.param_shape_utils import get_param_shape
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


@dataclass
class Range:
    start: int
    end: int


class ConcatenatedLoRALayer(BaseLayerPatch):
    """A patch layer that is composed of multiple sub-layers concatenated together.

    This class was created to handle a special case with FLUX LoRA models. In the BFL FLUX model format, the attention
    Q, K, V matrices are concatenated along the first dimension. In the diffusers LoRA format, the Q, K, V matrices are
    stored as separate tensors. This class enables diffusers LoRA layers to be used in BFL FLUX models.
    """

    def __init__(
        self,
        lora_layers: Sequence[BaseLayerPatch],
        ranges: Sequence[Range],
    ):
        super().__init__()

        self.lora_layers = lora_layers
        # self.ranges[i] is the range for the i'th lora layer along the 0'th weight dimension.
        self.ranges = ranges
        assert len(self.ranges) == len(self.lora_layers)

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

        out_parameters: dict[str, torch.Tensor] = {}

        for lora_layer, range in zip(self.lora_layers, self.ranges, strict=True):
            sliced_parameters: dict[str, torch.Tensor] = {
                n: p[range.start : range.end] for n, p in orig_parameters.items()
            }

            # Note that `weight` is applied in the sub-layers, no need to apply it in this function.
            layer_out_parameters = lora_layer.get_parameters(sliced_parameters, weight)

            for out_param_name, out_param in layer_out_parameters.items():
                if out_param_name not in out_parameters:
                    # If not already in the output dict, initialize an output tensor with the same shape as the full
                    # original parameter.
                    out_parameters[out_param_name] = torch.zeros(
                        get_param_shape(orig_parameters[out_param_name]),
                        dtype=out_param.dtype,
                        device=out_param.device,
                    )
                out_parameters[out_param_name][range.start : range.end] += out_param

        return out_parameters

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        for lora_layer in self.lora_layers:
            lora_layer.to(device=device, dtype=dtype)

    def calc_size(self) -> int:
        return sum(lora_layer.calc_size() for lora_layer in self.lora_layers)
