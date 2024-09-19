# Largely based on https://github.com/city96/ComfyUI-GGUF

from typing import Optional, Union

import gguf
from torch import Tensor, device, dtype, float32, nn, zeros_like

from invokeai.backend.quantization.gguf.utils import dequantize_tensor, is_quantized

PATCH_TYPES = Union[list[Tensor], tuple[Tensor]]


class GGUFTensor(Tensor):
    """
    Main tensor-like class for storing quantized weights
    """

    def __init__(self, *args, tensor_type, tensor_shape, patches=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches or []

    def __new__(cls, *args, tensor_type, tensor_shape, patches=None, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            print(f"ignoring 'copy_' on tensor: {e}")

    def __deepcopy__(self, *args, **kwargs):
        # Intel Arc fix, ref#50
        new = super().__deepcopy__(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


class GGUFLayer(nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """

    dequant_dtype = None
    patch_dtype = None
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None):
        if weight is None or bias is None:
            return False
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict: dict[str, Tensor], prefix: str, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight", None), state_dict.get(f"{prefix}bias", None)
        if self.is_ggml_quantized(weight=weight, bias=bias):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata,
        strict,
        missing_keys: list[str],
        unexpected_keys,
        error_msgs,
    ):
        for k, v in state_dict.items():
            if k.endswith("weight"):
                self.weight = nn.Parameter(v, requires_grad=False)
            elif k.endswith("bias") and v is not None:
                self.bias = nn.Parameter(v, requires_grad=False)
            else:
                missing_keys.append(k)

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination: dict[str, Tensor], prefix: str):
        # This is a fake state dict for vram estimation
        weight = zeros_like(self.weight, device=device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = zeros_like(self.bias, device=device("meta"))
            destination[prefix + "bias"] = bias
        return

    def get_weight(self, tensor: Optional[Tensor], dtype: dtype):
        if tensor is None:
            return

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
        return weight

    def calc_size(self) -> int:
        """Get the size of this model in bytes."""
        return self.bias.nelement() * self.bias.element_size()

    def cast_bias_weight(
        self,
        input: Tensor,
        dtype: Optional[dtype] = None,
        device: Optional[device] = None,
        bias_dtype: Optional[dtype] = None,
    ) -> tuple[Tensor, Tensor]:
        if dtype is None:
            dtype = getattr(input, "dtype", float32)
            if dtype is None:
                raise ValueError("dtype is required")
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

        bias = self.get_weight(self.bias.to(device), dtype)
        if bias is not None:
            bias = bias.to(dtype=bias_dtype, device=device, copy=False)

        weight = self.get_weight(self.weight.to(device), dtype)
        if weight is not None:
            weight = weight.to(dtype=dtype, device=device)
        if weight is None or bias is None:
            raise ValueError("Weight or bias is None")
        return weight, bias
