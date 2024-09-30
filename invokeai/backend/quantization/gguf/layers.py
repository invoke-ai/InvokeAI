# Largely based on https://github.com/city96/ComfyUI-GGUF

from typing import Callable, List, Optional, Union

import gguf
import torch

from invokeai.backend.quantization.gguf.utils import dequantize_tensor, is_quantized

PATCH_TYPES = Union[list[torch.Tensor], tuple[torch.Tensor]]


class GGUFTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights.
    Inherits from torch.Tensor and adds additional attributes.
    """

    tensor_type: Union[torch.dtype, gguf.GGMLQuantizationType, None]
    tensor_shape: torch.Size
    patches: List[Callable[[torch.Tensor], torch.Tensor]]

    def __new__(
        cls,
        data,
        tensor_type: Union[torch.dtype, gguf.GGMLQuantizationType],
        tensor_shape: torch.Size,
        patches: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        **kwargs,
    ):
        # Create a new tensor instance using the superclass method
        if isinstance(data, torch.Tensor):
            tensor = data.as_subclass(cls)
        else:
            tensor = torch.tensor(data, **kwargs).as_subclass(cls)
        # Set the additional attributes
        tensor.tensor_type = tensor_type
        tensor.tensor_shape = tensor_shape
        tensor.patches = patches or []
        return tensor

    def __init__(
        self,
        data,
        tensor_type: Union[torch.dtype, gguf.GGMLQuantizationType],
        tensor_shape: torch.Size,
        patches: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        **kwargs,
    ):
        # __init__ is not called for torch.Tensor subclasses
        pass

    def to(self, *args, **kwargs):
        # Create a new tensor with the desired type/device and copy attributes
        new = super().to(*args, **kwargs)
        new = new.as_subclass(GGUFTensor)
        new.tensor_type = getattr(self, "tensor_type", self.dtype)
        new.tensor_shape = getattr(self, "tensor_shape", self.size())
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # Attempt to copy data into the tensor; handle exceptions gracefully
        try:
            new = super().copy_(*args, **kwargs)
            new = new.as_subclass(GGUFTensor)
            new.tensor_type = getattr(self, "tensor_type", self.dtype)
            new.tensor_shape = getattr(self, "tensor_shape", self.size())
            new.patches = getattr(self, "patches", []).copy()
            return new
        except Exception as e:
            print(f"Ignoring 'copy_' on tensor: {e}")

    def __deepcopy__(self, memo):
        # Create a deep copy of the tensor and copy attributes
        new = super().__deepcopy__(memo)
        if isinstance(new, torch.Tensor):
            new = new.as_subclass(GGUFTensor)
            new.tensor_type = getattr(self, "tensor_type", self.dtype)
            new.tensor_shape = getattr(self, "tensor_shape", self.size())
            new.patches = getattr(self, "patches", []).copy()
        return new

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


class GGUFLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """

    dequant_dtype = None
    patch_dtype = None
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None):
        weight = weight if weight is not None else self.weight
        bias = bias if bias is not None else self.bias
        weight_quantized = is_quantized(weight)
        bias_quantized = is_quantized(bias)
        return weight_quantized or bias_quantized

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight", None), state_dict.get(f"{prefix}bias", None)
        if self.is_ggml_quantized(weight=weight, bias=bias):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata,
        strict,
        missing_keys: list[str],
        unexpected_keys,
        error_msgs,
    ):
        for k, v in state_dict.items():
            if k.endswith("weight"):
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k.endswith("bias") and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                missing_keys.append(k)

    def get_weight(self, tensor: Optional[torch.Tensor], dtype: torch.dtype):
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
        input: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        bias_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dtype is None:
            dtype = getattr(input, "dtype", torch.float32)
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
