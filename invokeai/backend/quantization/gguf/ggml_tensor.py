import gguf
import torch

from invokeai.backend.quantization.gguf.utils import (
    DEQUANTIZE_FUNCTIONS,
    TORCH_COMPATIBLE_QTYPES,
    dequantize,
)


def dequantize_and_run(func, args, kwargs):
    # TODO(ryand): Use the highest input precision of non-quantized inputs instead of hardcoding torch.float32.
    dequantized_args = [
        a.get_dequantized_tensor(dtype=torch.bfloat16) if hasattr(a, "get_dequantized_tensor") else a for a in args
    ]
    dequantized_kwargs = {
        k: v.get_dequantized_tensor(dtype=torch.bfloat16) if hasattr(v, "get_dequantized_tensor") else v
        for k, v in kwargs.items()
    }
    return func(*dequantized_args, **dequantized_kwargs)


def apply_to_quantized_tensor(func, args, kwargs):
    ggml_tensor = args[0]
    assert isinstance(ggml_tensor, GGMLTensor)
    new_data = func(ggml_tensor._data, *args[1:], **kwargs)
    return GGMLTensor(new_data, ggml_tensor._ggml_quantization_type, ggml_tensor._tensor_shape)


GGML_TENSOR_OP_TABLE = {
    torch.ops.aten.detach.default: apply_to_quantized_tensor,
    torch.ops.aten._to_copy.default: apply_to_quantized_tensor,
    # --
    torch.ops.aten.t.default: dequantize_and_run,
    torch.ops.aten.addmm.default: dequantize_and_run,
    torch.ops.aten.mul.Tensor: dequantize_and_run,
}


class GGMLTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data: torch.Tensor, ggml_quantization_type: gguf.GGMLQuantizationType, tensor_shape: torch.Size):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            dtype=data.dtype,
            layout=data.layout,
            device=data.device,
            strides=data.stride(),
            storage_offset=data.storage_offset(),
        )

    def __init__(self, data: torch.Tensor, ggml_quantization_type: gguf.GGMLQuantizationType, tensor_shape: torch.Size):
        self._data = data
        self._ggml_quantization_type = ggml_quantization_type
        # The dequantized shape of the tensor.
        self._tensor_shape = tensor_shape

    def __repr__(self):
        return f"GGMLTensor(type={self._ggml_quantization_type.name}, dequantized_shape=({self._tensor_shape})"

    def size(self):
        return self._tensor_shape

    @property
    def shape(self):
        return self.size()

    def requires_grad_(self, requires_grad: bool = True):
        # TODO(ryand): Think about whether we should set requires_grad on the underlying tensor.
        return self

    def get_dequantized_tensor(self, dtype: torch.dtype):
        """Return the dequantized tensor.

        Args:
            dtype: The dtype of the dequantized tensor.
        """
        if self._ggml_quantization_type in TORCH_COMPATIBLE_QTYPES:
            return self._data.to(dtype)
        elif self._ggml_quantization_type in DEQUANTIZE_FUNCTIONS:
            # TODO(ryand): Look into how the dtype param is intended to be used.
            return dequantize(
                data=self._data, qtype=self._ggml_quantization_type, oshape=self._tensor_shape, dtype=None
            ).to(dtype)
        else:
            # There is no GPU implementation for this quantization type, so fallback to the numpy implementation.
            new = gguf.quants.dequantize(self._data.cpu().numpy(), self._ggml_quantization_type)
            return torch.from_numpy(new).to(self._data.device, dtype=dtype)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in GGML_TENSOR_OP_TABLE:
            return GGML_TENSOR_OP_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"Unsupported function {func}")
