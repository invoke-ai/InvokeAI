import gguf
import torch

from invokeai.backend.quantization.gguf.utils import (
    DEQUANTIZE_FUNCTIONS,
    TORCH_COMPATIBLE_QTYPES,
    dequantize,
)


class GGMLTensor:
    def __init__(self, data: torch.Tensor, ggml_quantization_type: gguf.GGMLQuantizationType, tensor_shape: torch.Size):
        self._data = data
        self._ggml_quantization_type = ggml_quantization_type
        # The dequantized shape of the tensor.
        self._tensor_shape = tensor_shape

    def __repr__(self):
        return f"GGMLTensor(type={self._ggml_quantization_type.name}, dequantized_shape=({self._tensor_shape})"

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
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Most functions will work by simply running on the dequantized tensors, so we assume this as the default
        # behavior. Over time, we will have to add special handling for exceptions. For example, .to() will need special
        # handling.
        if func in []:
            return NotImplemented
        else:
            # TODO(ryand): Use the highest input precision of non-quantized inputs instead of hardcoding torch.float32.
            dequantized_args = [
                a.get_dequantized_tensor(dtype=torch.float32) if hasattr(a, "get_dequantized_tensor") else a
                for a in args
            ]
            dequantized_kwargs = {
                k: v.get_dequantized_tensor(dtype=torch.float32) if hasattr(v, "get_dequantized_tensor") else v
                for k, v in kwargs.items()
            }
            return func(*dequantized_args, **dequantized_kwargs)
