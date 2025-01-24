from typing import overload

import gguf
import torch

from invokeai.backend.quantization.gguf.utils import (
    DEQUANTIZE_FUNCTIONS,
    TORCH_COMPATIBLE_QTYPES,
    dequantize,
)


def dequantize_and_run(func, args, kwargs):
    """A helper function for running math ops on GGMLTensor inputs.

    Dequantizes the inputs, and runs the function.
    """
    dequantized_args = [a.get_dequantized_tensor() if hasattr(a, "get_dequantized_tensor") else a for a in args]
    dequantized_kwargs = {
        k: v.get_dequantized_tensor() if hasattr(v, "get_dequantized_tensor") else v for k, v in kwargs.items()
    }
    return func(*dequantized_args, **dequantized_kwargs)


def apply_to_quantized_tensor(func, args, kwargs):
    """A helper function to apply a function to a quantized GGML tensor, and re-wrap the result in a GGMLTensor.

    Assumes that the first argument is a GGMLTensor.
    """
    # We expect the first argument to be a GGMLTensor, and all other arguments to be non-GGMLTensors.
    ggml_tensor = args[0]
    assert isinstance(ggml_tensor, GGMLTensor)
    assert all(not isinstance(a, GGMLTensor) for a in args[1:])
    assert all(not isinstance(v, GGMLTensor) for v in kwargs.values())

    new_data = func(ggml_tensor.quantized_data, *args[1:], **kwargs)

    if new_data.dtype != ggml_tensor.quantized_data.dtype:
        # This is intended to catch calls such as `.to(dtype-torch.float32)`, which are not supported on GGMLTensors.
        raise ValueError("Operation changed the dtype of GGMLTensor unexpectedly.")

    return GGMLTensor(
        new_data, ggml_tensor._ggml_quantization_type, ggml_tensor.tensor_shape, ggml_tensor.compute_dtype
    )


GGML_TENSOR_OP_TABLE = {
    # Ops to run on the quantized tensor.
    torch.ops.aten.detach.default: apply_to_quantized_tensor,  # pyright: ignore
    torch.ops.aten._to_copy.default: apply_to_quantized_tensor,  # pyright: ignore
    torch.ops.aten.clone.default: apply_to_quantized_tensor,  # pyright: ignore
    # Ops to run on dequantized tensors.
    torch.ops.aten.t.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.addmm.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.mul.Tensor: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.add.Tensor: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.sub.Tensor: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.allclose.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.slice.Tensor: dequantize_and_run,  # pyright: ignore
}

if torch.backends.mps.is_available():
    GGML_TENSOR_OP_TABLE.update(
        {torch.ops.aten.linear.default: dequantize_and_run}  # pyright: ignore
    )


class GGMLTensor(torch.Tensor):
    """A torch.Tensor sub-class holding a quantized GGML tensor.

    The underlying tensor is quantized, but the GGMLTensor class provides a dequantized view of the tensor on-the-fly
    when it is used in operations.
    """

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        ggml_quantization_type: gguf.GGMLQuantizationType,
        tensor_shape: torch.Size,
        compute_dtype: torch.dtype,
    ):
        # Type hinting is not supported for torch.Tensor._make_wrapper_subclass, so we ignore the errors.
        return torch.Tensor._make_wrapper_subclass(  # pyright: ignore
            cls,
            data.shape,
            dtype=data.dtype,
            layout=data.layout,
            device=data.device,
            strides=data.stride(),
            storage_offset=data.storage_offset(),
        )

    def __init__(
        self,
        data: torch.Tensor,
        ggml_quantization_type: gguf.GGMLQuantizationType,
        tensor_shape: torch.Size,
        compute_dtype: torch.dtype,
    ):
        self.quantized_data = data
        self._ggml_quantization_type = ggml_quantization_type
        # The dequantized shape of the tensor.
        self.tensor_shape = tensor_shape
        self.compute_dtype = compute_dtype

    def __repr__(self, *, tensor_contents=None):
        return f"GGMLTensor(type={self._ggml_quantization_type.name}, dequantized_shape=({self.tensor_shape})"

    @overload
    def size(self, dim: None = None) -> torch.Size: ...

    @overload
    def size(self, dim: int) -> int: ...

    def size(self, dim: int | None = None):
        """Return the size of the tensor after dequantization. I.e. the shape that will be used in any math ops."""
        if dim is not None:
            return self.tensor_shape[dim]
        return self.tensor_shape

    @property
    def shape(self) -> torch.Size:  # pyright: ignore[reportIncompatibleVariableOverride] pyright doesn't understand this for some reason.
        """The shape of the tensor after dequantization. I.e. the shape that will be used in any math ops."""
        return self.size()

    @property
    def quantized_shape(self) -> torch.Size:
        """The shape of the quantized tensor."""
        return self.quantized_data.shape

    def requires_grad_(self, mode: bool = True) -> torch.Tensor:
        """The GGMLTensor class is currently only designed for inference (not training). Setting requires_grad to True
        is not supported. This method is a no-op.
        """
        return self

    def get_dequantized_tensor(self):
        """Return the dequantized tensor.

        Args:
            dtype: The dtype of the dequantized tensor.
        """
        if self._ggml_quantization_type in TORCH_COMPATIBLE_QTYPES:
            return self.quantized_data.to(self.compute_dtype)
        elif self._ggml_quantization_type in DEQUANTIZE_FUNCTIONS:
            # TODO(ryand): Look into how the dtype param is intended to be used.
            return dequantize(
                data=self.quantized_data, qtype=self._ggml_quantization_type, oshape=self.tensor_shape, dtype=None
            ).to(self.compute_dtype)
        else:
            # There is no GPU implementation for this quantization type, so fallback to the numpy implementation.
            new = gguf.quants.dequantize(self.quantized_data.cpu().numpy(), self._ggml_quantization_type)
            return torch.from_numpy(new).to(self.quantized_data.device, dtype=self.compute_dtype)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # We will likely hit cases here in the future where a new op is encountered that is not yet supported.
        # The new op simply needs to be added to the GGML_TENSOR_OP_TABLE.
        if func in GGML_TENSOR_OP_TABLE:
            return GGML_TENSOR_OP_TABLE[func](func, args, kwargs)
        return NotImplemented
