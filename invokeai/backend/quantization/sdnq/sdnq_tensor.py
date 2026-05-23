"""SDNQTensor - A torch.Tensor subclass for SDNQ quantized weights with on-the-fly dequantization."""

import logging
from typing import Optional, overload

import torch

from invokeai.backend.quantization.sdnq.utils import (
    SDNQQuantizationType,
    apply_svd_correction,
    dequantize_asymmetric,
    dequantize_symmetric,
    dequantize_uint4_per_group,
)

logger = logging.getLogger(__name__)


def dequantize_and_run(func, args, kwargs):
    """Helper function for running math ops on SDNQTensor inputs.

    Dequantizes the inputs and runs the function.
    Also casts other floating point tensors to match the compute_dtype of SDNQTensors.
    """
    compute_dtype = None
    target_device = None

    for a in args:
        if hasattr(a, "compute_dtype"):
            compute_dtype = a.compute_dtype
        if isinstance(a, torch.Tensor) and target_device is None:
            target_device = a.device
        if compute_dtype is not None and target_device is not None:
            break

    if compute_dtype is None or target_device is None:
        for v in kwargs.values():
            if hasattr(v, "compute_dtype") and compute_dtype is None:
                compute_dtype = v.compute_dtype
            if isinstance(v, torch.Tensor) and target_device is None:
                target_device = v.device
            if compute_dtype is not None and target_device is not None:
                break

    def process_tensor(t):
        if hasattr(t, "get_dequantized_tensor"):
            result = t.get_dequantized_tensor()
            if target_device is not None and result.device != target_device:
                result = result.to(target_device)
            return result
        elif isinstance(t, torch.Tensor) and compute_dtype is not None and t.is_floating_point():
            return t.to(compute_dtype)
        return t

    dequantized_args = [process_tensor(a) for a in args]
    dequantized_kwargs = {k: process_tensor(v) for k, v in kwargs.items()}
    return func(*dequantized_args, **dequantized_kwargs)


def apply_to_quantized_tensor(func, args, kwargs):
    """Apply function to quantized tensor and re-wrap result in SDNQTensor.

    Assumes that the first argument is an SDNQTensor.
    """
    sdnq_tensor = args[0]
    assert isinstance(sdnq_tensor, SDNQTensor)
    assert all(not isinstance(a, SDNQTensor) for a in args[1:])
    assert all(not isinstance(v, SDNQTensor) for v in kwargs.values())

    new_data = func(sdnq_tensor.quantized_data, *args[1:], **kwargs)

    if new_data.dtype != sdnq_tensor.quantized_data.dtype:
        raise ValueError("Operation changed the dtype of SDNQTensor unexpectedly.")

    return SDNQTensor(
        data=new_data,
        quantization_type=sdnq_tensor._quantization_type,
        tensor_shape=sdnq_tensor.tensor_shape,
        compute_dtype=sdnq_tensor.compute_dtype,
        scale=sdnq_tensor._scale,
        zero_point=sdnq_tensor._zero_point,
        svd_up=sdnq_tensor._svd_up,
        svd_down=sdnq_tensor._svd_down,
        group_size=sdnq_tensor._group_size,
    )


SDNQ_TENSOR_OP_TABLE = {
    # Ops to run on the quantized tensor (keep quantized).
    torch.ops.aten.detach.default: apply_to_quantized_tensor,  # pyright: ignore
    torch.ops.aten._to_copy.default: apply_to_quantized_tensor,  # pyright: ignore
    torch.ops.aten.clone.default: apply_to_quantized_tensor,  # pyright: ignore
    # Ops to run on dequantized tensors.
    torch.ops.aten.t.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.linear.default: dequantize_and_run,  # pyright: ignore - needed for F.linear
    torch.ops.aten.addmm.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.mm.default: dequantize_and_run,  # pyright: ignore - matrix multiply
    torch.ops.aten.bmm.default: dequantize_and_run,  # pyright: ignore - batch matrix multiply
    torch.ops.aten.baddbmm.default: dequantize_and_run,  # pyright: ignore - batch add batch mm
    torch.ops.aten.matmul.default: dequantize_and_run,  # pyright: ignore - general matmul
    # Element-wise ops
    torch.ops.aten.mul.Tensor: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.mul.Scalar: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.add.Tensor: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.add.Scalar: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.sub.Tensor: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.sub.Scalar: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.div.Tensor: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.div.Scalar: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.neg.default: dequantize_and_run,  # pyright: ignore
    # Shape manipulation ops
    torch.ops.aten.slice.Tensor: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.view.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.reshape.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten._unsafe_view.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.expand.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.permute.default: dequantize_and_run,  # pyright: ignore - attention reshaping
    torch.ops.aten.transpose.int: dequantize_and_run,  # pyright: ignore - attention
    torch.ops.aten.contiguous.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.squeeze.dim: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.squeeze.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.unsqueeze.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.select.int: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.split.Tensor: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.split_with_sizes.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.chunk.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.cat.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.stack.default: dequantize_and_run,  # pyright: ignore
    # Attention ops
    torch.ops.aten.scaled_dot_product_attention.default: dequantize_and_run,  # pyright: ignore - SDPA
    torch.ops.aten._scaled_dot_product_flash_attention.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten._scaled_dot_product_efficient_attention.default: dequantize_and_run,  # pyright: ignore
    # Normalization and activation ops
    torch.ops.aten.layer_norm.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.group_norm.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.native_layer_norm.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.native_group_norm.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.silu.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.gelu.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.relu.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.softmax.int: dequantize_and_run,  # pyright: ignore
    torch.ops.aten._softmax.default: dequantize_and_run,  # pyright: ignore
    # Reduction ops
    torch.ops.aten.mean.dim: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.sum.dim_IntList: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.var.correction: dequantize_and_run,  # pyright: ignore - for RMSNorm
    torch.ops.aten.std.correction: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.pow.Tensor_Scalar: dequantize_and_run,  # pyright: ignore - for RMSNorm
    torch.ops.aten.rsqrt.default: dequantize_and_run,  # pyright: ignore - for RMSNorm
    torch.ops.aten.sqrt.default: dequantize_and_run,  # pyright: ignore
    # Misc ops
    torch.ops.aten.allclose.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.index_put_.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.embedding.default: dequantize_and_run,  # pyright: ignore
    torch.ops.aten.copy_.default: dequantize_and_run,  # pyright: ignore
    # Conv ops - needed for VAE
    torch.ops.aten.convolution.default: dequantize_and_run,  # pyright: ignore - Conv2d uses this
    torch.ops.aten.conv2d.default: dequantize_and_run,  # pyright: ignore
}


class SDNQTensor(torch.Tensor):
    """A torch.Tensor subclass holding SDNQ quantized weights.

    Provides on-the-fly dequantization when used in operations.
    Supports symmetric/asymmetric quantization and optional SVD correction.
    """

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        quantization_type: SDNQQuantizationType,
        tensor_shape: torch.Size,
        compute_dtype: torch.dtype,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
        svd_up: Optional[torch.Tensor] = None,
        svd_down: Optional[torch.Tensor] = None,
        group_size: Optional[int] = None,
    ):
        # Use tensor_shape (the dequantized shape) for the wrapper, not data.shape (packed shape)
        # This ensures PyTorch's load_state_dict sees the correct shape for assignment
        # Compute strides for row-major (C-contiguous) layout
        strides = []
        stride = 1
        for dim in reversed(tensor_shape):
            strides.append(stride)
            stride *= dim
        strides = tuple(reversed(strides))

        return torch.Tensor._make_wrapper_subclass(  # pyright: ignore
            cls,
            tensor_shape,
            dtype=data.dtype,
            layout=data.layout,
            device=data.device,
            strides=strides,
            storage_offset=0,
        )

    def __init__(
        self,
        data: torch.Tensor,
        quantization_type: SDNQQuantizationType,
        tensor_shape: torch.Size,
        compute_dtype: torch.dtype,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor] = None,
        svd_up: Optional[torch.Tensor] = None,
        svd_down: Optional[torch.Tensor] = None,
        group_size: Optional[int] = None,
    ):
        self.quantized_data = data
        self._quantization_type = quantization_type
        self.tensor_shape = tensor_shape
        self.compute_dtype = compute_dtype
        self._scale = scale
        self._zero_point = zero_point
        self._svd_up = svd_up
        self._svd_down = svd_down
        self._group_size = group_size

    def __repr__(self, *, tensor_contents=None):
        return (
            f"SDNQTensor(type={self._quantization_type.value}, "
            f"dequantized_shape=({self.tensor_shape}), "
            f"has_svd={self._svd_up is not None})"
        )

    @overload
    def size(self, dim: None = None) -> torch.Size: ...

    @overload
    def size(self, dim: int) -> int: ...

    def size(self, dim: int | None = None):
        """Return the size of the tensor after dequantization."""
        if dim is not None:
            return self.tensor_shape[dim]
        return self.tensor_shape

    @property
    def shape(self) -> torch.Size:  # pyright: ignore[reportIncompatibleVariableOverride]
        """The shape of the tensor after dequantization."""
        return self.size()

    @property
    def quantized_shape(self) -> torch.Size:
        """The shape of the quantized tensor."""
        return self.quantized_data.shape

    @property
    def is_asymmetric(self) -> bool:
        """Whether this tensor uses asymmetric quantization."""
        return self._zero_point is not None

    @property
    def has_svd(self) -> bool:
        """Whether this tensor has SVD correction components."""
        return self._svd_up is not None and self._svd_down is not None

    def requires_grad_(self, mode: bool = True) -> torch.Tensor:
        """SDNQTensor is only for inference, not training. This is a no-op."""
        return self

    # Track which tensors we've logged (to avoid spam)
    _logged_tensors: set = set()

    def get_dequantized_tensor(self) -> torch.Tensor:
        """Return the dequantized tensor.

        Returns:
            Dequantized tensor with compute_dtype.
        """
        # Debug logging for first few tensors
        tensor_id = id(self)
        should_log = tensor_id not in SDNQTensor._logged_tensors and len(SDNQTensor._logged_tensors) < 5
        if should_log:
            SDNQTensor._logged_tensors.add(tensor_id)
            print(
                f"[SDNQ] dequantize: type={self._quantization_type.value}, "
                f"weight_shape={self.quantized_data.shape}, weight_dtype={self.quantized_data.dtype}, "
                f"scale_shape={self._scale.shape}, zp={self._zero_point is not None}, "
                f"svd={self.has_svd}, group_size={self._group_size}"
            )
            logger.info(
                f"SDNQ dequantize: type={self._quantization_type.value}, "
                f"weight_shape={self.quantized_data.shape}, weight_dtype={self.quantized_data.dtype}, "
                f"scale_shape={self._scale.shape}, scale_range=[{self._scale.min():.6f}, {self._scale.max():.6f}], "
                f"zp={self._zero_point is not None}, svd={self.has_svd}, group_size={self._group_size}"
            )
            if self._zero_point is not None:
                logger.info(
                    f"  zero_point_shape={self._zero_point.shape}, "
                    f"zp_range=[{self._zero_point.min():.6f}, {self._zero_point.max():.6f}]"
                )

        # Perform dequantization based on quantization type
        if self._quantization_type == SDNQQuantizationType.UINT4_ASYM:
            # uint4 with per-group quantization
            assert self._zero_point is not None
            assert self._group_size is not None
            dequantized = dequantize_uint4_per_group(
                self.quantized_data,
                self._scale,
                self._zero_point,
                self.tensor_shape,
                self._group_size,
                dtype=self.compute_dtype,
            )
        elif self.is_asymmetric:
            assert self._zero_point is not None
            dequantized = dequantize_asymmetric(
                self.quantized_data,
                self._scale,
                self._zero_point,
                dtype=self.compute_dtype,
            )
        else:
            dequantized = dequantize_symmetric(
                self.quantized_data,
                self._scale,
                dtype=self.compute_dtype,
            )

        # Apply SVD correction if present
        if self.has_svd:
            dequantized = apply_svd_correction(
                dequantized,
                self._svd_up,
                self._svd_down,
                dtype=self.compute_dtype,
            )

        # Reshape to original tensor shape if needed
        if dequantized.shape != self.tensor_shape:
            dequantized = dequantized.view(self.tensor_shape)

        result = dequantized.to(self.compute_dtype)

        # Log output range for debugging
        if should_log:
            logger.info(f"  -> output_shape={result.shape}, output_range=[{result.min():.6f}, {result.max():.6f}]")

        return result

    # Track unknown ops to avoid spamming warnings
    _warned_ops: set = set()

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in SDNQ_TENSOR_OP_TABLE:
            return SDNQ_TENSOR_OP_TABLE[func](func, args, kwargs)

        # Fallback: dequantize and run for unknown operations
        # This ensures computation works, but may indicate missing ops
        if func not in cls._warned_ops:
            cls._warned_ops.add(func)
            import logging

            logging.getLogger(__name__).warning(
                f"SDNQTensor: unknown op {func}, dequantizing (add to op table for efficiency)"
            )

        return dequantize_and_run(func, args, kwargs)
