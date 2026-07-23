"""SDNQ (SD.Next Quantization) utility functions and enums."""

from enum import Enum
from typing import Optional

import torch


class SDNQQuantizationType(str, Enum):
    """SDNQ Quantization types from SD.Next."""

    INT8_SYM = "int8_sym"  # Symmetric Int8 quantization
    INT8_ASYM = "int8_asym"  # Asymmetric Int8 quantization
    UINT8_SYM = "uint8_sym"  # Symmetric UInt8 quantization
    UINT8_ASYM = "uint8_asym"  # Asymmetric UInt8 quantization
    UINT4_ASYM = "uint4_asym"  # Asymmetric UInt4 quantization (packed in uint8)
    INT5_ASYM = "int5_asym"  # Asymmetric signed 5-bit quantization (range -16..15, packed 8/5 bytes)
    FP8_E4M3 = "fp8_e4m3"  # FP8 E4M3 format
    FP8_E5M2 = "fp8_e5m2"  # FP8 E5M2 format


def unpack_uint4(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Unpack uint4 values from packed uint8 tensor.

    SDNQ stores uint4 values packed as 2 values per uint8 byte:
    - Lower 4 bits: first value (packed & 0x0F)
    - Upper 4 bits: second value (packed >> 4)

    Args:
        packed: Packed uint8 tensor with shape [..., N/2] or flattened 1D.
        original_shape: Original tensor shape before packing.

    Returns:
        Unpacked tensor with shape original_shape containing uint4 values (0-15).
    """
    # If packed is 1D but original_shape is 2D, reshape to packed 2D first
    # This ensures correct unpacking order
    if packed.dim() == 1 and len(original_shape) == 2:
        out_features, in_features = original_shape
        packed_in_features = in_features // 2
        packed = packed.view(out_features, packed_in_features)

    # Extract lower and upper 4 bits
    lower = torch.bitwise_and(packed, 15)
    upper = torch.bitwise_right_shift(packed, 4)

    # Standard packing order: lower nibble first, upper nibble second
    # byte = (upper << 4) | lower means lower is at even indices, upper at odd
    unpacked = torch.stack((lower, upper), dim=-1).view(original_shape)
    return unpacked


def unpack_uint5(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Unpack 5-bit values from packed uint8 tensor.

    SDNQ packs 8 5-bit values into 5 bytes (40 bits). The packed tensor has its
    last dim equal to 5, where each (..., 5) row encodes 8 unpacked values.
    Bit layout matches Disty0/sdnq's ``unpack_uint5`` so the produced order is the
    same as the producer's pack order.

    Args:
        packed: Packed uint8 tensor with shape (..., 5).
        original_shape: Original tensor shape before packing (must total to packed.numel() * 8 // 5).

    Returns:
        Unpacked tensor with shape ``original_shape`` containing values 0-31.
    """
    if packed.dim() < 2 or packed.shape[-1] != 5:
        raise ValueError(f"unpack_uint5 expects packed tensor with last dim = 5; got shape {tuple(packed.shape)}")
    result_bitwise_right_shift = torch.bitwise_right_shift(packed[..., :3], 5)
    result = torch.cat(
        (
            torch.bitwise_and(packed[..., :5], 31),
            torch.bitwise_or(
                result_bitwise_right_shift[..., :2],
                torch.bitwise_and(torch.bitwise_right_shift(packed[..., 3:5], 2), 24),
            ),
            torch.bitwise_or(
                result_bitwise_right_shift[..., 2:3],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_right_shift(packed[..., 3:4], 3), 16),
                    torch.bitwise_and(torch.bitwise_right_shift(packed[..., 4:5], 4), 8),
                ),
            ),
        ),
        dim=-1,
    ).view(original_shape)
    return result


def dequantize_symmetric(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Symmetric dequantization: result = weight * scale.

    Args:
        weight: Quantized weight tensor (int8, uint8, or fp8).
        scale: Scale factor tensor.
        dtype: Target dtype for the result.

    Returns:
        Dequantized tensor.
    """
    # Ensure scale is on the same device as weight
    if scale.device != weight.device:
        scale = scale.to(weight.device)

    # Handle scale broadcasting for different shapes
    # Scale might be: scalar, [1], [out_features], [out_features, 1], etc.
    scale = scale.to(dtype)
    weight = weight.to(dtype)

    # Reshape scale for broadcasting if needed
    if scale.dim() == 1 and weight.dim() == 2:
        # Per-channel scale for Linear: [out_features] -> [out_features, 1]
        scale = scale.unsqueeze(-1)
    elif scale.dim() == 1 and weight.dim() == 4:
        # Per-channel scale for Conv2d: [out_channels] -> [out_channels, 1, 1, 1]
        scale = scale.view(-1, 1, 1, 1)

    return weight * scale


def dequantize_asymmetric(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Asymmetric dequantization: result = weight * scale + zero_point.

    Note: SDNQ uses a different convention where zero_point is a pre-computed bias.
    Standard formula: result = (weight - zp) * scale
    SDNQ formula: result = weight * scale + zero_point (where zero_point = -zp * scale)

    Args:
        weight: Quantized weight tensor.
        scale: Scale factor tensor.
        zero_point: Zero point/bias tensor for asymmetric quantization.
        dtype: Target dtype for the result.

    Returns:
        Dequantized tensor.
    """
    # Ensure scale and zero_point are on the same device as weight
    if scale.device != weight.device:
        scale = scale.to(weight.device)
    if zero_point.device != weight.device:
        zero_point = zero_point.to(weight.device)

    # Convert to compute dtype
    scale = scale.to(dtype)
    zero_point = zero_point.to(dtype)
    weight = weight.to(dtype)

    # Reshape scale and zero_point for broadcasting if needed
    if scale.dim() == 1 and weight.dim() == 2:
        # Per-channel for Linear: [out_features] -> [out_features, 1]
        scale = scale.unsqueeze(-1)
    elif scale.dim() == 1 and weight.dim() == 4:
        # Per-channel for Conv2d: [out_channels] -> [out_channels, 1, 1, 1]
        scale = scale.view(-1, 1, 1, 1)

    if zero_point.dim() == 1 and weight.dim() == 2:
        zero_point = zero_point.unsqueeze(-1)
    elif zero_point.dim() == 1 and weight.dim() == 4:
        zero_point = zero_point.view(-1, 1, 1, 1)

    # SDNQ formula: x = q * scale + zero_point (zero_point is actually a bias)
    return weight * scale + zero_point


# Track whether we've done diagnostic logging
_uint4_diagnostic_done = False


def dequantize_uint4_per_group(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    original_shape: torch.Size,
    group_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize uint4 weights with per-group scaling.

    SDNQ uint4 quantization uses:
    - Packed uint8 storage (2 uint4 values per byte)
    - Per-group scale factors
    - Per-group zero points for asymmetric quantization

    The scale tensor has shape [out_features, num_groups, 1] where
    num_groups = in_features / group_size.

    Args:
        packed_weight: Packed uint8 tensor with shape [out_features, in_features/2].
        scale: Per-group scale tensor with shape [out_features, num_groups, 1].
        zero_point: Per-group zero point tensor with shape [out_features, num_groups, 1].
        original_shape: Original weight shape [out_features, in_features].
        group_size: Number of elements per quantization group.
        dtype: Target dtype for the result.

    Returns:
        Dequantized tensor with shape original_shape.
    """
    global _uint4_diagnostic_done

    # Ensure scale and zero_point are on the same device as packed_weight
    device = packed_weight.device
    if scale.device != device:
        scale = scale.to(device)
    if zero_point.device != device:
        zero_point = zero_point.to(device)

    # Per-group scale/zero_point may be stored as [out_features, num_groups, 1] or, without the
    # trailing singleton, as [out_features, num_groups]. The grouped weight below is 3D
    # ([out_features, num_groups, group_size]); a 2D scale would right-align (group_size vs
    # num_groups) and fail broadcasting, so normalize 2D params with a trailing singleton dim.
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    if zero_point.dim() == 2:
        zero_point = zero_point.unsqueeze(-1)

    # Unpack uint4 values
    unpacked = unpack_uint4(packed_weight, original_shape)

    out_features, in_features = original_shape
    num_groups = in_features // group_size

    # Reshape for per-group operations: [out_features, num_groups, group_size]
    weight_grouped = unpacked.view(out_features, num_groups, group_size).to(dtype)

    # SDNQ uses pre-computed bias: x = q * scale + zero_point
    # where zero_point = -original_zp * scale (already a floating point bias)
    scale_f = scale.to(dtype)
    zp_f = zero_point.to(dtype)
    dequantized = weight_grouped * scale_f + zp_f

    # Diagnostic logging (once)
    if not _uint4_diagnostic_done:
        _uint4_diagnostic_done = True
        print("[SDNQ uint4] Diagnostic:")
        print(f"  packed_weight: shape={packed_weight.shape}, dtype={packed_weight.dtype}")
        print(f"  unpacked: min={unpacked.min().item()}, max={unpacked.max().item()}, unique={len(unpacked.unique())}")
        print(f"  scale: shape={scale.shape}, dtype={scale.dtype}, range=[{scale.min():.6f}, {scale.max():.6f}]")
        print(
            f"  zero_point: shape={zero_point.shape}, dtype={zero_point.dtype}, range=[{zero_point.min():.6f}, {zero_point.max():.6f}]"
        )
        print(f"  group_size={group_size}, num_groups={num_groups}")
        # Sample values
        zp_sample = zero_point.flatten()[:10].tolist()
        print(f"  zero_point sample: {zp_sample}")
        sample_dequant = dequantized.flatten()[:10].tolist()
        print(f"  dequantized sample: {sample_dequant}")
        print(f"  dequantized range: [{dequantized.min():.6f}, {dequantized.max():.6f}]")

    # Reshape back to original shape
    return dequantized.view(original_shape)


def dequantize_int5_per_group(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    original_shape: torch.Size,
    group_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize signed 5-bit weights with per-group scaling.

    SDNQ packs 8 int5 values into 5 bytes. The raw unpacked bytes give 0..31; for signed
    int5 storage we shift by ``min = -16`` to get the actual range -16..15 (matches Disty0
    upstream's ``unpack_int`` sign-extension step).

    ``zero_point`` is optional: dynamic-mixed-precision SDNQ can emit scale-only 5-bit
    tensors when the per-group loss budget allows it.
    """
    device = packed_weight.device
    if scale.device != device:
        scale = scale.to(device)
    if zero_point is not None and zero_point.device != device:
        zero_point = zero_point.to(device)

    # Normalize [out_features, num_groups] scale/zero_point to [out_features, num_groups, 1] so they
    # broadcast against the 3D grouped weight below (a 2D param would right-align group_size against
    # num_groups and fail broadcasting).
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    if zero_point is not None and zero_point.dim() == 2:
        zero_point = zero_point.unsqueeze(-1)

    unpacked = unpack_uint5(packed_weight, original_shape)
    # Sign-extend: SDNQ stores signed int5 (-16..15) in unsigned 5-bit (0..31) by adding 16
    # at pack time. We subtract it back here. We cast to int8 first so the subtraction stays
    # within an exact integer domain before dtype promotion.
    signed = unpacked.to(torch.int8) - 16

    out_features, in_features = original_shape
    num_groups = in_features // group_size
    weight_grouped = signed.view(out_features, num_groups, group_size).to(dtype)
    scale_f = scale.to(dtype)
    dequantized = weight_grouped * scale_f
    if zero_point is not None:
        dequantized = dequantized + zero_point.to(dtype)
    return dequantized.view(original_shape)


def apply_svd_correction(
    dequantized: torch.Tensor,
    svd_up: Optional[torch.Tensor],
    svd_down: Optional[torch.Tensor],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Apply SVD correction: result = dequantized + svd_up @ svd_down.

    SVD (Singular Value Decomposition) correction adds a low-rank approximation
    to improve the accuracy of quantized weights.

    Args:
        dequantized: Already dequantized tensor.
        svd_up: SVD up matrix (U * S component).
        svd_down: SVD down matrix (V^T component).
        dtype: Target dtype for the result.

    Returns:
        Tensor with SVD correction applied.
    """
    if svd_up is not None and svd_down is not None:
        # Ensure SVD matrices are on the same device as dequantized tensor
        device = dequantized.device
        if svd_up.device != device:
            svd_up = svd_up.to(device)
        if svd_down.device != device:
            svd_down = svd_down.to(device)
        svd_correction = svd_up.to(dtype) @ svd_down.to(dtype)
        return dequantized + svd_correction
    return dequantized
