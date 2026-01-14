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
    FP8_E4M3 = "fp8_e4m3"  # FP8 E4M3 format
    FP8_E5M2 = "fp8_e5m2"  # FP8 E5M2 format


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
    return weight.to(dtype) * scale.to(dtype)


def dequantize_asymmetric(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Asymmetric dequantization: result = (weight - zero_point) * scale.

    Args:
        weight: Quantized weight tensor.
        scale: Scale factor tensor.
        zero_point: Zero point tensor for asymmetric quantization.
        dtype: Target dtype for the result.

    Returns:
        Dequantized tensor.
    """
    return (weight.to(dtype) - zero_point.to(dtype)) * scale.to(dtype)


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
        svd_correction = svd_up.to(dtype) @ svd_down.to(dtype)
        return dequantized + svd_correction
    return dequantized
