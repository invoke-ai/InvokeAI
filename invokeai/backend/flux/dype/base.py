"""DyPE base configuration and utilities."""

from dataclasses import dataclass
from typing import Literal

import math
import torch
from torch import Tensor


@dataclass
class DyPEConfig:
    """Configuration for Dynamic Position Extrapolation."""

    enable_dype: bool = True
    base_resolution: int = 1024  # Native training resolution
    method: Literal["vision_yarn", "yarn", "ntk", "base"] = "vision_yarn"
    dype_scale: float = 2.0  # Magnitude λs (0.0-8.0)
    dype_exponent: float = 2.0  # Decay speed λt (0.0-1000.0)
    dype_start_sigma: float = 1.0  # When DyPE decay starts


def get_mscale(scale: float, mscale_factor: float = 1.0) -> float:
    """Calculate magnitude scaling factor.

    Args:
        scale: The resolution scaling factor
        mscale_factor: Adjustment factor for the scaling

    Returns:
        The magnitude scaling factor
    """
    if scale <= 1.0:
        return 1.0
    return mscale_factor * math.log(scale) + 1.0


def get_timestep_mscale(
    scale: float,
    current_sigma: float,
    dype_scale: float,
    dype_exponent: float,
    dype_start_sigma: float,
) -> float:
    """Calculate timestep-dependent magnitude scaling.

    The key insight of DyPE: early steps focus on low frequencies (global structure),
    late steps on high frequencies (details). This function modulates the scaling
    based on the current timestep/sigma.

    Args:
        scale: Resolution scaling factor
        current_sigma: Current noise level (1.0 = full noise, 0.0 = clean)
        dype_scale: DyPE magnitude (λs)
        dype_exponent: DyPE decay speed (λt)
        dype_start_sigma: Sigma threshold to start decay

    Returns:
        Timestep-modulated scaling factor
    """
    if scale <= 1.0:
        return 1.0

    # Normalize sigma to [0, 1] range relative to start_sigma
    if current_sigma >= dype_start_sigma:
        t_normalized = 1.0
    else:
        t_normalized = current_sigma / dype_start_sigma

    # Apply exponential decay: stronger extrapolation early, weaker late
    # decay = exp(-λt * (1 - t))  where t=1 is early (high sigma), t=0 is late
    decay = math.exp(-dype_exponent * (1.0 - t_normalized))

    # Base mscale from resolution
    base_mscale = get_mscale(scale)

    # Interpolate between base_mscale and 1.0 based on decay and dype_scale
    # When decay=1 (early): use scaled value
    # When decay=0 (late): use base value
    scaled_mscale = 1.0 + (base_mscale - 1.0) * dype_scale * decay

    return scaled_mscale


def compute_vision_yarn_freqs(
    pos: Tensor,
    dim: int,
    theta: int,
    scale_h: float,
    scale_w: float,
    current_sigma: float,
    dype_config: DyPEConfig,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE frequencies using Vision-YARN method.

    Vision-YARN is optimized for image models with separate H/W axes.
    It applies different scaling to height and width dimensions.

    Args:
        pos: Position tensor
        dim: Embedding dimension
        theta: RoPE base frequency
        scale_h: Height scaling factor
        scale_w: Width scaling factor
        current_sigma: Current noise level
        dype_config: DyPE configuration

    Returns:
        Tuple of (cos, sin) frequency tensors
    """
    assert dim % 2 == 0

    # Use the larger scale for mscale calculation
    scale = max(scale_h, scale_w)

    # Get timestep-dependent mscale
    mscale = get_timestep_mscale(
        scale=scale,
        current_sigma=current_sigma,
        dype_scale=dype_config.dype_scale,
        dype_exponent=dype_config.dype_exponent,
        dype_start_sigma=dype_config.dype_start_sigma,
    )

    # Compute base frequencies with scaling
    device = pos.device
    dtype = torch.float64 if device.type != "mps" else torch.float32

    # Scale theta based on resolution scaling (NTK-aware)
    scaled_theta = theta * (scale ** (dim / (dim - 2)))

    freq_seq = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
    freqs = 1.0 / (scaled_theta**freq_seq)

    # Apply mscale to frequencies
    freqs = freqs * mscale

    # Compute position * frequency
    # pos shape: (..., ) -> expand for broadcasting
    angles = torch.einsum("...n,d->...nd", pos.to(dtype), freqs)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos.to(pos.dtype), sin.to(pos.dtype)


def compute_yarn_freqs(
    pos: Tensor,
    dim: int,
    theta: int,
    scale: float,
    current_sigma: float,
    dype_config: DyPEConfig,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE frequencies using YARN method.

    Standard YARN with isotropic scaling.

    Args:
        pos: Position tensor
        dim: Embedding dimension
        theta: RoPE base frequency
        scale: Uniform scaling factor
        current_sigma: Current noise level
        dype_config: DyPE configuration

    Returns:
        Tuple of (cos, sin) frequency tensors
    """
    assert dim % 2 == 0

    mscale = get_timestep_mscale(
        scale=scale,
        current_sigma=current_sigma,
        dype_scale=dype_config.dype_scale,
        dype_exponent=dype_config.dype_exponent,
        dype_start_sigma=dype_config.dype_start_sigma,
    )

    device = pos.device
    dtype = torch.float64 if device.type != "mps" else torch.float32

    # YARN scaling with aggressive mscale
    scaled_theta = theta * (scale ** (dim / (dim - 2)))

    freq_seq = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
    freqs = 1.0 / (scaled_theta**freq_seq) * mscale

    angles = torch.einsum("...n,d->...nd", pos.to(dtype), freqs)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos.to(pos.dtype), sin.to(pos.dtype)


def compute_ntk_freqs(
    pos: Tensor,
    dim: int,
    theta: int,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE frequencies using NTK method.

    Neural Tangent Kernel approach - continuous frequency scaling without
    timestep dependency.

    Args:
        pos: Position tensor
        dim: Embedding dimension
        theta: RoPE base frequency
        scale: Scaling factor

    Returns:
        Tuple of (cos, sin) frequency tensors
    """
    assert dim % 2 == 0

    device = pos.device
    dtype = torch.float64 if device.type != "mps" else torch.float32

    # NTK scaling
    scaled_theta = theta * (scale ** (dim / (dim - 2)))

    freq_seq = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
    freqs = 1.0 / (scaled_theta**freq_seq)

    angles = torch.einsum("...n,d->...nd", pos.to(dtype), freqs)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos.to(pos.dtype), sin.to(pos.dtype)
