"""DyPE base configuration and utilities for FLUX vision_yarn RoPE."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DyPEConfig:
    """Configuration for Dynamic Position Extrapolation."""

    enable_dype: bool = True
    base_resolution: int = 1024  # Native training resolution
    dype_scale: float = 2.0  # Magnitude λs (0.0-8.0)
    dype_exponent: float = 2.0  # Decay speed λt (0.0-1000.0)
    dype_start_sigma: float = 1.0  # When DyPE decay starts


def get_timestep_kappa(
    current_sigma: float,
    dype_scale: float,
    dype_exponent: float,
    dype_start_sigma: float,
) -> float:
    """Calculate the paper-style DyPE scheduler value κ(t).

    The key insight of DyPE: early steps focus on low frequencies (global structure),
    late steps on high frequencies (details). DyPE expresses this as a direct
    timestep scheduler over the positional extrapolation strength:

        κ(t) = λs * t^λt

    Args:
        current_sigma: Current noise level (1.0 = full noise, 0.0 = clean)
        dype_scale: DyPE magnitude (λs)
        dype_exponent: DyPE decay speed (λt)
        dype_start_sigma: Sigma threshold to start decay

    Returns:
        Timestep scheduler value κ(t)
    """
    if dype_scale <= 0.0 or dype_start_sigma <= 0.0:
        return 0.0

    t_normalized = max(0.0, min(current_sigma / dype_start_sigma, 1.0))
    return dype_scale * (t_normalized**dype_exponent)


def compute_vision_yarn_freqs(
    pos: Tensor,
    dim: int,
    theta: int,
    scale_h: float,
    scale_w: float,
    current_sigma: float,
    dype_config: DyPEConfig,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE frequencies using NTK-aware scaling for high-resolution.

    This method extends FLUX's position encoding to handle resolutions beyond
    the 1024px training resolution by scaling the base frequency (theta).

    The NTK-aware approach smoothly interpolates frequencies to cover larger
    position ranges without breaking the attention patterns.

    DyPE (Dynamic Position Extrapolation) modulates the NTK scaling based on
    the current timestep - stronger extrapolation in early steps (global structure),
    weaker in late steps (fine details).

    Args:
        pos: Position tensor
        dim: Embedding dimension
        theta: RoPE base frequency
        scale_h: Height scaling factor
        scale_w: Width scaling factor
        current_sigma: Current noise level (1.0 = full noise, 0.0 = clean)
        dype_config: DyPE configuration

    Returns:
        Tuple of (cos, sin) frequency tensors
    """
    assert dim % 2 == 0

    scale = max(scale_h, scale_w)

    device = pos.device
    dtype = torch.float64 if device.type != "mps" else torch.float32

    # DyPE applies a direct timestep scheduler to the NTK extrapolation exponent.
    # Early steps keep strong extrapolation; late steps relax smoothly back
    # toward the training-time RoPE.
    if scale > 1.0:
        ntk_exponent = dim / (dim - 2)
        kappa = get_timestep_kappa(
            current_sigma=current_sigma,
            dype_scale=dype_config.dype_scale,
            dype_exponent=dype_config.dype_exponent,
            dype_start_sigma=dype_config.dype_start_sigma,
        )
        scaled_theta = theta * (scale ** (ntk_exponent * kappa))
    else:
        scaled_theta = theta

    # Standard RoPE frequency computation
    freq_seq = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
    freqs = 1.0 / (scaled_theta**freq_seq)

    # Compute angles = position * frequency
    angles = torch.einsum("...n,d->...nd", pos.to(dtype), freqs)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos.to(pos.dtype), sin.to(pos.dtype)
