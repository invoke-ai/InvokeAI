"""DyPE base configuration and utilities.

Implements Dynamic Position Extrapolation (DyPE) with YaRN-style frequency blending.
Based on ComfyUI-DyPE: https://github.com/wildminder/ComfyUI-DyPE
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

# YaRN default parameters for FLUX
# These define the frequency correction ranges for blending
YARN_BETA_0 = 1.25  # Low-frequency ratio (β₀)
YARN_BETA_1 = 0.75  # High-frequency ratio (β₁)
YARN_GAMMA_0 = 16.0  # Original position range (γ₀)
YARN_GAMMA_1 = 2.0  # Extended position range (γ₁)

# FLUX model constants
# Base position embedding length = base_resolution / patch_size / packing = 1024/8/2 = 64
FLUX_BASE_PE_LEN = 64


@dataclass
class DyPEConfig:
    """Configuration for Dynamic Position Extrapolation."""

    enable_dype: bool = True
    base_resolution: int = 1024  # Native training resolution
    method: Literal["vision_yarn", "yarn", "ntk", "base"] = "vision_yarn"
    dype_scale: float = 2.0  # Magnitude λs (0.0-8.0)
    dype_exponent: float = 2.0  # Decay speed λt (0.0-1000.0)
    dype_start_sigma: float = 1.0  # When DyPE decay starts


def get_mscale(scale: float) -> float:
    """Calculate magnitude scaling factor (mscale).

    Uses the formula from YaRN paper: mscale = 1 + 0.1 * log(s) / sqrt(s)
    This provides better attention score normalization for high-resolution.

    Args:
        scale: The resolution scaling factor (NTK scale)

    Returns:
        The magnitude scaling factor
    """
    if scale <= 1.0:
        return 1.0
    return 1.0 + 0.1 * math.log(scale) / math.sqrt(scale)


def find_correction_factor(
    num_rotations: int,
    dim: int,
    base: int,
    max_position_embeddings: int,
) -> float:
    """Calculate correction factor for YaRN frequency masking.

    Finds the dimension index where the wavelength equals a given number of rotations.
    Used to determine which frequency components need interpolation.

    Args:
        num_rotations: Number of rotations to find the factor for
        dim: Embedding dimension
        base: RoPE base frequency (theta)
        max_position_embeddings: Original maximum position embeddings

    Returns:
        The dimension index (can be fractional) where wavelength matches num_rotations
    """
    # Wavelength at dimension d = 2π * base^(d/dim)
    # We want to find d where: wavelength / max_pe_len = num_rotations
    # => 2π * base^(d/dim) = num_rotations * max_pe_len
    # => d = dim * log(num_rotations * max_pe_len / 2π) / log(base)
    return (dim * math.log(max_position_embeddings / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(base))


def find_correction_range(
    low_ratio: float,
    high_ratio: float,
    dim: int,
    base: int,
    ori_max_pe_len: int,
) -> tuple[float, float]:
    """Find the dimension range for frequency correction.

    Determines the range of dimensions that need interpolation between
    different frequency scaling methods.

    Args:
        low_ratio: Low frequency ratio (beta or gamma low)
        high_ratio: High frequency ratio (beta or gamma high)
        dim: Embedding dimension
        base: RoPE base frequency (theta)
        ori_max_pe_len: Original maximum position embedding length

    Returns:
        Tuple of (low_dim, high_dim) indices for the correction range
    """
    low = math.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = math.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0.0), min(high, dim - 1.0)


def linear_ramp_mask(
    min_val: float,
    max_val: float,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create linear interpolation mask between frequency bands.

    Creates a tensor that ramps linearly from 0 to 1 between min_val and max_val,
    with values clamped to [0, 1] outside that range.

    Args:
        min_val: Dimension index where mask starts (becomes > 0)
        max_val: Dimension index where mask ends (becomes 1)
        dim: Number of frequency components (half the embedding dimension)
        device: Target device for the tensor
        dtype: Target dtype for the tensor

    Returns:
        Tensor of shape (dim,) with values in [0, 1]
    """
    if max_val <= min_val:
        # Degenerate case: no interpolation range, return step function
        indices = torch.arange(dim, device=device, dtype=dtype)
        return (indices >= min_val).to(dtype)

    # Linear ramp: (i - min) / (max - min), clamped to [0, 1]
    indices = torch.arange(dim, device=device, dtype=dtype)
    mask = (indices - min_val) / (max_val - min_val)
    return torch.clamp(mask, 0.0, 1.0)


def compute_dype_k_t(
    current_sigma: float,
    dype_scale: float,
    dype_exponent: float,
    dype_start_sigma: float,
) -> float:
    """Compute the DyPE timestep modulation factor k_t.

    The key insight of DyPE: early steps focus on low frequencies (global structure),
    late steps on high frequencies (details). This function computes k_t which
    modulates the YaRN correction parameters (beta, gamma).

    Formula: k_t = dype_scale * (timestep ^ dype_exponent)

    Where timestep is the normalized sigma value (0 at end, 1 at start of denoising).

    Args:
        current_sigma: Current noise level (1.0 = full noise, 0.0 = clean)
        dype_scale: DyPE magnitude (λs, 0.0-8.0)
        dype_exponent: DyPE decay speed (λt, 0.0-1000.0)
        dype_start_sigma: Sigma threshold to start decay

    Returns:
        k_t modulation factor - larger values mean stronger extrapolation
    """
    # Normalize sigma to [0, 1] range relative to start_sigma
    if current_sigma >= dype_start_sigma:
        timestep = 1.0
    else:
        timestep = current_sigma / dype_start_sigma

    # DyPE formula: k_t = scale * (timestep ^ exponent)
    # At timestep=1 (early, high sigma): k_t = dype_scale
    # At timestep=0 (late, low sigma): k_t = 0
    k_t = dype_scale * (timestep**dype_exponent)

    return k_t


def compute_timestep_mscale(
    ntk_scale: float,
    current_sigma: float,
    dype_config: DyPEConfig,
) -> float:
    """Compute timestep-dependent magnitude scaling.

    Interpolates from aggressive mscale at early steps to 1.0 at late steps.
    Matches ComfyUI-DyPE's _get_mscale behavior.

    Args:
        ntk_scale: Global NTK scaling factor
        current_sigma: Current noise level (1.0 = full noise, 0.0 = clean)
        dype_config: DyPE configuration

    Returns:
        Timestep-modulated magnitude scaling factor
    """
    if ntk_scale <= 1.0:
        return 1.0

    # Aggressive mscale formula (start value at high sigma)
    mscale_start = 0.1 * math.log(ntk_scale) + 1.0
    mscale_end = 1.0

    # Normalize sigma
    if current_sigma >= dype_config.dype_start_sigma:
        t_norm = 1.0
    else:
        t_norm = current_sigma / dype_config.dype_start_sigma

    # Interpolate: full mscale at early steps, 1.0 at late steps
    return mscale_end + (mscale_start - mscale_end) * (t_norm**dype_config.dype_exponent)


def compute_vision_yarn_freqs(
    pos: Tensor,
    dim: int,
    theta: int,
    linear_scale: float,
    ntk_scale: float,
    current_sigma: float,
    dype_config: DyPEConfig,
    ori_max_pe_len: int = FLUX_BASE_PE_LEN,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE frequencies using DyPE-modulated YaRN 3-band blending.

    Uses three frequency bands (base, linear, NTK) blended via beta/gamma masks.
    DyPE modulates the correction ranges so early denoising steps use stronger
    extrapolation (global structure) and late steps use weaker (fine details).

    Args:
        pos: Position tensor
        dim: Embedding dimension
        theta: RoPE base frequency
        linear_scale: Per-axis linear scaling factor (height or width ratio)
        ntk_scale: Global NTK scaling factor (max of height/width ratios)
        current_sigma: Current noise level (1.0 = full noise, 0.0 = clean)
        dype_config: DyPE configuration
        ori_max_pe_len: Original maximum position embedding length for YaRN correction

    Returns:
        Tuple of (cos, sin) frequency tensors
    """
    assert dim % 2 == 0
    n_freqs = dim // 2

    device = pos.device
    dtype = torch.float64 if device.type != "mps" else torch.float32

    linear_scale = max(linear_scale, 1.0)
    ntk_scale = max(ntk_scale, 1.0)

    if ntk_scale <= 1.0:
        # No scaling needed - use base frequencies
        freq_seq = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
        freqs = 1.0 / (theta**freq_seq)
        angles = torch.einsum("...n,d->...nd", pos.to(dtype), freqs)
        return torch.cos(angles).to(pos.dtype), torch.sin(angles).to(pos.dtype)

    # === Step 1: Compute DyPE modulation factor k_t ===
    k_t = compute_dype_k_t(
        current_sigma=current_sigma,
        dype_scale=dype_config.dype_scale,
        dype_exponent=dype_config.dype_exponent,
        dype_start_sigma=dype_config.dype_start_sigma,
    )

    # === Step 2: DyPE-modulate YaRN correction parameters ===
    beta_0: float = YARN_BETA_0
    beta_1: float = YARN_BETA_1
    gamma_0: float = YARN_GAMMA_0
    gamma_1: float = YARN_GAMMA_1

    if dype_config.enable_dype:
        beta_0 = beta_0**k_t
        beta_1 = beta_1**k_t
        gamma_0 = gamma_0**k_t
        gamma_1 = gamma_1**k_t

    # === Step 3: Compute three frequency bands ===
    freq_seq = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim

    # Band 1: Base frequencies (original, unscaled)
    freqs_base = 1.0 / (theta**freq_seq)

    # Band 2: Linearly scaled frequencies (per-axis)
    freqs_linear = freqs_base / linear_scale

    # Band 3: NTK-scaled frequencies (global)
    new_base = theta * (ntk_scale ** (dim / (dim - 2)))
    freqs_ntk = 1.0 / (new_base**freq_seq)

    # === Step 4: Beta mask - blend linear <-> NTK ===
    low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
    low = max(0, low)
    high = min(n_freqs, high)
    mask_beta = 1.0 - linear_ramp_mask(low, high, n_freqs, device, dtype)
    freqs = freqs_linear * (1.0 - mask_beta) + freqs_ntk * mask_beta

    # === Step 5: Gamma mask - blend result <-> base ===
    low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
    low = max(0, low)
    high = min(n_freqs, high)
    mask_gamma = 1.0 - linear_ramp_mask(low, high, n_freqs, device, dtype)
    freqs = freqs * (1.0 - mask_gamma) + freqs_base * mask_gamma

    # === Step 6: Compute angles ===
    angles = torch.einsum("...n,d->...nd", pos.to(dtype), freqs)

    # === Step 7: Apply timestep-dependent mscale ===
    mscale = compute_timestep_mscale(ntk_scale, current_sigma, dype_config)

    cos = torch.cos(angles) * mscale
    sin = torch.sin(angles) * mscale

    return cos.to(pos.dtype), sin.to(pos.dtype)


def compute_yarn_freqs(
    pos: Tensor,
    dim: int,
    theta: int,
    scale: float,
    current_sigma: float,
    dype_config: DyPEConfig,
    ori_max_pe_len: int = FLUX_BASE_PE_LEN,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE frequencies using DyPE-modulated YaRN with uniform scale.

    Uses the same 3-band blending as vision_yarn but with a uniform scale
    factor for both linear and NTK components.

    Args:
        pos: Position tensor
        dim: Embedding dimension
        theta: RoPE base frequency
        scale: Uniform scaling factor (used for both linear and NTK)
        current_sigma: Current noise level (1.0 = full noise, 0.0 = clean)
        dype_config: DyPE configuration
        ori_max_pe_len: Original maximum position embedding length

    Returns:
        Tuple of (cos, sin) frequency tensors
    """
    return compute_vision_yarn_freqs(
        pos=pos,
        dim=dim,
        theta=theta,
        linear_scale=scale,
        ntk_scale=scale,
        current_sigma=current_sigma,
        dype_config=dype_config,
        ori_max_pe_len=ori_max_pe_len,
    )


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
