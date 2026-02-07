"""DyPE-enhanced RoPE (Rotary Position Embedding) functions."""

import torch
from einops import rearrange
from torch import Tensor

from invokeai.backend.flux.dype.base import (
    FLUX_BASE_PE_LEN,
    DyPEConfig,
    compute_ntk_freqs,
    compute_timestep_mscale,
    compute_vision_yarn_freqs,
    compute_yarn_freqs,
)


def rope_dype(
    pos: Tensor,
    dim: int,
    theta: int,
    current_sigma: float,
    target_height: int,
    target_width: int,
    dype_config: DyPEConfig,
    axis_index: int = 0,
    ori_max_pe_len: int = FLUX_BASE_PE_LEN,
) -> Tensor:
    """Compute RoPE with Dynamic Position Extrapolation.

    This is the core DyPE function that replaces the standard rope() function.
    It applies resolution-aware and timestep-aware scaling to position embeddings.

    DyPE scaling is only applied to spatial axes (axis_index > 0). Axis 0
    (time/channel) always uses plain RoPE to avoid distorting temporal attention.

    Args:
        pos: Position indices tensor
        dim: Embedding dimension per axis
        theta: RoPE base frequency (typically 10000)
        current_sigma: Current noise level (1.0 = full noise, 0.0 = clean)
        target_height: Target image height in pixels
        target_width: Target image width in pixels
        dype_config: DyPE configuration
        axis_index: Which axis this is (0=time/channel, 1=height, 2=width).
            Axis 0 always uses plain RoPE without DyPE scaling.
        ori_max_pe_len: Original maximum position embedding length for YaRN correction

    Returns:
        Rotary position embedding tensor with shape suitable for FLUX attention
    """
    assert dim % 2 == 0

    # Axis 0 (time/channel) never gets DyPE scaling - only spatial axes do
    if axis_index == 0:
        return _rope_base(pos, dim, theta)

    # Calculate scaling factors
    base_res = dype_config.base_resolution
    scale_h = target_height / base_res
    scale_w = target_width / base_res
    scale = max(scale_h, scale_w)

    # If no scaling needed and DyPE disabled, use base method
    if not dype_config.enable_dype or scale <= 1.0:
        return _rope_base(pos, dim, theta)

    # Compute per-axis linear_scale and global ntk_scale
    # linear_scale: the scale for THIS specific axis (height or width)
    # ntk_scale: the global scale = max(scale_h, scale_w)
    if axis_index == 1:
        linear_scale = scale_h
    elif axis_index == 2:
        linear_scale = scale_w
    else:
        linear_scale = scale
    ntk_scale = scale

    # Select method and compute frequencies
    method = dype_config.method

    if method == "vision_yarn":
        # Compute timestep-dependent mscale (matches ComfyUI-DyPE's _get_mscale)
        mscale = compute_timestep_mscale(ntk_scale, current_sigma, dype_config)
        cos, sin = compute_vision_yarn_freqs(
            pos=pos,
            dim=dim,
            theta=theta,
            linear_scale=linear_scale,
            ntk_scale=ntk_scale,
            current_sigma=current_sigma,
            dype_config=dype_config,
            ori_max_pe_len=ori_max_pe_len,
            mscale_override=mscale,
        )
    elif method == "yarn":
        cos, sin = compute_yarn_freqs(
            pos=pos,
            dim=dim,
            theta=theta,
            scale=scale,
            current_sigma=current_sigma,
            dype_config=dype_config,
            ori_max_pe_len=ori_max_pe_len,
        )
    elif method == "ntk":
        cos, sin = compute_ntk_freqs(
            pos=pos,
            dim=dim,
            theta=theta,
            scale=scale,
        )
    else:  # "base"
        return _rope_base(pos, dim, theta)

    # Construct rotation matrix from cos/sin
    # Output shape: (batch, seq_len, dim/2, 2, 2)
    out = torch.stack([cos, -sin, sin, cos], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)

    return out.to(dtype=pos.dtype, device=pos.device)


def _rope_base(pos: Tensor, dim: int, theta: int) -> Tensor:
    """Standard RoPE without DyPE scaling.

    This matches the original rope() function from invokeai.backend.flux.math.
    """
    assert dim % 2 == 0

    device = pos.device
    dtype = torch.float64 if device.type != "mps" else torch.float32

    scale = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
    omega = 1.0 / (theta**scale)

    out = torch.einsum("...n,d->...nd", pos.to(dtype), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)

    return out.to(dtype=pos.dtype, device=pos.device)
