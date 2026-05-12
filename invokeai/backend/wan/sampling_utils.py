"""Sampling utilities for Wan 2.2 image generation.

Single-frame inference uses 5D ``[B, C, T=1, H, W]`` latent tensors. The
scale factors are dictated by the model variant:

* A14B  — standard Wan VAE: spatial 8x, latent channels 16
* TI2V-5B — Wan2.2-VAE: spatial 16x, latent channels 48
"""

from __future__ import annotations

import torch

from invokeai.backend.model_manager.taxonomy import WanVariantType


def get_spatial_scale_factor(variant: WanVariantType) -> int:
    """Return the VAE spatial downsampling factor for a Wan variant."""
    if variant == WanVariantType.TI2V_5B:
        return 16
    return 8  # A14B and any future single-expert variant default to standard Wan VAE.


def get_default_latent_channels(variant: WanVariantType) -> int:
    """Return the default latent-channel count for a Wan variant.

    Use the actual transformer ``in_channels`` from the loaded model when
    possible; this helper is for cases where we need the count before the
    transformer is on device (e.g. building the noise tensor before entering
    the model-on-device context).
    """
    if variant == WanVariantType.TI2V_5B:
        return 48
    return 16


def make_noise(
    *,
    batch_size: int,
    latent_channels: int,
    height: int,
    width: int,
    spatial_scale_factor: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    num_latent_frames: int = 1,
) -> torch.Tensor:
    """Generate Wan-shaped noise: ``[B, C, T_lat, H/s, W/s]``.

    For single-frame image generation the default ``num_latent_frames=1`` yields
    a temporal dim of 1 (matching the original behaviour). Video generation
    passes the latent-space frame count computed from the pixel-frame count via
    :func:`num_latent_frames_for` (or directly).

    Mirrors Anima's ``_get_noise``: noise is generated on CPU (deterministic
    across CUDA / ROCm / MPS) and moved to ``device`` afterwards.
    """
    return torch.randn(
        batch_size,
        latent_channels,
        num_latent_frames,
        height // spatial_scale_factor,
        width // spatial_scale_factor,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device=device, dtype=dtype)


# Wan 2.2 VAE temporal compression ratio. 4 pixel-frames collapse to 1 latent-
# temporal slice (this is also why the I2V conditioning mask has 4 channels).
WAN_VAE_TEMPORAL_SCALE_FACTOR = 4


def num_latent_frames_for(num_frames: int, vae_scale_factor_temporal: int = WAN_VAE_TEMPORAL_SCALE_FACTOR) -> int:
    """Convert a pixel-frame count to latent-frame count for the Wan VAE.

    Matches Diffusers ``WanPipeline.prepare_latents``: ``(num_frames - 1) // s + 1``
    (e.g. ``81 -> 21`` for the standard Wan VAE).
    """
    return (num_frames - 1) // vae_scale_factor_temporal + 1
