"""Sampling utilities for ERNIE-Image.

Mirrors the static helpers in `diffusers.pipelines.ernie_image.pipeline_ernie_image.ErnieImagePipeline`
so we can drive the transformer ourselves while remaining wire-compatible with upstream weights.
"""

from typing import List

import torch

# Latent channels of the ERNIE-Image VAE (AutoencoderKLFlux2). The transformer's
# `in_channels` is this value times 4 (after a 2x2 patchify).
LATENT_CHANNELS: int = 32

# Total downscale factor between the image and the latent grid. Matches
# `2 ** len(vae.config.block_out_channels)` for AutoencoderKLFlux2.
VAE_SCALE_FACTOR: int = 16


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """2x2 patchify: [B, 32, H, W] -> [B, 128, H/2, W/2]."""
    b, c, h, w = latents.shape
    if h % 2 or w % 2:
        raise ValueError(f"Latent spatial dims must be even, got {h}x{w}")
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Reverse 2x2 patchify: [B, 128, H/2, W/2] -> [B, 32, H, W]."""
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def pad_text(
    text_hiddens: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    text_in_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length text-embedding tensors into [B, Tmax, text_in_dim]."""
    if not text_hiddens:
        return (
            torch.zeros((0, 0, text_in_dim), device=device, dtype=dtype),
            torch.zeros((0,), device=device, dtype=torch.long),
        )

    normalized = [
        th.squeeze(1).to(device).to(dtype) if th.dim() == 3 else th.to(device).to(dtype) for th in text_hiddens
    ]
    lens = torch.tensor([t.shape[0] for t in normalized], device=device, dtype=torch.long)
    t_max = int(lens.max().item())
    text_bth = torch.zeros((len(normalized), t_max, text_in_dim), device=device, dtype=dtype)
    for i, t in enumerate(normalized):
        text_bth[i, : t.shape[0], :] = t
    return text_bth, lens


def get_schedule(num_steps: int, denoising_start: float = 0.0, denoising_end: float = 1.0) -> torch.Tensor:
    """Linear sigma schedule from 1.0 -> 0.0, same convention as the upstream pipeline."""
    if not 0.0 <= denoising_start < denoising_end <= 1.0:
        raise ValueError(f"Invalid denoising window: start={denoising_start}, end={denoising_end}")
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1)
    start = int(num_steps * denoising_start)
    end = int(num_steps * denoising_end)
    # Slice to [start, end] inclusive of both ends so the caller can use adjacent pairs.
    return sigmas[start : end + 1]


def vae_normalize(latents: torch.Tensor, bn: torch.nn.Module, eps: float = 1e-5) -> torch.Tensor:
    """Apply the VAE's BatchNorm statistics to map encoder output -> transformer input.

    The ERNIE-Image VAE wraps a BN layer that the upstream pipeline uses to normalize
    latents before patchify (during img2img/inpaint encode) and to denormalize after
    the denoise loop (before decode). This is the encode-side direction.
    """
    mean = bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + eps).to(latents.device, latents.dtype)
    return (latents - mean) / std


def vae_denormalize(latents: torch.Tensor, bn: torch.nn.Module, eps: float = 1e-5) -> torch.Tensor:
    """Reverse of `vae_normalize` -- decode-side post-denoise denormalization."""
    mean = bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + eps).to(latents.device, latents.dtype)
    return latents * std + mean
