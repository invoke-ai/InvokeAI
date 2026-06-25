"""Sampling helpers for Ideogram 4: packed-sequence construction and latent unpacking.

These wrap the vendored reference model (``modeling_ideogram4`` etc.) for use in
InvokeAI invocations. They mirror the logic of ``Ideogram4Pipeline._build_inputs``
and ``Ideogram4Pipeline._decode`` from the reference implementation, specialised
to the single-image (batch size 1) case InvokeAI generates, where there is no
left-padding so the packed layout is simply ``[text tokens][image tokens]``.
"""

from __future__ import annotations

from typing import TypedDict

import torch

from invokeai.backend.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)

# Latent patch size (each transformer image token covers a patch_size x patch_size
# block of VAE latents) and the VAE's spatial downscale factor. A single image
# token therefore covers ``PATCH_SIZE * AE_SCALE_FACTOR`` pixels per side.
PATCH_SIZE = 2
AE_SCALE_FACTOR = 8
PIXELS_PER_IMAGE_TOKEN = PATCH_SIZE * AE_SCALE_FACTOR  # 16

# Packed-latent channel count: ae z_channels (32) * patch_size**2 (4) = 128.
LATENT_DIM = 128


class Ideogram4DenoiseInputs(TypedDict):
    """The packed-sequence tensors fed to the transformer during denoising."""

    position_ids: torch.Tensor  # (1, L, 3) int64 — (t, h, w) positions for MRoPE
    segment_ids: torch.Tensor  # (1, L) int64 — sample id within the packed batch
    indicator: torch.Tensor  # (1, L) int64 — LLM_TOKEN_INDICATOR or OUTPUT_IMAGE_INDICATOR
    num_text_tokens: int
    num_image_tokens: int
    grid_h: int
    grid_w: int


def validate_dimensions(height: int, width: int) -> None:
    """Ensure the requested resolution is compatible with the patch/VAE grid."""
    if height % PIXELS_PER_IMAGE_TOKEN != 0 or width % PIXELS_PER_IMAGE_TOKEN != 0:
        raise ValueError(
            f"height and width must be divisible by {PIXELS_PER_IMAGE_TOKEN}, got {height}x{width}"
        )


def build_denoise_inputs(
    num_text_tokens: int,
    height: int,
    width: int,
    device: torch.device,
) -> Ideogram4DenoiseInputs:
    """Build the packed ``[text][image]`` position/segment/indicator tensors for one image.

    Mirrors ``Ideogram4Pipeline._build_inputs`` for batch size 1 (no padding).
    """
    validate_dimensions(height, width)
    grid_h = height // PIXELS_PER_IMAGE_TOKEN
    grid_w = width // PIXELS_PER_IMAGE_TOKEN
    num_image_tokens = grid_h * grid_w
    total_seq_len = num_text_tokens + num_image_tokens

    # Image grid positions (t=0, h, w), offset so they never collide with text positions.
    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    position_ids = torch.zeros(1, total_seq_len, 3, dtype=torch.long)
    text_pos = torch.arange(num_text_tokens)
    text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
    position_ids[0, :num_text_tokens] = text_pos_3d
    position_ids[0, num_text_tokens:] = image_pos

    # Single sample, no padding -> every position belongs to segment 1.
    segment_ids = torch.ones(1, total_seq_len, dtype=torch.long)

    indicator = torch.zeros(1, total_seq_len, dtype=torch.long)
    indicator[0, :num_text_tokens] = LLM_TOKEN_INDICATOR
    indicator[0, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

    return Ideogram4DenoiseInputs(
        position_ids=position_ids.to(device),
        segment_ids=segment_ids.to(device),
        indicator=indicator.to(device),
        num_text_tokens=num_text_tokens,
        num_image_tokens=num_image_tokens,
        grid_h=grid_h,
        grid_w=grid_w,
    )


def pack_latents_to_grid(z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Reshape sampled latents ``(1, grid_h*grid_w, LATENT_DIM)`` to ``(1, LATENT_DIM, grid_h, grid_w)``.

    Stores the packed latent in a channels-first 4-D tensor so the grid dimensions
    survive in the latent shape (the L2I node recovers them).
    """
    batch_size = z.shape[0]
    z = z.reshape(batch_size, grid_h, grid_w, LATENT_DIM)
    return z.permute(0, 3, 1, 2).contiguous()


def unpatchify_and_denormalize(
    packed: torch.Tensor,
    latent_shift: torch.Tensor,
    latent_scale: torch.Tensor,
) -> torch.Tensor:
    """Convert a packed latent ``(1, LATENT_DIM, grid_h, grid_w)`` to a VAE latent ``(1, 32, H/8, W/8)``.

    Applies the per-channel latent denormalization (``z * scale + shift``) in the
    packed space, then unpatchifies, exactly as ``Ideogram4Pipeline._decode`` does.
    """
    batch_size, channels, grid_h, grid_w = packed.shape
    if channels != LATENT_DIM:
        raise ValueError(f"expected {LATENT_DIM} packed channels, got {channels}")

    # (B, grid_h, grid_w, LATENT_DIM)
    z = packed.permute(0, 2, 3, 1)
    z = z * latent_scale.to(z.device, z.dtype) + latent_shift.to(z.device, z.dtype)

    ae_channels = LATENT_DIM // (PATCH_SIZE * PATCH_SIZE)  # 32
    z = z.reshape(batch_size, grid_h, grid_w, PATCH_SIZE, PATCH_SIZE, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    z = z.reshape(batch_size, ae_channels, grid_h * PATCH_SIZE, grid_w * PATCH_SIZE)
    return z
