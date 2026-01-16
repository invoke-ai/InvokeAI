"""FLUX.2 Klein Sampling Utilities.

FLUX.2 Klein uses a 32-channel VAE (AutoencoderKLFlux2) instead of the 16-channel VAE
used by FLUX.1. This module provides sampling utilities adapted for FLUX.2.
"""

import math

import torch
from einops import rearrange


def get_noise_flux2(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Generate noise for FLUX.2 Klein (32 channels).

    FLUX.2 uses a 32-channel VAE, so noise must have 32 channels.
    The spatial dimensions are calculated to allow for packing.

    Args:
        num_samples: Batch size.
        height: Target image height in pixels.
        width: Target image width in pixels.
        device: Target device.
        dtype: Target dtype.
        seed: Random seed.

    Returns:
        Noise tensor of shape (num_samples, 32, latent_h, latent_w).
    """
    # We always generate noise on the same device and dtype then cast to ensure consistency.
    rand_device = "cpu"
    rand_dtype = torch.float16

    # FLUX.2 uses 32 latent channels
    # Latent dimensions: height/8, width/8 (from VAE downsampling)
    # Must be divisible by 2 for packing (patchify step)
    latent_h = 2 * math.ceil(height / 16)
    latent_w = 2 * math.ceil(width / 16)

    return torch.randn(
        num_samples,
        32,  # FLUX.2 uses 32 latent channels (vs 16 for FLUX.1)
        latent_h,
        latent_w,
        device=rand_device,
        dtype=rand_dtype,
        generator=torch.Generator(device=rand_device).manual_seed(seed),
    ).to(device=device, dtype=dtype)


def pack_flux2(x: torch.Tensor) -> torch.Tensor:
    """Pack latent image to flattened array of patch embeddings for FLUX.2.

    This performs the patchify + pack operation in one step:
    1. Patchify: Group 2x2 spatial patches into channels (C*4)
    2. Pack: Flatten spatial dimensions to sequence

    For 32-channel input: (B, 32, H, W) -> (B, H/2*W/2, 128)

    Args:
        x: Latent tensor of shape (B, 32, H, W).

    Returns:
        Packed tensor of shape (B, H/2*W/2, 128).
    """
    # Same operation as FLUX.1 pack, but input has 32 channels -> output has 128
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpack_flux2(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack flat array of patch embeddings back to latent image for FLUX.2.

    This reverses the pack_flux2 operation:
    1. Unpack: Restore spatial dimensions from sequence
    2. Unpatchify: Restore 32 channels from 128

    Args:
        x: Packed tensor of shape (B, H/2*W/2, 128).
        height: Target image height in pixels.
        width: Target image width in pixels.

    Returns:
        Latent tensor of shape (B, 32, H, W).
    """
    # Calculate latent dimensions
    latent_h = 2 * math.ceil(height / 16)
    latent_w = 2 * math.ceil(width / 16)

    # Packed dimensions (after patchify)
    packed_h = latent_h // 2
    packed_w = latent_w // 2

    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=packed_h,
        w=packed_w,
        ph=2,
        pw=2,
    )


def generate_img_ids_flux2(h: int, w: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate tensor of image position ids for FLUX.2.

    FLUX.2 uses 4D position coordinates (T, H, W, L) instead of 3D.

    Args:
        h: Height of image in latent space.
        w: Width of image in latent space.
        batch_size: Batch size.
        device: Device.
        dtype: dtype.

    Returns:
        Image position ids tensor of shape (batch_size, h*w, 4).
    """
    # After packing, spatial dims are h/2 x w/2
    packed_h = h // 2
    packed_w = w // 2

    # Create coordinate grids
    # T (time/batch), H, W, L (layer/channel)
    img_ids = torch.zeros(packed_h, packed_w, 4, device=device, dtype=dtype)

    # H coordinates
    img_ids[:, :, 1] = torch.arange(packed_h, device=device, dtype=dtype)[:, None]
    # W coordinates
    img_ids[:, :, 2] = torch.arange(packed_w, device=device, dtype=dtype)[None, :]

    # Flatten and expand for batch
    img_ids = img_ids.reshape(1, packed_h * packed_w, 4)
    img_ids = img_ids.expand(batch_size, -1, -1)

    return img_ids
