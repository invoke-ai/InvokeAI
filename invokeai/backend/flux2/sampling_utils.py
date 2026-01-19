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


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for FLUX.2 schedule shifting.

    This matches the diffusers Flux2Pipeline implementation.
    The mu value controls how much the schedule is shifted towards higher timesteps.

    Args:
        image_seq_len: Number of image tokens (packed_h * packed_w).
        num_steps: Number of denoising steps.

    Returns:
        The empirical mu value.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_schedule_flux2(
    num_steps: int,
    image_seq_len: int,
    shift: bool = True,
) -> list[float]:
    """Get timestep schedule for FLUX.2 matching diffusers implementation.

    Key differences from FLUX.1 schedule:
    1. Ends at 1/num_steps instead of 0.0
    2. Uses empirical mu computation specific to FLUX.2

    Args:
        num_steps: Number of denoising steps.
        image_seq_len: Number of image tokens (packed_h * packed_w).
        shift: Whether to apply schedule shifting. Set to False for distilled models.

    Returns:
        List of timesteps from ~1.0 to ~1/num_steps.
    """
    import numpy as np

    # Create sigmas from 1.0 to 1/num_steps (NOT including 0.0)
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)

    if shift:
        # Compute mu for schedule shifting
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)

        # Apply time shift
        # Formula: exp(mu) / (exp(mu) + (1/t - 1))
        shifted_sigmas = []
        for sigma in sigmas:
            if sigma > 0:
                shifted = math.exp(mu) / (math.exp(mu) + (1 / sigma - 1))
                shifted_sigmas.append(float(shifted))
            else:
                shifted_sigmas.append(0.0)

        print(f"[FLUX.2] Schedule (shifted): mu={mu:.4f}, num_steps={num_steps}")
    else:
        # Linear schedule for distilled models
        shifted_sigmas = [float(s) for s in sigmas]
        print(f"[FLUX.2] Schedule (linear, no shift): num_steps={num_steps}")

    # Add final 0.0 for the last step (scheduler needs n+1 timesteps for n steps)
    shifted_sigmas.append(0.0)

    print(f"[FLUX.2] Schedule (first 5): {shifted_sigmas[:5]}")
    print(f"[FLUX.2] Schedule (last 5): {shifted_sigmas[-5:]}")

    return shifted_sigmas


def generate_img_ids_flux2(h: int, w: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate tensor of image position ids for FLUX.2.

    FLUX.2 uses 4D position coordinates (T, H, W, L) for its rotary position embeddings.
    This is different from FLUX.1 which uses 3D coordinates.

    IMPORTANT: Position IDs must use int64 (long) dtype like diffusers, not bfloat16.
    Using floating point dtype for position IDs can cause NaN in rotary embeddings.

    Args:
        h: Height of image in latent space.
        w: Width of image in latent space.
        batch_size: Batch size.
        device: Device.

    Returns:
        Image position ids tensor of shape (batch_size, h/2*w/2, 4) with int64 dtype.
    """
    # After packing, spatial dims are h/2 x w/2
    packed_h = h // 2
    packed_w = w // 2

    # Create coordinate grids - 4D: (T, H, W, L)
    # T = time/batch index, H = height, W = width, L = layer/channel
    # Use int64 (long) dtype like diffusers
    img_ids = torch.zeros(packed_h, packed_w, 4, device=device, dtype=torch.long)

    # T (time/batch) coordinate - set to 0 (already initialized)
    # H coordinates
    img_ids[..., 1] = torch.arange(packed_h, device=device, dtype=torch.long)[:, None]
    # W coordinates
    img_ids[..., 2] = torch.arange(packed_w, device=device, dtype=torch.long)[None, :]
    # L (layer) coordinate - set to 0 (already initialized)

    # Flatten and expand for batch
    img_ids = img_ids.reshape(1, packed_h * packed_w, 4)
    img_ids = img_ids.expand(batch_size, -1, -1)

    return img_ids
