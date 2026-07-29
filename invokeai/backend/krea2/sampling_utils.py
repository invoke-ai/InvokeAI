"""Sampling/packing utilities for Krea-2 (Krea2Pipeline) inference.

InvokeAI hand-writes its own denoise loop for Qwen-family models rather than calling the
diffusers pipeline ``__call__``. These helpers replicate the Krea-2 sampling math so the
``Krea2Transformer2DModel`` (loaded from diffusers) can be driven directly.

Reference: ``diffusers/pipelines/krea2/pipeline_krea2.py`` (diffusers main / 0.39.0.dev0).
"""

from typing import List

import numpy as np
import torch

# Krea-2 packs latents into 2x2 patches; the VAE (AutoencoderKLQwenImage) is f8.
PATCH_SIZE = 2
VAE_SCALE_FACTOR = 8

# Hidden-state layers tapped from the Qwen3-VL text encoder (model_index.json
# text_encoder_select_layers). Stacked per token into prompt_embeds (B, seq, 12, hidden).
KREA2_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

# Text template constants (diffusers Krea2Pipeline.get_text_hidden_states).
KREA2_MAX_SEQ_LEN = 512
KREA2_START_IDX = 34  # drop the system-prompt prefix tokens
KREA2_NUM_SUFFIX_TOKENS = 5

# Resolution-aware time-shift parameters (scheduler_config.json).
KREA2_BASE_SHIFT = 0.5
KREA2_MAX_SHIFT = 1.15
KREA2_BASE_IMAGE_SEQ_LEN = 256
KREA2_MAX_IMAGE_SEQ_LEN = 6400
# Fixed timestep shift for the distilled (Turbo) checkpoint.
KREA2_DISTILLED_MU = 1.15


def pack_latents(latents: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int) -> torch.Tensor:
    """Pack 4D latents (B, C, H, W) into 2x2-patched 3D (B, H/2*W/2, C*4).

    Identical to the Qwen-Image / Krea-2 ``_pack_latents`` (patch_size=2).
    """
    p = PATCH_SIZE
    latents = latents.view(batch_size, num_channels, height // p, p, width // p, p)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // p) * (width // p), num_channels * p * p)
    return latents


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack 3D patched latents (B, seq, C*4) back to 4D (B, C, H, W).

    ``height``/``width`` are in latent space (i.e. pixels // vae_scale_factor).
    """
    p = PATCH_SIZE
    batch_size, _num_patches, channels = latents.shape
    h = p * (height // p)
    w = p * (width // p)
    latents = latents.view(batch_size, h // p, w // p, channels // (p * p), p, p)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (p * p), h, w)
    return latents


def prepare_position_ids(text_seq_len: int, grid_height: int, grid_width: int, device: torch.device) -> torch.Tensor:
    """Build the (text_seq_len + grid_h*grid_w, 3) rotary coordinates.

    Text tokens sit at the origin (0, 0, 0); image tokens carry their (0, h, w) latent-grid
    coordinates. Matches diffusers ``Krea2Pipeline.prepare_position_ids``.
    """
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    image_ids = image_ids.reshape(grid_height * grid_width, 3)
    return torch.cat([text_ids, image_ids], dim=0)


def calculate_shift(
    image_seq_len: int,
    base_image_seq_len: int = KREA2_BASE_IMAGE_SEQ_LEN,
    max_image_seq_len: int = KREA2_MAX_IMAGE_SEQ_LEN,
    base_shift: float = KREA2_BASE_SHIFT,
    max_shift: float = KREA2_MAX_SHIFT,
) -> float:
    """Resolution-aware mu (linear interpolation by sequence length).

    NOTE: mu is fed straight into ``FlowMatchEulerDiscreteScheduler.set_timesteps(..., mu=mu)``;
    the exponential time-shift happens inside the scheduler. Do NOT ``exp()`` this value.
    """
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    return image_seq_len * m + b


def build_sigmas(steps: int) -> List[float]:
    """Krea-2 sigma schedule: linspace(1.0, 1/steps, steps)."""
    return np.linspace(1.0, 1.0 / steps, steps).tolist()
