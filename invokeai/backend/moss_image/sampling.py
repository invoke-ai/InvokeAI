"""Sigma schedule, initial noise and latent layout for MOSS-Image.

MOSS-Image denoises in the space the FLUX.2 transformer works in: the FLUX.2 VAE latent, 2x2-patchified to 128
channels at H/16 x W/16 and normalized with the VAE's BatchNorm running statistics. Between nodes InvokeAI keeps
latents in FLUX.2's raw layout (32 channels at H/8 x W/8, denormalized), so the FLUX.2 VAE nodes handle MOSS latents
unchanged and the denoise node converts at its boundary.
"""

import torch

from invokeai.backend.ernie_image.sampling_utils import (
    patchify_latents,
    unpatchify_latents,
    vae_denormalize,
    vae_normalize,
)
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageScheduler


def get_moss_sigmas(num_steps: int, denoising_start: float = 0.0, denoising_end: float = 1.0) -> list[float]:
    """Descending sigmas for a denoising window, terminal sigma included.

    The full window is the reference scheduler's schedule as is: shifted, snapped to the 1000-step training grid and
    de-duplicated, so a large `num_steps` takes fewer steps than requested (250 -> 246). A partial window is clipped the
    way the FLUX denoise nodes clip theirs and, if that leaves fewer than `num_steps` steps, re-spaced to `num_steps`
    along the scheduler's shift curve with both endpoints kept exactly.
    """
    if not 0.0 <= denoising_start < denoising_end <= 1.0:
        raise ValueError(
            f"Invalid denoising window: start={denoising_start}, end={denoising_end}. Require 0 <= start < end <= 1."
        )
    scheduler = MOSSImageScheduler()
    scheduler.set_timesteps(num_steps)
    sigmas: list[float] = scheduler.sigmas.tolist()
    if denoising_start == 0.0 and denoising_end == 1.0:
        return sigmas

    window = clip_timestep_schedule_fractional(sigmas, denoising_start, denoising_end)
    if len(window) < 2:
        raise ValueError(f"The denoising window [{denoising_start}, {denoising_end}] is too narrow to hold a step.")
    if len(window) - 1 >= num_steps:
        return window

    # Space the steps evenly in unshifted space and map them through the scheduler's shift
    # sigma = s * v / (1 + (s - 1) * v): the full schedule's curve through this window, sampled more finely. Unlike the
    # full schedule the interior is not snapped to the training grid. The window's endpoints are off-grid in general
    # anyway, an off-grid sigma is at most half a grid cell (0.0005) from a trained one, and snapping a dense window
    # would merge neighbouring steps again, losing the steps this re-spacing exists to restore.
    shift = scheduler.config.shift
    v_start, v_end = (sigma / (shift - (shift - 1.0) * sigma) for sigma in (window[0], window[-1]))
    respaced = []
    for i in range(num_steps + 1):
        v = v_start + (v_end - v_start) * i / num_steps
        respaced.append(shift * v / (1.0 + (shift - 1.0) * v))
    # The round trip through the shift is not exact in floating point, and both endpoints are load-bearing: img2img
    # start latents are blended at the first sigma, and the last one ends the requested range.
    respaced[0], respaced[-1] = window[0], window[-1]
    return respaced


def get_moss_noise(seed: int, height: int, width: int) -> torch.Tensor:
    """Initial model-space noise (1, 128, height // 16, width // 16), drawn as the reference pipeline draws it.

    The reference samples float32 noise directly in model space from the generator it is given, a seeded CPU generator
    on its model card, so a seed reproduces the same start on every device. The noise is already N(0, 1) in the
    normalized space: it must not go through `to_model_space`.
    """
    return torch.randn(
        (1, 128, height // 16, width // 16),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    )


def to_model_space(raw: torch.Tensor, bn: torch.nn.BatchNorm2d) -> torch.Tensor:
    """FLUX.2 raw latent (B, 32, H/8, W/8) -> MOSS model space (B, 128, H/16, W/16), in float32.

    `bn` is the FLUX.2 VAE's BatchNorm; its own eps (1e-4 for AutoencoderKLFlux2) is used.
    """
    return vae_normalize(patchify_latents(raw.float()), bn, eps=bn.eps)


def to_raw_latent(packed: torch.Tensor, bn: torch.nn.BatchNorm2d) -> torch.Tensor:
    """MOSS model space -> FLUX.2 raw latent, in float32: the reference pipeline's pre-decode denormalization."""
    return unpatchify_latents(vae_denormalize(packed.float(), bn, eps=bn.eps))
