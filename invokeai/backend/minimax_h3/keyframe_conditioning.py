"""MiniMax H3 keyframe (first/last frame) VAE conditioning.

First-party port of ``MiniMaxH3KeyframeVaeEncoderStep`` and the keyframe half of
``MiniMaxH3SetupStep`` from the diffusers MiniMax-H3 integration (commit recorded in
``__init__``). The keyframes are encoded by the video VAE's spatial encoder alone (they are
single frames), the posterior is *sampled* under a generator seeded with 42 independently of
the request seed, and the sampled latent is rounded to float16 before normalization — all
three are part of reproducing the released model's conditioning.

The rows returned here are CLEAN (not yet noise-augmented): the denoise state noises them to
``t = 0.999`` with the request generator's first draws, keeping the one-generator-three-draws
reproducibility order in one place (see :mod:`invokeai.backend.minimax_h3.sampling`).
"""

import contextlib

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image, ImageOps

from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from invokeai.backend.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    patchify_video_latents,
    prepare_keyframe_image,
)
from invokeai.backend.minimax_h3.sampling import MINIMAX_H3_PATCH_SIZE


def prepare_keyframes(
    first_image: Image.Image | None,
    last_image: Image.Image | None,
    height: int,
    width: int,
) -> tuple[list[Image.Image], tuple[str, ...]]:
    """Put the keyframes onto the target canvas, in packed order.

    The first keyframe *in packed order* is the geometry anchor and is stretched onto the
    canvas (upstream stretches index 0 even when it is a lone last-frame); any second keyframe
    follows the canvas and is cover-cropped. Returns the prepared images and their anchors
    (``"first"`` / ``"last"``), both in packed order. Must be applied identically by the text
    encoder (vision context) and the VAE conditioning encoder.
    """
    keyframes: list[Image.Image] = []
    anchors: list[str] = []
    for anchor, image in (("first", first_image), ("last", last_image)):
        if image is None:
            continue
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        keyframes.append(prepare_keyframe_image(prepared, height, width, stretch=len(keyframes) == 0))
        anchors.append(anchor)
    return keyframes, tuple(anchors)


def vae_encode_autocast(device: torch.device) -> contextlib.AbstractContextManager[None]:
    """The precision context every H3 video-VAE *encode* runs under.

    The encoder is a 180M-parameter 3D-conv ResNet whose weights are pinned float32. On CUDA-type
    devices (NVIDIA and ROCm both report ``"cuda"``) the convolutions run under float16 autocast
    over those float32 weights — the same recipe the decode path uses. Measured on a 5060 Ti for one
    17-frame 768x448 chunk: 4.6 s -> 3.2 s with TF32 convs, 9.2 s -> 3.2 s without (ROCm has no
    TF32 path), and ~0.5 GiB less peak VRAM. ``cache_enabled=False`` keeps autocast from pinning
    float16 copies of the whole encoder for the duration of a long reference encode.

    Numerics: the autocast encode drifts ~1.5e-3 RMS (relative) from the float32 one. Visual
    conditioning rows are noise-augmented to ``t = 0.999`` before the transformer sees them, so
    the clean encode contributes 0.1% of each row and the drift lands at ~1e-6 of the row
    magnitude. MPS/CPU encode in the weights' dtype.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=False)
    return contextlib.nullcontext()


@torch.no_grad()
def encode_keyframes(
    vae: AutoencoderKLMiniMaxH3,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    """Encode prepared keyframes into clean, normalized, packed conditioning rows (float32, CPU)."""
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)

    rows = []
    for image in images:
        pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        # A keyframe is one frame: the (tiled) spatial encoder alone, no 17-frame temporal chunking.
        with vae_encode_autocast(device):
            moments = vae._encode_clip(pixels)
        moments = moments.float()
        posterior = DiagonalGaussianDistribution(moments)
        latents = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
        # The fp16 rounding before normalization is a checkpoint contract (~11 bits of every
        # conditioning latent); without it the released model's conditioning is not reproduced.
        latents = latents.to(torch.float16).float().cpu()
        rows.append(patchify_video_latents((latents - latents_mean) / latents_std, MINIMAX_H3_PATCH_SIZE))
    return torch.cat(rows)
