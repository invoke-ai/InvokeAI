"""Rendering a denoising preview with a Tiny AutoEncoder, and choosing which installed TAE renders it."""

import math
from collections.abc import Iterable
from typing import Final, TypeVar

import torch
import torch.nn.functional as F
from PIL import Image

from invokeai.backend.architectures import LatentSpace, latent_space_with_channels
from invokeai.backend.model_manager.configs.tiny_autoencoder import TinyAutoencoder_Checkpoint_Config_Base
from invokeai.backend.tiny_autoencoder.taesd import TaesdDecoder

MAX_PREVIEW_EDGE: Final = 1024
"""Longest edge of a TAE preview, in pixels. Bounds the decode's memory and the frame's payload.

At 1024 a decode costs ~640 MiB of VRAM for the 8x TAEs (~160 MiB for TAEQI2.1) and ~25 ms (RTX 4090,
fp16) at any generation size; uncapped, 2048 costs ~2.5 GiB and ~100 ms. Decoding at full size and then
scaling down would look better (SDXL 36.3 vs 30.7 dB against the VAE) but carries that uncapped cost, which
is why the latent is area-downsampled before the decode instead.
"""

TinyAutoencoderConfig = TypeVar("TinyAutoencoderConfig", bound=TinyAutoencoder_Checkpoint_Config_Base)


def decode_preview(decoder: TaesdDecoder, sample: torch.Tensor, device: torch.device | None = None) -> Image.Image:
    """Decode the first latent of `sample` into an RGB preview no larger than `MAX_PREVIEW_EDGE`.

    `sample` is `(C, h, w)` or `(B, C, h, w)` in the decoder's latent space. A latent whose decode would
    exceed the cap is area-downsampled first, which is far cheaper than decoding at full size and scaling
    the image. Alpha, where the decoder has it, is dropped.

    The decode runs on `device`, which a caller holding the model through the model cache passes as the
    cache's compute device: a partially loaded decoder keeps some weights elsewhere, so its parameters
    do not say where it runs. Without one, the first parameter's device is used. The dtype is always the
    decoder's.
    """
    if sample.dim() == 3:
        sample = sample.unsqueeze(0)
    elif sample.dim() != 4:
        raise ValueError(f"expected a (C, h, w) or (B, C, h, w) latent, got shape {tuple(sample.shape)}")
    if sample.shape[1] != decoder.latent_channels:
        raise ValueError(
            f"the decoder reads {decoder.latent_channels} latent channels, the sample has {sample.shape[1]}"
        )

    parameter = next(decoder.parameters())
    with torch.inference_mode():
        latents = sample[0:1].to(device=device or parameter.device, dtype=parameter.dtype)
        height, width = latents.shape[-2:]
        longest_edge = max(height, width) * decoder.spatial_compression
        if longest_edge > MAX_PREVIEW_EDGE:
            scale = MAX_PREVIEW_EDGE / longest_edge
            size = (max(1, math.floor(height * scale)), max(1, math.floor(width * scale)))
            latents = F.interpolate(latents, size=size, mode="area")
        image = decoder(latents)[0, :3].clamp(0, 1)
        pixels = image.mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu()
    return Image.fromarray(pixels.numpy())


def select_tiny_autoencoder(
    configs: Iterable[TinyAutoencoderConfig], space: LatentSpace
) -> TinyAutoencoderConfig | None:
    """The installed TAE that decodes `space`, or None.

    A record serves every base that shares its latent space, matched by identity. Among several that fit,
    the choice is deterministic -- by name, then key -- so a preview does not change decoder between runs.
    """
    fitting = [c for c in configs if latent_space_with_channels(c.base, c.latent_channels) is space]
    return min(fitting, key=lambda c: (c.name, c.key), default=None)
