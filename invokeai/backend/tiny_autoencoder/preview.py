"""Rendering a denoising preview with a Tiny AutoEncoder, and choosing which installed TAE renders it."""

import math
from collections.abc import Iterable
from typing import TypeVar

import torch
import torch.nn.functional as F
from PIL import Image

from invokeai.backend.architectures import LatentSpace, latent_space_with_channels
from invokeai.backend.model_manager.configs.tiny_autoencoder import TinyAutoencoder_Checkpoint_Config_Base
from invokeai.backend.tiny_autoencoder.decoder import TinyAutoencoderDecoder

TinyAutoencoderConfig = TypeVar("TinyAutoencoderConfig", bound=TinyAutoencoder_Checkpoint_Config_Base)


def decode_preview(
    decoder: TinyAutoencoderDecoder, sample: torch.Tensor, device: torch.device | None = None
) -> Image.Image:
    """Decode the first latent of `sample` into an RGB preview no larger than `decoder.max_preview_edge`.

    `sample` is `(C, h, w)` or `(B, C, h, w)` in the decoder's latent space, or a `(B, C, T, h, w)` window
    of latent frames, which a video decoder renders as its last frame with the earlier ones as temporal
    context (a TAESD decoder refuses it). A latent whose decode would exceed the cap is area-downsampled
    first, which is far cheaper than decoding at full size and scaling the image. Alpha, where the decoder
    has it, is dropped.

    The decode runs on `device`, which a caller holding the model through the model cache passes as the
    cache's compute device: a partially loaded decoder keeps some weights elsewhere, so its parameters
    do not say where it runs. Without one, the first parameter's device is used. The dtype is always the
    decoder's.
    """
    if sample.dim() == 3:
        sample = sample.unsqueeze(0)
    elif sample.dim() not in (4, 5):
        raise ValueError(
            f"expected a (C, h, w) or (B, C, h, w) latent or a (B, C, T, h, w) window, got shape {tuple(sample.shape)}"
        )
    if sample.shape[1] != decoder.latent_channels:
        raise ValueError(
            f"the decoder reads {decoder.latent_channels} latent channels, the sample has {sample.shape[1]}"
        )

    parameter = next(decoder.parameters())
    with torch.inference_mode():
        latents = sample[0:1].to(device=device or parameter.device, dtype=parameter.dtype)
        height, width = latents.shape[-2:]
        longest_edge = max(height, width) * decoder.spatial_compression
        if longest_edge > decoder.max_preview_edge:
            scale = decoder.max_preview_edge / longest_edge
            size = (max(1, math.floor(height * scale)), max(1, math.floor(width * scale)))
            if latents.dim() == 5:
                # A window's frames are folded into the batch, so they are pooled in space only and through
                # the same 2-D path as single latents (3-D adaptive pooling is not verified on every backend).
                frames = latents.transpose(1, 2)
                pooled = F.interpolate(frames.flatten(0, 1), size=size, mode="area")
                latents = pooled.unflatten(0, frames.shape[:2]).transpose(1, 2)
            else:
                latents = F.interpolate(latents, size=size, mode="area")
        image = decoder.decode_window(latents) if latents.dim() == 5 else decoder(latents)
        pixels = image[0, :3].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu()
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
