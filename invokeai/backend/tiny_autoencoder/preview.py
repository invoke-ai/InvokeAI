"""Rendering a denoising preview with a Tiny AutoEncoder, and choosing which installed TAE renders it."""

import math
from collections.abc import Iterable
from typing import Final, TypeVar

import torch
import torch.nn.functional as F
from PIL import Image

from invokeai.backend.architectures import LatentSpace, latent_space_with_channels
from invokeai.backend.model_manager.configs.tiny_autoencoder import TinyAutoencoder_Checkpoint_Config_Base
from invokeai.backend.tiny_autoencoder.decoder import TinyAutoencoderDecoder

_LAST_STAGE_CHANNELS: Final = 64
"""Width of a TAESD decoder's full-resolution stage, the last `Block` and the convolution before it."""

_LIVE_ACTIVATIONS: Final = 3
"""Full-resolution activations a TAESD decode holds at once, read off the vendored decoder.

The peak is in the last `Block`, the only one at full resolution. Its input stays referenced while its conv branch
runs, and `nn.ReLU` is not in place, so each layer of the branch has its own input and output live next to the block's
input: three tensors. Adding the branch's output to the input (the skip is an identity) and fusing the sum holds three
again. Every earlier stage runs at half resolution or less. TAEQI2.1's last stage runs at half resolution and holds a
quarter of this; the bound reserves it the same.
"""

_CONV_WORKSPACE_ACTIVATIONS: Final = 2
"""The convolutions' workspace on top, in activation-sized buffers. Measured, and it depends on the algorithm the
backend picks: on an RTX 4090 in fp16, `max_memory_allocated` grew by 640 MiB over a 1024x1024 decode, five
activations, two more than the structure holds.
"""


_POOLING_NEEDS_DIVISIBLE_SIZES: Final = frozenset({"mps"})
"""Device types whose adaptive average pooling, behind `F.interpolate(mode="area")`, refuses an output size that does
not divide the input ("input sizes must be divisible by output sizes", pytorch#96056); macOS stays on torch 2.7."""

TinyAutoencoderConfig = TypeVar("TinyAutoencoderConfig", bound=TinyAutoencoder_Checkpoint_Config_Base)


def _preview_latent_size(decoder: TinyAutoencoderDecoder, height: int, width: int) -> tuple[int, int]:
    """The latent size a `(height, width)` latent is decoded at: downsampled until the image fits the decoder's
    `max_preview_edge`."""
    longest_edge = max(height, width) * decoder.spatial_compression
    if longest_edge <= decoder.max_preview_edge:
        return height, width
    scale = decoder.max_preview_edge / longest_edge
    return max(1, math.floor(height * scale)), max(1, math.floor(width * scale))


def decode_working_memory_bytes(decoder: TinyAutoencoderDecoder, sample: torch.Tensor) -> int:
    """The working memory `decode_preview(decoder, sample)` needs on its device besides the decoder's weights.

    The live activations plus the convolutions' workspace, at the capped output size and the decoder's element size:
    640 MiB at 1024x1024 in fp16. The cache's working-memory floor (`device_working_mem_gb`, 3 GB by default) covers
    it, so the reservation matters only where that floor is set lower.
    """
    height, width = _preview_latent_size(decoder, *sample.shape[-2:])
    activation_bytes = height * width * decoder.spatial_compression**2 * _LAST_STAGE_CHANNELS
    activation_bytes *= next(decoder.parameters()).element_size()
    return (_LIVE_ACTIVATIONS + _CONV_WORKSPACE_ACTIVATIONS) * activation_bytes


def _area_downsample(latents: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = latents.shape[-2:]
    if latents.device.type in _POOLING_NEEDS_DIVISIBLE_SIZES and (height % size[0] or width % size[1]):
        # Pooled on the CPU instead, in float32 for CPU builds without half-precision pooling. The latent is small.
        pooled = F.interpolate(latents.cpu().float(), size=size, mode="area")
        return pooled.to(device=latents.device, dtype=latents.dtype)
    return F.interpolate(latents, size=size, mode="area")


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
        size = _preview_latent_size(decoder, *latents.shape[-2:])
        if size != tuple(latents.shape[-2:]):
            if latents.dim() == 5:
                # A window's frames are folded into the batch, so they are pooled in space only and through
                # the same 2-D path as single latents (3-D adaptive pooling is not verified on every backend).
                frames = latents.transpose(1, 2)
                pooled = _area_downsample(frames.flatten(0, 1), size)
                latents = pooled.unflatten(0, frames.shape[:2]).transpose(1, 2)
            else:
                latents = _area_downsample(latents, size)
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
