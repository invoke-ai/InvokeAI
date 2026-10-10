"""Qwen-Image-2.1's RGBA VAE: working memory, tile choice, and how an RGBA decode becomes an image.

The tile geometry itself is applied by `patch_qwen_image_vae_tiling`, which this VAE shares with the Qwen-Image
and Wan VAEs (same tiling attributes, 16x instead of 8x).
"""

import math

import numpy as np
import torch
from PIL import Image

from invokeai.backend.util.vae_working_memory import should_pretile_vae_decode

SPATIAL_SCALE = 16

# Measured on an RTX 4090 in bf16 (torch's reserved memory, which is what Windows pages against): an untiled
# 1024x1024 decode peaks at 8.38 GiB, a 2048x2048 decode in 1024px tiles at the same 8.38 GiB, in 768px tiles
# at 5.27 GiB, in 512px tiles at 2.68 GiB. ~4000 bytes per pixel per byte of the VAE's dtype (8000 in bf16) of
# the frame (or tile) plus a fixed 0.6 GiB covers all four. The encoder is narrower than the decoder (96 vs 144
# base channels); it is priced the same.
_BYTES_PER_PIXEL_PER_ELEMENT_BYTE = 4000
_FIXED_BYTES = int(0.6 * 1024**3)

# Tiles of 256px (diffusers' default) leave visible seams: against an untiled fp32 decode a 2048x2048 image came
# to 35.6 dB with alpha off by up to 73/255; 1024px tiles reach 47.7 dB and 16/255. Smaller tiles only where
# memory forces them.
_TILE_SIZES = (1024, 768, 512)
# A cost floor for the node field: at 16x a 256px tile is 16 latents, and the tile count grows with the inverse
# square of the size -- a 32px tile would decode a 2048x2048 image as ~16k tiles.
MIN_TILE_SIZE = 256


def working_memory_bytes(height: int, width: int, tile_size: int | None, element_size: int) -> int:
    """Peak bytes an encode or decode of a `height` x `width` image needs, untiled or in `tile_size` tiles.

    `element_size` is the byte width of the VAE's dtype: a float32 VAE (`precision: float32`) needs twice the
    bfloat16 figure.
    """
    pixels = height * width if tile_size is None else min(height * width, tile_size * tile_size)
    return pixels * _BYTES_PER_PIXEL_PER_ELEMENT_BYTE * element_size + _FIXED_BYTES


def default_tile_size(device: torch.device, element_size: int) -> int:
    """The largest tile whose decode takes at most half of the device's memory."""
    if device.type != "cuda":
        return _TILE_SIZES[0]
    total = torch.cuda.get_device_properties(device).total_memory
    for size in _TILE_SIZES:
        if working_memory_bytes(size, size, size, element_size) <= total // 2:
            return size
    return _TILE_SIZES[-1]


def choose_tile_size(
    height: int,
    width: int,
    element_size: int,
    device: torch.device,
    *,
    tiled: bool,
    tile_size: int,
    auto_tile: bool,
) -> int | None:
    """The tile size an encode or decode runs at, or None to run it in one piece.

    `tiled` is the node's request (or `force_tiled_decode`); `auto_tile` (`auto_tiled_decode`) also tiles a frame
    whose untiled working memory would claim too much of the device. `tile_size` is the node field: 0 picks the
    largest tile for the device, anything else is floored at `MIN_TILE_SIZE`.
    """
    if not tiled and not (
        auto_tile and should_pretile_vae_decode(device, working_memory_bytes(height, width, None, element_size))
    ):
        return None
    if tile_size <= 0:
        return default_tile_size(device, element_size)
    return max(tile_size, MIN_TILE_SIZE)


# Opaque content decodes with alpha just under 255, and tiling adds seam noise: untiled fp32 decodes of opaque
# images bottom out at 252, 1024px tiles at 239, 256px tiles at 181 on 0.03 % of pixels. A transparent prompt
# leaves ~40 % of pixels near 0. So "opaque" is: at most 0.1 % of pixels below 224.
_OPAQUE_ALPHA = 224
_MAX_TRANSPARENT_FRACTION = 0.001


def to_image(decoded: torch.Tensor) -> Image.Image:
    """A `(4, H, W)` decode in [-1, 1] as an RGB image when it is opaque, RGBA when it is not."""
    pixels = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    alpha = pixels[..., 3]
    if np.count_nonzero(alpha < _OPAQUE_ALPHA) <= _MAX_TRANSPARENT_FRACTION * alpha.size:
        return Image.fromarray(np.ascontiguousarray(pixels[..., :3]), mode="RGB")
    return Image.fromarray(pixels, mode="RGBA")


def to_vae_input(image: Image.Image, keep_alpha: bool = False) -> torch.Tensor:
    """An image as the VAE's `(1, 4, H, W)` input in [-1, 1].

    For image-to-image and the canvas, alpha is set to 1 rather than read from the image: the canvas hands its masks
    over in alpha, and a transparent region there means "repaint", not "transparent pixels". A reference image keeps
    its alpha (`keep_alpha`), as the pipeline encodes it.
    """
    rgba = image.convert("RGBA")
    # In the pipeline's `VaeImageProcessor` order, so the result matches it to the bit.
    pixels = torch.from_numpy(np.asarray(rgba).astype(np.float32) / 255.0).permute(2, 0, 1) * 2.0 - 1.0
    if not keep_alpha:
        pixels[3] = 1.0
    return pixels[None]


# The area every reference image is resized to, as the pipeline's default `output_resolution` sets it.
REFERENCE_AREA = 1024 * 1024


def reference_size(width: int, height: int) -> tuple[int, int]:
    """The size a `width` x `height` reference is read at: `REFERENCE_AREA` at its aspect ratio, on the 32px grid.

    The text encoder and the VAE must both read it at this size: each `<|image_pad|>` slot the encoder emits
    stands for 2x2 of the reference's latents, and the transformer drops the latents into those slots.
    """
    ratio = width / height
    scaled_width = math.sqrt(REFERENCE_AREA * ratio)
    # Python's round (half to even) is the pipeline's `calculate_dimensions`.
    return round(scaled_width / 32) * 32, round(scaled_width / ratio / 32) * 32


def to_reference(image: Image.Image) -> Image.Image:
    """A reference image as both the text encoder and the VAE read it: RGBA, resized to `reference_size`."""
    rgba = image.convert("RGBA")
    return rgba.resize(reference_size(*rgba.size), resample=Image.LANCZOS)
