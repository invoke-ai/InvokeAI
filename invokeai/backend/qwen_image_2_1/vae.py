"""Qwen-Image-2.1's RGBA VAE: working memory, tile choice, and how an RGBA decode becomes an image.

The tile geometry itself is applied by `patch_qwen_image_vae_tiling`, which this VAE shares with the Qwen-Image
and Wan VAEs (same tiling attributes, 16x instead of 8x).
"""

import functools
import math
from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
import torch
from PIL import Image

from invokeai.backend.util.devices import device_has_dedicated_vram
from invokeai.backend.util.vae_working_memory import should_pretile_vae_decode
from invokeai.backend.working_memory.coefficients import shipped_data
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import convolution_line
from invokeai.backend.working_memory.platform import platform_for, priced_as
from invokeai.backend.working_memory.spec import EstimatorSpec, register

SPATIAL_SCALE = 16

# Peak reserved memory of an encode or decode, on the AutoencoderKL convolution line of `backend/util/vae_working_memory.py`:
# `k_<operation>_<dtype>` bytes per output pixel plus `b_<operation>_<dtype>` bytes, per convolution backend and
# allocator (`platform_for`). The coefficients and the measured points that bound them are in
# `backend/working_memory/data/qwen_image_2_1_vae.json`; `backend/working_memory/fit.py` explains how they are fitted.
# This VAE is far hungrier than the family's (a 1024px bf16 decode reserves ~9 GiB on cuDNN and ~12 GiB on MIOpen), and
# on MIOpen the per-pixel cost steps up past ~1.5 Mpx, which the line covers by being fitted on the envelope.
#
# A tiled operation is priced as one tile of the line, widened for the overlap the tile blend adds, which differs by
# convolution backend (the `conv_backend` feature, as `priced_as` names it). MIOpen's is fitted on its tiled points per
# line, and its tiles cost more per pixel the smaller they are (17.4 KB at 512px, ~13.6 KB at 768px in bf16), so a tile
# below the smallest measured one is priced as that one. cuDNN keeps the original 4000-bytes-per-element-byte pricing,
# widened just enough to cover the RTX 4090's 768px and 512px tiles, which that pricing fell ~5% short of.
_COEFFICIENTS = frozenset(
    f"{part}_{operation}_{dtype}"
    for part in ("k", "b")
    for operation in ("decode", "encode")
    for dtype in ("bf16", "fp32")
)
_DTYPES = {2: "bf16", 4: "fp32"}
_MIOPEN_TILE_OVERLAP = {"decode_bf16": 1.1175, "decode_fp32": 1.18, "encode_bf16": 1.2, "encode_fp32": 1.45}
_MIOPEN_SMALLEST_MEASURED_TILE = 512
_CUDNN_TILE_OVERLAP = 1.065


def _terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    height, width = cast(int, features["height"]), cast(int, features["width"])
    tile = cast(int | None, features["tile"])
    pixels: float = height * width
    line = f"{features['operation']}_{features['dtype']}"
    # The VAE tiles only when a side is larger than the tile; a tile covering the frame runs the single pass.
    if tile is not None and (tile < height or tile < width):
        if features["conv_backend"] == "miopen":
            side = max(tile, _MIOPEN_SMALLEST_MEASURED_TILE)
            pixels = side * side * _MIOPEN_TILE_OVERLAP[line]
        else:
            pixels = tile * tile * _CUDNN_TILE_OVERLAP
    return (
        WorkingMemoryTerm.fitted("convolutions", f"k_{line}", pixels, values),
        WorkingMemoryTerm.fitted("convolution_base", f"b_{line}", 1, values),
    )


SPEC = register(
    EstimatorSpec(
        id="vae.qwen_image_2_1",
        version=1,
        coefficients=_COEFFICIENTS,
        platform_keys=("conv_backend", "allocator"),
        terms=_terms,
        line=convolution_line,
    )
)

# Tiles of 256px (diffusers' default) leave visible seams: against an untiled fp32 decode a 2048x2048 image came
# to 35.6 dB with alpha off by up to 73/255; 1024px tiles reach 47.7 dB and 16/255. Smaller tiles only where
# memory forces them.
_TILE_SIZES = (1024, 768, 512)
# A cost floor for the node field: at 16x a 256px tile is 16 latents, and the tile count grows with the inverse
# square of the size -- a 32px tile would decode a 2048x2048 image as ~16k tiles.
MIN_TILE_SIZE = 256


def estimate_working_memory(
    operation: Literal["encode", "decode"],
    height: int,
    width: int,
    tile_size: int | None,
    element_size: int,
    device: torch.device,
) -> WorkingMemoryEstimate:
    """The working memory an encode or decode of a `height` x `width` image needs on `device`, untiled or in
    `tile_size` tiles. `element_size` is the byte width of the VAE's dtype: 2 is bf16, 4 fp32."""
    features = {
        "operation": operation,
        "height": height,
        "width": width,
        "tile": tile_size,
        "dtype": _DTYPES[element_size],
        "conv_backend": priced_as(platform_for(device)).conv_backend,
    }
    return SPEC.estimate(features, device)


@functools.cache
def _measured_untiled_peaks(operation: str, conv_backend: str, dtype: str) -> tuple[tuple[int, int], ...]:
    """The shipped untiled points of `operation` on `conv_backend` under the native allocator in `dtype`, as
    (area, bytes) by area; the larger where two shapes share an area."""
    peaks: dict[int, int] = {}
    for data in shipped_data().values():
        for point in data.points:
            features = point.features
            if (
                point.estimator == SPEC.id
                and point.platform == {"conv_backend": conv_backend, "allocator": "native"}
                and features["operation"] == operation
                and features["tile"] is None
                and features["dtype"] == dtype
            ):
                area = cast(int, features["height"]) * cast(int, features["width"])
                peaks[area] = max(peaks.get(area, 0), point.measured_bytes)
    return tuple(sorted(peaks.items()))


def untiled_peak_bytes(
    operation: Literal["encode", "decode"], height: int, width: int, element_size: int, device: torch.device
) -> int:
    """What an untiled encode or decode is expected to peak at, for the up-front tiling decision only.

    The reservation carries up to 1.3x headroom over the measured points; deciding to tile on it would tile frames
    that fit, and tiled output is not pixel-identical. So within the measured untiled points of the device's backend
    this is the chord between the two neighbouring points, in bytes, and below the first one that point's bytes per
    pixel, as `qwen_image_untiled_decode_peak_bytes` does for Qwen-Image. Past the last point, or with fewer than two
    points (cuDNN, and every backend priced as it), it is the reservation.

    The points are the native allocator's under every allocator: the other allocators' entries (``cudaMallocAsync``,
    MIOpen's expandable segments on Windows) carry the native entry's values as stand-ins and no points of their own.
    """
    peaks = _measured_untiled_peaks(operation, priced_as(platform_for(device)).conv_backend, _DTYPES[element_size])
    area = height * width
    if len(peaks) < 2 or area > peaks[-1][0]:
        return estimate_working_memory(operation, height, width, None, element_size, device).reservation_bytes
    previous_area, previous_bytes = 0, 0
    for measured_area, measured_bytes in peaks:
        if measured_area >= area:
            if previous_area == 0:
                return measured_bytes * area // measured_area
            span = (measured_bytes - previous_bytes) / (measured_area - previous_area)
            return int(previous_bytes + span * (area - previous_area))
        previous_area, previous_bytes = measured_area, measured_bytes
    raise AssertionError("areas past the measured range are handled above")


def default_tile_size(operation: Literal["encode", "decode"], device: torch.device, element_size: int) -> int:
    """The largest tile whose encode or decode takes at most half of the device's memory."""
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
    elif device.type == "xpu" and device_has_dedicated_vram(device):
        total = torch.xpu.get_device_properties(device).total_memory
    else:
        return _TILE_SIZES[0]
    for size in _TILE_SIZES:
        # Priced as one tile of a larger frame, as it will run and be reserved, tiling overhead included.
        tiled = estimate_working_memory(operation, 2 * size, 2 * size, size, element_size, device)
        if tiled.reservation_bytes <= total // 2:
            return size
    return _TILE_SIZES[-1]


def choose_tile_size(
    operation: Literal["encode", "decode"],
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
    whose untiled peak (`untiled_peak_bytes`) would claim too much of the device. `tile_size` is the node field: 0 picks the
    largest tile for the device, anything else is floored at `MIN_TILE_SIZE`.
    """
    if not tiled and not (
        auto_tile
        and should_pretile_vae_decode(device, untiled_peak_bytes(operation, height, width, element_size, device))
    ):
        return None
    if tile_size <= 0:
        return default_tile_size(operation, device, element_size)
    return max(tile_size, MIN_TILE_SIZE)


# Opaque content decodes with alpha just under 255, and tiling adds seam noise: untiled fp32 decodes of opaque
# images bottom out at 252, 1024px tiles at 239, 256px tiles at 181 on 0.03 % of pixels. A transparent prompt
# leaves ~40 % of pixels near 0. So "opaque" is: at most 0.1 % of pixels below 224.
_OPAQUE_ALPHA = 224
_MAX_TRANSPARENT_FRACTION = 0.001


def to_image(decoded: torch.Tensor) -> Image.Image:
    """A `(4, H, W)` decode in [-1, 1] as an RGB image when it is opaque, RGBA when it is not.

    Mapped to [0, 1] in the decode's own dtype and only then widened, as the pipeline's `VaeImageProcessor` does:
    from a bf16 decode, widening first rounds 1 in ~8 % of pixels the other way.
    """
    unit = (decoded * 0.5 + 0.5).clamp(0, 1).float()
    pixels = (unit * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
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
