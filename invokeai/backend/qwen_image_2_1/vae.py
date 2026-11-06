"""Qwen-Image-2.1's RGBA VAE: working memory, tile choice, and how an RGBA decode becomes an image.

The tile geometry itself is applied by `patch_qwen_image_vae_tiling`, which this VAE shares with the Qwen-Image
and Wan VAEs (same tiling attributes, 16x instead of 8x).
"""

import dataclasses
import functools
import math
from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
import torch
from PIL import Image

from invokeai.backend.util.devices import device_has_dedicated_vram
from invokeai.backend.util.vae_working_memory import should_pretile_vae_decode
from invokeai.backend.working_memory.coefficients import find_entry, shipped_data
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import Line, convolution_line
from invokeai.backend.working_memory.platform import Platform, platform_for, priced_as
from invokeai.backend.working_memory.spec import EstimatorSpec, register

SPATIAL_SCALE = 16

# Peak reserved memory of an encode or decode, on the convolution lines of `backend/util/vae_working_memory.py`:
# `k_<line>` bytes per output pixel plus `b_<line>` bytes, per convolution backend, allocator (`platform_for`) and, where
# an entry names it, OS. The coefficients and the measured points that bound them are in
# `backend/working_memory/data/qwen_image_2_1_vae.json`; `backend/working_memory/fit.py` explains how they are fitted.
# This VAE is far hungrier than the family's: a 1024px bf16 decode peaks at 8.38 GiB on an RTX 4090, 11.81 GiB on a W7900
# on Linux, and 14.06 GiB (10.49 GiB with expandable segments) on an RX 9060 XT on Windows.
#
# On MIOpen the bf16 decode steps up from ~12.5 KB per pixel to ~14.4 KB between 1264px and 1280px, so frames past the
# step price on a line of their own, `decode_large_bf16`. A tiled operation prices its tile on the untiled line, the tile clamped to the frame
# per axis, plus `tiled_<operation>_<dtype>` over the frame's pixels for what tiling adds (`fit.Line.sits_on`).
_LINES = ("decode_bf16", "decode_fp32", "encode_bf16", "encode_fp32")
_COEFFICIENTS = frozenset(
    f"{part}_{line}"
    for part in ("k", "b")
    for line in (*_LINES, *(f"tiled_{line}" for line in _LINES), "decode_large_bf16")
)
_DTYPES = {2: "bf16", 4: "fp32"}
_MIOPEN_DECODE_STEP_PIXELS = 1_580_000


def _dtype(element_size: int) -> str:
    """The dtype of a VAE with elements this wide. Two bytes are bf16: the loader never holds this VAE in fp16."""
    dtype = _DTYPES.get(element_size)
    if dtype is None:
        raise ValueError(f"Qwen-Image-2.1's VAE has no working-memory lines for {element_size}-byte elements.")
    return dtype


def _untiled_line(suffix: str, pixels: int, features: Mapping[str, FeatureValue]) -> str:
    """The untiled line `pixels` price on: past MIOpen's step, a bf16 decode's own."""
    stepped = (
        suffix == "decode_bf16" and features.get("conv_backend") == "miopen" and pixels > _MIOPEN_DECODE_STEP_PIXELS
    )
    return "decode_large_bf16" if stepped else suffix


def _line(features: Mapping[str, FeatureValue]) -> Line | None:
    """`convolution_line`, without the assembly phase, and with what lies past MIOpen's bf16 decode step on its own line:
    an untiled frame, and a tile that large, which costs what the frame of its size does.

    The features' `tile` is None where the tile covers the frame (`estimate_working_memory`)."""
    line = convolution_line(features)
    if line is None:
        return None
    if line.sits_on is not None:
        beneath, tile_area = line.sits_on
        # shortcut: a tiled decode's assembly is not priced separately, negligible next to a tile up to the measured
        # 3072px; give this VAE an assembly line if frames past ~8K get measured.
        return line._replace(sits_on=(_untiled_line(beneath, tile_area, features), tile_area), assembly=None)
    return line._replace(suffix=_untiled_line(line.suffix, line.size, features))


def _terms(features: Mapping[str, FeatureValue], values: Mapping[str, float]) -> tuple[WorkingMemoryTerm, ...]:
    line = _line(features)
    assert line is not None, "an estimate's features always place it on a line"
    if line.sits_on is None:
        return (
            WorkingMemoryTerm.fitted("convolutions", f"k_{line.suffix}", line.size, values),
            WorkingMemoryTerm.fitted("convolution_base", f"b_{line.suffix}", 1, values),
        )
    beneath, tile_area = line.sits_on
    return (
        WorkingMemoryTerm.fitted("convolutions", f"k_{beneath}", tile_area, values),
        WorkingMemoryTerm.fitted("convolution_base", f"b_{beneath}", 1, values),
        WorkingMemoryTerm.fitted("tiled", f"k_{line.suffix}", line.size, values),
        WorkingMemoryTerm.fitted("tiled_base", f"b_{line.suffix}", 1, values),
    )


SPEC = register(
    EstimatorSpec(
        id="vae.qwen_image_2_1",
        version=1,
        coefficients=_COEFFICIENTS,
        platform_keys=("conv_backend", "allocator"),
        terms=_terms,
        line=_line,
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
    covering = tile_size is None or (tile_size >= height and tile_size >= width)
    features = {
        "operation": operation,
        "height": height,
        "width": width,
        # The VAE tiles only when a side is larger than the tile; a tile covering the frame runs the single pass.
        "tile": None if covering else tile_size,
        "dtype": _dtype(element_size),
        "conv_backend": priced_as(platform_for(device)).conv_backend,
    }
    return SPEC.estimate(features, device)


@functools.cache
def _measured_untiled_peaks(operation: str, platform: Platform, dtype: str) -> tuple[tuple[int, int], ...]:
    """The shipped untiled points of `operation` in `dtype` that cap the line where `platform` is priced, as (area,
    bytes) by area; the larger where two shapes share an area."""
    data = shipped_data().values()
    entries = [entry for file in data for entry in file.coefficients]
    priced = {key: str(value) for key, value in dataclasses.asdict(priced_as(platform)).items()}
    measured_on = find_entry(entries, SPEC.id, SPEC.version, priced, str(priced)).platform
    peaks: dict[int, int] = {}
    for file in data:
        for point in file.points:
            features = point.features
            if (
                point.estimator == SPEC.id
                and point.version == SPEC.version
                and point.in_fit
                and point.holds_upper_bound
                and point.platform == measured_on
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
    that fit, and tiled output is not pixel-identical. So within the measured untiled points of the platform the device
    is priced as (each OS-specific entry has its own) this is the chord between the two neighbouring points, in bytes,
    and below the first one that point's bytes per pixel, as `qwen_image_untiled_decode_peak_bytes` does for Qwen-Image.
    Past the last point, or with no point at all, it is the reservation. One point decides up to its own size: cuDNN's,
    so a 1024px decode on NVIDIA tiles on its measured 8.38 GiB rather than on what it reserves.
    """
    peaks = _measured_untiled_peaks(operation, platform_for(device), _dtype(element_size))
    area = height * width
    if not peaks or area > peaks[-1][0]:
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
