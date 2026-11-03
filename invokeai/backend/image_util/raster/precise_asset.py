"""Versioned safetensors file for an exact working-precision raster.

Layout: tensor `pixels` (H, W, C) in the raster dtype, optional tensor `icc_profile` (uint8, 1-D), and string
metadata `format`, `version` and `descriptor` (JSON). Generation metadata is not stored here. The path comes
from the caller; atomic publication (no partial file after a crash) belongs to the storage layer.

The writer and the reader enforce the same bounds, so a file this module writes can always be read back.
The reader validates header, bounds and tensor offsets against the file size before reading pixel bytes, then
reads them with one `readinto` into a preallocated array: peak memory is one copy of the pixel data. Values are
little-endian, as in every safetensors file.
"""

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import save_file

from invokeai.backend.image_util.raster.descriptor import (
    ColorEncoding,
    RasterDescriptor,
    RasterError,
    ValueRange,
)
from invokeai.backend.image_util.raster.raster import RasterImage

PRECISE_ASSET_FORMAT = "raster_safetensors"  # asset_format column value
PRECISE_ASSET_VERSION = 1  # format_version column value

_FILE_FORMAT = "invokeai.raster"  # value of the `format` metadata entry inside the file
_PIXELS = "pixels"
_ICC = "icc_profile"
_DESCRIPTOR_KEYS = frozenset({"channels", "channel_count", "color", "value_range", "alpha"})
_DTYPE_CODES = {
    "U8": np.dtype(np.uint8),
    "U16": np.dtype(np.uint16),
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
}

# The JSON header holds two tensors and three short strings; 64 KiB is far above that and far below a DoS.
MAX_HEADER_BYTES = 64 * 1024
# Longest side. Also keeps H*W*C comfortably inside integer range.
MAX_DIMENSION = 16384
# Decoded pixel bytes: 16384 x 16384 RGBA float32 (e.g. a 4x upscale of a 4096 square). Bounds a read's allocation.
MAX_PIXEL_BYTES = 1 << 32
# Real-world ICC profiles are a few KiB to a few MiB.
MAX_ICC_BYTES = 16 * 1024 * 1024
MAX_CHANNELS = 64


class PreciseAssetError(RasterError):
    """The precise asset file is unwritable, unreadable, corrupt, out of bounds or of an unsupported version."""


def _check_bounds(height: int, width: int, channels: int, itemsize: int, icc_length: int | None) -> None:
    """The one place the size bounds live; used for what is written and for what a header declares."""
    if not (1 <= height <= MAX_DIMENSION and 1 <= width <= MAX_DIMENSION):
        raise PreciseAssetError(f"image size {width}x{height} is outside 1..{MAX_DIMENSION} (limit)")
    if not 1 <= channels <= MAX_CHANNELS:
        raise PreciseAssetError(f"{channels} channels is outside 1..{MAX_CHANNELS} (limit)")
    if height * width * channels * itemsize > MAX_PIXEL_BYTES:
        raise PreciseAssetError(f"pixel data exceeds the limit of {MAX_PIXEL_BYTES} bytes")
    if icc_length is not None and icc_length > MAX_ICC_BYTES:
        raise PreciseAssetError(f"icc_profile exceeds the limit of {MAX_ICC_BYTES} bytes")


def check_precise_asset_bounds(raster: RasterImage) -> None:
    """Raise PreciseAssetError when `raster` is outside the bounds `write_precise_asset` enforces.

    Lets a caller refuse an oversized raster before it persists anything else.
    """
    icc = raster.descriptor.icc_profile
    _check_bounds(
        raster.height, raster.width, raster.channel_count, raster.dtype.itemsize, None if icc is None else len(icc)
    )


def write_precise_asset(raster: RasterImage, path: Path) -> None:
    """Write `raster` to `path` (created or overwritten).

    Rasters outside the reader's bounds are refused with PreciseAssetError before anything is written.
    I/O failures surface as OSError and MemoryError propagates unchanged; publication policy is the caller's.
    """
    check_precise_asset_bounds(raster)
    descriptor = raster.descriptor
    icc = descriptor.icc_profile
    # safetensors writes raw bytes without checking strides, so the array must be C-contiguous.
    tensors = {_PIXELS: np.ascontiguousarray(raster.view())}
    if icc is not None:
        tensors[_ICC] = np.frombuffer(icc, dtype=np.uint8)
    descriptor_json = json.dumps(
        {
            "channels": descriptor.channels.value,
            "channel_count": descriptor.channel_count,
            "color": descriptor.color.value,
            "value_range": descriptor.value_range.value,
            "alpha": descriptor.alpha.value,
        },
        sort_keys=True,
    )
    metadata = {"format": _FILE_FORMAT, "version": str(PRECISE_ASSET_VERSION), "descriptor": descriptor_json}
    try:
        save_file(tensors, str(path), metadata=metadata)
    except SafetensorError as e:
        # The inputs were validated above, so safetensors can only fail on I/O (it reports that as its own type).
        raise OSError(f"could not write precise asset {path}: {e}") from e


def read_precise_asset(path: Path) -> RasterImage:
    """Read an owned raster from `path`.

    PreciseAssetError means the content is corrupt, out of bounds or of another version. I/O failures (OSError,
    including a missing file or a sharing violation) and MemoryError propagate unchanged, so callers never
    mistake an I/O hiccup for a damaged asset.

    Only PRECISE_ASSET_VERSION is accepted; callers compare a stored version with it to report a mismatch.
    """
    with open(path, "rb") as f:
        header_length, header = _read_header(f)
        file_size = f.seek(0, 2)
        descriptor, dtype, shape, pixel_range, icc_range = _validate_header(header)
        if file_size != 8 + header_length + max(pixel_range[1], icc_range[1] if icc_range else 0):
            raise PreciseAssetError("file size does not match the tensor offsets")
        data_start = 8 + header_length

        pixels = np.empty(shape, dtype=dtype)
        f.seek(data_start + pixel_range[0])
        # The short-read checks below only fire if the file changed after the size check.
        if f.readinto(memoryview(pixels).cast("B")) != pixels.nbytes:
            raise PreciseAssetError("file ends inside the pixel data")
        icc_bytes = None
        if icc_range is not None:
            f.seek(data_start + icc_range[0])
            icc_bytes = f.read(icc_range[1] - icc_range[0])
            if len(icc_bytes) != icc_range[1] - icc_range[0]:
                raise PreciseAssetError("file ends inside the icc_profile")
    try:
        descriptor = RasterDescriptor(
            descriptor.channels,
            descriptor.color,
            descriptor.value_range,
            descriptor.alpha,
            descriptor.channel_count,
            icc_bytes,
        )
    except RasterError as e:
        raise PreciseAssetError(f"invalid descriptor: {e}") from e
    return RasterImage._adopt(pixels, descriptor)


def _read_header(f) -> tuple[int, dict[str, Any]]:
    prefix = f.read(8)
    if len(prefix) < 8:
        raise PreciseAssetError("file is too short to be a precise asset")
    (header_length,) = struct.unpack("<Q", prefix)
    if header_length > MAX_HEADER_BYTES:
        raise PreciseAssetError(f"header length {header_length} exceeds the limit of {MAX_HEADER_BYTES} bytes")
    raw = f.read(header_length)
    if len(raw) != header_length:
        raise PreciseAssetError("file ends inside the header")
    try:
        header = json.loads(
            raw.decode("utf-8")
        )  # safetensors headers are UTF-8; json.loads(bytes) would sniff UTF-16/32
    except (ValueError, RecursionError) as e:
        raise PreciseAssetError(f"header is not valid JSON: {e}") from e
    if not isinstance(header, dict):
        raise PreciseAssetError("header is not a JSON object")
    return header_length, header


def _validate_header(
    header: dict[str, Any],
) -> tuple[RasterDescriptor, np.dtype, tuple[int, int, int], tuple[int, int], tuple[int, int] | None]:
    """Check metadata, descriptor, tensor declarations and bounds.

    Returns (descriptor without ICC, pixel dtype, pixel shape, pixel byte range, icc byte range or None).
    Ranges are relative to the data section; together they must tile it without gaps or overlap.
    """
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or set(metadata) != {"format", "version", "descriptor"}:
        raise PreciseAssetError("metadata must contain exactly format, version and descriptor")
    if metadata["format"] != _FILE_FORMAT:
        raise PreciseAssetError(f"unexpected format {metadata['format']!r}")
    version = metadata["version"]
    if not isinstance(version, str) or version != str(PRECISE_ASSET_VERSION):
        raise PreciseAssetError(f"unsupported precise asset version {version!r}, expected '{PRECISE_ASSET_VERSION}'")

    try:
        fields = json.loads(metadata["descriptor"])
    except (TypeError, ValueError, RecursionError) as e:
        raise PreciseAssetError(f"descriptor is not valid JSON: {e}") from e
    if not isinstance(fields, dict) or set(fields) != _DESCRIPTOR_KEYS:
        raise PreciseAssetError(f"descriptor must contain exactly {sorted(_DESCRIPTOR_KEYS)}")
    try:
        descriptor = RasterDescriptor(**fields)
    except RasterError as e:
        raise PreciseAssetError(f"invalid descriptor: {e}") from e

    tensors = {name: entry for name, entry in header.items() if name != "__metadata__"}
    if set(tensors) not in ({_PIXELS}, {_PIXELS, _ICC}):
        raise PreciseAssetError(f"tensors must be {_PIXELS} and optionally {_ICC}, got {sorted(tensors)}")

    dtype, shape, pixel_range = _tensor_declaration(tensors[_PIXELS], _PIXELS)
    if len(shape) != 3:
        raise PreciseAssetError(f"pixels must have shape (H, W, C), got {shape}")
    height, width, channels = shape
    if channels != descriptor.channel_count:
        raise PreciseAssetError(f"pixels have {channels} channels, descriptor says {descriptor.channel_count}")
    if descriptor.value_range is ValueRange.UNBOUNDED and dtype.kind != "f":
        raise PreciseAssetError("unbounded value range requires a float dtype")

    icc_range = None
    icc_length = None
    if _ICC in tensors:
        if descriptor.color is ColorEncoding.NONE:
            raise PreciseAssetError("icc_profile requires a color encoding")
        icc_dtype, icc_shape, icc_range = _tensor_declaration(tensors[_ICC], _ICC)
        if icc_dtype != np.dtype(np.uint8) or len(icc_shape) != 1:
            raise PreciseAssetError("icc_profile must be a 1-D uint8 tensor")
        icc_length = icc_shape[0]
    _check_bounds(height, width, channels, dtype.itemsize, icc_length)

    if pixel_range[1] - pixel_range[0] != height * width * channels * dtype.itemsize:
        raise PreciseAssetError("pixel data_offsets do not match the declared shape")
    if icc_range is not None:
        if icc_range[1] - icc_range[0] != icc_length:
            raise PreciseAssetError("icc_profile data_offsets do not match the declared shape")
        # safetensors orders equal-sized dtypes by name, so with uint8 pixels the profile comes first.
        first, second = sorted((pixel_range, icc_range))
        if first[0] != 0 or second[0] != first[1]:
            raise PreciseAssetError("tensor data_offsets must tile the data section")
    elif pixel_range[0] != 0:
        raise PreciseAssetError("tensor data_offsets must tile the data section")
    return descriptor, dtype, (height, width, channels), pixel_range, icc_range


def _tensor_declaration(entry: Any, name: str) -> tuple[np.dtype, tuple[int, ...], tuple[int, int]]:
    if not isinstance(entry, dict):
        raise PreciseAssetError(f"{name} declaration is malformed")
    code = entry.get("dtype")
    dtype = _DTYPE_CODES.get(code) if isinstance(code, str) else None
    if dtype is None:
        raise PreciseAssetError(f"{name} has unsupported dtype {entry.get('dtype')!r}")
    shape = entry.get("shape")
    if not isinstance(shape, list) or not all(_is_count(d) for d in shape):
        raise PreciseAssetError(f"{name} has a malformed shape")
    offsets = entry.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2 or not all(_is_count(o) for o in offsets):
        raise PreciseAssetError(f"{name} has malformed data_offsets")
    if offsets[0] > offsets[1]:
        raise PreciseAssetError(f"{name} has malformed data_offsets")
    return dtype, tuple(shape), (offsets[0], offsets[1])


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
