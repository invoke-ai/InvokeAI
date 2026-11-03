import json
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from invokeai.backend.image_util.raster import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    RasterError,
    RasterImage,
    ValueRange,
)
from invokeai.backend.image_util.raster.precise_asset import (
    MAX_CHANNELS,
    MAX_DIMENSION,
    MAX_HEADER_BYTES,
    MAX_ICC_BYTES,
    PreciseAssetError,
    read_precise_asset,
    write_precise_asset,
)

CHANNEL_CASES = [
    (RasterChannels.L, 1, AlphaMode.NONE, ColorEncoding.SRGB),
    (RasterChannels.LA, 2, AlphaMode.STRAIGHT, ColorEncoding.LINEAR_SRGB),
    (RasterChannels.RGB, 3, AlphaMode.NONE, ColorEncoding.SRGB),
    (RasterChannels.RGBA, 4, AlphaMode.PREMULTIPLIED, ColorEncoding.LINEAR_SRGB),
    (RasterChannels.DATA, 5, AlphaMode.NONE, ColorEncoding.NONE),
]
DTYPES = [np.uint8, np.uint16, np.float16, np.float32]
UINT_VIEW = {np.dtype(np.uint8): np.uint8, np.dtype(np.uint16): np.uint16, np.dtype(np.float16): np.uint16}
UINT_VIEW[np.dtype(np.float32)] = np.uint32


def make_descriptor(channels, count, alpha, color, value_range=ValueRange.UNIT, icc=None) -> RasterDescriptor:
    return RasterDescriptor(channels, color, value_range, alpha, count, icc)


def sample_array(dtype, shape) -> np.ndarray:
    """Deterministic values; float arrays include negative, >1, subnormal, NaN and +-inf entries."""
    count = int(np.prod(shape))
    dtype = np.dtype(dtype)
    if dtype.kind == "u":
        return (np.arange(count, dtype=np.uint64) * 40503 % (np.iinfo(dtype).max + 1)).astype(dtype).reshape(shape)
    special = np.array([-1.5, 7.25, 0.0, -0.0, np.nan, np.inf, -np.inf, np.finfo(dtype).smallest_subnormal])
    values = np.resize(special, count).astype(dtype)
    values[::3] = (np.arange(count)[::3] / 7.0 - 3.0).astype(dtype)
    return values.reshape(shape)


def bits(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array).view(UINT_VIEW[array.dtype])


def read_header(path: Path) -> dict:
    data = path.read_bytes()
    (length,) = struct.unpack("<Q", data[:8])
    return json.loads(data[8 : 8 + length])


def write_crafted(path: Path, header: dict, payload: bytes | None = None, prefix_length: int | None = None) -> None:
    """Write a file with `header`; the payload defaults to zeros up to the largest declared end offset."""
    if payload is None:
        ends = [e["data_offsets"][1] for k, e in header.items() if k != "__metadata__"]
        payload = bytes(max(ends, default=0))
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw) if prefix_length is None else prefix_length) + raw + payload)


def good_header(**overrides) -> dict:
    """Header of a valid 1x2 RGB uint8 asset, with selected top-level entries replaced."""
    descriptor = {
        "channels": "RGB",
        "channel_count": 3,
        "color": "srgb",
        "value_range": "unit",
        "alpha": "none",
    }
    header = {
        "__metadata__": {"format": "invokeai.raster", "version": "1", "descriptor": json.dumps(descriptor)},
        "pixels": {"dtype": "U8", "shape": [1, 2, 3], "data_offsets": [0, 6]},
    }
    header.update(overrides)
    return header


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("channels", "count", "alpha", "color"), CHANNEL_CASES)
def test_round_trip_is_bit_exact(tmp_path, dtype, channels, count, alpha, color):
    value_range = ValueRange.UNBOUNDED if np.dtype(dtype).kind == "f" else ValueRange.UNIT
    icc = b"\x00icc-profile\xff" if color is not ColorEncoding.NONE else None
    descriptor = make_descriptor(channels, count, alpha, color, value_range, icc)
    source = sample_array(dtype, (3, 5, count))  # non-square
    path = tmp_path / "a.safetensors"

    write_precise_asset(RasterImage.from_array(source, descriptor), path)
    loaded = read_precise_asset(path)

    assert loaded.descriptor == descriptor
    assert loaded.dtype == source.dtype
    assert loaded.view().shape == source.shape
    assert np.array_equal(bits(loaded.view()), bits(source))
    assert loaded.is_owned
    assert loaded.retained_nbytes == loaded.nbytes
    assert not loaded.view().flags.writeable


def test_empty_icc_profile_survives(tmp_path):
    descriptor = make_descriptor(RasterChannels.L, 1, AlphaMode.NONE, ColorEncoding.SRGB, icc=b"")
    write_precise_asset(RasterImage.from_array(sample_array(np.uint8, (2, 2, 1)), descriptor), tmp_path / "a.st")
    assert read_precise_asset(tmp_path / "a.st").descriptor.icc_profile == b""


def test_borrowed_non_contiguous_source_writes_the_visible_pixels(tmp_path):
    descriptor = make_descriptor(RasterChannels.RGB, 3, AlphaMode.NONE, ColorEncoding.SRGB)
    big = sample_array(np.uint16, (8, 10, 3))
    crop = big[1:7:2, 2:9, :]
    assert not crop.flags.c_contiguous
    write_precise_asset(RasterImage.borrow(crop, descriptor), tmp_path / "a.st")
    loaded = read_precise_asset(tmp_path / "a.st")
    assert np.array_equal(loaded.view(), crop)


def test_file_matches_the_documented_on_disk_format(tmp_path):
    icc = b"profile-bytes"
    descriptor = make_descriptor(
        RasterChannels.RGBA, 4, AlphaMode.STRAIGHT, ColorEncoding.SRGB, ValueRange.UNBOUNDED, icc
    )
    source = sample_array(np.float16, (2, 3, 4))
    path = tmp_path / "a.st"
    write_precise_asset(RasterImage.from_array(source, descriptor), path)

    tensors = load_file(str(path))
    assert set(tensors) == {"pixels", "icc_profile"}
    assert np.array_equal(bits(tensors["pixels"]), bits(source))
    assert tensors["pixels"].shape == (2, 3, 4)
    assert tensors["icc_profile"].dtype == np.uint8
    assert tensors["icc_profile"].tobytes() == icc

    metadata = read_header(path)["__metadata__"]
    assert set(metadata) == {"format", "version", "descriptor"}
    assert metadata["format"] == "invokeai.raster"
    assert metadata["version"] == "1"
    assert json.loads(metadata["descriptor"]) == {
        "channels": "RGBA",
        "channel_count": 4,
        "color": "srgb",
        "value_range": "unbounded",
        "alpha": "straight",
    }


def test_asset_without_icc_has_only_the_pixel_tensor(tmp_path):
    descriptor = make_descriptor(RasterChannels.L, 1, AlphaMode.NONE, ColorEncoding.SRGB)
    write_precise_asset(RasterImage.from_array(sample_array(np.uint8, (2, 2, 1)), descriptor), tmp_path / "a.st")
    assert set(load_file(str(tmp_path / "a.st"))) == {"pixels"}


def test_writer_io_failure_propagates_as_oserror(tmp_path):
    descriptor = make_descriptor(RasterChannels.L, 1, AlphaMode.NONE, ColorEncoding.SRGB)
    raster = RasterImage.from_array(sample_array(np.uint8, (2, 2, 1)), descriptor)
    with pytest.raises(OSError) as info:
        write_precise_asset(raster, tmp_path / "missing-dir" / "a.st")
    assert not isinstance(info.value, RasterError)


def test_non_canonical_nan_payloads_survive(tmp_path):
    descriptor = make_descriptor(RasterChannels.L, 1, AlphaMode.NONE, ColorEncoding.LINEAR_SRGB, ValueRange.UNBOUNDED)
    patterns = {
        np.float32: np.array([0x7FC00001, 0xFFC00000, 0x7F800001, 0x7FC00000], dtype=np.uint32),
        np.float16: np.array([0x7E01, 0xFC01, 0x7C01, 0xFE00], dtype=np.uint16),
    }
    for dtype, raw in patterns.items():
        source = raw.view(dtype).reshape(2, 2, 1)
        write_precise_asset(RasterImage.from_array(source, descriptor), tmp_path / "a.st")
        assert np.array_equal(read_precise_asset(tmp_path / "a.st").view().view(raw.dtype).ravel(), raw)


def test_strided_owned_raster_is_written_in_logical_order(tmp_path):
    descriptor = make_descriptor(RasterChannels.RGB, 3, AlphaMode.NONE, ColorEncoding.SRGB)
    source = sample_array(np.uint16, (3, 4, 3))
    reversed_channels = source[:, :, ::-1]
    assert not reversed_channels.flags.c_contiguous
    # The internal adopt path takes the array as is, so this raster is "owned" yet strided.
    strided = RasterImage._adopt(source.copy()[:, :, ::-1], descriptor)
    for raster in (strided, RasterImage.borrow(reversed_channels, descriptor)):
        write_precise_asset(raster, tmp_path / "a.st")
        assert np.array_equal(read_precise_asset(tmp_path / "a.st").view(), reversed_channels)
        assert np.array_equal(load_file(str(tmp_path / "a.st"))["pixels"], reversed_channels)


def zero_stride(dtype, shape) -> np.ndarray:
    """A huge-looking array that occupies no memory."""
    return np.broadcast_to(np.zeros((), dtype), shape)


UNWRITABLE = {
    "width": (zero_stride(np.uint8, (1, MAX_DIMENSION + 1, 3)), {}, "outside"),
    "height": (zero_stride(np.uint8, (MAX_DIMENSION + 1, 1, 3)), {}, "outside"),
    "channels": (zero_stride(np.uint8, (1, 1, MAX_CHANNELS + 1)), {"data": True}, "channels"),
    "icc": (zero_stride(np.uint8, (1, 1, 1)), {"icc": bytes(MAX_ICC_BYTES + 1)}, "icc_profile"),
    "bytes": (zero_stride(np.float32, (MAX_DIMENSION, MAX_DIMENSION, 5)), {"data": True}, "pixel data"),
}


@pytest.mark.parametrize(("array", "options", "message"), UNWRITABLE.values(), ids=UNWRITABLE.keys())
def test_writer_enforces_the_reader_bounds(tmp_path, array, options, message):
    count = array.shape[2]
    if options.get("data"):
        descriptor = make_descriptor(RasterChannels.DATA, count, AlphaMode.NONE, ColorEncoding.NONE)
    else:
        channels = {1: RasterChannels.L, 3: RasterChannels.RGB}[count]
        descriptor = make_descriptor(channels, count, AlphaMode.NONE, ColorEncoding.SRGB, icc=options.get("icc"))
    path = tmp_path / "a.st"
    with pytest.raises(PreciseAssetError, match=message):
        write_precise_asset(RasterImage.borrow(array, descriptor), path)
    assert not path.exists()


def test_writer_accepts_the_largest_dimension(tmp_path):
    descriptor = make_descriptor(RasterChannels.L, 1, AlphaMode.NONE, ColorEncoding.SRGB)
    raster = RasterImage.from_array(np.zeros((1, MAX_DIMENSION, 1), np.uint8), descriptor)
    write_precise_asset(raster, tmp_path / "a.st")
    assert read_precise_asset(tmp_path / "a.st").width == MAX_DIMENSION


def test_crafted_valid_file_reads(tmp_path):
    # Guards the rejection tests below: their base header is accepted unchanged.
    path = tmp_path / "valid.st"
    write_crafted(path, good_header(), payload=bytes(range(6)))
    assert read_precise_asset(path).view().tolist() == [[[0, 1, 2], [3, 4, 5]]]


def test_icc_after_pixels_is_accepted(tmp_path):
    # Base for the icc cases below: the same header with a correctly placed profile reads.
    path = tmp_path / "icc.st"
    write_crafted(path, good_header(**icc()), payload=bytes(range(10)))
    assert read_precise_asset(path).descriptor.icc_profile == bytes([6, 7, 8, 9])


def with_metadata(**changes) -> dict:
    header = good_header()
    header["__metadata__"] = {**header["__metadata__"], **changes}
    return header


def with_descriptor(**changes) -> dict:
    descriptor = {
        "channels": "RGB",
        "channel_count": 3,
        "color": "srgb",
        "value_range": "unit",
        "alpha": "none",
    }
    descriptor.update(changes)
    return with_metadata(descriptor=json.dumps({k: v for k, v in descriptor.items() if v is not ...}))


def pixels(dtype="U8", shape=(1, 2, 3), offsets=(0, 6)) -> dict:
    return {"pixels": {"dtype": dtype, "shape": list(shape), "data_offsets": list(offsets)}}


def icc(dtype="U8", shape=(4,), offsets=(6, 10)) -> dict:
    return {"icc_profile": {"dtype": dtype, "shape": list(shape), "data_offsets": list(offsets)}}


# name: (header, expected message fragment). Each case differs from the accepted base header in one rule.
REJECTED_HEADERS = {
    "wrong format": (with_metadata(format="something.else"), "format"),
    "version 2": (with_metadata(version="2"), "version '2'"),
    "version 0": (with_metadata(version="0"), "version '0'"),
    "version not a string": (with_metadata(version=1), "version 1"),
    "descriptor not json": (with_metadata(descriptor="{not json"), "not valid JSON"),
    "descriptor not an object": (with_metadata(descriptor="[1, 2]"), "exactly"),
    "missing descriptor key": (with_descriptor(alpha=...), "exactly"),
    "extra descriptor key": (with_descriptor(icc_profile="abc"), "exactly"),
    "invalid channels enum": (with_descriptor(channels="CMYK"), "invalid channels"),
    "invalid color enum": (with_descriptor(color="adobe"), "invalid color"),
    "channel count mismatch": (with_descriptor(channel_count=4), "channel_count"),
    "channel count not int": (with_descriptor(channel_count=3.0), "channel_count"),
    "alpha does not fit channels": (with_descriptor(alpha="straight"), "alpha"),
    "unbounded integer": (with_descriptor(value_range="unbounded"), "unbounded"),
    "icc on data": (
        {**with_descriptor(channels="data", color="none", channel_count=3), **icc()},
        "icc_profile requires",
    ),
    "missing metadata": ({k: v for k, v in good_header().items() if k != "__metadata__"}, "metadata"),
    "extra metadata entry": (with_metadata(extra="x"), "metadata"),
    "float64": (good_header(**pixels("F64", offsets=(0, 48))), "dtype"),
    "int32": (good_header(**pixels("I32", offsets=(0, 24))), "dtype"),
    "not hwc": (good_header(**pixels(shape=(2, 3))), "H, W, C"),
    "shape channel mismatch": (good_header(**pixels(shape=(1, 3, 2))), "channels"),
    "zero height": (good_header(**pixels(shape=(0, 2, 3), offsets=(0, 0))), "outside"),
    "negative dimension": (good_header(**pixels(shape=(1, -2, 3))), "malformed shape"),
    "extra tensor": (good_header(other={"dtype": "U8", "shape": [1], "data_offsets": [6, 7]}), "tensors must be"),
    "missing pixels": ({"__metadata__": good_header()["__metadata__"], **icc(offsets=(0, 4))}, "tensors must be"),
    "icc wrong dtype": (good_header(**icc("U16", (2,), (6, 10))), "1-D uint8"),
    "icc not 1-d": (good_header(**icc(shape=(2, 2))), "1-D uint8"),
    "dimension over bound": (
        good_header(**pixels(shape=(MAX_DIMENSION + 1, 1, 3), offsets=(0, (MAX_DIMENSION + 1) * 3))),
        "outside",
    ),
    "channels over bound": (
        {
            **with_descriptor(channels="data", color="none", channel_count=MAX_CHANNELS + 1),
            **pixels(shape=(1, 1, MAX_CHANNELS + 1), offsets=(0, MAX_CHANNELS + 1)),
        },
        "channels",
    ),
    "offsets do not match shape": (good_header(**pixels(offsets=(0, 5))), "do not match"),
    "pixels do not start at 0": (good_header(**pixels(offsets=(2, 8))), "tile"),
    "gap before icc": (good_header(**icc(offsets=(7, 11))), "tile"),
    "overlapping icc": (good_header(**icc(offsets=(4, 8))), "tile"),
    "nothing at offset 0": (good_header(**pixels(offsets=(2, 8)), **icc(offsets=(8, 12))), "tile"),
    "icc offsets longer than shape": (good_header(**icc(shape=(4,), offsets=(6, 11))), "icc_profile data_offsets"),
    "icc offsets shorter than shape": (good_header(**icc(shape=(4,), offsets=(6, 8))), "icc_profile data_offsets"),
    "unhashable dtype": (good_header(pixels={"dtype": [], "shape": [1, 2, 3], "data_offsets": [0, 6]}), "dtype"),
    "reversed offsets": (good_header(**pixels(offsets=(6, 0))), "data_offsets"),
}


@pytest.mark.parametrize(("header", "message"), REJECTED_HEADERS.values(), ids=REJECTED_HEADERS.keys())
def test_invalid_headers_are_rejected(tmp_path, header, message):
    path = tmp_path / "bad.st"
    write_crafted(path, header)
    with pytest.raises(PreciseAssetError, match=message):
        read_precise_asset(path)


@pytest.mark.parametrize(
    ("declared_header", "message"),
    [
        # 16384 x 16384 x 5 float32 = 5 GiB declared (over the byte bound); the file holds no pixel bytes at all.
        (
            {
                **with_descriptor(channels="data", color="none", channel_count=5),
                **pixels("F32", (MAX_DIMENSION, MAX_DIMENSION, 5), (0, MAX_DIMENSION * MAX_DIMENSION * 20)),
            },
            "pixel data exceeds",
        ),
        (good_header(**icc(shape=(MAX_ICC_BYTES + 1,), offsets=(6, 6 + MAX_ICC_BYTES + 1))), "icc_profile exceeds"),
    ],
    ids=["pixel bytes", "icc bytes"],
)
def test_oversized_declaration_is_rejected_from_a_tiny_file(tmp_path, declared_header, message):
    path = tmp_path / "huge.st"
    write_crafted(path, declared_header, payload=b"")
    assert path.stat().st_size < 4096
    with pytest.raises(PreciseAssetError, match=message):
        read_precise_asset(path)


@pytest.mark.parametrize("payload_length", [0, 5, 7], ids=["empty", "short", "trailing bytes"])
def test_file_size_must_match_the_declared_offsets(tmp_path, payload_length):
    path = tmp_path / "size.st"
    write_crafted(path, good_header(), payload=bytes(payload_length))
    with pytest.raises(PreciseAssetError, match="file size"):
        read_precise_asset(path)


def test_oversized_header_length_prefix_is_rejected_before_parsing(tmp_path):
    path = tmp_path / "prefix.st"
    write_crafted(path, good_header(), prefix_length=MAX_HEADER_BYTES + 1)
    with pytest.raises(PreciseAssetError, match="header length"):
        read_precise_asset(path)
    path.write_bytes(struct.pack("<Q", 2**63) + b"{}")
    with pytest.raises(PreciseAssetError, match="header length"):
        read_precise_asset(path)


def test_file_written_with_another_version_is_rejected_naming_it(tmp_path):
    path = tmp_path / "v2.st"
    save_file(
        {"pixels": np.zeros((1, 2, 3), np.uint8)},
        str(path),
        metadata={**good_header()["__metadata__"], "version": "2"},
    )
    with pytest.raises(PreciseAssetError, match="version '2'"):
        read_precise_asset(path)


def test_truncated_file_is_rejected(tmp_path):
    source = tmp_path / "full.st"
    descriptor = make_descriptor(RasterChannels.RGB, 3, AlphaMode.NONE, ColorEncoding.SRGB, icc=b"icc")
    write_precise_asset(RasterImage.from_array(sample_array(np.uint16, (2, 3, 3)), descriptor), source)
    data = source.read_bytes()
    for cut in (len(data) - 1, len(data) - 40, 8 + 10, 4):
        path = tmp_path / f"cut{cut}.st"
        path.write_bytes(data[:cut])
        with pytest.raises(PreciseAssetError):
            read_precise_asset(path)


@pytest.mark.parametrize(
    "content", [b"", b"\x01\x02\x03", bytes(range(256)) * 4, b"{" * 100, struct.pack("<Q", 5) + b"[1,2]"]
)
def test_garbage_files_are_rejected(tmp_path, content):
    path = tmp_path / "garbage.st"
    path.write_bytes(content)
    with pytest.raises(PreciseAssetError):
        read_precise_asset(path)


def test_missing_file_is_an_io_error_not_corruption(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        read_precise_asset(tmp_path / "nope.st")
    assert not isinstance(info.value, RasterError)


def test_directory_in_place_of_a_file_is_an_io_error(tmp_path):
    with pytest.raises(OSError) as info:
        read_precise_asset(tmp_path)
    assert not isinstance(info.value, RasterError)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-32"])
def test_header_must_be_utf8(tmp_path, encoding):
    raw = json.dumps(good_header()).encode(encoding)
    path = tmp_path / "enc.st"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(6))
    with pytest.raises(PreciseAssetError, match="not valid JSON"):
        read_precise_asset(path)


def test_deeply_nested_header_is_corruption_not_a_crash(tmp_path):
    raw = b"[" * 60000
    path = tmp_path / "deep.st"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(PreciseAssetError):
        read_precise_asset(path)
