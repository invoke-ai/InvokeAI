import io
import json
import logging
import struct
import time
import tracemalloc
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from invokeai.app.api.extract_metadata import extract_metadata_from_image
from invokeai.backend.image_util.raster import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    RasterImage,
    ValueRange,
)
from invokeai.backend.image_util.raster.png_codec import (
    DEFAULT_PNG_LIMITS,
    PngCodecError,
    PngLimits,
    PngText,
    decode_png,
    encode_png,
    png_ihdr,
    read_png_text,
)

SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHANNEL_COUNTS = {RasterChannels.L: 1, RasterChannels.RGB: 3, RasterChannels.RGBA: 4}
# IHDR color type per channel layout, written down here rather than taken from the codec.
COLOR_TYPES = {RasterChannels.L: 0, RasterChannels.RGB: 2, RasterChannels.RGBA: 6}

METADATA = '{"model": "sdxl", "seed": 123}'
WORKFLOW = json.dumps(
    {
        "name": "wf",
        "author": "",
        "description": "",
        "version": "1.0.0",
        "contact": "",
        "tags": "",
        "notes": "",
        "exposedFields": [],
        "meta": {"version": "3.0.0", "category": "user"},
        "nodes": [],
        "edges": [],
    }
)
GRAPH = '{"nodes": {}, "edges": []}'
RESOLVED = '{"nodes": {"a": {}}, "edges": []}'


def descriptor(channels=RasterChannels.RGB, *, color=ColorEncoding.SRGB, alpha=None, icc=None):
    if alpha is None:
        alpha = AlphaMode.STRAIGHT if channels in (RasterChannels.LA, RasterChannels.RGBA) else AlphaMode.NONE
    count = {RasterChannels.LA: 2, **CHANNEL_COUNTS}[channels]
    return RasterDescriptor(channels, color, ValueRange.UNIT, alpha, count, icc)


def sample_raster(channels, dtype, *, shape=(7, 13), icc=None, seed=0):
    """Random full-range samples, so 16-bit low bytes are unrelated to the high bytes."""
    rng = np.random.default_rng(seed)
    count = CHANNEL_COUNTS[channels]
    info = np.iinfo(dtype)
    pixels = rng.integers(info.min, int(info.max) + 1, size=(*shape, count), dtype=np.int64).astype(dtype)
    return RasterImage.from_array(pixels, descriptor(channels, icc=icc))


def parse_chunks(data):
    """Independent chunk walker: [(type, payload)], verifying signature and CRCs."""
    assert data[:8] == SIGNATURE
    chunks, pos = [], 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind, payload = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(kind + payload)
        chunks.append((kind, payload))
        pos += 12 + length
    return chunks


def make_chunk(kind, payload, *, crc=None):
    crc = zlib.crc32(kind + payload) if crc is None else crc
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


AFTER_IHDR = 8 + 25


def insert_before_idat(png, *chunks):
    return png[:AFTER_IHDR] + b"".join(chunks) + png[AFTER_IHDR:]


def insert_before_iend(png, *chunks):
    return png[:-12] + b"".join(chunks) + png[-12:]


def idat_payload(png):
    return b"".join(payload for kind, payload in parse_chunks(png) if kind == b"IDAT")


def with_idat_chunks(png, *payloads):
    """The same image with its IDAT stream replaced by the given chunk payloads (placed in order)."""
    chunks = [make_chunk(b"IDAT", payload) for payload in payloads]
    return png[:AFTER_IHDR] + b"".join(chunks) + png[-12:]


def pillow_pixels(data, dtype, channels):
    """Pixels as Pillow reads them. Pillow opens 16-bit RGB(A) PNGs as 8-bit (high byte), 16-bit gray as I;16."""
    pixels = np.asarray(Image.open(io.BytesIO(data)))
    if pixels.ndim == 2:
        pixels = pixels[:, :, np.newaxis]
    return pixels.astype(np.uint16) if dtype == np.uint16 and channels is RasterChannels.L else pixels


def ihdr_of(png):
    kind, payload = parse_chunks(png)[0]
    assert kind == b"IHDR"
    width, height, depth, color_type = struct.unpack(">IIBB", payload[:10])
    return width, height, depth, color_type


def pillow_png(image, **save_args):
    buffer = io.BytesIO()
    image.save(buffer, "PNG", **save_args)
    return buffer.getvalue()


def small_png():
    return encode_png(sample_raster(RasterChannels.RGB, np.uint8, shape=(4, 5)))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [RasterChannels.L, RasterChannels.RGB, RasterChannels.RGBA])
class TestPixelRoundTrip:
    def test_exact_round_trip_and_ihdr(self, channels, dtype):
        raster = sample_raster(channels, dtype, seed=3)
        data = encode_png(raster)

        _, _, depth, color_type = ihdr_of(data)
        assert depth == np.dtype(dtype).itemsize * 8
        assert color_type == COLOR_TYPES[channels]

        decoded = decode_png(data)
        assert decoded.raster.dtype == dtype
        assert decoded.raster.descriptor == raster.descriptor
        assert decoded.raster.is_owned
        np.testing.assert_array_equal(decoded.raster.view(), raster.view())

    def test_pillow_reads_the_same_pixels(self, channels, dtype):
        raster = sample_raster(channels, dtype, seed=4)
        expected = raster.view()
        if dtype == np.uint16 and channels is not RasterChannels.L:
            expected = expected >> 8  # Pillow keeps the high byte of 16-bit color images
        np.testing.assert_array_equal(pillow_pixels(encode_png(raster), dtype, channels), expected)

    @pytest.mark.parametrize("level", [0, 1, 9])
    def test_compress_levels_are_lossless(self, channels, dtype, level):
        raster = sample_raster(channels, dtype, shape=(5, 4), seed=level)
        decoded = decode_png(encode_png(raster, compress_level=level)).raster
        np.testing.assert_array_equal(decoded.view(), raster.view())


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [RasterChannels.RGB, RasterChannels.RGBA])
def test_channel_order_is_kept(channels, dtype):
    peak = np.iinfo(dtype).max
    pixels = np.zeros((2, 3, CHANNEL_COUNTS[channels]), dtype)
    pixels[0, 1, 0] = peak  # red
    if channels is RasterChannels.RGBA:
        pixels[:, :, 3] = peak
    data = encode_png(RasterImage.from_array(pixels, descriptor(channels)))

    np.testing.assert_array_equal(decode_png(data).raster.view(), pixels)
    assert tuple(pillow_pixels(data, dtype, channels)[0, 1][:3]) == (255, 0, 0)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_incompressible_image_fits_the_idat_bound(dtype):
    # Level 0 stores the data, the largest IDAT a real writer produces for this size.
    raster = sample_raster(RasterChannels.RGBA, dtype, shape=(200, 100), seed=9)
    decoded = decode_png(encode_png(raster, compress_level=0)).raster
    np.testing.assert_array_equal(decoded.view(), raster.view())


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_hidden_rgb_under_zero_alpha_survives(dtype):
    pixels = sample_raster(RasterChannels.RGBA, dtype, shape=(6, 9), seed=11).to_array()
    pixels[:, ::2, 3] = 0
    pixels[:, ::2, :3] |= 1  # guarantee non-zero hidden color
    raster = RasterImage.from_array(pixels, descriptor(RasterChannels.RGBA))

    decoded = decode_png(encode_png(raster)).raster.view()
    np.testing.assert_array_equal(decoded, pixels)


def test_alpha_mode_follows_the_color_type():
    rgb = decode_png(small_png()).raster.descriptor
    assert rgb.alpha is AlphaMode.NONE
    rgba = decode_png(encode_png(sample_raster(RasterChannels.RGBA, np.uint8))).raster.descriptor
    assert rgba.alpha is AlphaMode.STRAIGHT


class TestIcc:
    def test_profile_round_trips_and_pillow_sees_it(self):
        profile = bytes(np.random.default_rng(5).integers(0, 256, 3000, dtype=np.uint8))
        raster = sample_raster(RasterChannels.RGB, np.uint16, icc=profile)
        data = encode_png(raster, PngText(invokeai_metadata=METADATA))

        assert decode_png(data).raster.descriptor.icc_profile == profile
        assert Image.open(io.BytesIO(data)).info["icc_profile"] == profile
        kinds = [kind for kind, _ in parse_chunks(data)]
        assert kinds.index(b"iCCP") < kinds.index(b"IDAT")

    def test_absent_profile_is_none(self):
        assert decode_png(small_png()).raster.descriptor.icc_profile is None

    def test_pillow_written_profile_is_read(self):
        profile = b"fake-icc-profile" * 40
        data = pillow_png(Image.new("RGB", (3, 2), (1, 2, 3)), icc_profile=profile)
        assert decode_png(data).raster.descriptor.icc_profile == profile


class TestText:
    FULL = PngText(METADATA, WORKFLOW, GRAPH, RESOLVED)

    def test_full_partial_and_absent(self):
        raster = sample_raster(RasterChannels.RGB, np.uint8, shape=(3, 3))
        for text in (self.FULL, PngText(invokeai_graph=GRAPH), PngText(), None):
            assert decode_png(encode_png(raster, text)).text == (text or PngText())

    def test_text_precedes_pixel_data(self):
        data = encode_png(sample_raster(RasterChannels.RGB, np.uint8), self.FULL)
        kinds = [kind for kind, _ in parse_chunks(data)]
        assert kinds[0] == b"IHDR" and kinds[-1] == b"IEND"
        assert kinds[1] == b"tEXt" and kinds.index(b"tEXt") < kinds.index(b"IDAT")

    def test_chunk_type_follows_latin1_rule(self):
        raster = sample_raster(RasterChannels.L, np.uint8, shape=(2, 2))
        latin1 = "café ÿ"
        unicode_text = "Gemälde — 日本語 \U0001f600"
        nul_text = "a\x00b"
        text = PngText(invokeai_metadata=latin1, invokeai_workflow=unicode_text, invokeai_graph=nul_text)
        data = encode_png(raster, text)

        chunks = {
            payload.split(b"\x00")[0]: (kind, payload)
            for kind, payload in parse_chunks(data)
            if kind in (b"tEXt", b"iTXt")
        }
        assert chunks[b"invokeai_metadata"][0] == b"tEXt"
        assert chunks[b"invokeai_metadata"][1].endswith(latin1.encode("latin-1"))
        assert chunks[b"invokeai_workflow"][0] == b"iTXt"
        assert chunks[b"invokeai_workflow"][1].endswith(unicode_text.encode("utf-8"))
        assert chunks[b"invokeai_graph"][0] == b"iTXt"  # a NUL cannot live in tEXt
        assert decode_png(data).text == text

    @pytest.mark.parametrize("unicode_payload", [False, True])
    def test_multi_megabyte_text(self, unicode_payload):
        unit = '{"k": "ä日€"}, ' if unicode_payload else '{"key": "value"}, '
        big = unit * (3_000_000 // len(unit))
        raster = sample_raster(RasterChannels.L, np.uint8, shape=(2, 2))
        data = encode_png(raster, PngText(invokeai_metadata=big, invokeai_graph=big[::-1]))
        decoded = decode_png(data).text
        assert decoded.invokeai_metadata == big
        assert decoded.invokeai_graph == big[::-1]
        assert Image.open(io.BytesIO(data)).info["invokeai_metadata"] == big

    def test_zlib_text_from_pillow_is_decoded(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("invokeai_metadata", METADATA, zip=True)  # zTXt
        info.add_text("invokeai_workflow", PngImagePlugin.iTXt(WORKFLOW + "ü日", lang="de", tkey="x"), zip=True)
        info.add_text("invokeai_graph", GRAPH + "—")  # plain iTXt
        info.add_text("comment", "ignored")
        data = pillow_png(Image.new("RGB", (4, 3), (9, 8, 7)), pnginfo=info)
        kinds = {kind for kind, _ in parse_chunks(data)}
        assert {b"zTXt", b"iTXt"} <= kinds

        text = decode_png(data).text
        assert text == PngText(METADATA, WORKFLOW + "ü日", GRAPH + "—", None)

    def test_unknown_keys_are_ignored_even_if_repeated(self):
        png = insert_before_idat(
            small_png(),
            make_chunk(b"tEXt", b"Comment\x00one"),
            make_chunk(b"tEXt", b"Comment\x00two"),
            make_chunk(b"zTXt", b"Other\x00\x00" + zlib.compress(b"x")),
        )
        assert decode_png(png).text == PngText()

    @pytest.mark.parametrize(
        "chunks",
        [
            [make_chunk(b"tEXt", b"invokeai_graph\x00a"), make_chunk(b"tEXt", b"invokeai_graph\x00b")],
            [
                make_chunk(b"tEXt", b"invokeai_graph\x00a"),
                make_chunk(b"zTXt", b"invokeai_graph\x00\x00" + zlib.compress(b"b")),
            ],
        ],
    )
    def test_duplicate_invoke_key_is_an_error(self, chunks):
        with pytest.raises(PngCodecError, match="duplicate"):
            decode_png(insert_before_idat(small_png(), *chunks))

    def test_invalid_utf8_in_itxt_is_an_error(self):
        chunk = make_chunk(b"iTXt", b"invokeai_graph\x00\x00\x00\x00\x00\xff\xfe")
        with pytest.raises(PngCodecError):
            decode_png(insert_before_idat(small_png(), chunk))

    def test_uncompressed_itxt_ignores_the_method_byte(self):
        chunk = make_chunk(b"iTXt", b"invokeai_graph\x00\x00\x07\x00\x00hello")
        assert decode_png(insert_before_idat(small_png(), chunk)).text.invokeai_graph == "hello"

    def test_compressed_itxt_needs_method_zero(self):
        chunk = make_chunk(b"iTXt", b"invokeai_graph\x00\x01\x07\x00\x00" + zlib.compress(b"hello"))
        with pytest.raises(PngCodecError, match="compression method"):
            decode_png(insert_before_idat(small_png(), chunk))

    def test_text_after_the_image_data_is_read(self):
        png = insert_before_iend(small_png(), make_chunk(b"tEXt", b"invokeai_graph\x00late"))
        assert decode_png(png).text.invokeai_graph == "late"


class TestInterop:
    def test_pillow_reads_the_same_text_and_extractor_agrees(self):
        text = PngText(METADATA, WORKFLOW, GRAPH, RESOLVED)
        raster = sample_raster(RasterChannels.RGBA, np.uint8)
        ours = Image.open(io.BytesIO(encode_png(raster, text)))

        info = PngImagePlugin.PngInfo()
        for key in ("invokeai_metadata", "invokeai_workflow", "invokeai_graph", "invokeai_resolved_inputs"):
            info.add_text(key, getattr(text, key))
        reference = Image.open(io.BytesIO(pillow_png(Image.fromarray(raster.view(), "RGBA"), pnginfo=info)))

        for key in ("invokeai_metadata", "invokeai_workflow", "invokeai_graph", "invokeai_resolved_inputs"):
            assert ours.info[key] == reference.info[key] == getattr(text, key)

        logger = logging.getLogger("test")
        assert extract_metadata_from_image(ours, None, None, None, logger) == extract_metadata_from_image(
            reference, None, None, None, logger
        )
        extracted = extract_metadata_from_image(ours, None, None, None, logger)
        assert (extracted.invokeai_metadata, extracted.invokeai_workflow) == (METADATA, WORKFLOW)
        assert extracted.invokeai_graph == GRAPH and extracted.invokeai_resolved_inputs == RESOLVED

    @pytest.mark.parametrize(
        ("mode", "channels"), [("L", RasterChannels.L), ("RGB", RasterChannels.RGB), ("RGBA", RasterChannels.RGBA)]
    )
    def test_decodes_pillow_8_bit(self, mode, channels):
        pixels = sample_raster(channels, np.uint8, shape=(6, 10), seed=8).to_array()
        data = pillow_png(Image.fromarray(pixels[:, :, 0] if mode == "L" else pixels, mode))
        assert ihdr_of(data)[2] == 8
        np.testing.assert_array_equal(decode_png(data).raster.view(), pixels)

    def test_decodes_pillow_16_bit_gray(self):
        pixels = np.random.default_rng(2).integers(0, 65536, (5, 9), dtype=np.uint16)
        pil = Image.fromarray(pixels)
        assert pil.mode == "I;16"
        data = pillow_png(pil)
        assert ihdr_of(data)[2] == 16
        decoded = decode_png(data).raster
        assert decoded.dtype == np.uint16
        np.testing.assert_array_equal(decoded.view()[:, :, 0], pixels)

    @pytest.mark.parametrize("depth", [8, 16])
    def test_gray_alpha_decodes_exactly(self, depth):
        # LA cannot be written (OpenCV has no gray+alpha encoder) but files from other writers are exact.
        dtype = np.uint8 if depth == 8 else np.uint16
        gray = np.array([[1, 40000 % (2**depth), 77]], dtype=dtype)
        alpha = np.array([[250 % (2**depth), 3, 0]], dtype=dtype)
        width, height = 3, 1
        rows = b"\x00" + np.stack([gray[0], alpha[0]], axis=1).astype(">u2" if depth == 16 else "u1").tobytes()
        ihdr = struct.pack(">IIBBBBB", width, height, depth, 4, 0, 0, 0)
        png = (
            SIGNATURE + make_chunk(b"IHDR", ihdr) + make_chunk(b"IDAT", zlib.compress(rows)) + make_chunk(b"IEND", b"")
        )

        raster = decode_png(png).raster
        assert raster.descriptor.channels is RasterChannels.LA
        assert raster.descriptor.alpha is AlphaMode.STRAIGHT
        np.testing.assert_array_equal(raster.view()[:, :, 0], gray)
        np.testing.assert_array_equal(raster.view()[:, :, 1], alpha)


class TestRejectedEncodes:
    def test_float_rasters(self):
        pixels = np.zeros((2, 2, 3), np.float32)
        with pytest.raises(PngCodecError):
            encode_png(RasterImage.from_array(pixels, descriptor()))

    def test_unbounded_float(self):
        unbounded = RasterDescriptor(RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNBOUNDED, AlphaMode.NONE, 3)
        with pytest.raises(PngCodecError):
            encode_png(RasterImage.from_array(np.zeros((2, 2, 3), np.float32), unbounded))

    def test_data_raster(self):
        data = RasterDescriptor(RasterChannels.DATA, ColorEncoding.NONE, ValueRange.UNIT, AlphaMode.NONE, 3)
        with pytest.raises(PngCodecError):
            encode_png(RasterImage.from_array(np.zeros((2, 2, 3), np.uint8), data))

    def test_linear_srgb(self):
        linear = descriptor(color=ColorEncoding.LINEAR_SRGB)
        with pytest.raises(PngCodecError):
            encode_png(RasterImage.from_array(np.zeros((2, 2, 3), np.uint8), linear))

    def test_premultiplied_alpha(self):
        premultiplied = descriptor(RasterChannels.RGBA, alpha=AlphaMode.PREMULTIPLIED)
        with pytest.raises(PngCodecError):
            encode_png(RasterImage.from_array(np.zeros((2, 2, 4), np.uint8), premultiplied))

    def test_gray_alpha(self):
        with pytest.raises(PngCodecError):
            encode_png(RasterImage.from_array(np.zeros((2, 2, 2), np.uint8), descriptor(RasterChannels.LA)))

    @pytest.mark.parametrize("level", [-1, 10])
    def test_bad_compress_level(self, level):
        with pytest.raises(PngCodecError):
            encode_png(sample_raster(RasterChannels.L, np.uint8), compress_level=level)


class TestMalformedInput:
    def test_bad_signature_and_empty(self):
        for data in (b"", b"GIF89a" + b"\x00" * 40, small_png()[1:]):
            with pytest.raises(PngCodecError, match="signature"):
                decode_png(data)

    @pytest.mark.parametrize("keep", [8, 12, 25, 40, -12, -5, -1])
    def test_truncated(self, keep):
        png = small_png()
        with pytest.raises(PngCodecError):
            decode_png(png[:keep])

    def test_crc_mismatch(self):
        png = bytearray(small_png())
        index = png.index(b"IDAT") + 6
        png[index] ^= 0xFF
        with pytest.raises(PngCodecError, match="CRC"):
            decode_png(bytes(png))

    def test_crc_mismatch_in_ignored_chunk(self):
        png = insert_before_idat(small_png(), make_chunk(b"tEXt", b"Comment\x00x", crc=1))
        with pytest.raises(PngCodecError, match="CRC"):
            decode_png(png)

    def test_corrupt_pixel_stream_with_valid_crc(self):
        # Valid structure, but IDAT is not zlib data.
        ihdr = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
        png = SIGNATURE + make_chunk(b"IHDR", ihdr) + make_chunk(b"IDAT", b"garbage" * 5) + make_chunk(b"IEND", b"")
        with pytest.raises(PngCodecError):
            decode_png(png)

    def test_missing_idat(self):
        ihdr = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
        with pytest.raises(PngCodecError):
            decode_png(SIGNATURE + make_chunk(b"IHDR", ihdr) + make_chunk(b"IEND", b""))

    @pytest.mark.parametrize(("width", "height"), [(100_000, 1), (1, 100_000), (20_000, 20_000)])
    def test_oversized_dimensions_are_rejected_from_the_header(self, width, height):
        # IDAT is garbage with a valid CRC: only a header check can produce a "size"/"pixels" error.
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
        png = SIGNATURE + make_chunk(b"IHDR", ihdr) + make_chunk(b"IDAT", b"x") + make_chunk(b"IEND", b"")
        with pytest.raises(PngCodecError, match="exceeds|pixels"):
            decode_png(png)

    def test_dimension_limits_are_inclusive(self):
        limits = PngLimits(**{**DEFAULT_PNG_LIMITS.__dict__, "max_side": 8, "max_pixels": 40})
        decode_png(pillow_png(Image.new("RGB", (8, 5))), limits)
        decode_png(pillow_png(Image.new("RGB", (5, 8))), limits)
        with pytest.raises(PngCodecError, match="per side"):
            decode_png(pillow_png(Image.new("RGB", (9, 4))), limits)
        with pytest.raises(PngCodecError, match="pixels"):
            decode_png(pillow_png(Image.new("RGB", (8, 6))), limits)

    @pytest.mark.parametrize(("width", "height"), [(0, 5), (5, 0)])
    def test_zero_dimension(self, width, height):
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        png = SIGNATURE + make_chunk(b"IHDR", ihdr) + make_chunk(b"IDAT", b"x") + make_chunk(b"IEND", b"")
        with pytest.raises(PngCodecError, match="at least 1x1"):
            decode_png(png)

    @staticmethod
    def peak_of(call):
        tracemalloc.start()
        try:
            call()
        except PngCodecError as error:
            failure = error
        else:
            failure = None
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return failure, peak

    def test_text_bomb_is_stopped_inside_the_inflate(self):
        # The compressed bomb is below the budget, so only a bounded inflate can stop the 50 MB expansion.
        limits = PngLimits(**{**DEFAULT_PNG_LIMITS.__dict__, "max_text_bytes": 100_000})
        bomb = zlib.compress(b"a" * 50_000_000)
        assert len(bomb) < limits.max_text_bytes
        base = small_png()
        for chunk in (
            make_chunk(b"zTXt", b"invokeai_metadata\x00\x00" + bomb),
            make_chunk(b"iTXt", b"invokeai_metadata\x00\x01\x00\x00\x00" + bomb),
        ):
            png = insert_before_idat(base, chunk)
            error, peak = self.peak_of(lambda png=png: decode_png(png, limits))
            assert error is not None and "decompressed" in str(error)
            assert peak < 5_000_000

    def test_icc_bomb_is_stopped_inside_the_inflate(self):
        limits = PngLimits(**{**DEFAULT_PNG_LIMITS.__dict__, "max_icc_bytes": 100_000})
        bomb = zlib.compress(b"\x00" * 50_000_000)
        assert len(bomb) < limits.max_icc_bytes
        png = insert_before_idat(small_png(), make_chunk(b"iCCP", b"p\x00\x00" + bomb))
        error, peak = self.peak_of(lambda png=png: decode_png(png, limits))
        assert error is not None and "decompressed" in str(error)
        assert peak < 5_000_000

    def test_oversized_compressed_payloads_are_rejected_before_inflating(self):
        limits = PngLimits(**{**DEFAULT_PNG_LIMITS.__dict__, "max_text_bytes": 1000, "max_icc_bytes": 1000})
        noise = np.random.default_rng(0).bytes(200_000)  # incompressible, far over both budgets
        for chunk in (
            make_chunk(b"zTXt", b"invokeai_graph\x00\x00" + zlib.compress(noise)),
            make_chunk(b"iCCP", b"p\x00\x00" + zlib.compress(noise)),
        ):
            with pytest.raises(PngCodecError, match="size limit"):
                decode_png(insert_before_idat(small_png(), chunk), limits)

    def test_truncated_zlib_stream_in_text_is_rejected(self):
        compressor = zlib.compressobj()
        prefix = compressor.compress(b"hello world " * 20) + compressor.flush(zlib.Z_SYNC_FLUSH)  # valid, no end
        for chunk in (
            make_chunk(b"zTXt", b"invokeai_graph\x00\x00" + prefix),
            make_chunk(b"iCCP", b"p\x00\x00" + prefix),
        ):
            with pytest.raises(PngCodecError, match="complete zlib"):
                decode_png(insert_before_idat(small_png(), chunk))

    def test_uncompressed_text_over_limit(self):
        limits = PngLimits(**{**DEFAULT_PNG_LIMITS.__dict__, "max_text_bytes": 100})
        chunk = make_chunk(b"tEXt", b"invokeai_metadata\x00" + b"a" * 200)
        with pytest.raises(PngCodecError):
            decode_png(insert_before_idat(small_png(), chunk), limits)

    def test_text_limit_is_shared_between_fields(self):
        limits = PngLimits(**{**DEFAULT_PNG_LIMITS.__dict__, "max_text_bytes": 100})
        chunks = [make_chunk(b"tEXt", key + b"\x00" + b"a" * 60) for key in (b"invokeai_graph", b"invokeai_workflow")]
        decode_png(insert_before_idat(small_png(), chunks[0]), limits)
        with pytest.raises(PngCodecError):
            decode_png(insert_before_idat(small_png(), *chunks), limits)

    def test_text_limit_counts_utf8_bytes(self):
        limits = PngLimits(**{**DEFAULT_PNG_LIMITS.__dict__, "max_text_bytes": 100})
        body = ("é" * 40).encode("utf-8")  # 40 characters, 80 bytes
        chunks = [
            make_chunk(b"iTXt", key + b"\x00\x00\x00\x00\x00" + body)
            for key in (b"invokeai_graph", b"invokeai_workflow")
        ]
        decode_png(insert_before_idat(small_png(), chunks[0]), limits)
        with pytest.raises(PngCodecError, match="size limit"):
            decode_png(insert_before_idat(small_png(), *chunks), limits)

    def test_duplicate_iccp(self):
        chunk = make_chunk(b"iCCP", b"p\x00\x00" + zlib.compress(b"abc"))
        with pytest.raises(PngCodecError, match="duplicate"):
            decode_png(insert_before_idat(small_png(), chunk, chunk))

    def test_iccp_method_byte_is_checked(self):
        chunk = make_chunk(b"iCCP", b"p\x00\x01" + zlib.compress(b"abc"))
        with pytest.raises(PngCodecError, match="compression method"):
            decode_png(insert_before_idat(small_png(), chunk))

    def test_iccp_after_the_image_data_is_rejected(self):
        chunk = make_chunk(b"iCCP", b"p\x00\x00" + zlib.compress(b"abc"))
        with pytest.raises(PngCodecError, match="misplaced"):
            decode_png(insert_before_iend(small_png(), chunk))

    def test_too_many_ancillary_chunks(self):
        limits = PngLimits(**{**DEFAULT_PNG_LIMITS.__dict__, "max_ancillary_chunks": 3})
        chunk = make_chunk(b"tEXt", b"Comment\x00x")
        decode_png(insert_before_idat(small_png(), *[chunk] * 3), limits)
        with pytest.raises(PngCodecError, match="ancillary"):
            decode_png(insert_before_idat(small_png(), *[chunk] * 4), limits)

    def test_malformed_text_chunks(self):
        for chunk in (
            make_chunk(b"zTXt", b"invokeai_graph\x00\x00not-zlib"),
            make_chunk(b"zTXt", b"invokeai_graph\x00\x07" + zlib.compress(b"x")),
            make_chunk(b"iTXt", b"invokeai_graph\x00\x00\x00"),
            make_chunk(b"tEXt", b"invokeai_graph"),
        ):
            with pytest.raises(PngCodecError):
                decode_png(insert_before_idat(small_png(), chunk))

    def test_palette_png_is_rejected(self):
        image = Image.new("P", (4, 4))
        image.putpalette([255, 0, 0, 0, 255, 0])
        with pytest.raises(PngCodecError, match="palette"):
            decode_png(pillow_png(image))

    def test_one_bit_png_is_rejected(self):
        with pytest.raises(PngCodecError, match="bit depth"):
            decode_png(pillow_png(Image.new("1", (8, 8))))

    def test_color_key_transparency_is_rejected(self):
        with pytest.raises(PngCodecError, match="tRNS"):
            decode_png(pillow_png(Image.new("RGB", (4, 4), (1, 2, 3)), transparency=(1, 2, 3)))

    def test_junk_after_iend_is_ignored(self):
        raster = sample_raster(RasterChannels.RGB, np.uint8)
        decoded = decode_png(encode_png(raster) + b"trailing junk")
        np.testing.assert_array_equal(decoded.raster.view(), raster.view())


def build_png(*chunks):
    """A file from (type, payload) pairs or ready-made chunk bytes."""
    return SIGNATURE + b"".join(c if isinstance(c, bytes) else make_chunk(*c) for c in chunks)


def ihdr_chunk(width, height, depth, color_type, interlace=0):
    return make_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, interlace))


class TestChunkStructure:
    def test_idat_may_be_split_into_chunks(self):
        raster = sample_raster(RasterChannels.RGB, np.uint8, shape=(9, 9))
        stream = idat_payload(encode_png(raster))
        png = with_idat_chunks(encode_png(raster), stream[:7], stream[7:20], stream[20:])
        np.testing.assert_array_equal(decode_png(png).raster.view(), raster.view())

    def test_many_empty_idat_chunks_are_rejected_cheaply(self):
        base = small_png()
        png = base[:AFTER_IHDR] + make_chunk(b"IDAT", b"") * 200_000 + base[AFTER_IHDR:]
        started = time.perf_counter()
        error, peak = TestMalformedInput.peak_of(lambda: decode_png(png))
        assert error is not None and "empty IDAT" in str(error)
        assert peak < 5_000_000
        assert time.perf_counter() - started < 5

    def test_idat_far_beyond_the_image_size_is_rejected(self):
        # A 1x1 RGBA8 image needs a few dozen bytes of zlib data; 100k single-byte chunks are padding.
        png = build_png(ihdr_chunk(1, 1, 8, 6), *[(b"IDAT", b"x")] * 100_000, (b"IEND", b""))
        error, peak = TestMalformedInput.peak_of(lambda: decode_png(png))
        assert error is not None and "implausibly large" in str(error)
        assert peak < 5_000_000

    def test_non_contiguous_idat_is_rejected(self):
        raster = sample_raster(RasterChannels.RGB, np.uint8, shape=(9, 9))
        stream = idat_payload(encode_png(raster))
        png = build_png(
            ihdr_chunk(9, 9, 8, 2),
            (b"IDAT", stream[:10]),
            (b"tEXt", b"Comment\x00x"),
            (b"IDAT", stream[10:]),
            (b"IEND", b""),
        )
        with pytest.raises(PngCodecError, match="contiguous"):
            decode_png(png)

    def test_unknown_critical_chunk_is_rejected_but_unknown_ancillary_is_not(self):
        with pytest.raises(PngCodecError, match="critical"):
            decode_png(insert_before_idat(small_png(), make_chunk(b"ABCD", b"x")))
        decode_png(insert_before_idat(small_png(), make_chunk(b"abcd", b"x")))

    @pytest.mark.parametrize("kind", [b"ab1d", b"a b!", b"\xc3\xa4bc", b"tEX\x00"])
    def test_chunk_type_must_be_ascii_letters(self, kind):
        with pytest.raises(PngCodecError, match="letters"):
            decode_png(insert_before_idat(small_png(), make_chunk(kind, b"x")))

    def test_ihdr_must_come_first_and_only_once(self):
        png = small_png()
        ihdr = png[8:AFTER_IHDR]
        with pytest.raises(PngCodecError, match="IHDR"):
            decode_png(SIGNATURE + png[AFTER_IHDR:])
        with pytest.raises(PngCodecError, match="duplicate IHDR"):
            decode_png(insert_before_idat(png, ihdr))

    def test_plte_is_ignored_for_color_images_only_before_the_image_data(self):
        raster = sample_raster(RasterChannels.RGB, np.uint8, shape=(4, 5))
        png = encode_png(raster)
        palette = make_chunk(b"PLTE", bytes(range(12)))
        np.testing.assert_array_equal(decode_png(insert_before_idat(png, palette)).raster.view(), raster.view())
        for bad in (
            insert_before_iend(png, palette),
            insert_before_idat(png, palette, palette),
            insert_before_idat(png, make_chunk(b"PLTE", bytes(range(4)))),
        ):
            with pytest.raises(PngCodecError, match="PLTE"):
                decode_png(bad)
        gray = encode_png(sample_raster(RasterChannels.L, np.uint8, shape=(4, 5)))
        with pytest.raises(PngCodecError, match="PLTE"):
            decode_png(insert_before_idat(gray, palette))

    def test_iend_must_be_empty(self):
        png = small_png()
        with pytest.raises(PngCodecError, match="IEND"):
            decode_png(png[:-12] + make_chunk(b"IEND", b"x"))

    def test_interlaced_rgba16_decodes_exactly(self):
        # Hand-built Adam7 stream (filter 0), independent of the codec's bound arithmetic.
        height, width = 6, 5
        pixels = np.random.default_rng(12).integers(0, 65536, (height, width, 4), dtype=np.uint16)
        rows = []
        for x0, y0, dx, dy in (
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        ):
            for row in pixels[y0::dy, x0::dx]:
                if row.size:
                    rows.append(b"\x00" + row.astype(">u2").tobytes())
        png = build_png(
            ihdr_chunk(width, height, 16, 6, interlace=1), (b"IDAT", zlib.compress(b"".join(rows))), (b"IEND", b"")
        )
        assert Image.open(io.BytesIO(png)).size == (width, height)  # a well-formed interlaced file
        np.testing.assert_array_equal(decode_png(png).raster.view(), pixels)


def noise_rows(depth, width, height, samples, seed=0):
    """(pixels, filtered raw stream with filter 0) of incompressible noise."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 2**depth, (height, width, samples)).astype(">u2" if depth == 16 else "u1")
    return pixels, b"".join(b"\x00" + row.tobytes() for row in pixels)


def png_from_stream(width, height, depth, color_type, stream):
    return build_png(ihdr_chunk(width, height, depth, color_type), (b"IDAT", stream), (b"IEND", b""))


class _BitWriter:
    def __init__(self):
        self.out, self.acc, self.count = bytearray(), 0, 0

    def put(self, value, bits):  # LSB first
        self.acc |= value << self.count
        self.count += bits
        while self.count >= 8:
            self.out.append(self.acc & 255)
            self.acc >>= 8
            self.count -= 8

    def put_code(self, code, bits):  # Huffman codes are packed MSB first
        self.put(int(format(code, f"0{bits}b")[::-1], 2), bits)

    def finish(self):
        return bytes(self.out) + (bytes([self.acc & 255]) if self.count else b"")


def fixed_huffman_literals(raw):
    """zlib stream of one fixed-Huffman block with literals only (stb_image_write style): up to 12.5% larger."""
    bits = _BitWriter()
    bits.put(1, 1)  # final block
    bits.put(1, 2)  # fixed Huffman
    for byte in raw:
        bits.put_code(0x30 + byte, 8) if byte < 144 else bits.put_code(0x190 + byte - 144, 9)
    bits.put_code(0, 7)  # end of block
    return b"x" + bits.finish() + struct.pack(">I", zlib.adler32(raw))


class TestValidStreamsOfAnySize:
    """Valid writers beat zlib's default worst case; the IDAT cap must not reject what other decoders accept."""

    def test_pillow_level_1_noise(self):
        rgb, _ = noise_rows(8, 40, 30, 3, seed=1)
        data = pillow_png(Image.fromarray(rgb, "RGB"), compress_level=1)
        np.testing.assert_array_equal(decode_png(data).raster.view(), rgb)

        gray16 = np.random.default_rng(2).integers(0, 65536, (30, 40), dtype=np.uint16)
        data = pillow_png(Image.fromarray(gray16), compress_level=1)
        np.testing.assert_array_equal(decode_png(data).raster.view()[:, :, 0], gray16)

    @pytest.mark.parametrize("options", [{"wbits": 9}, {"memLevel": 1}])
    def test_small_window_or_memory_level(self, options):
        pixels, raw = noise_rows(8, 40, 30, 3, seed=3)
        compressor = zlib.compressobj(6, zlib.DEFLATED, options.get("wbits", 15), options.get("memLevel", 8))
        stream = compressor.compress(raw) + compressor.flush()
        np.testing.assert_array_equal(decode_png(png_from_stream(40, 30, 8, 2, stream)).raster.view(), pixels)

    def test_sync_flush_after_every_row(self):
        pixels, _ = noise_rows(8, 40, 30, 3, seed=4)
        compressor = zlib.compressobj(6)
        stream = b"".join(
            compressor.compress(b"\x00" + row.tobytes()) + compressor.flush(zlib.Z_SYNC_FLUSH) for row in pixels
        )
        stream += compressor.flush()
        np.testing.assert_array_equal(decode_png(png_from_stream(40, 30, 8, 2, stream)).raster.view(), pixels)

    @pytest.mark.parametrize(("depth", "color_type", "samples"), [(8, 2, 3), (16, 6, 4)])
    def test_fixed_huffman_literals(self, depth, color_type, samples):
        pixels, raw = noise_rows(depth, 24, 20, samples, seed=5)
        stream = fixed_huffman_literals(raw)
        assert zlib.decompress(stream) == raw
        assert len(stream) > len(raw)  # larger than the data it carries
        png = png_from_stream(24, 20, depth, color_type, stream)
        np.testing.assert_array_equal(decode_png(png).raster.view(), pixels)
        Image.open(io.BytesIO(png)).load()  # Pillow accepts it too


def test_read_png_text_returns_the_text_fields_without_decoding_pixels():
    raster = RasterImage.from_array(np.arange(24, dtype=np.uint16).reshape(2, 4, 3), descriptor())
    text = PngText(METADATA, WORKFLOW, GRAPH, RESOLVED)
    png = encode_png(raster, text)
    # Garbage inside the (CRC-protected) IDAT stream would fail a decode; the chunk reader never looks at it.
    idat = png.index(b"IDAT") - 4  # start of the IDAT chunk
    iend = png.index(b"IEND") - 4
    payload = bytes(iend - idat - 12)
    broken = png[:idat] + struct.pack(">I", len(payload)) + b"IDAT" + payload
    broken += struct.pack(">I", zlib.crc32(b"IDAT" + payload)) + png[iend:]

    assert read_png_text(png) == text
    assert read_png_text(broken) == text
    with pytest.raises(PngCodecError):
        decode_png(broken)


def test_read_png_text_enforces_the_text_limits():
    png = encode_png(
        RasterImage.from_array(np.zeros((2, 2, 3), np.uint8), descriptor()), PngText(invokeai_metadata="x" * 100)
    )

    with pytest.raises(PngCodecError):
        read_png_text(
            png, PngLimits(max_side=16, max_pixels=256, max_text_bytes=10, max_icc_bytes=10, max_ancillary_chunks=8)
        )


@pytest.mark.parametrize(
    ("channels", "dtype", "depth"),
    [(RasterChannels.RGB, np.uint16, 16), (RasterChannels.RGBA, np.uint8, 8), (RasterChannels.L, np.uint16, 16)],
)
def test_png_ihdr_reads_size_depth_and_channels(channels, dtype, depth):
    png = encode_png(RasterImage.from_array(np.zeros((3, 5, CHANNEL_COUNTS[channels]), dtype), descriptor(channels)))

    header = png_ihdr(png)

    assert (header.width, header.height, header.bit_depth, header.channels) == (5, 3, depth, channels)


def test_png_ihdr_returns_none_for_anything_that_is_not_a_png_start():
    assert png_ihdr(b"") is None
    assert png_ihdr(b"GIF89a" + bytes(40)) is None
    assert png_ihdr(SIGNATURE + bytes(30)) is None


def test_an_opencv_failure_is_reported_as_a_codec_error(monkeypatch):
    def failing_imencode(*args, **kwargs):
        raise cv2.error("imencode failed")

    monkeypatch.setattr(cv2, "imencode", failing_imencode)

    with pytest.raises(PngCodecError, match="imencode failed"):
        encode_png(sample_raster(RasterChannels.RGB, np.uint16, shape=(3, 3), seed=0))
