"""Bounded PNG8/PNG16 codec for RasterImage with Invoke text chunks and an ICC profile.

Pixels go through OpenCV (`imencode`/`imdecode`); chunks (text, ICC) are handled by one small parser here. Decode
validates the chunk structure, every CRC and the IHDR before any pixel decode, caps the IDAT payload at a generous
multiple of the image size, and hands OpenCV only the critical chunks, so a hostile ancillary chunk never reaches
libpng. Works on bytes, never on paths.

Not interpreted: sRGB/gAMA/cHRM/cICP/eXIf (the raster is declared sRGB; the ICC profile, if any, is carried).
Not decoded: palette and sub-8-bit images, and gray/RGB images with a tRNS color key. Their exact meaning needs a
conversion this codec does not own (gallery files are read through Pillow), so they are rejected, not approximated.

libpng writes diagnostics to stderr when it meets a corrupt pixel stream; the call still ends in a PngCodecError.
"""

import struct
import zlib
from dataclasses import dataclass

import cv2
import numpy as np

from invokeai.backend.image_util.raster.descriptor import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    RasterError,
    ValueRange,
)
from invokeai.backend.image_util.raster.raster import RasterImage

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_CHUNK_LENGTH = 2**31 - 1  # PNG spec limit
_AFTER_IHDR = len(_SIGNATURE) + 12 + 13  # signature + IHDR chunk: where our chunks are inserted

# Color type -> sample count. Palette (3) is rejected; see module docstring.
_SAMPLES_BY_COLOR_TYPE = {0: 1, 2: 3, 4: 2, 6: 4}
_CHANNELS_BY_SAMPLES = {1: RasterChannels.L, 2: RasterChannels.LA, 3: RasterChannels.RGB, 4: RasterChannels.RGBA}
# Adam7 passes: (x start, y start, x step, y step).
_ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))

_TEXT_KEYS = ("invokeai_metadata", "invokeai_workflow", "invokeai_graph", "invokeai_resolved_inputs")
_ICC_PROFILE_NAME = b"ICC Profile"  # the name Pillow writes; readers ignore it
_MAX_KEYWORD = 79


class PngCodecError(RasterError):
    """Malformed, truncated or over-limit PNG input, or a raster this codec does not write."""


@dataclass(frozen=True)
class PngText:
    """The Invoke text fields of a PNG. Transport only: validation stays with `extract_metadata`."""

    invokeai_metadata: str | None = None
    invokeai_workflow: str | None = None
    invokeai_graph: str | None = None
    invokeai_resolved_inputs: str | None = None


@dataclass(frozen=True)
class PngLimits:
    max_side: int  # IHDR width/height
    max_pixels: int  # IHDR width * height: bounds the decoded allocation and, through it, the IDAT payload
    max_text_bytes: int  # decoded size of all Invoke text fields together, so a zlib bomb cannot expand past it
    max_icc_bytes: int  # decoded ICC profile size
    max_ancillary_chunks: int  # every chunk except IHDR/PLTE/IDAT/IEND; text and unknown chunks cost parse time


DEFAULT_PNG_LIMITS = PngLimits(
    max_side=16384,  # same bounds as precise assets
    max_pixels=2**28,
    max_text_bytes=64 * 1024 * 1024,  # graphs/workflows reach several MB; matches Pillow's MAX_TEXT_MEMORY
    max_icc_bytes=16 * 1024 * 1024,  # real profiles are a few hundred KB at most
    max_ancillary_chunks=256,
)


@dataclass(frozen=True)
class DecodedPng:
    raster: RasterImage  # uint8 for 8-bit files, uint16 for 16-bit files
    text: PngText


def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(payload, zlib.crc32(chunk_type))
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def _text_chunk(key: str, value: str) -> bytes:
    keyword = key.encode("latin-1")
    try:
        encoded = value.encode("latin-1")
    except UnicodeEncodeError:
        encoded = None
    # A NUL would end the tEXt value early, so such values travel as iTXt like non-Latin-1 ones.
    if encoded is not None and b"\x00" not in encoded:
        return _chunk(b"tEXt", keyword + b"\x00" + encoded)
    # iTXt: keyword, NUL, compression flag 0, method 0, empty language tag, NUL, empty translated keyword, NUL, text.
    return _chunk(b"iTXt", keyword + b"\x00\x00\x00\x00\x00" + value.encode("utf-8"))


def png_descriptor_refusal(descriptor: RasterDescriptor) -> str | None:
    """Why a uint8/uint16 raster with this descriptor cannot be written as PNG, or None when it can."""
    what = f"{descriptor.channels.value}/{descriptor.color.value}/{descriptor.alpha.value}"
    if descriptor.channels not in (RasterChannels.L, RasterChannels.RGB, RasterChannels.RGBA):
        # OpenCV cannot write gray+alpha PNGs, and widening to RGBA would change the file's meaning.
        return f"cannot write {what} as PNG: only L, RGB and RGBA"
    if (
        descriptor.color is not ColorEncoding.SRGB
        or descriptor.value_range is not ValueRange.UNIT
        or descriptor.alpha is AlphaMode.PREMULTIPLIED
    ):
        return f"cannot write {what} as PNG: only sRGB, unit-range, straight-alpha rasters"
    return None


def _check_encodable(raster: RasterImage) -> None:
    if raster.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise PngCodecError(f"cannot write {raster.dtype} raster as PNG: only uint8 and uint16 rasters")
    refusal = png_descriptor_refusal(raster.descriptor)
    if refusal is not None:
        raise PngCodecError(refusal)


def encode_png(raster: RasterImage, text: PngText | None = None, *, compress_level: int = 1) -> bytes:
    """PNG bytes for a uint8/uint16 L, RGB or RGBA sRGB raster; text and ICC follow the IHDR.

    Level 1 is OpenCV's fast profile (SUB filter, zlib RLE). Passing any compression level makes OpenCV drop that
    profile, so level 1 passes the strategy only; other levels use the default strategy.
    """
    _check_encodable(raster)
    if not isinstance(compress_level, int) or isinstance(compress_level, bool) or not 0 <= compress_level <= 9:
        raise PngCodecError(f"compress_level must be 0-9, got {compress_level!r}")

    pixels = raster.view()
    if pixels.shape[2] == 3:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    elif pixels.shape[2] == 4:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGRA)
    else:
        pixels = pixels[:, :, 0]
    if compress_level == 1:
        params = [cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
    ok, encoded = cv2.imencode(".png", pixels, params)
    del pixels
    if not ok:
        raise PngCodecError("OpenCV failed to encode the raster")

    extra = []
    icc_profile = raster.descriptor.icc_profile
    if icc_profile:
        extra.append(_chunk(b"iCCP", _ICC_PROFILE_NAME + b"\x00\x00" + zlib.compress(icc_profile)))
    if text is not None:
        for key in _TEXT_KEYS:
            value = getattr(text, key)
            if value is not None:
                extra.append(_text_chunk(key, value))
    if not extra:
        return encoded.tobytes()

    # OpenCV writes the IHDR first, so the chunks go at a fixed offset (a byte search could hit the IHDR CRC).
    png = memoryview(encoded)
    if bytes(png[:8]) != _SIGNATURE or bytes(png[12:16]) != b"IHDR":
        raise PngCodecError("OpenCV produced an unexpected PNG layout")
    return b"".join([png[:_AFTER_IHDR], *extra, png[_AFTER_IHDR:]])


def _size_cap(size: int) -> int:
    """Generous bound on a compressed stream of `size` decoded bytes.

    Valid writers exceed zlib's own worst case (small windows, flush per row, fixed-Huffman literals reach +12%), so
    this only stops floods and absurd payloads. Memory stays bounded by tracking offsets and by bounded inflation.
    """
    return 2 * size + 64 * 1024


def _inflate(data, limit: int, what: str) -> bytes:
    decompressor = zlib.decompressobj()
    try:
        out = decompressor.decompress(data, limit + 1)
        if len(out) > limit:
            raise PngCodecError(f"{what} exceeds {limit} bytes when decompressed")
        if not decompressor.eof or decompressor.unconsumed_tail:
            raise PngCodecError(f"{what} is not a complete zlib stream")
    except zlib.error as error:
        raise PngCodecError(f"{what} is not valid zlib data: {error}") from None
    return out


def _split_keyword(payload, what: str) -> tuple[bytes, memoryview]:
    """(keyword, rest after its NUL) of a payload that starts with a 1-79 byte keyword."""
    payload = memoryview(payload)
    separator = bytes(payload[: _MAX_KEYWORD + 1]).find(b"\x00")
    if separator < 1:
        raise PngCodecError(f"malformed {what} chunk")
    return bytes(payload[:separator]), payload[separator + 1 :]


def _parse_text_chunk(chunk_type: bytes, payload, budget: int) -> tuple[str, str]:
    """(keyword, value) of a tEXt/zTXt/iTXt chunk; compressed text may expand to at most `budget` bytes."""
    what = chunk_type.decode("ascii")
    keyword, rest = _split_keyword(payload, what)
    key = keyword.decode("latin-1")
    try:
        if chunk_type == b"tEXt":
            return key, bytes(rest).decode("latin-1")
        if chunk_type == b"zTXt":
            if not rest or rest[0] != 0:
                raise PngCodecError("zTXt chunk has an unknown compression method")
            return key, _inflate(rest[1:], budget, "zTXt text").decode("latin-1")
        # iTXt: compression flag, compression method, language tag NUL, translated keyword NUL, text.
        if len(rest) < 2 or rest[0] not in (0, 1):
            raise PngCodecError("malformed iTXt chunk")
        compressed = rest[0] == 1
        if compressed and rest[1] != 0:  # the method byte only matters for compressed text
            raise PngCodecError("iTXt chunk has an unknown compression method")
        _, separator, tail = bytes(rest[2:]).partition(b"\x00")
        if not separator:
            raise PngCodecError("malformed iTXt chunk")
        _, separator, body = tail.partition(b"\x00")
        if not separator:
            raise PngCodecError("malformed iTXt chunk")
        if compressed:
            body = _inflate(body, budget, "iTXt text")
        return key, body.decode("utf-8")
    except UnicodeDecodeError:
        raise PngCodecError(f"{what} text is not valid UTF-8") from None


def _max_idat_bytes(width: int, height: int, samples: int, bit_depth: int, interlace: int) -> int:
    """Size cap for the IDAT stream of the filtered image (one filter byte per scanline)."""
    bytes_per_pixel = samples * bit_depth // 8
    if interlace:
        raw = 0
        for x0, y0, dx, dy in _ADAM7:
            pass_width, pass_height = -(-(width - x0) // dx), -(-(height - y0) // dy)
            if pass_width > 0 and pass_height > 0:
                raw += pass_height * (pass_width * bytes_per_pixel + 1)
    else:
        raw = height * (width * bytes_per_pixel + 1)
    return _size_cap(raw)


def _validate_ihdr(ihdr: bytes, limits: PngLimits) -> tuple[int, int, int, int, int]:
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", ihdr)
    if width < 1 or height < 1:
        raise PngCodecError(f"image size {width}x{height} must be at least 1x1")
    if width > limits.max_side or height > limits.max_side:
        raise PngCodecError(f"image size {width}x{height} exceeds the limit of {limits.max_side} per side")
    if width * height > limits.max_pixels:
        raise PngCodecError(f"image has {width * height} pixels, limit is {limits.max_pixels}")
    if color_type == 3:
        raise PngCodecError("palette PNGs are not supported")
    if color_type not in _SAMPLES_BY_COLOR_TYPE:
        raise PngCodecError(f"invalid PNG color type {color_type}")
    if bit_depth not in (8, 16):
        raise PngCodecError(f"unsupported PNG bit depth {bit_depth}")
    if compression != 0 or filter_method != 0 or interlace not in (0, 1):
        raise PngCodecError("unsupported PNG compression, filter or interlace method")
    return width, height, _SAMPLES_BY_COLOR_TYPE[color_type], bit_depth, interlace


@dataclass(frozen=True)
class PngHeader:
    width: int
    height: int
    bit_depth: int
    channels: RasterChannels | None  # None for palette and invalid color types


def png_ihdr(data: bytes) -> PngHeader | None:
    """The IHDR fields of a PNG, or None when `data` does not start with a signature and an IHDR chunk. A cheap sniff:
    nothing is validated beyond that, so it never replaces `decode_png`."""
    if len(data) < _AFTER_IHDR or data[: len(_SIGNATURE)] != _SIGNATURE or data[12:16] != b"IHDR":
        return None
    width, height, bit_depth, color_type = struct.unpack_from(">IIBB", data, 16)
    samples = _SAMPLES_BY_COLOR_TYPE.get(color_type)
    return PngHeader(width, height, bit_depth, _CHANNELS_BY_SAMPLES[samples] if samples else None)


def read_png_text(data: bytes, limits: PngLimits = DEFAULT_PNG_LIMITS) -> PngText:
    """The Invoke text fields of a PNG, read from its chunks under the same structure, text and chunk limits as
    `decode_png` but without decoding the pixels."""
    return _parse_png(data, limits).text


@dataclass
class _ParsedPng:
    shape: tuple[int, int, int, int, int]
    text: PngText
    icc_profile: bytes | None
    view: memoryview
    ihdr_chunk: bytes
    idat_span: list[int]
    other_chunks_present: bool
    end: int  # position after IEND


def decode_png(data: bytes, limits: PngLimits = DEFAULT_PNG_LIMITS) -> DecodedPng:
    """Decode an 8- or 16-bit gray, gray+alpha, RGB or RGBA PNG. Anything else, or over `limits`, is rejected."""
    parsed = _parse_png(data, limits)
    width, height, samples, bit_depth, _ = parsed.shape
    view, position, idat_span = parsed.view, parsed.end, parsed.idat_span
    iend_chunk = bytes(view[position - 12 : position])
    if (
        not parsed.other_chunks_present
        and idat_span[0] == len(_SIGNATURE) + len(parsed.ihdr_chunk)
        and idat_span[1] == position - 12
    ):
        png = data  # nothing to strip
    else:
        png = b"".join([_SIGNATURE, parsed.ihdr_chunk, view[idat_span[0] : idat_span[1]], iend_chunk])
    pixels = _decode_pixels(png, width, height, samples, bit_depth)

    descriptor = RasterDescriptor(
        channels=_CHANNELS_BY_SAMPLES[samples],
        color=ColorEncoding.SRGB,
        value_range=ValueRange.UNIT,
        alpha=AlphaMode.STRAIGHT if samples in (2, 4) else AlphaMode.NONE,
        channel_count=samples,
        icc_profile=parsed.icc_profile,
    )
    return DecodedPng(raster=RasterImage._adopt(pixels, descriptor), text=parsed.text)


def _parse_png(data: bytes, limits: PngLimits) -> _ParsedPng:
    """Walks and validates every chunk (structure, CRCs, limits) and collects text and ICC; decodes no pixels."""
    if data[: len(_SIGNATURE)] != _SIGNATURE:
        raise PngCodecError("not a PNG file (bad signature)")

    view = memoryview(data)
    end = len(data)
    position = len(_SIGNATURE)
    shape: tuple[int, int, int, int, int] | None = None
    max_idat = 0
    idat_bytes = 0
    idat_span = [0, 0]  # first byte of the first IDAT chunk, end of the last
    idat_state = 0  # 0 before the image data, 1 inside the run of IDAT chunks, 2 after it
    seen_plte = False
    seen_iend = False
    ancillary_count = 0
    text_budget = limits.max_text_bytes
    texts: dict[str, str] = {}
    icc_profile: bytes | None = None
    seen_icc = False
    ihdr_chunk = b""
    other_chunks_present = False  # anything between IHDR and IEND besides the IDAT run

    while position < end and not seen_iend:
        if end - position < 12:
            raise PngCodecError("truncated PNG: incomplete chunk header")
        (length,) = struct.unpack_from(">I", data, position)
        chunk_type = bytes(view[position + 4 : position + 8])
        if not chunk_type.isalpha() or not chunk_type.isascii():
            raise PngCodecError("chunk type is not four ASCII letters")
        if length > _MAX_CHUNK_LENGTH:
            raise PngCodecError("chunk length exceeds the PNG limit")
        if end - position < 12 + length:
            raise PngCodecError("truncated PNG: chunk extends past the end of the file")
        chunk_start = position
        payload_start = position + 8
        payload = view[payload_start : payload_start + length]
        (stored_crc,) = struct.unpack_from(">I", data, payload_start + length)
        if zlib.crc32(payload, zlib.crc32(chunk_type)) != stored_crc:
            raise PngCodecError(f"CRC mismatch in {chunk_type!r} chunk")
        position = payload_start + length + 4

        if shape is None:
            if chunk_type != b"IHDR" or length != 13:
                raise PngCodecError("first chunk is not a valid IHDR")
            shape = _validate_ihdr(bytes(payload), limits)
            max_idat = _max_idat_bytes(*shape)
            ihdr_chunk = bytes(view[chunk_start:position])
            continue
        if chunk_type == b"IHDR":
            raise PngCodecError("duplicate IHDR")

        if chunk_type == b"IDAT":
            if idat_state == 2:
                raise PngCodecError("IDAT chunks are not contiguous")
            if length == 0:
                raise PngCodecError("empty IDAT chunk")  # also keeps the chunk count below the payload bound
            idat_bytes += length
            if idat_bytes > max_idat:
                raise PngCodecError("IDAT data is implausibly large for the image size")
            if idat_state == 0:
                idat_span[0] = chunk_start
            idat_span[1] = position
            idat_state = 1
            continue
        if idat_state == 1:
            idat_state = 2

        if chunk_type == b"IEND":
            if length:
                raise PngCodecError("IEND chunk has data")
            seen_iend = True
        elif chunk_type == b"PLTE":
            # Suggested palette of a color image: ignored. It is critical and must precede the image data.
            if shape[2] not in (3, 4) or seen_plte or idat_state or length % 3 or not 3 <= length <= 768:
                raise PngCodecError("misplaced or invalid PLTE chunk")
            seen_plte = True
            other_chunks_present = True
        elif chunk_type == b"tRNS":
            raise PngCodecError("tRNS transparency is not supported")
        elif chunk_type[0] < 0x61:  # uppercase first letter: critical
            raise PngCodecError(f"unknown critical chunk {chunk_type!r}")
        else:
            other_chunks_present = True
            ancillary_count += 1
            if ancillary_count > limits.max_ancillary_chunks:
                raise PngCodecError(f"more than {limits.max_ancillary_chunks} ancillary chunks")
            if chunk_type == b"iCCP":
                if seen_icc or idat_state:
                    raise PngCodecError("duplicate or misplaced iCCP chunk")
                seen_icc = True
                if length > _size_cap(limits.max_icc_bytes) + _MAX_KEYWORD + 2:
                    raise PngCodecError("ICC profile exceeds the size limit")
                _, rest = _split_keyword(payload, "iCCP")
                if not rest or rest[0] != 0:
                    raise PngCodecError("iCCP chunk has an unknown compression method")
                icc_profile = _inflate(rest[1:], limits.max_icc_bytes, "ICC profile") or None
            elif chunk_type in (b"tEXt", b"zTXt", b"iTXt"):
                keyword = bytes(payload[: _MAX_KEYWORD + 1]).partition(b"\x00")[0]
                if keyword.decode("latin-1") not in _TEXT_KEYS:
                    continue  # unknown keys are not even decompressed
                if length > _size_cap(text_budget) + 3 * _MAX_KEYWORD:
                    raise PngCodecError("Invoke text exceeds the size limit")
                key, value = _parse_text_chunk(chunk_type, payload, text_budget)
                if key in texts:
                    raise PngCodecError(f"duplicate {key} text chunk")
                texts[key] = value
                text_budget -= len(value.encode("utf-8")) if chunk_type == b"iTXt" else len(value)
                if text_budget < 0:
                    raise PngCodecError("Invoke text exceeds the size limit")

    if shape is None or not seen_iend:
        raise PngCodecError("truncated PNG: missing IEND")
    if not idat_state:
        raise PngCodecError("PNG has no image data")

    return _ParsedPng(shape, PngText(**texts), icc_profile, view, ihdr_chunk, idat_span, other_chunks_present, position)


def _decode_pixels(png: bytes, width: int, height: int, samples: int, bit_depth: int) -> np.ndarray:
    """Pixels as a fresh C-contiguous (H, W, samples) array in RGB(A) order."""
    try:
        decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        raise PngCodecError(f"OpenCV could not decode the PNG: {error}") from None
    if decoded is None:
        raise PngCodecError("OpenCV could not decode the PNG")
    expected_dtype = np.uint8 if bit_depth == 8 else np.uint16
    if decoded.dtype != expected_dtype or decoded.shape[:2] != (height, width):
        raise PngCodecError("decoded pixels do not match the IHDR")
    channels = 1 if decoded.ndim == 2 else decoded.shape[2]
    if channels != (4 if samples == 2 else samples):
        raise PngCodecError("decoded pixels do not match the IHDR")
    if samples == 1:
        return decoded[:, :, np.newaxis]
    if samples == 2:
        # OpenCV returns gray+alpha as BGRA with the gray replicated; keep one gray copy and the alpha.
        return np.ascontiguousarray(decoded[:, :, [0, 3]])
    # In place: OpenCV returns BGR(A).
    cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB if samples == 3 else cv2.COLOR_BGRA2RGBA, dst=decoded)
    return decoded
