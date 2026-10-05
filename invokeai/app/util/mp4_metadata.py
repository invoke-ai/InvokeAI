"""Generation metadata carried inside MP4 files.

Images carry their ``invokeai_metadata`` / ``invokeai_workflow`` / ``invokeai_graph`` JSON as
PNG text chunks, so a file dragged into any Invoke instance is recallable. This module gives
MP4s the same property: the three strings are stored as QuickTime *keyed metadata* — the
``moov/udta/meta`` box with an ``mdta`` ``keys`` table and an ``ilst`` of UTF-8 ``data``
atoms. That is the layout ffmpeg writes for ``-movflags use_metadata_tags`` and the one
ffprobe, exiftool and mediainfo display, so the tags are inspectable with ordinary tools. Like
PNG text chunks they survive a container-aware copy — an ffmpeg remux that passes
``-movflags use_metadata_tags`` — and not a plain ``-c copy`` or a re-encode.

Writing goes through the bundled ffmpeg as a stream-copy remux: hand-inserting a box into
``moov`` would mean re-pointing every chunk offset table when ``moov`` precedes ``mdat``.
The file's own global tags are replaced by ours (``-map_metadata``); per-stream metadata is
kept. The values travel in an *ffmetadata* file rather than ``-metadata`` arguments because
a graph JSON routinely exceeds the per-argument (Linux, 128 KiB) and whole-command-line
(Windows, 32 K) limits.

Reading is a pure-Python box walk: the bundled ffmpeg ships without ffprobe, and walking by
box sizes costs a handful of seeks regardless of file size (``mdat`` is skipped, never read).
"""

import struct
import subprocess
import tempfile
from collections.abc import Collection, Iterator, Mapping
from pathlib import Path
from typing import BinaryIO, Optional

import imageio_ffmpeg

# A single tag is a JSON string the app itself produced; a graph JSON is the largest and
# runs to a few hundred KiB. The cap exists only so a hostile file cannot make the reader
# allocate its declared box size.
MAX_TAG_BYTES = 8 * 1024 * 1024
# ``keys`` is a table of short names; a hostile count or name length is refused, not read.
_MAX_KEYS = 4096
_MAX_KEY_NAME_BYTES = 1024
# A real file has a dozen top-level boxes at most, and ``moov`` precedes every fragment in a
# fragmented file, so a raw upload padded with tiny top-level boxes is given up on rather than
# walked. Children of ``moov`` are bounded by ``moov``'s own size.
_MAX_TOP_LEVEL_BOXES = 1024
# Same wall-clock bound as the audio mux: a stream copy of a 1 GB upload is disk-bound and
# finishes in seconds, so this only stops a wedged child from holding the save forever.
REMUX_TIMEOUT_SECONDS = 600

_UTF8_TYPE_INDICATOR = 1
_FFMETADATA_HEADER = ";FFMETADATA1\n"


class Mp4MetadataError(RuntimeError):
    """The MP4 could not be rewritten with the requested tags."""


def _ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _iter_boxes(
    fh: BinaryIO, start: int, end: int, max_boxes: Optional[int] = None
) -> Iterator[tuple[bytes, int, int]]:
    """Yield ``(type, payload_start, payload_end)`` for each box in ``[start, end)``.

    Stops silently at the first structurally impossible box (size smaller than its own
    header, or one that overruns ``end``), which is how a truncated or non-MP4 file reads as
    "no tags" rather than raising. ``max_boxes`` stops the walk after that many boxes.
    """
    position = start
    seen = 0
    while position + 8 <= end and (max_boxes is None or seen < max_boxes):
        seen += 1
        fh.seek(position)
        header = fh.read(8)
        if len(header) < 8:
            return
        size = int.from_bytes(header[:4], "big")
        box_type = header[4:8]
        header_size = 8
        if size == 1:
            extended = fh.read(8)
            if len(extended) < 8:
                return
            size = int.from_bytes(extended, "big")
            header_size = 16
        elif size == 0:
            # "Extends to end of file" — legal only for the last top-level box.
            size = end - position
        if size < header_size or position + size > end:
            return
        yield box_type, position + header_size, position + size
        position += size


def _find_box(
    fh: BinaryIO, start: int, end: int, box_type: bytes, max_boxes: Optional[int] = None
) -> Optional[tuple[int, int]]:
    for found_type, payload_start, payload_end in _iter_boxes(fh, start, end, max_boxes):
        if found_type == box_type:
            return payload_start, payload_end
    return None


def _meta_payload_start(fh: BinaryIO, start: int, end: int) -> int:
    """``meta`` is a FullBox (4 bytes of version/flags) in ISO BMFF but a plain box in
    QuickTime files. Disambiguate the way ffmpeg does: if the bytes right after the header
    already spell a child box type, there is no version/flags word."""
    if start + 8 > end:
        return start
    fh.seek(start + 4)
    if fh.read(4) in (b"hdlr", b"keys", b"ilst", b"mhdr"):
        return start
    return start + 4


def _read_keys(fh: BinaryIO, start: int, end: int) -> dict[int, str]:
    """The ``keys`` table: 1-based index → key name, ``mdta`` namespace only."""
    if start + 8 > end:
        return {}
    fh.seek(start)
    _version_flags, count = struct.unpack(">II", fh.read(8))
    if count > _MAX_KEYS:
        return {}
    keys: dict[int, str] = {}
    position = start + 8
    for index in range(1, count + 1):
        if position + 8 > end:
            break
        fh.seek(position)
        size, namespace = struct.unpack(">I4s", fh.read(8))
        if size < 8 or position + size > end:
            break
        if namespace == b"mdta" and size - 8 <= _MAX_KEY_NAME_BYTES:
            name = fh.read(size - 8)
            try:
                keys[index] = name.decode("utf-8")
            except UnicodeDecodeError:
                pass
        position += size
    return keys


def _read_utf8_data(fh: BinaryIO, start: int, end: int) -> Optional[str]:
    """The first UTF-8 ``data`` atom of an ``ilst`` item, or None."""
    for box_type, payload_start, payload_end in _iter_boxes(fh, start, end):
        if box_type != b"data" or payload_end - payload_start < 8:
            continue
        fh.seek(payload_start)
        type_indicator, _locale = struct.unpack(">II", fh.read(8))
        if type_indicator != _UTF8_TYPE_INDICATOR:
            continue
        length = payload_end - payload_start - 8
        if length > MAX_TAG_BYTES:
            return None
        try:
            return fh.read(length).decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def read_mp4_tags(path: Path, keys: Optional[Collection[str]] = None) -> dict[str, str]:
    """Return the file's ``mdta`` keyed metadata as ``{key: value}`` for UTF-8 values.

    ``keys`` restricts the result (and the bytes read) to the named tags. A file that is not
    an MP4, has no keyed metadata, or is truncated yields ``{}``; only I/O errors propagate.
    """
    wanted = set(keys) if keys is not None else None
    result: dict[str, str] = {}
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        file_size = fh.tell()
        moov = _find_box(fh, 0, file_size, b"moov", max_boxes=_MAX_TOP_LEVEL_BOXES)
        if moov is None:
            return result
        udta = _find_box(fh, *moov, b"udta")
        if udta is None:
            return result
        meta = _find_box(fh, *udta, b"meta")
        if meta is None:
            return result
        meta_start = _meta_payload_start(fh, *meta)
        keys_box = _find_box(fh, meta_start, meta[1], b"keys")
        ilst = _find_box(fh, meta_start, meta[1], b"ilst")
        if keys_box is None or ilst is None:
            return result
        names = _read_keys(fh, *keys_box)
        for item_type, item_start, item_end in _iter_boxes(fh, *ilst):
            name = names.get(int.from_bytes(item_type, "big"))
            if name is None or (wanted is not None and name not in wanted):
                continue
            value = _read_utf8_data(fh, item_start, item_end)
            if value is not None:
                result[name] = value
    return result


def read_ftyp_major_brand(path: Path, search_limit: int = 64 * 1024) -> Optional[bytes]:
    """The ``ftyp`` major brand if a well-formed one appears within the first ``search_limit``
    bytes, else None."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        end = min(fh.tell(), search_limit)
        for box_type, payload_start, payload_end in _iter_boxes(fh, 0, end):
            if box_type == b"ftyp":
                if payload_end - payload_start < 4:
                    return None
                fh.seek(payload_start)
                return fh.read(4)
    return None


def _escape_ffmetadata(value: str) -> str:
    # The ffmetadata grammar: '=', ';', '#', '\\' and line breaks are special in keys and
    # values; a backslash before a CR or LF continues the line, which is how both are carried.
    return (
        value.replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("\n", "\\\n")
        .replace("\r", "\\\r")
    )


def write_mp4_tags(src: Path, dst: Path, tags: Mapping[str, str]) -> None:
    """Write ``dst`` = ``src``'s first video stream and every audio stream, stream-copied, with
    ``tags`` as the file's keyed metadata (replacing any global tags ``src`` had). An empty
    mapping therefore strips ``src``'s global tags.

    Subtitle, data and further video tracks are not carried: InvokeAI stores only what it
    serves, and the ingest path already narrows uploads the same way. ``dst`` is written with
    ``+faststart`` so progressive playback is preserved. Raises Mp4MetadataError when ffmpeg
    fails and ValueError for a tag the format cannot carry; ``dst`` may then be absent or
    partial and the caller owns cleanup.
    """
    if any(not key or "\r" in key or "\n" in key or value.endswith("\\") for key, value in tags.items()):
        # ffmpeg's ffmetadata reader treats any backslash right before the line break as a
        # continuation, escaped or not, so a value ending in a backslash would swallow the
        # next tag. JSON text never ends that way; the caller falls back to a sidecar.
        raise ValueError("MP4 tag keys must be single-line and non-empty; values may not end in a backslash")
    body = "".join(f"{_escape_ffmetadata(key)}={_escape_ffmetadata(value)}\n" for key, value in tags.items())
    # newline="\n": the escaped line breaks must reach ffmpeg byte-for-byte, never as CRLF.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".ffmeta", delete=False) as ffmeta:
        ffmeta.write(_FFMETADATA_HEADER + body)
        ffmeta_path = Path(ffmeta.name)
    try:
        try:
            proc = subprocess.run(
                [
                    _ffmpeg_exe(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(src),
                    "-i",
                    str(ffmeta_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-map_metadata",
                    "1",
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart+use_metadata_tags",
                    "-f",
                    "mp4",
                    str(dst),
                ],
                capture_output=True,
                timeout=REMUX_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as e:
            raise Mp4MetadataError(f"ffmpeg timed out writing metadata into {dst.name}") from e
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise Mp4MetadataError(f"ffmpeg could not write metadata into {dst.name}: {stderr[-500:]}")
        try:
            if dst.stat().st_size == 0:
                raise Mp4MetadataError(f"ffmpeg wrote an empty file for {dst.name}")
        except OSError as e:
            raise Mp4MetadataError(f"ffmpeg produced no output for {dst.name}") from e
    finally:
        ffmeta_path.unlink(missing_ok=True)
