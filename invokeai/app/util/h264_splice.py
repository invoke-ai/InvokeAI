"""Lossless splicing of H.264 MP4s by stream copy.

Re-encoding a clip that is already H.264 costs quality on every pass, and the losses
compound: a video extended N times by decode-everything-and-re-encode carries its first
segment through N generations of artifacts. When the head of a clip can be reused as-is,
:func:`copy_frames` cuts out a keyframe-aligned run of frames without decoding, and
:func:`concat_stream_copy` joins it to freshly encoded neighbours.

That join is only valid when every part carries the *same* sample entry: an MP4 stores
one per track — the ``avcC`` (profile, level, SPS and PPS) a decoder needs, and boxes such
as ``colr`` and ``pasp`` that say how to present the frames — so a part encoded or tagged
differently would be decoded or displayed against the wrong ones.
:func:`read_h264_layout` exposes those bytes so callers can require an exact match. Parts written by this
app's own encoder (``make_mp4_writer``) at the same canvas and frame rate match; anything
else is reported as not spliceable rather than guessed at.

Cuts are made only at IDR pictures. Nothing after an IDR in decode order references
anything before it, and everything before it is displayed before it, so the samples
between two IDRs decode to exactly the frames displayed between them. A plain sync
sample is not enough: an open-GOP I-frame is marked as one too, and the frames that
follow it in decode order but display before it would be lost from the copied run.

A joined file also has to keep one timeline. x264 gives a part of fewer than
``MIN_ENCODED_FRAMES`` frames a shorter B-frame reorder delay than a full one, and the
concat demuxer then squeezes timestamps at the seam; callers keep encoded parts out of
that range and check :attr:`H264Layout.frame_duration` of the result.

Sample indices only name displayed frames when nothing in the container reinterprets the
samples, so two more things make a track unspliceable: an edit list that hides samples (a
``-c copy`` trim keeps every sample and cuts with an edit list instead, so decoding yields
fewer frames than the sample table lists), and a display matrix other than the identity (a
rotation applied as metadata, which decoding applies to the frames this app re-encodes but
a stream copy carries into the joined file for all of them).
"""

import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable, NamedTuple, Optional

import numpy as np

from invokeai.app.util.mp4_metadata import find_box, iter_boxes
from invokeai.app.util.video_audio import run_ffmpeg_process

# A stream copy is disk-bound; this only stops a wedged child from holding the node forever.
SPLICE_TIMEOUT_SECONDS = 600
# ``moov`` precedes or follows ``mdat`` in a real file; a dozen top-level boxes at most.
_MAX_TOP_LEVEL_BOXES = 1024
# Children walked per container (moov, trak, mdia, minf, stbl, edts): real ones hold about a
# dozen, and the boxes are the uploader's, so a flood is given up on rather than scanned.
_MAX_CHILD_BOXES = 64
# Keyframes examined per track: at the 2 s interval this app writes, nine hours of video.
_MAX_SYNC_ENTRIES = 1 << 14
# VisualSampleEntry fields before its child boxes (SampleEntry 8 + visual header 70), and where
# among them the 32-byte compressor name sits.
_VISUAL_SAMPLE_ENTRY_HEADER = 78
_COMPRESSOR_NAME = slice(42, 74)
# A real sample entry holds a handful of boxes (avcC, btrt, colr, pasp...) in a few hundred
# bytes; more of either is hostile, and refused before any of it is read.
_MAX_ENTRY_CHILDREN = 32
_MAX_SAMPLE_ENTRY_BYTES = 64 * 1024
# NAL units examined per keyframe before its first slice (AUD, SEI, SPS, PPS precede it).
_MAX_NALS_BEFORE_SLICE = 64
# Shortest part x264 (bframes=3, as make_mp4_writer leaves it) encodes with its full B-frame
# reorder delay; a shorter part joined to full-delay parts distorts the joined timestamps.
MIN_ENCODED_FRAMES = 3
# Samples per track: over four hours at 60 fps. Bounds every table read before it is allocated.
_MAX_SAMPLES = 1 << 20
# NAL unit types: 1-5 are coded slices, 5 an IDR picture's.
_NAL_IDR = 5
_NAL_VCL = range(1, 6)
# The 3x3 ``tkhd`` display matrix that leaves frames as decoded (16.16 / 2.30 fixed point).
_IDENTITY_MATRIX = (0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


class H264AccessPoint(NamedTuple):
    """An IDR sample and its exact position on both the packet and presentation timelines."""

    sample_index: int
    frame_index: int
    dts: int
    pts: int


class H264Layout(NamedTuple):
    """What a splice needs to know about an MP4's first video track."""

    sample_count: int
    """Number of video samples (frames) in the track."""
    sync_samples: tuple[int, ...]
    """0-based decode-order indices of the IDR pictures — the only clean cut points — ascending."""
    decoder_config: bytes
    """The raw ``avcC`` payload: profile, level, SPS and PPS."""
    sample_entry: bytes
    """What a joined track inherits from this one's sample entry: its fixed fields and every
    child box — ``avcC``, and any ``colr``, ``pasp``, ``clap`` and the like, which change how
    frames are presented without changing ``avcC`` — except ``btrt``, a per-file bitrate.
    Two tracks can share a joined track only if these bytes match."""
    frame_duration: Optional[int]
    """Every sample's duration in media ticks, or None if they are not all equal."""
    time_scale: int
    """The media clock rate in ticks per second."""
    access_points: tuple[H264AccessPoint, ...]
    """IDR points with sample and presentation indices, DTS and PTS; empty unless timing is verified CFR."""


class SpliceError(RuntimeError):
    """ffmpeg failed to copy or join the streams."""


def _read_full_box_table(fh: BinaryIO, start: int, end: int, header: str) -> Optional[tuple[int, ...]]:
    """Read a FullBox's fixed header fields (after version/flags) as big-endian uint32s."""
    size = struct.calcsize(header)
    if start + 4 + size > end:
        return None
    fh.seek(start + 4)
    return struct.unpack(header, fh.read(size))


def read_h264_layout(path: Path) -> Optional[H264Layout]:
    """Return the first video track's layout if it is spliceable H.264, else None.

    None covers anything this module will not splice: no video track, a codec other than
    ``avc1``, a fragmented file (its samples live in ``moof`` boxes, not the sample table),
    an edit list that hides samples, a non-identity display matrix (see the module
    docstring), and truncated or malformed boxes. Only I/O errors propagate.
    """
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        file_size = fh.tell()
        moov = find_box(fh, 0, file_size, b"moov", max_boxes=_MAX_TOP_LEVEL_BOXES)
        if moov is None:
            return None
        for box_type, trak_start, trak_end in iter_boxes(fh, *moov, max_boxes=_MAX_CHILD_BOXES):
            if box_type != b"trak":
                continue
            mdia = find_box(fh, trak_start, trak_end, b"mdia", max_boxes=_MAX_CHILD_BOXES)
            if mdia is None:
                continue
            hdlr = find_box(fh, *mdia, b"hdlr", max_boxes=_MAX_CHILD_BOXES)
            if hdlr is None or hdlr[0] + 12 > hdlr[1]:
                continue
            fh.seek(hdlr[0] + 8)
            if fh.read(4) != b"vide":
                continue
            # First video track only: that is the stream every ffmpeg call here maps (0:v:0).
            minf = find_box(fh, *mdia, b"minf", max_boxes=_MAX_CHILD_BOXES)
            stbl = find_box(fh, *minf, b"stbl", max_boxes=_MAX_CHILD_BOXES) if minf is not None else None
            movie_timescale = _read_timescale(fh, find_box(fh, *moov, b"mvhd", max_boxes=_MAX_CHILD_BOXES))
            media_timescale = _read_timescale(fh, find_box(fh, *mdia, b"mdhd", max_boxes=_MAX_CHILD_BOXES))
            if stbl is None or movie_timescale is None or media_timescale is None:
                return None
            if not _has_identity_matrix(fh, find_box(fh, trak_start, trak_end, b"tkhd", max_boxes=_MAX_CHILD_BOXES)):
                return None
            edts = find_box(fh, trak_start, trak_end, b"edts", max_boxes=_MAX_CHILD_BOXES)
            if edts is not None and not _edits_show_every_sample(
                fh, find_box(fh, *edts, b"elst", max_boxes=_MAX_CHILD_BOXES), stbl, movie_timescale, media_timescale
            ):
                return None
            return _read_video_stbl(fh, *stbl, media_timescale)
    return None


def _read_timescale(fh: BinaryIO, box: Optional[tuple[int, int]]) -> Optional[int]:
    """The timescale of an ``mvhd`` or ``mdhd`` box (same layout up to that field)."""
    if box is None or box[0] + 24 > box[1]:
        return None
    fh.seek(box[0])
    offset = 20 if fh.read(1)[0] == 1 else 12
    fh.seek(box[0] + offset)
    (timescale,) = struct.unpack(">I", fh.read(4))
    return timescale or None


def _has_identity_matrix(fh: BinaryIO, tkhd: Optional[tuple[int, int]]) -> bool:
    if tkhd is None or tkhd[0] >= tkhd[1]:
        return False
    fh.seek(tkhd[0])
    offset = 52 if fh.read(1)[0] == 1 else 40
    if tkhd[0] + offset + 36 > tkhd[1]:
        return False
    fh.seek(tkhd[0] + offset)
    return struct.unpack(">9i", fh.read(36)) == _IDENTITY_MATRIX


def _edits_show_every_sample(
    fh: BinaryIO,
    elst: Optional[tuple[int, int]],
    stbl: tuple[int, int],
    movie_timescale: int,
    media_timescale: int,
) -> bool:
    """True if the edit list only shifts presentation to the first frame, as ffmpeg writes
    for B-frame reordering delay, rather than hiding any samples.

    That edit is a single normal-rate entry starting at the first sample's composition
    offset and lasting the whole track. A trim (later start or shorter duration), an empty
    leading edit, or several edits all change which samples are displayed.
    """
    if elst is None or elst[0] + 8 > elst[1]:
        return False
    fh.seek(elst[0])
    version = fh.read(1)[0]
    entry_format = ">QqhH" if version == 1 else ">IihH"
    entry_size = struct.calcsize(entry_format)
    fh.seek(elst[0] + 4)
    (entry_count,) = struct.unpack(">I", fh.read(4))
    if entry_count != 1 or elst[0] + 8 + entry_size > elst[1]:
        return False
    segment_duration, media_time, rate_integer, rate_fraction = struct.unpack(entry_format, fh.read(entry_size))
    if (rate_integer, rate_fraction) != (1, 0):
        return False

    runs = _read_stts(fh, stbl)
    if runs is None:
        return False
    track_duration = sum(count * delta for count, delta in runs)
    first_delta = runs[0][1]

    first_offset = 0
    ctts = find_box(fh, *stbl, b"ctts", max_boxes=_MAX_CHILD_BOXES)
    if ctts is not None:
        if ctts[0] + 16 > ctts[1]:
            return False
        fh.seek(ctts[0] + 12)
        (first_offset,) = struct.unpack(">i", fh.read(4))
    if media_time != first_offset:
        return False
    # The segment is in movie units, usually rounded to milliseconds: allow that rounding,
    # but not a missing frame.
    return segment_duration * media_timescale >= (track_duration - first_delta // 2) * movie_timescale


def _read_stts(fh: BinaryIO, stbl: tuple[int, int]) -> Optional[list[tuple[int, int]]]:
    """The ``(sample_count, sample_delta)`` runs of the time-to-sample table."""
    stts = find_box(fh, *stbl, b"stts", max_boxes=_MAX_CHILD_BOXES)
    if stts is None or stts[0] + 8 > stts[1]:
        return None
    fh.seek(stts[0] + 4)
    (count,) = struct.unpack(">I", fh.read(4))
    if count == 0 or count > _MAX_SAMPLES or stts[0] + 8 + 8 * count > stts[1]:
        return None
    flat = struct.unpack(f">{2 * count}I", fh.read(8 * count))
    return list(zip(flat[::2], flat[1::2], strict=True))


def _sample_sizes(fh: BinaryIO, stsz: tuple[int, int], sample_count: int, file_size: int) -> Optional[np.ndarray]:
    """Every sample's size, or None unless the table is plausible for a file of ``file_size``.

    The count comes from the file, so it is bounded before anything is allocated by it: by
    ``_MAX_SAMPLES``, by the table actually holding that many entries, and by the samples
    fitting in the file.
    """
    if not 0 < sample_count <= _MAX_SAMPLES or stsz[0] + 12 > stsz[1]:
        return None
    fh.seek(stsz[0] + 4)
    (uniform_size,) = struct.unpack(">I", fh.read(4))
    if uniform_size:
        if uniform_size * sample_count > file_size:
            return None
        return np.full(sample_count, uniform_size, dtype=np.int64)
    if stsz[0] + 12 + 4 * sample_count > stsz[1]:
        return None
    fh.seek(stsz[0] + 12)
    sizes = np.frombuffer(fh.read(4 * sample_count), dtype=">u4").astype(np.int64)
    return sizes if int(sizes.sum()) <= file_size else None


def _sample_locations(
    fh: BinaryIO, stbl: tuple[int, int], sizes: np.ndarray, wanted: tuple[int, ...], file_size: int
) -> Optional[list[tuple[int, int]]]:
    """``(file_offset, size)`` of each sample in ``wanted`` (ascending), from the chunk tables."""
    stsc = find_box(fh, *stbl, b"stsc", max_boxes=_MAX_CHILD_BOXES)
    stco, co64 = (
        find_box(fh, *stbl, b"stco", max_boxes=_MAX_CHILD_BOXES),
        find_box(fh, *stbl, b"co64", max_boxes=_MAX_CHILD_BOXES),
    )
    offsets_box, offset_format = (stco, "I") if stco is not None else (co64, "Q")
    # Each table opens with version/flags and a 4-byte count.
    if stsc is None or offsets_box is None or any(box[0] + 8 > box[1] for box in (stsc, offsets_box)):
        return None
    sample_count = len(sizes)
    fh.seek(stsc[0] + 4)
    (run_count,) = struct.unpack(">I", fh.read(4))
    if not 0 < run_count <= _MAX_SAMPLES or stsc[0] + 8 + 12 * run_count > stsc[1]:
        return None
    flat = struct.unpack(f">{3 * run_count}I", fh.read(12 * run_count))
    runs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 3)]
    fh.seek(offsets_box[0] + 4)
    (chunk_count,) = struct.unpack(">I", fh.read(4))
    entry = struct.calcsize(offset_format)
    if chunk_count > sample_count or offsets_box[0] + 8 + entry * chunk_count > offsets_box[1]:
        return None
    chunk_offsets = struct.unpack(f">{chunk_count}{offset_format}", fh.read(entry * chunk_count))
    # Byte position of each sample relative to the start of the sample table's data.
    starts = np.concatenate([[0], np.cumsum(sizes)])

    locations: list[tuple[int, int]] = []
    pending = iter(wanted)
    target = next(pending, None)
    first_sample = 0
    for run, (first_chunk, per_chunk) in enumerate(runs):
        last_chunk = runs[run + 1][0] - 1 if run + 1 < len(runs) else chunk_count
        if not 1 <= first_chunk <= last_chunk + 1 or last_chunk > chunk_count or per_chunk > sample_count:
            return None
        for chunk in range(first_chunk - 1, last_chunk):
            end_sample = first_sample + per_chunk
            if end_sample > sample_count:
                return None
            while target is not None and target < end_sample:
                offset = chunk_offsets[chunk] + int(starts[target] - starts[first_sample])
                if offset + int(sizes[target]) > file_size:
                    return None
                locations.append((offset, int(sizes[target])))
                target = next(pending, None)
            first_sample = end_sample
    return locations if first_sample == sample_count and target is None else None


def _is_idr(fh: BinaryIO, offset: int, size: int, length_size: int) -> bool:
    """True if the sample's first coded slice is an IDR slice (AUD, SEI, SPS and PPS may precede it)."""
    position, end = offset, offset + size
    for _ in range(_MAX_NALS_BEFORE_SLICE):
        if position + length_size + 1 > end:
            return False
        fh.seek(position)
        header = fh.read(length_size + 1)
        if len(header) < length_size + 1:
            return False
        nal_length = int.from_bytes(header[:length_size], "big")
        nal_type = header[length_size] & 0x1F
        if nal_type in _NAL_VCL:
            return nal_type == _NAL_IDR
        if nal_length == 0:
            return False
        position += length_size + nal_length
    return False


def _read_video_stbl(fh: BinaryIO, stbl_start: int, stbl_end: int, time_scale: int) -> Optional[H264Layout]:
    stbl = (stbl_start, stbl_end)
    stsd = find_box(fh, *stbl, b"stsd", max_boxes=_MAX_CHILD_BOXES)
    stsz = find_box(fh, *stbl, b"stsz", max_boxes=_MAX_CHILD_BOXES)
    if stsd is None or stsz is None:
        return None

    # stsd: version/flags, entry_count, then sample entries. One entry is all ffmpeg writes;
    # several would mean samples switch configuration mid-track, which a splice cannot match.
    entry_count = _read_full_box_table(fh, *stsd, ">I")
    if entry_count != (1,):
        return None
    entries = list(iter_boxes(fh, stsd[0] + 8, stsd[1], max_boxes=1))
    if not entries or entries[0][0] != b"avc1":
        return None
    _, entry_start, entry_end = entries[0]
    # Everything read below lies inside the entry, so one size bound covers every child box.
    if entry_end - entry_start > _MAX_SAMPLE_ENTRY_BYTES:
        return None
    children = list(
        iter_boxes(fh, entry_start + _VISUAL_SAMPLE_ENTRY_HEADER, entry_end, max_boxes=_MAX_ENTRY_CHILDREN + 1)
    )
    if len(children) > _MAX_ENTRY_CHILDREN:
        return None
    avcc = next(((start, end) for child, start, end in children if child == b"avcC"), None)
    if avcc is None or avcc[0] >= avcc[1]:
        return None
    fh.seek(avcc[0])
    decoder_config = fh.read(avcc[1] - avcc[0])
    fh.seek(entry_start)
    header_fields = bytearray(fh.read(_VISUAL_SAMPLE_ENTRY_HEADER))
    # The compressor name records the encoder's version: masked, so files this app wrote before
    # an ffmpeg upgrade still match what it writes after one.
    header_fields[_COMPRESSOR_NAME] = bytes(_COMPRESSOR_NAME.stop - _COMPRESSOR_NAME.start)
    parts = [bytes(header_fields)]
    for child, child_start, child_end in children:
        if child != b"btrt":
            fh.seek(child_start)
            parts.append(struct.pack(">I", child_end - child_start) + child + fh.read(child_end - child_start))
    sample_entry = b"".join(parts)

    header = _read_full_box_table(fh, *stsz, ">II")
    if header is None:
        return None
    fh.seek(0, 2)
    file_size = fh.tell()
    sizes = _sample_sizes(fh, stsz, header[1], file_size)
    if sizes is None:
        return None
    sample_count = len(sizes)

    stss = find_box(fh, *stbl, b"stss", max_boxes=_MAX_CHILD_BOXES)
    if stss is None:
        # No sync-sample table means every sample is a sync sample (an all-intra stream), each
        # of which would be examined below: only a short one is worth splicing.
        if sample_count > _MAX_SYNC_ENTRIES:
            return None
        sync_samples = tuple(range(sample_count))
    else:
        count = _read_full_box_table(fh, *stss, ">I")
        if count is None or count[0] > _MAX_SYNC_ENTRIES or stss[0] + 8 + 4 * count[0] > stss[1]:
            return None
        fh.seek(stss[0] + 8)
        one_based = struct.unpack(f">{count[0]}I", fh.read(4 * count[0]))
        if any(not 1 <= n <= sample_count for n in one_based) or list(one_based) != sorted(set(one_based)):
            return None
        sync_samples = tuple(n - 1 for n in one_based)

    if len(decoder_config) < 5:
        return None
    length_size = (decoder_config[4] & 0x03) + 1
    locations = _sample_locations(fh, stbl, sizes, sync_samples, file_size)
    if locations is None:
        return None
    idr_samples = tuple(
        k for k, location in zip(sync_samples, locations, strict=True) if _is_idr(fh, *location, length_size)
    )

    runs = _read_stts(fh, stbl)
    if runs is None or sum(count for count, _ in runs) != sample_count or any(delta <= 0 for _, delta in runs):
        return None
    deltas = {delta for _, delta in runs}
    frame_duration = deltas.pop() if len(deltas) == 1 else None
    access_points: tuple[H264AccessPoint, ...] = ()
    if frame_duration is not None:
        offsets = _read_composition_offsets(fh, stbl, sample_count)
        if offsets is not None:
            dts = np.arange(sample_count, dtype=np.int64) * frame_duration
            pts = dts + offsets
            sorted_pts = np.sort(pts)
            if sample_count == 1 or np.all(np.diff(sorted_pts) == frame_duration):
                access_points = tuple(
                    H264AccessPoint(
                        sample_index=k,
                        frame_index=int(np.searchsorted(sorted_pts, pts[k])),
                        dts=int(dts[k]),
                        pts=int(pts[k]),
                    )
                    for k in idr_samples
                )
    return H264Layout(
        sample_count=sample_count,
        sync_samples=idr_samples,
        decoder_config=decoder_config,
        sample_entry=sample_entry,
        frame_duration=frame_duration,
        time_scale=time_scale,
        access_points=access_points,
    )


def _read_composition_offsets(fh: BinaryIO, stbl: tuple[int, int], sample_count: int) -> Optional[np.ndarray]:
    """Expand the bounded ``ctts`` runs, or return zero offsets when the table is absent."""
    ctts = find_box(fh, *stbl, b"ctts", max_boxes=_MAX_CHILD_BOXES)
    if ctts is None:
        return np.zeros(sample_count, dtype=np.int64)
    if ctts[0] + 8 > ctts[1]:
        return None
    fh.seek(ctts[0])
    version = fh.read(1)[0]
    if version not in (0, 1):
        return None
    fh.seek(ctts[0] + 4)
    (run_count,) = struct.unpack(">I", fh.read(4))
    if not 0 < run_count <= _MAX_SAMPLES or ctts[0] + 8 + 8 * run_count > ctts[1]:
        return None
    fh.seek(ctts[0] + 8)
    fmt = ">II" if version == 0 else ">Ii"
    runs = [struct.unpack(fmt, fh.read(8)) for _ in range(run_count)]
    if any(count == 0 for count, _ in runs) or sum(count for count, _ in runs) != sample_count:
        return None
    return np.repeat(
        np.fromiter((offset for _, offset in runs), dtype=np.int64, count=run_count),
        np.fromiter((count for count, _ in runs), dtype=np.int64, count=run_count),
    )


def run_ffmpeg(args: list[str], description: str, is_canceled: Optional[Callable[[], bool]] = None) -> None:
    """Run the bundled ffmpeg, raising SpliceError if it fails or outlasts the timeout.

    ``is_canceled`` stops the child promptly and raises CanceledException (see
    ``run_ffmpeg_process``), so a canceled node never waits on a long copy.
    """
    try:
        proc = run_ffmpeg_process(args, SPLICE_TIMEOUT_SECONDS, is_canceled)
    except subprocess.TimeoutExpired as e:
        raise SpliceError(f"ffmpeg timed out {description}") from e
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace").strip()
        raise SpliceError(f"ffmpeg failed {description}: {message[-500:]}")


def copy_frames(
    src: Path,
    start: int,
    end: int,
    sample_count: int,
    dst: Path,
    is_canceled: Optional[Callable[[], bool]] = None,
) -> None:
    """Write ``dst`` = frames ``[start, end)`` of ``src``'s video, stream-copied.

    ``start`` and ``end`` are decode-order sample indices that must each be a keyframe (or
    0, or ``sample_count`` for the end of the track) so the cuts are clean: with closed
    GOPs, the packets between two keyframes decode to exactly the frames displayed between
    them. Audio is dropped; callers rebuild the soundtrack for the new timeline.
    """
    description = f"copying frames {start}-{end - 1} of {src.name}"
    if start == 0:
        run_ffmpeg(
            ["-i", str(src), "-map", "0:v:0", "-c", "copy", "-frames:v", str(end), "-f", "mp4", str(dst)],
            description,
            is_canceled,
        )
        return
    # The segment muxer splits at exact packet counts; the wanted range is the second segment.
    split_points = [start] + ([end] if end < sample_count else [])
    with tempfile.TemporaryDirectory(prefix="invokeai_h264_splice_") as work:
        pattern = Path(work) / "segment%d.mp4"
        run_ffmpeg(
            [
                "-i",
                str(src),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-f",
                "segment",
                "-segment_format",
                "mp4",
                "-segment_frames",
                ",".join(str(n) for n in split_points),
                "-reset_timestamps",
                "1",
                str(pattern),
            ],
            description,
            is_canceled,
        )
        segment = Path(work) / "segment1.mp4"
        if not segment.exists():
            raise SpliceError(f"ffmpeg wrote no segment {description}")
        shutil.move(str(segment), str(dst))


def probes_as(path: Path, fps: float, frames: int) -> bool:
    """Whether a player's probe of ``path`` agrees with the timeline it should hold: ``frames``
    at ``fps``. A last check on a joined file, independent of the sample tables it was built
    from — a squeezed or stretched timestamp shows up as a wrong rate or duration."""
    from invokeai.app.util.video_thumbnails import probe_video

    _, _, duration, probed_fps = probe_video(path)
    # The probe reports duration to the hundredth of a second.
    return (
        probed_fps is not None
        and math.isclose(probed_fps, fps, rel_tol=1e-3)
        and abs(duration - frames / fps) <= 0.5 / fps + 0.01
    )


def _concat_list_entry(path: Path) -> str:
    # The concat demuxer's quoting: close the quote, emit an escaped quote, reopen.
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'\n"


def concat_stream_copy(parts: list[Path], dst: Path, is_canceled: Optional[Callable[[], bool]] = None) -> None:
    """Write ``dst`` = ``parts``' video streams joined end to end, stream-copied.

    Every part must share one decoder configuration (see :func:`read_h264_layout`) and
    start on a keyframe; the concat demuxer offsets each part's timestamps by the duration
    of the parts before it.
    """
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as listing:
        listing.write("ffconcat version 1.0\n" + "".join(_concat_list_entry(p) for p in parts))
        list_path = Path(listing.name)
    try:
        run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-f",
                "mp4",
                str(dst),
            ],
            f"joining {len(parts)} stream-copied parts",
            is_canceled,
        )
    finally:
        list_path.unlink(missing_ok=True)
