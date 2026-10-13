"""Lossless splicing of an MP4's AAC soundtrack by stream copy.

The audio counterpart of ``h264_splice``. Re-encoding an AAC track costs quality on every
pass just as re-encoding video does, so a soundtrack rebuilt for a join keeps the leading
packets of the clip it starts with and encodes only what follows.

AAC packets are not independent: each decodes to the overlap-add of its own inverse MDCT
with the second half of the previous packet's. Copying packets ``[0, m)`` therefore
reproduces the source's first ``m`` output frames exactly, and the first freshly encoded
packet after them blends with the last copied one. That blend cancels cleanly only when
both packets were coded from the same signal across their shared window, so two things
keep the seam at coding-noise level rather than a glitch: the new encode starts
``PREROLL_FRAMES`` frames before the cut (one is not enough — the encoder's first frames
after a cold start make different window decisions), and callers keep the cut that many
frames clear of anything the joined soundtrack changes. Callers also verify the seam.

Packets are joined as ADTS, which carries no timing, so two things the MP4 encodes in
its edit list are restored afterwards: the encoder priming the source's first packet
holds (an ``-itsoffset`` when muxing, from which ffmpeg writes the leading edit), and the
exact end of the soundtrack (patched into the edit list, as ffmpeg's own encodes trim
the last packet's padding).

Only a track this app's encoder could have written is spliced — the same
``AudioSpecificConfig`` and priming, constant 1024-sample packets — and callers verify
the copied head decodes bit-exact before trusting the result.
"""

import struct
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable, NamedTuple, Optional

import numpy as np

from invokeai.app.util.h264_splice import SpliceError, run_ffmpeg
from invokeai.app.util.mp4_metadata import find_box, iter_boxes
from invokeai.app.util.video_audio import AAC_ENCODE_ARGS, extract_audio_pcm

AAC_FRAME_SAMPLES = 1024
# Frames of already-known audio the new encode starts with before the cut, and the clearance
# the cut keeps from the first changed sample.
PREROLL_FRAMES = 2
_MAX_TOP_LEVEL_BOXES = 1024
_MAX_PACKETS = 1 << 24
# AudioSampleEntry fields before its child boxes (SampleEntry 8 + version-0 audio header 20).
_AUDIO_SAMPLE_ENTRY_HEADER = 28
_ES_DESCRIPTOR, _DECODER_CONFIG_DESCRIPTOR, _DECODER_SPECIFIC_INFO = 3, 4, 5
# syncExtensionType 0x2b7, extension object type 5 (SBR), sbrPresentFlag 0, as ffmpeg writes it
# after a two-byte AAC-LC config.
_NO_SBR_EXTENSION = b"\x56\xe5\x00"


class AacLayout(NamedTuple):
    """What an audio splice needs to know about an MP4's first audio track."""

    packet_count: int
    sample_rate: int
    """The media timescale, which for AAC is the sample rate."""
    priming: int
    """Leading samples the edit list skips: the encoder delay held by the first packet."""
    decoder_config: bytes
    """The ``AudioSpecificConfig`` (object type, sample rate, channels), without the
    backward-compatible "no SBR" extension ffmpeg appends to its own encodes but cannot
    write for a stream muxed from ADTS (both describe the same plain AAC-LC stream)."""


def _read_descriptor(data: bytes, position: int) -> Optional[tuple[int, int, int]]:
    """``(tag, payload_start, payload_end)`` of the MPEG-4 descriptor at ``position``."""
    if position >= len(data):
        return None
    tag, length, position = data[position], 0, position + 1
    for _ in range(4):
        if position >= len(data):
            return None
        byte = data[position]
        position += 1
        length = (length << 7) | (byte & 0x7F)
        if not byte & 0x80:
            break
    if position + length > len(data):
        return None
    return tag, position, position + length


def _audio_specific_config(esds: bytes) -> Optional[bytes]:
    es = _read_descriptor(esds, 4)  # after the FullBox version/flags
    if es is None or es[0] != _ES_DESCRIPTOR:
        return None
    flags = esds[es[1] + 2] if es[1] + 2 < es[2] else 0
    position = es[1] + 3
    if flags & 0x80:  # streamDependenceFlag
        position += 2
    if flags & 0x40:  # URL_Flag
        position += 1 + (esds[position] if position < es[2] else 0)
    if flags & 0x20:  # OCRstreamFlag
        position += 2
    config = _read_descriptor(esds, position)
    if config is None or config[0] != _DECODER_CONFIG_DESCRIPTOR:
        return None
    specific = _read_descriptor(esds, config[1] + 13)
    if specific is None or specific[0] != _DECODER_SPECIFIC_INFO:
        return None
    return esds[specific[1] : specific[2]]


def read_aac_layout(path: Path) -> Optional[AacLayout]:
    """Return the first audio track's layout if it is spliceable AAC, else None.

    Spliceable means one ``mp4a`` sample entry, 1024-sample packets (the last may be
    shorter), and an edit list that only skips priming. Only I/O errors propagate.
    """
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        moov = find_box(fh, 0, fh.tell(), b"moov", max_boxes=_MAX_TOP_LEVEL_BOXES)
        if moov is None:
            return None
        for box_type, trak_start, trak_end in iter_boxes(fh, *moov):
            if box_type != b"trak":
                continue
            mdia = find_box(fh, trak_start, trak_end, b"mdia")
            hdlr = find_box(fh, *mdia, b"hdlr") if mdia is not None else None
            if hdlr is None or hdlr[0] + 12 > hdlr[1]:
                continue
            fh.seek(hdlr[0] + 8)
            if fh.read(4) != b"soun":
                continue
            return _read_audio_track(fh, (trak_start, trak_end), mdia)
    return None


def _read_audio_track(fh: BinaryIO, trak: tuple[int, int], mdia: tuple[int, int]) -> Optional[AacLayout]:
    mdhd = find_box(fh, *mdia, b"mdhd")
    minf = find_box(fh, *mdia, b"minf")
    stbl = find_box(fh, *minf, b"stbl") if minf is not None else None
    if mdhd is None or stbl is None or mdhd[0] + 24 > mdhd[1]:
        return None
    fh.seek(mdhd[0])
    fh.seek(mdhd[0] + (20 if fh.read(1)[0] == 1 else 12))
    (sample_rate,) = struct.unpack(">I", fh.read(4))

    stsd = find_box(fh, *stbl, b"stsd")
    if stsd is None or stsd[0] + 8 > stsd[1]:
        return None
    fh.seek(stsd[0] + 4)
    if struct.unpack(">I", fh.read(4)) != (1,):
        return None
    entries = list(iter_boxes(fh, stsd[0] + 8, stsd[1], max_boxes=1))
    if not entries or entries[0][0] != b"mp4a":
        return None
    _, entry_start, entry_end = entries[0]
    fh.seek(entry_start + 8)
    if fh.read(2) != b"\x00\x00":  # QuickTime v1/v2 sound entries lay their fields out differently
        return None
    # Anything beyond the decoder config (and a per-file bitrate) — a channel layout, say —
    # would describe the copied packets differently from the ones encoded beside them. Read
    # lazily and bounded: the entry is the uploader's.
    # Each allowed box at most once, so a third box of any kind is refused and no more are read.
    esds, seen = None, set()
    for child, start, end in iter_boxes(fh, entry_start + _AUDIO_SAMPLE_ENTRY_HEADER, entry_end, max_boxes=3):
        if child not in (b"esds", b"btrt") or child in seen:
            return None
        seen.add(child)
        if child == b"esds":
            esds = (start, end)
    if esds is None:
        return None
    fh.seek(esds[0])
    decoder_config = _audio_specific_config(fh.read(esds[1] - esds[0]))
    if not decoder_config:
        return None
    if len(decoder_config) == 5 and decoder_config.endswith(_NO_SBR_EXTENSION):
        decoder_config = decoder_config[:2]

    stts = find_box(fh, *stbl, b"stts")
    if stts is None or stts[0] + 8 > stts[1]:
        return None
    fh.seek(stts[0] + 4)
    (runs,) = struct.unpack(">I", fh.read(4))
    if not 1 <= runs <= 2 or stts[0] + 8 + 8 * runs > stts[1]:
        return None
    flat = struct.unpack(f">{2 * runs}I", fh.read(8 * runs))
    counts, deltas = flat[::2], flat[1::2]
    # Every packet holds one AAC frame; ffmpeg shortens only the last one, to its real samples.
    if deltas[0] != AAC_FRAME_SAMPLES or (runs == 2 and (counts[1] != 1 or deltas[1] > AAC_FRAME_SAMPLES)):
        return None
    packet_count = sum(counts)
    if packet_count > _MAX_PACKETS:
        return None

    edts = find_box(fh, *trak, b"edts")
    elst = find_box(fh, *edts, b"elst") if edts is not None else None
    priming = 0
    if elst is not None:
        if elst[0] + 8 > elst[1]:
            return None
        fh.seek(elst[0])
        version = fh.read(1)[0]
        entry_format = ">QqhH" if version == 1 else ">IihH"
        fh.seek(elst[0] + 4)
        if struct.unpack(">I", fh.read(4)) != (1,) or elst[0] + 8 + struct.calcsize(entry_format) > elst[1]:
            return None
        _, priming, rate_integer, rate_fraction = struct.unpack(entry_format, fh.read(struct.calcsize(entry_format)))
        if (rate_integer, rate_fraction) != (1, 0) or priming < 0:
            return None
    return AacLayout(packet_count, sample_rate, priming, decoder_config)


_encoder_layouts: dict[int, AacLayout] = {}


def encoder_aac_layout(sample_rate: int) -> Optional[AacLayout]:
    """The layout ``mux_audio_into_video``'s encoder writes at ``sample_rate`` (stereo), or
    None if the probe fails. Only a successful probe is remembered."""
    if sample_rate not in _encoder_layouts:
        layout = _probe_encoder_layout(sample_rate)
        if layout is None:
            return None
        _encoder_layouts[sample_rate] = layout
    return _encoder_layouts[sample_rate]


def _probe_encoder_layout(sample_rate: int) -> Optional[AacLayout]:
    from invokeai.app.util.video_encoding import write_stereo_wav

    with tempfile.TemporaryDirectory(prefix="invokeai_aac_probe_") as work:
        wav, probe = Path(work) / "probe.wav", Path(work) / "probe.m4a"
        write_stereo_wav(wav, np.zeros((2, 4 * AAC_FRAME_SAMPLES), dtype=np.float32), sample_rate)
        try:
            run_ffmpeg(["-i", str(wav), *AAC_ENCODE_ARGS, str(probe)], "probing the AAC encoder")
        except SpliceError:
            return None
        return read_aac_layout(probe)


def _adts_frames(data: bytes) -> list[bytes]:
    frames: list[bytes] = []
    position = 0
    while position < len(data):
        header = data[position : position + 7]
        if len(header) < 7 or header[0] != 0xFF or header[1] & 0xF6 != 0xF0:
            raise SpliceError("ffmpeg wrote a malformed ADTS stream")
        length = ((header[3] & 0x03) << 11) | (header[4] << 3) | (header[5] >> 5)
        if length < 7 or position + length > len(data):
            raise SpliceError("ffmpeg wrote a truncated ADTS frame")
        frames.append(data[position : position + length])
        position += length
    return frames


def _set_track_duration(path: Path, handler: bytes, samples: int, sample_rate: int) -> None:
    """Trim the ``handler`` track's presentation to ``samples`` via its edit list, and keep
    ``tkhd`` and ``mvhd`` durations consistent with it. All fields are rewritten in place."""
    with open(path, "r+b") as fh:
        fh.seek(0, 2)
        moov = find_box(fh, 0, fh.tell(), b"moov", max_boxes=_MAX_TOP_LEVEL_BOXES)
        mvhd = find_box(fh, *moov, b"mvhd") if moov is not None else None
        if mvhd is None:
            raise SpliceError(f"{path.name} has no movie header")
        fh.seek(mvhd[0])
        mvhd_version = fh.read(1)[0]
        fh.seek(mvhd[0] + (20 if mvhd_version == 1 else 12))
        (movie_timescale,) = struct.unpack(">I", fh.read(4))
        duration = round(samples * movie_timescale / sample_rate)
        track_durations: list[int] = []
        patched = False
        for box_type, trak_start, trak_end in iter_boxes(fh, *moov):
            if box_type != b"trak":
                continue
            tkhd = find_box(fh, trak_start, trak_end, b"tkhd")
            mdia = find_box(fh, trak_start, trak_end, b"mdia")
            hdlr = find_box(fh, *mdia, b"hdlr") if mdia is not None else None
            if tkhd is None or hdlr is None:
                raise SpliceError(f"{path.name} has a malformed track")
            fh.seek(hdlr[0] + 8)
            is_target = fh.read(4) == handler
            fh.seek(tkhd[0])
            wide = fh.read(1)[0] == 1
            tkhd_duration_at, field = tkhd[0] + (28 if wide else 20), ">Q" if wide else ">I"
            if is_target and not patched:
                edts = find_box(fh, trak_start, trak_end, b"edts")
                elst = find_box(fh, *edts, b"elst") if edts is not None else None
                if elst is None:
                    raise SpliceError(f"{path.name}'s {handler.decode()} track has no edit list")
                fh.seek(elst[0])
                elst_field = ">Q" if fh.read(1)[0] == 1 else ">I"
                fh.seek(elst[0] + 8)
                fh.write(struct.pack(elst_field, duration))
                fh.seek(tkhd_duration_at)
                fh.write(struct.pack(field, duration))
                patched = True
            fh.seek(tkhd_duration_at)
            track_durations.append(struct.unpack(field, fh.read(struct.calcsize(field)))[0])
        if not patched:
            raise SpliceError(f"{path.name} has no {handler.decode()} track")
        fh.seek(mvhd[0] + (24 if mvhd_version == 1 else 16))
        fh.write(struct.pack(">Q" if mvhd_version == 1 else ">I", max(track_durations)))


def mux_spliced_aac(
    video_path: Path,
    source: Path,
    head_packets: int,
    priming: int,
    tail_pcm: np.ndarray,
    sample_rate: int,
    total_samples: int,
    out_path: Path,
    is_canceled: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Write ``out_path`` = ``video_path``'s video plus a soundtrack spliced from ``source``.

    The soundtrack is ``source``'s first ``head_packets`` AAC packets, stream-copied —
    ``head_packets * 1024 - priming`` samples after priming — followed by ``tail_pcm``
    encoded from ``PREROLL_FRAMES`` frames *before* that cut (``tail_pcm`` must start
    there), presented for ``total_samples`` samples in all.

    Returns the new encode's own decode of the ``PREROLL_FRAMES`` frames after the cut:
    the coding noise this content gets from an ordinary encode, which is what a clean
    seam over the same samples should be measured against.
    """
    from invokeai.app.util.video_encoding import write_stereo_wav

    if priming % AAC_FRAME_SAMPLES:
        raise SpliceError(f"priming of {priming} samples is not a whole AAC frame")
    with tempfile.TemporaryDirectory(prefix="invokeai_aac_splice_") as work:
        head, tail_wav, tail = Path(work) / "head.aac", Path(work) / "tail.wav", Path(work) / "tail.aac"
        run_ffmpeg(
            ["-i", str(source), "-map", "0:a:0", "-c", "copy", "-frames:a", str(head_packets), "-f", "adts", str(head)],
            f"copying the first {head_packets} audio packets of {source.name}",
            is_canceled,
        )
        write_stereo_wav(tail_wav, tail_pcm, sample_rate)
        run_ffmpeg(
            ["-i", str(tail_wav), *AAC_ENCODE_ARGS, "-f", "adts", str(tail)], "encoding the new audio", is_canceled
        )
        head_frames = _adts_frames(head.read_bytes())
        if len(head_frames) != head_packets:
            raise SpliceError(f"copied {len(head_frames)} audio packets, expected {head_packets}")
        # Drop the new encode's priming packets and its pre-roll.
        dropped = priming // AAC_FRAME_SAMPLES + PREROLL_FRAMES
        tail_frames = _adts_frames(tail.read_bytes())[dropped:]
        decoded = extract_audio_pcm(tail)
        if decoded is None:
            raise SpliceError("the new audio decoded to nothing")
        # An ADTS decode knows nothing of priming, so it starts with the priming frame.
        seam_start = priming + PREROLL_FRAMES * AAC_FRAME_SAMPLES
        seam_reference = decoded[0][:, seam_start : seam_start + PREROLL_FRAMES * AAC_FRAME_SAMPLES]
        joined = Path(work) / "joined.aac"
        joined.write_bytes(b"".join(head_frames + tail_frames))
        run_ffmpeg(
            [
                "-i",
                str(video_path),
                # The first copied packet still holds the source's priming; presenting it from
                # -priming makes ffmpeg write the leading edit that skips it.
                "-itsoffset",
                repr(-priming / sample_rate),
                "-i",
                str(joined),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                "-f",
                "mp4",
                str(out_path),
            ],
            "muxing the spliced audio",
            is_canceled,
        )
    _set_track_duration(out_path, b"soun", total_samples, sample_rate)
    return seam_reference
