"""read_h264_layout must refuse tracks whose sample indices are not the decoder's frames.

A splice copies samples by index and re-encodes the neighbouring frames from a decode, so
the two must name the same frames. A ``-c copy`` trim keeps every sample and hides some
with an edit list, and a metadata rotation is applied to decoded frames but carried as-is
by a stream copy; either would misalign or misorient a join. Real ffmpeg throughout.
"""

import io
import os
import struct
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import imageio.v2 as iio2
import imageio_ffmpeg
import numpy as np
import pytest

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.util.aac_splice import read_aac_layout
from invokeai.app.util.h264_splice import read_h264_layout, run_ffmpeg
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav, writer_sample_entry

FPS = 16.0


def _encode(path: Path, n: int, width: int = 96, height: int = 64) -> Path:
    rng = np.random.default_rng(0)
    writer = make_mp4_writer(path, FPS)
    for _ in range(n):
        writer.append_data(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    writer.close()
    return path


def _remux(src: Path, dst: Path, *args: str) -> Path:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *args, "-i", str(src), "-c", "copy", str(dst)],
        check=True,
        capture_output=True,
    )
    return dst


def _decoded_count(path: Path) -> int:
    reader = iio2.get_reader(str(path), format="FFMPEG")
    try:
        return sum(1 for _ in reader)
    finally:
        reader.close()


def test_own_output_lists_exactly_the_decoded_frames(tmp_path: Path) -> None:
    path = _encode(tmp_path / "a.mp4", 70)
    layout = read_h264_layout(path)
    assert layout is not None
    assert layout.sample_count == _decoded_count(path) == 70
    assert layout.sync_samples[0] == 0 and len(layout.sync_samples) > 1
    assert layout.sample_entry == writer_sample_entry(96, 64, FPS)
    assert layout.time_scale / layout.frame_duration == FPS
    assert [point.sample_index for point in layout.access_points] == list(layout.sync_samples)
    assert [point.frame_index for point in layout.access_points] == list(layout.sync_samples)
    assert all(point.pts - point.dts == 2 * layout.frame_duration for point in layout.access_points)


def test_edit_list_trim_is_refused(tmp_path: Path) -> None:
    # A stream-copied trim starting off a keyframe keeps every sample from the keyframe before
    # it and hides the leading ones with an edit list.
    trimmed = _remux(_encode(tmp_path / "a.mp4", 70), tmp_path / "trimmed.mp4", "-ss", "0.3")
    assert _decoded_count(trimmed) < 70
    assert read_h264_layout(trimmed) is None


@pytest.mark.parametrize("trim", [("-ss", "0.3"), ("-ss", "2.0"), ("-t", "2.1"), ("-ss", "0.5", "-t", "1.3")])
def test_an_accepted_layout_lists_exactly_the_decoded_frames(tmp_path: Path, trim: tuple[str, ...]) -> None:
    trimmed = _remux(_encode(tmp_path / "a.mp4", 70), tmp_path / "trimmed.mp4", *trim)
    layout = read_h264_layout(trimmed)
    assert layout is None or layout.sample_count == _decoded_count(trimmed)


def test_rotation_metadata_is_refused(tmp_path: Path) -> None:
    source = _encode(tmp_path / "sq.mp4", 40, width=64, height=64)
    rotated = _remux(source, tmp_path / "rot.mp4", "-display_rotation", "90")
    assert read_h264_layout(rotated) is None


def test_non_mp4_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.mp4"
    path.write_bytes(b"not a video")
    assert read_h264_layout(path) is None


def test_open_gop_i_frames_are_not_cut_points(tmp_path: Path) -> None:
    path = tmp_path / "og.mp4"
    rng = np.random.default_rng(0)
    writer = iio2.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=FPS,
        codec="libx264",
        macro_block_size=1,
        quality=None,
        output_params=["-g", "32", "-x264-params", "open-gop=1"],
    )
    for _ in range(100):
        writer.append_data(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8))
    writer.close()
    layout = read_h264_layout(path)
    assert layout is not None
    assert layout.sync_samples == (0,)


def _patch_box(data: bytearray, box: bytes, offset: int, value: bytes) -> None:
    at = data.find(box)
    assert at > 0, box
    data[at + 4 + offset : at + 4 + offset + len(value)] = value


@pytest.mark.skipif(
    sys.platform != "linux", reason="bounds the child's address space with RLIMIT_AS, which only Linux enforces"
)
@pytest.mark.parametrize("uniform_size", [0, 1], ids=["size-table", "uniform-size"])
def test_oversized_sample_count_is_refused_without_allocating(tmp_path: Path, uniform_size: int) -> None:
    """Uploads that are already H.264 MP4s are stored as-is, so the sample tables are the
    uploader's. A count claiming ~4 billion samples, in a track with no sync-sample table
    (so every sample would be a keyframe), must be refused before anything is sized by it."""
    data = bytearray(_encode(tmp_path / "a.mp4", 40).read_bytes())
    # stsz payload: version/flags, uniform sample size, sample count.
    _patch_box(data, b"stsz", 4, struct.pack(">II", uniform_size, 0xFFFFFFF0))
    # Hide the sync-sample table: an all-sync track is the case that sized a tuple by the count.
    at = data.find(b"stss")
    assert at > 0
    data[at : at + 4] = b"free"
    path = tmp_path / "hostile.mp4"
    path.write_bytes(bytes(data))

    script = (
        "import resource, sys\n"
        # Far below the ~32 GB a count-sized tuple would take, with room for numpy's import.
        "resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))\n"
        "from pathlib import Path\n"
        "from invokeai.app.util.h264_splice import read_h264_layout\n"
        "print(read_h264_layout(Path(sys.argv[1])))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        # One BLAS thread: each one reserves address space the limit would otherwise spend.
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "None"


@pytest.mark.parametrize(
    "tag",
    [
        ("-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"),
        ("-aspect", "16:9"),
    ],
    ids=["colr", "pasp"],
)
def test_presentation_tags_set_the_track_apart(tmp_path: Path, tag: tuple[str, ...]) -> None:
    """A remux can add colour or pixel-aspect boxes while ``avcC`` stays byte-identical; a
    joined track would present every frame with the copied part's tags."""
    source = _encode(tmp_path / "a.mp4", 40)
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source), "-c", "copy", *tag]
        + [str(tmp_path / "tagged.mp4")],
        check=True,
    )
    plain, tagged = read_h264_layout(source), read_h264_layout(tmp_path / "tagged.mp4")
    assert plain is not None and tagged is not None
    assert tagged.decoder_config == plain.decoder_config
    assert tagged.sample_entry != writer_sample_entry(96, 64, FPS) == plain.sample_entry


def test_cancellation_kills_a_running_ffmpeg(tmp_path: Path) -> None:
    """A canceled node must not wait out a long copy: the child is killed and reaped."""
    started = time.monotonic()
    calls = [0]

    def is_canceled() -> bool:
        calls[0] += 1
        return time.monotonic() - started > 0.5

    children: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def spawn(*args, **kwargs) -> subprocess.Popen:
        children.append(real_popen(*args, **kwargs))
        return children[-1]

    # An endless synthetic source that only stops when killed.
    with patch("invokeai.app.util.h264_splice.subprocess.Popen", side_effect=spawn):
        with pytest.raises(CanceledException):
            run_ffmpeg(["-f", "lavfi", "-i", "anullsrc", "-f", "null", "-"], "running forever", is_canceled)
    assert time.monotonic() - started < 10
    # The last child is the copy (locating the binary may spawn a version probe first).
    assert children and children[-1].returncode is not None, "the child was left running"


def _grow_sample_entry(
    data: bytes,
    handler: bytes,
    blob: bytes,
    first: bool = False,
    inside: bytes | None = None,
    in_stbl: bool = False,
) -> bytes:
    """Insert ``blob`` inside the ``handler`` track's sample entry — after its other child boxes,
    before them if ``first``, or at the end of its ``inside`` child box — or, with ``in_stbl``,
    at the start of the track's sample table, growing every ancestor box. The file must keep
    ``moov`` after ``mdat`` so no chunk offset moves."""

    def boxes(start: int, end: int):
        while start + 8 <= end:
            size, kind = struct.unpack(">I4s", data[start : start + 8])
            assert size >= 8, "the fixture must be a well-formed MP4"
            yield kind, start, size
            start += size

    _, moov, moov_size = next(b for b in boxes(0, len(data)) if b[0] == b"moov")
    for kind, trak, trak_size in boxes(moov + 8, moov + moov_size):
        if kind != b"trak":
            continue
        chain = [(moov, moov_size), (trak, trak_size)]
        start, end = trak + 8, trak + trak_size
        for name in (b"mdia", b"minf", b"stbl", b"stsd"):
            _, at, size = next(b for b in boxes(start, end) if b[0] == name)
            chain.append((at, size))
            start, end = at + 8, at + size
        hdlr = data.find(b"hdlr", trak, trak + trak_size)
        if data[hdlr + 12 : hdlr + 16] != handler:
            continue
        grown = bytearray(data)
        if in_stbl:
            # Ahead of stsd and every other table.
            chain.pop()
            stbl_at = chain[-1][0]
            grown[stbl_at + 8 : stbl_at + 8] = blob
            for box_at, size in chain:
                struct.pack_into(">I", grown, box_at, size + len(blob))
            return bytes(grown)
        _, entry, entry_size = next(boxes(start + 8, end))
        chain.append((entry, entry_size))
        # Child boxes follow the entry's box header and fixed fields: 78 bytes for avc1, 28 for mp4a.
        children_at = entry + 8 + (78 if handler == b"vide" else 28)
        at = children_at if first else entry + entry_size
        if inside is not None:
            _, child, child_size = next(b for b in boxes(children_at, entry + entry_size) if b[0] == inside)
            chain.append((child, child_size))
            at = child + child_size
        grown[at:at] = blob
        for at, size in chain:
            struct.pack_into(">I", grown, at, size + len(blob))
        return bytes(grown)
    raise AssertionError(f"no {handler!r} track")


def _audio_clip(tmp_path: Path) -> Path:
    """A clip of this app's own encoding, with a soundtrack, and moov after mdat."""
    wav = tmp_path / "a.wav"
    write_stereo_wav(wav, np.zeros((2, 48000), dtype=np.float32), 48000)
    source = tmp_path / "a.mp4"
    writer = make_mp4_writer(source, FPS, audio_path=wav)
    for _ in range(16):
        writer.append_data(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.close()
    return source


@pytest.mark.parametrize("handler", [b"vide", b"soun"])
def test_sample_entry_stuffed_with_boxes_is_refused_cheaply(tmp_path: Path, handler: bytes) -> None:
    """The sample entry is the uploader's: a flood of child boxes must be refused after a
    bounded look, not collected (memory) or concatenated (quadratic time)."""
    source = _audio_clip(tmp_path)
    hostile = tmp_path / "hostile.mp4"
    hostile.write_bytes(_grow_sample_entry(source.read_bytes(), handler, struct.pack(">I4s", 8, b"free") * 200_000))

    read = read_h264_layout if handler == b"vide" else read_aac_layout
    tracemalloc.start()
    try:
        assert read(hostile) is None
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # 200k boxes are 1.6 MB of file; reading a bounded handful of them allocates almost nothing.
    assert peak < 1 << 20


@pytest.mark.parametrize(
    "handler,inside", [(b"vide", None), (b"vide", b"avcC"), (b"soun", b"esds")], ids=["free", "avcC", "esds"]
)
def test_oversized_sample_entry_is_refused_unread(tmp_path: Path, handler: bytes, inside: bytes | None) -> None:
    """One huge child box — a free box, or the decoder config padded out — must be refused by
    size, not read into memory."""
    source = _audio_clip(tmp_path)
    hostile = tmp_path / "hostile.mp4"
    padding = 16 << 20
    blob = bytes(padding) if inside else struct.pack(">I4s", 8 + padding, b"free") + bytes(padding)
    hostile.write_bytes(_grow_sample_entry(source.read_bytes(), handler, blob, inside=inside))
    read = read_h264_layout if handler == b"vide" else read_aac_layout
    tracemalloc.start()
    try:
        assert read(hostile) is None
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20


def test_box_flood_before_avcc_is_not_scanned(tmp_path: Path) -> None:
    """``avcC`` is found among a bounded number of child boxes, so a flood placed before it is
    refused after a fixed look rather than walked in full."""
    source = _audio_clip(tmp_path)
    hostile = tmp_path / "hostile.mp4"
    hostile.write_bytes(
        _grow_sample_entry(source.read_bytes(), b"vide", struct.pack(">I4s", 8, b"free") * 2_000_000, first=True)
    )
    seeks = [0]

    class CountingFile(io.BufferedReader):
        def seek(self, *args):
            seeks[0] += 1
            return super().seek(*args)

    def counting_open(path, mode="r", *args, **kwargs):
        return CountingFile(io.FileIO(path, "r"))

    # The flood really sits inside the sample entry, ahead of its avcC.
    data = hostile.read_bytes()
    entry = data.find(b"avc1")
    assert entry < data.find(b"\x00\x00\x00\x08free", entry) < data.find(b"avcC", entry)
    with patch("invokeai.app.util.h264_splice.open", counting_open, create=True):
        assert read_h264_layout(hostile) is None
    # Walking the flood would take a seek per box: two million of them.
    assert seeks[0] < 10_000


@pytest.mark.parametrize("handler", [b"vide", b"soun"])
def test_box_flood_in_a_container_is_not_scanned(tmp_path: Path, handler: bytes) -> None:
    """Every container walk (moov, trak, mdia, minf, stbl) stops after a fixed number of boxes:
    a flood ahead of the sample tables would otherwise be walked once per table looked up."""
    source = _audio_clip(tmp_path)
    hostile = tmp_path / "hostile.mp4"
    hostile.write_bytes(
        _grow_sample_entry(source.read_bytes(), handler, struct.pack(">I4s", 8, b"free") * 200_000, in_stbl=True)
    )
    module = "h264_splice" if handler == b"vide" else "aac_splice"
    read = read_h264_layout if handler == b"vide" else read_aac_layout
    seeks = [0]

    class CountingFile(io.BufferedReader):
        def seek(self, *args):
            seeks[0] += 1
            return super().seek(*args)

    with patch(f"invokeai.app.util.{module}.open", lambda path, *a, **k: CountingFile(io.FileIO(path)), create=True):
        assert read(hostile) is None
    # Walking the flood would take a seek per box per lookup: hundreds of thousands at least.
    assert seeks[0] < 10_000
