"""read_h264_layout must refuse tracks whose sample indices are not the decoder's frames.

A splice copies samples by index and re-encodes the neighbouring frames from a decode, so
the two must name the same frames. A ``-c copy`` trim keeps every sample and hides some
with an edit list, and a metadata rotation is applied to decoded frames but carried as-is
by a stream copy; either would misalign or misorient a join. Real ffmpeg throughout.
"""

import subprocess
from pathlib import Path

import imageio.v2 as iio2
import imageio_ffmpeg
import numpy as np
import pytest

from invokeai.app.util.h264_splice import read_h264_layout
from invokeai.app.util.video_encoding import make_mp4_writer, writer_decoder_config

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
    assert layout.decoder_config == writer_decoder_config(96, 64, FPS)


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
