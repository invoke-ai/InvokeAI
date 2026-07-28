"""Regression tests for VideoFrameExtractInvocation negative-index resolution.

Covers JPPhoto's code-review finding (PR #9163): the old code computed
``n_frames = round(duration * fps)`` to resolve ``frame_index=-1``. For uploads
with inexact metadata that can overshoot the decoded frame count, requesting
the last frame would fail. The fix queries ``iio.improps(...).shape[0]`` for
the exact decoder count.

We exercise the ``decoder_frame_count`` helper (now subprocess-backed) with a real synthetic
MP4 so the iio integration is actually validated.
"""

import imageio.v3 as iio
import numpy as np
import pytest

from invokeai.app.util.video_thumbnails import decoder_frame_count


def _write_mp4(tmp_path, n_frames: int):
    """Encode a tiny synthetic MP4 with exactly ``n_frames`` frames at 8 fps."""
    path = tmp_path / "synth.mp4"
    frames = [np.full((32, 32, 3), 64 + i * 8, dtype=np.uint8) for i in range(n_frames)]
    iio.imwrite(path, frames, plugin="FFMPEG", codec="libx264", fps=8.0, macro_block_size=1)
    return path


class TestDecoderFrameCountExact:
    """decoder_frame_count returns the actual decoded count from the container."""

    @pytest.mark.parametrize("n", [1, 5, 16, 33])
    def test_matches_encoded_frame_count(self, tmp_path, n: int) -> None:
        path = _write_mp4(tmp_path, n)
        assert decoder_frame_count(path) == n


class TestDecoderFrameCountGracefulFallback:
    """decoder_frame_count returns None on unreadable inputs so the caller can fall back."""

    def test_missing_path_returns_none(self, tmp_path) -> None:
        bogus = tmp_path / "does_not_exist.mp4"
        # Either iio raises (caught) or returns props without shape — both must yield None.
        assert decoder_frame_count(bogus) is None

    def test_non_video_file_returns_none(self, tmp_path) -> None:
        not_a_video = tmp_path / "junk.mp4"
        not_a_video.write_bytes(b"not actually an mp4")
        assert decoder_frame_count(not_a_video) is None
