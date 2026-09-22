"""Regression tests for make_mp4_writer (PR #9163 review).

The bug: the production encoders relied on imageio's default ``macro_block_size=16``,
which makes ffmpeg silently *rescale* frames to the next multiple of 16 — a 1920x1080
upload trimmed by Frame Range from Video came back as 1920x1088 while the DTO recorded
1080, so concatenating the trim with its own source failed the same-dimensions check.
``make_mp4_writer`` pins ``macro_block_size=1`` so encoded dimensions always match the
input frames exactly.
"""

from pathlib import Path

import imageio.v2 as iio2
import numpy as np
import pytest

from invokeai.app.invocations.video_frame_extract_range import _validate_even_dimensions
from invokeai.app.util.video_encoding import make_mp4_writer


def test_non_multiple_of_16_dimensions_are_preserved(tmp_path: Path) -> None:
    # 120x84: even (so yuv420p-encodable) but not a multiple of 16 — the imageio
    # default would silently rescale this to 128x96.
    width, height = 120, 84
    path = tmp_path / "out.mp4"
    writer = make_mp4_writer(path, fps=8.0)
    try:
        for i in range(4):
            writer.append_data(np.full((height, width, 3), i * 10, dtype=np.uint8))
    finally:
        writer.close()

    reader = iio2.get_reader(str(path))
    try:
        first = reader.get_data(0)
    finally:
        reader.close()
    assert first.shape[:2] == (height, width)


def test_validate_even_dimensions_accepts_even_and_rejects_odd() -> None:
    _validate_even_dimensions(1920, 1080, "ok.mp4")
    with pytest.raises(ValueError, match="even dimensions"):
        _validate_even_dimensions(833, 480, "odd-width.mp4")
    with pytest.raises(ValueError, match="even dimensions"):
        _validate_even_dimensions(832, 481, "odd-height.mp4")
