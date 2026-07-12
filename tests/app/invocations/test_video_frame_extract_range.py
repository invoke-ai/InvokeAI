"""Regression tests for ExtractVideoRangeInvocation streaming (PR #9163 review).

The bug: the node collected every selected frame into a list before encoding, so the
default ``start_frame=0, end_frame=-1`` materialized the whole source in RAM — and the
upload API admits 1 GB compressed files whose decoded frames can run to tens of
gigabytes. ``_write_frame_range`` now streams each frame straight into the encoder and
stops decoding as soon as the range is written.
"""

from typing import Iterator

import numpy as np

from invokeai.app.invocations.video_frame_extract_range import _write_frame_range


class RecordingWriter:
    """Stands in for the imageio writer; records what was appended and when."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.pulled_at_first_append: int | None = None
        self._pulled_ref: list[int] | None = None

    def watch(self, pulled: list[int]) -> "RecordingWriter":
        self._pulled_ref = pulled
        return self

    def append_data(self, frame: np.ndarray) -> None:
        if self.pulled_at_first_append is None and self._pulled_ref is not None:
            self.pulled_at_first_append = self._pulled_ref[0]
        self.frames.append(frame)


def _lazy_frames(n: int, pulled: list[int]) -> Iterator[np.ndarray]:
    for i in range(n):
        pulled[0] += 1
        yield np.full((4, 4, 3), i % 255, dtype=np.uint8)


class TestWriteFrameRangeStreams:
    def test_encoding_begins_without_materializing_the_iterator(self) -> None:
        pulled = [0]
        writer = RecordingWriter().watch(pulled)
        written = _write_frame_range(_lazy_frames(1000, pulled), writer, start=0, end=99)
        assert written == 100
        # The first frame must reach the encoder after a single decode, not after the
        # whole range (let alone the whole file) has been buffered.
        assert writer.pulled_at_first_append == 1

    def test_decoding_stops_after_the_range(self) -> None:
        pulled = [0]
        writer = RecordingWriter().watch(pulled)
        written = _write_frame_range(_lazy_frames(1000, pulled), writer, start=5, end=9)
        assert written == 5
        # Frames 0..9 pass through, frame 10 triggers the break — the remaining 989
        # frames are never decoded.
        assert pulled[0] == 11

    def test_range_to_final_frame_consumes_input_exactly_once(self) -> None:
        pulled = [0]
        writer = RecordingWriter().watch(pulled)
        written = _write_frame_range(_lazy_frames(24, pulled), writer, start=0, end=23)
        assert written == 24
        assert pulled[0] == 24
        assert [int(f[0, 0, 0]) for f in writer.frames] == list(range(24))
