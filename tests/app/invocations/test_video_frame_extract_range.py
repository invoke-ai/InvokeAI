"""Regression tests for ExtractVideoRangeInvocation streaming (PR #9163 review).

The bug: the node collected every selected frame into a list before encoding, so the
default ``start_frame=0, end_frame=-1`` materialized the whole source in RAM — and the
upload API admits 1 GB compressed files whose decoded frames can run to tens of
gigabytes. ``_write_frame_range`` now streams each frame straight into the encoder and
stops decoding as soon as the range is written.
"""

from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_frame_extract_range import ExtractVideoRangeInvocation, _write_frame_range
from invokeai.app.services.session_processor.session_processor_common import CanceledException


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


def _fail_after(n: int) -> Iterator[np.ndarray]:
    yield from _lazy_frames(n, [0])
    raise RuntimeError("decoder advanced past requested range")


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
        assert pulled[0] == 10

    def test_does_not_pull_the_frame_after_end(self) -> None:
        writer = RecordingWriter()
        assert _write_frame_range(_fail_after(10), writer, start=5, end=9) == 5

    def test_range_to_final_frame_consumes_input_exactly_once(self) -> None:
        pulled = [0]
        writer = RecordingWriter().watch(pulled)
        written = _write_frame_range(_lazy_frames(24, pulled), writer, start=0, end=23)
        assert written == 24
        assert pulled[0] == 24
        assert [int(f[0, 0, 0]) for f in writer.frames] == list(range(24))

    def test_cancellation_stops_before_writing_more_frames(self) -> None:
        pulled = [0]
        writer = RecordingWriter()
        with pytest.raises(CanceledException):
            _write_frame_range(_lazy_frames(24, pulled), writer, start=0, end=23, is_canceled=lambda: pulled[0] >= 2)
        assert len(writer.frames) == 1
        assert pulled[0] == 2


@pytest.mark.parametrize("written,should_raise", [(5, False), (3, True)])
def test_invocation_only_saves_complete_requested_range(written: int, should_raise: bool) -> None:
    invocation = ExtractVideoRangeInvocation(video=VideoField(video_name="input.mp4"), start_frame=0, end_frame=4)
    context = MagicMock()
    context.videos.get_path.return_value = Path("input.mp4")
    context.util.is_canceled.return_value = False
    base_output = MagicMock(
        video=VideoField(video_name="output.mp4"), width=32, height=32, num_frames=5, fps=8.0, duration=0.625
    )

    with (
        patch("invokeai.app.invocations.video_frame_extract_range.probe_video", return_value=(32, 32, 0.625, 8.0)),
        patch("invokeai.app.invocations.video_frame_extract_range.decoder_frame_count", return_value=5),
        patch("invokeai.app.invocations.video_frame_extract_range.make_mp4_writer", return_value=MagicMock()),
        patch("invokeai.app.invocations.video_frame_extract_range._write_frame_range", return_value=written),
        patch("invokeai.app.invocations.video_frame_extract_range.extract_audio_pcm", return_value=None),
        patch("invokeai.app.invocations.video_frame_extract_range.VideoOutput.build", return_value=base_output),
    ):
        if should_raise:
            with pytest.raises(ValueError, match="Decoded only 3 of 5 requested frames"):
                invocation.invoke(context)
            context.videos.save.assert_not_called()
        else:
            output = invocation.invoke(context)
            assert output.end_frame == 4
            context.videos.save.assert_called_once()


class TestWriteFrameRangeResizes:
    """`resize_to` conforms the trimmed clip to a target canvas (see the concat pairing below)."""

    def test_frames_are_resampled_to_the_requested_canvas(self) -> None:
        writer = RecordingWriter()
        written = _write_frame_range(_lazy_frames(4, [0]), writer, start=0, end=3, resize_to=(8, 6))
        assert written == 4
        assert [f.shape for f in writer.frames] == [(6, 8, 3)] * 4

    def test_matching_frames_pass_through_untouched(self) -> None:
        writer = RecordingWriter()
        _write_frame_range(_lazy_frames(2, [0]), writer, start=0, end=1, resize_to=(4, 4))
        assert [f.shape for f in writer.frames] == [(4, 4, 3)] * 2


def _resize_invocation(**kwargs: object) -> tuple[ExtractVideoRangeInvocation, MagicMock]:
    invocation = ExtractVideoRangeInvocation(
        video=VideoField(video_name="input.mp4"), start_frame=0, end_frame=4, **kwargs
    )
    context = MagicMock()
    context.videos.get_path.return_value = Path("input.mp4")
    context.util.is_canceled.return_value = False
    return invocation, context


def test_invocation_saves_the_requested_canvas() -> None:
    """A 1920x1080 source trimmed onto a 1344x768 canvas must be recorded as 1344x768.

    This is what lets Extend Video concatenate the trimmed source with the generated
    clip -- `video_concat` rejects mismatched dimensions outright.
    """
    invocation, context = _resize_invocation(width=1344, height=768)
    base_output = MagicMock(
        video=VideoField(video_name="output.mp4"), width=1344, height=768, num_frames=5, fps=8.0, duration=0.625
    )

    with (
        patch("invokeai.app.invocations.video_frame_extract_range.probe_video", return_value=(1920, 1080, 0.625, 8.0)),
        patch("invokeai.app.invocations.video_frame_extract_range.decoder_frame_count", return_value=5),
        patch("invokeai.app.invocations.video_frame_extract_range.make_mp4_writer", return_value=MagicMock()),
        patch("invokeai.app.invocations.video_frame_extract_range._write_frame_range", return_value=5) as write,
        patch("invokeai.app.invocations.video_frame_extract_range.extract_audio_pcm", return_value=None),
        patch("invokeai.app.invocations.video_frame_extract_range.VideoOutput.build", return_value=base_output),
    ):
        output = invocation.invoke(context)

    assert write.call_args.kwargs["resize_to"] == (1344, 768)
    saved = context.videos.save.call_args.kwargs
    assert (saved["width"], saved["height"]) == (1344, 768)
    assert output.video.video_name == "output.mp4"


def test_invocation_keeps_source_dimensions_when_canvas_is_unset() -> None:
    invocation, context = _resize_invocation()
    base_output = MagicMock(
        video=VideoField(video_name="output.mp4"), width=32, height=32, num_frames=5, fps=8.0, duration=0.625
    )

    with (
        patch("invokeai.app.invocations.video_frame_extract_range.probe_video", return_value=(32, 32, 0.625, 8.0)),
        patch("invokeai.app.invocations.video_frame_extract_range.decoder_frame_count", return_value=5),
        patch("invokeai.app.invocations.video_frame_extract_range.make_mp4_writer", return_value=MagicMock()),
        patch("invokeai.app.invocations.video_frame_extract_range._write_frame_range", return_value=5) as write,
        patch("invokeai.app.invocations.video_frame_extract_range.extract_audio_pcm", return_value=None),
        patch("invokeai.app.invocations.video_frame_extract_range.VideoOutput.build", return_value=base_output),
    ):
        invocation.invoke(context)

    assert write.call_args.kwargs["resize_to"] is None
    saved = context.videos.save.call_args.kwargs
    assert (saved["width"], saved["height"]) == (32, 32)


def test_invocation_rejects_an_odd_requested_canvas() -> None:
    invocation, context = _resize_invocation(width=1345, height=768)

    with (
        patch("invokeai.app.invocations.video_frame_extract_range.probe_video", return_value=(1920, 1080, 0.625, 8.0)),
        patch("invokeai.app.invocations.video_frame_extract_range.decoder_frame_count", return_value=5),
        pytest.raises(ValueError, match="Requested output size is 1345x768"),
    ):
        invocation.invoke(context)
    context.videos.save.assert_not_called()
