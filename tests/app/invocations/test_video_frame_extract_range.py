"""Regression tests for ExtractVideoRangeInvocation streaming (PR #9163 review).

The bug: the node collected every selected frame into a list before encoding, so the
default ``start_frame=0, end_frame=-1`` materialized the whole source in RAM — and the
upload API admits 1 GB compressed files whose decoded frames can run to tens of
gigabytes. ``_write_frame_range`` now streams each frame straight into the encoder and
stops decoding as soon as the range is written.
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import imageio.v2 as iio2
import imageio_ffmpeg
import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_frame_extract_range import ExtractVideoRangeInvocation, _write_frame_range
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.util.h264_splice import MIN_ENCODED_FRAMES, read_h264_layout
from invokeai.app.util.mp4_metadata import read_mp4_tags, write_mp4_tags
from invokeai.app.util.video_encoding import make_mp4_writer
from invokeai.app.util.video_thumbnails import probe_video


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
        patch("invokeai.app.invocations.video_frame_extract_range.decoder_frame_count", return_value=10),
        patch("invokeai.app.invocations.video_frame_extract_range.make_mp4_writer", return_value=MagicMock()),
        patch("invokeai.app.invocations.video_frame_extract_range._write_frame_range", return_value=written),
        patch("invokeai.app.invocations.video_frame_extract_range.extract_audio_pcm", return_value=None),
        patch("invokeai.app.invocations.video_frame_extract_range.read_h264_layout", return_value=None),
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


def test_invocation_echoes_the_source_video_it_was_given() -> None:
    """`video` is a freshly-encoded trim; `source_video` is what the user actually picked.

    Extend Video records the latter as the generation's provenance -- the trim is an
    intermediate that never surfaces in the gallery.
    """
    invocation = ExtractVideoRangeInvocation(video=VideoField(video_name="input.mp4"), start_frame=0, end_frame=4)
    context = MagicMock()
    context.videos.get_path.return_value = Path("input.mp4")
    context.util.is_canceled.return_value = False
    base_output = MagicMock(
        video=VideoField(video_name="output.mp4"), width=32, height=32, num_frames=5, fps=8.0, duration=0.625
    )

    with (
        patch("invokeai.app.invocations.video_frame_extract_range.probe_video", return_value=(32, 32, 0.625, 8.0)),
        patch("invokeai.app.invocations.video_frame_extract_range.decoder_frame_count", return_value=10),
        patch("invokeai.app.invocations.video_frame_extract_range.make_mp4_writer", return_value=MagicMock()),
        patch("invokeai.app.invocations.video_frame_extract_range._write_frame_range", return_value=5),
        patch("invokeai.app.invocations.video_frame_extract_range.extract_audio_pcm", return_value=None),
        patch("invokeai.app.invocations.video_frame_extract_range.read_h264_layout", return_value=None),
        patch("invokeai.app.invocations.video_frame_extract_range.VideoOutput.build", return_value=base_output),
    ):
        output = invocation.invoke(context)

    assert output.source_video.video_name == "input.mp4"
    assert output.video.video_name == "output.mp4"


def _key_frames(path: Path) -> list[int]:
    """Display indices of the key frames, as the decoder reports them."""
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path), "-vf", "showinfo", "-f", "null", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return [int(n) for n, key in re.findall(r"\bn:\s*(\d+)\b.*?\biskey:(\d)", result.stderr) if key == "1"]


def _decode(path: Path) -> list[np.ndarray]:
    reader = iio2.get_reader(str(path), format="FFMPEG")
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


@pytest.mark.parametrize("fps", [0, 16])
def test_whole_clip_at_its_own_rate_is_copied_without_re_encoding(tmp_path: Path, fps: int) -> None:
    """Extend Video passes an untrimmed source through this node on every extension; a
    re-encode there would cost the whole source one more lossy generation each time."""
    rng = np.random.default_rng(0)
    source = tmp_path / "source.mp4"
    writer = make_mp4_writer(source, 16.0)
    for _ in range(24):
        writer.append_data(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    writer.close()

    saved = tmp_path / "saved.mp4"
    context = MagicMock()
    context.videos.get_path.return_value = source
    context.util.is_canceled.return_value = False

    def _save(source_path: Path, **_: object) -> MagicMock:
        shutil.copyfile(source_path, saved)
        return MagicMock()

    context.videos.save.side_effect = _save
    invocation = ExtractVideoRangeInvocation(
        video=VideoField(video_name="source.mp4"), start_frame=0, end_frame=-1, fps=fps
    )
    base_output = MagicMock(
        video=VideoField(video_name="saved.mp4"), width=64, height=48, num_frames=24, fps=16.0, duration=1.5
    )
    with patch("invokeai.app.invocations.video_frame_extract_range.VideoOutput.build", return_value=base_output):
        output = invocation.invoke(context)

    source_frames, saved_frames = _decode(source), _decode(saved)
    assert len(saved_frames) == 24
    assert all(np.array_equal(a, b) for a, b in zip(source_frames, saved_frames, strict=True))
    assert (output.start_frame, output.end_frame) == (0, 23)
    assert context.videos.save.call_args.kwargs["duration"] == pytest.approx(24 / 16.0)


def _textured_source(path: Path, n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    texture = rng.integers(0, 256, (48 + n, 64 + n, 3), dtype=np.uint8)
    frames = [texture[i : i + 48, i : i + 64].copy() for i in range(n)]
    writer = make_mp4_writer(path, 16.0)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return frames


def _invoke_trim(tmp_path: Path, source: Path, start: int, end: int) -> tuple[Path, MagicMock]:
    saved = tmp_path / "saved.mp4"

    def _save(source_path: Path, **_: object) -> MagicMock:
        shutil.copyfile(source_path, saved)
        return MagicMock()

    context = MagicMock()
    context.videos.get_path.return_value = source
    context.util.is_canceled.return_value = False
    context.videos.save.side_effect = _save
    invocation = ExtractVideoRangeInvocation(
        video=VideoField(video_name="source.mp4"), start_frame=start, end_frame=end
    )
    base_output = MagicMock(
        video=VideoField(video_name="saved.mp4"), width=64, height=48, num_frames=1, fps=16.0, duration=1.0
    )
    with patch("invokeai.app.invocations.video_frame_extract_range.VideoOutput.build", return_value=base_output):
        invocation.invoke(context)
    return saved, context


@pytest.mark.parametrize("start,end", [(0, 80), (10, 80), (10, 99), (32, 63)])
def test_trim_copies_the_keyframe_aligned_frames_inside_the_range(tmp_path: Path, start: int, end: int) -> None:
    """Trimming the source before extending it must not re-encode the frames that survive
    the trim, apart from the partial keyframe intervals at its ends."""
    source = tmp_path / "source.mp4"
    _textured_source(source, 100)
    layout = read_h264_layout(source)
    assert layout is not None
    k1 = min(k for k in layout.sync_samples if k >= start)
    k2 = max(k for k in [*layout.sync_samples, layout.sample_count] if k <= end + 1)
    assert k1 < k2, "fixture must contain a keyframe interval inside the range"

    saved, context = _invoke_trim(tmp_path, source, start, end)

    source_frames, saved_frames = _decode(source), _decode(saved)
    assert len(saved_frames) == end - start + 1
    copied = slice(k1 - start, k2 - start)
    assert all(np.array_equal(a, b) for a, b in zip(source_frames[k1:k2], saved_frames[copied], strict=True))
    # The re-encoded ends are the right frames, just not bit-exact.
    for offset in (0, len(saved_frames) - 1):
        error = np.abs(saved_frames[offset].astype(int) - source_frames[start + offset].astype(int)).mean()
        assert error < 8
    assert f"Stream-copied frames {k1}-{k2 - 1}" in str(context.logger.info.call_args_list)
    assert context.videos.save.call_args.kwargs["duration"] == pytest.approx((end - start + 1) / 16.0)


def test_trim_within_one_keyframe_interval_is_re_encoded(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _textured_source(source, 100)
    saved, context = _invoke_trim(tmp_path, source, 5, 20)
    assert len(_decode(saved)) == 16
    assert "Stream-copied" not in str(context.logger.info.call_args_list)


def test_whole_clip_copy_does_not_carry_the_sources_generation_metadata(tmp_path: Path) -> None:
    """save() rewrites the record into the file, but falls back to a sidecar if that fails,
    and embedded tags win on read: a copy still carrying the source's would be read back
    as this output's."""
    plain = tmp_path / "plain.mp4"
    _textured_source(plain, 24)
    source = tmp_path / "source.mp4"
    write_mp4_tags(plain, source, {"invokeai_metadata": '{"from": "source"}'})
    assert "invokeai_metadata" in read_mp4_tags(source)

    saved, _ = _invoke_trim(tmp_path, source, 0, -1)
    assert "invokeai_metadata" not in read_mp4_tags(saved)
    assert len(_decode(saved)) == 24


@pytest.mark.parametrize("start,end", [(31, 66), (30, 63), (32, 64), (32, 65), (31, 64), (63, 65)])
def test_short_encoded_ends_keep_one_timeline(tmp_path: Path, start: int, end: int) -> None:
    """x264 encodes a 1-2 frame part with a shorter B-frame delay than a full one, and joining
    it to a copied run squeezed the timestamps: the trim probed at the wrong fps, and the
    extension built on it was refused for mismatched frame rates."""
    source = tmp_path / "source.mp4"
    _textured_source(source, 100)
    saved, _ = _invoke_trim(tmp_path, source, start, end)

    n = end - start + 1
    width, height, duration, fps = probe_video(saved)
    assert fps == pytest.approx(16.0, rel=1e-3)
    assert duration == pytest.approx(n / 16.0, abs=0.5 / 16.0)
    assert len(_decode(saved)) == n


def test_open_gop_source_trim_keeps_every_frame(tmp_path: Path) -> None:
    """An open-GOP I-frame is a sync sample but not a clean cut: the frames after it in
    decode order that display before it would be lost from a copied run."""
    rng = np.random.default_rng(3)
    texture = rng.integers(0, 256, (48 + 100, 64 + 100, 3), dtype=np.uint8)
    frames = [texture[i : i + 48, i : i + 64].copy() for i in range(100)]
    source = tmp_path / "source.mp4"
    writer = iio2.get_writer(
        str(source),
        format="FFMPEG",
        mode="I",
        fps=16.0,
        codec="libx264",
        macro_block_size=1,
        quality=None,
        output_params=["-crf", "18", "-g", "32", "-x264-params", "open-gop=1"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    layout = read_h264_layout(source)
    assert layout is not None
    open_gop_frames = [n for n in _key_frames(source) if n >= 60]
    assert open_gop_frames, "x264 placed no keyframe in the last two GOPs"
    # Only the IDR is a clean cut; offering an open-GOP I-frame as one is the regression this guards.
    assert layout.sync_samples == (0,), f"cut points {layout.sync_samples} include an open-GOP I-frame"

    # Around each open-GOP I-frame of the last two GOPs: a cut on it, a head of MIN_ENCODED_FRAMES
    # or more before it, a head too short to keep, and a start just past it. The window is wide
    # enough for the I-frame's decode position, which leading B-frames put up to two before its
    # display position, so every way of treating it as a cut point is exercised.
    starts = sorted({s for p in open_gop_frames for s in range(p - MIN_ENCODED_FRAMES - 2, p + 2) if s < 100})
    for start in starts:
        saved, _ = _invoke_trim(tmp_path, source, start, 99)
        assert len(_decode(saved)) == 100 - start, f"trim from {start}"


@pytest.mark.parametrize("start,end", [(31, 130), (40, 129)])
def test_cut_moves_inward_rather_than_leave_a_short_encoded_end(tmp_path: Path, start: int, end: int) -> None:
    source = tmp_path / "source.mp4"
    _textured_source(source, 200)
    saved, context = _invoke_trim(tmp_path, source, start, end)
    assert "Stream-copied frames" in str(context.logger.info.call_args_list)
    _, _, _, fps = probe_video(saved)
    assert fps == pytest.approx(16.0, rel=1e-3)
    assert len(_decode(saved)) == end - start + 1


def test_squeezed_join_falls_back_to_re_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The joined file's timeline is checked, not assumed: with the short-end rule disabled,
    a 1-frame encoded head squeezes the timestamps and the node must re-encode instead."""
    monkeypatch.setattr("invokeai.app.invocations.video_frame_extract_range.MIN_ENCODED_FRAMES", 1)
    source = tmp_path / "source.mp4"
    _textured_source(source, 100)
    saved, context = _invoke_trim(tmp_path, source, 31, 95)
    assert "Re-encoding the whole range" in str(context.logger.info.call_args_list)
    _, _, _, fps = probe_video(saved)
    assert fps == pytest.approx(16.0, rel=1e-3)
    assert len(_decode(saved)) == 65


def test_cancellation_during_a_trim_splice_saves_nothing(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _textured_source(source, 200)
    canceled = [False]
    raised: list[BaseException] = []
    from invokeai.app.invocations import video_frame_extract_range

    real = video_frame_extract_range.copy_frames

    def cancel_then_copy(*args, **kwargs):
        canceled[0] = True
        try:
            return real(*args, **kwargs)
        except BaseException as e:
            raised.append(e)
            raise

    context = MagicMock()
    context.videos.get_path.return_value = source
    context.util.is_canceled.side_effect = lambda: canceled[0]
    invocation = ExtractVideoRangeInvocation(video=VideoField(video_name="source.mp4"), start_frame=31, end_frame=130)
    with patch("invokeai.app.invocations.video_frame_extract_range.copy_frames", side_effect=cancel_then_copy):
        with pytest.raises(CanceledException):
            invocation.invoke(context)
    assert len(raised) == 1 and isinstance(raised[0], CanceledException)
    context.videos.save.assert_not_called()
