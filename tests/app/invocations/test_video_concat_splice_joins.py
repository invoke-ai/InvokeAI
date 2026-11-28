"""Verify the first clip of a splice is copied, re-encoded or extended as its source and the join require."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.util.aac_splice import PREROLL_FRAMES
from invokeai.app.util.h264_splice import MIN_ENCODED_FRAMES, read_h264_layout
from invokeai.app.util.video_audio import extract_audio_pcm
from invokeai.app.util.video_encoding import (
    KEYFRAME_INTERVAL_SECONDS,
    make_mp4_writer,
)
from invokeai.app.util.video_thumbnails import probe_video
from tests.app.invocations.video_concat_splice_helpers import (
    FPS,
    HEIGHT,
    RATE,
    WIDTH,
    _concat,
    _decode,
    _dominant_freq,
    _encode,
    _frames,
    _last_usable_keyframe,
    _logged,
    _movie_duration,
    _remux,
    _sample_payload_hashes,
)

pytestmark = pytest.mark.ffmpeg


@pytest.mark.parametrize(
    "transition,transition_frames,expected_len",
    [("cut", 0, 100 + 40), ("crossfade", 2, 100 + 40 - 2), ("fade_through_black", 3, 100 + 40)],
)
def test_first_clip_head_is_copied_bit_exact(tmp_path, transition, transition_frames, expected_len):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1))
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2))
    out, context = _concat(tmp_path, [a, b], "out", transition=transition, transition_frames=transition_frames)

    tail = {"crossfade": transition_frames, "fade_through_black": transition_frames // 2}.get(transition, 0)
    a_frames, out_frames = _decode(a), _decode(out)
    keyframe = len(a_frames) if transition == "cut" else _last_usable_keyframe(a, tail)
    assert len(out_frames) == expected_len
    assert all(np.array_equal(x, y) for x, y in zip(a_frames[:keyframe], out_frames[:keyframe], strict=True))
    assert f"Stream-copied the first {keyframe} frames" in _logged(context)
    saved = context.videos.save.call_args.kwargs
    assert saved["duration"] == pytest.approx(expected_len / FPS)


def test_spliced_join_keeps_the_audio_timeline(tmp_path):
    a = _encode(tmp_path / "a.mp4", _frames(64, seed=1), tone=440.0)
    b = _encode(tmp_path / "b.mp4", _frames(64, seed=2), tone=880.0)
    out, context = _concat(tmp_path, [a, b], "out", transition="cut", transition_frames=0)
    assert "Stream-copied" in _logged(context)
    assert _sample_payload_hashes(a, 0, 64) + _sample_payload_hashes(b, 0, 64) == _sample_payload_hashes(out, 0, 128)

    extracted = extract_audio_pcm(out)
    assert extracted is not None
    pcm, rate = extracted
    assert pcm.shape[1] / rate == pytest.approx(128 / FPS, abs=0.1)
    half, margin = pcm.shape[1] // 2, rate // 10
    assert _dominant_freq(pcm[:, : half - margin], rate) == pytest.approx(440.0, abs=20.0)
    assert _dominant_freq(pcm[:, half + margin :], rate) == pytest.approx(880.0, abs=20.0)


def test_first_clip_from_another_encoder_is_re_encoded(tmp_path):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1), legacy=True)
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2))
    out, context = _concat(tmp_path, [a, b], "out", transition="crossfade", transition_frames=2)
    assert "Stream-copied" not in _logged(context)
    assert len(_decode(out)) == 138


def test_retimed_join_is_re_encoded(tmp_path):
    # A stream copy cannot change the frame rate, so an fps override never splices.
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1))
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2))
    out, context = _concat(tmp_path, [a, b], "out", transition="cut", fps=24)
    assert "Stream-copied" not in _logged(context)
    assert len(_decode(out)) == 140


def test_stream_copied_trim_as_first_clip_joins(tmp_path):
    """A lossless trim (LosslessCut, `ffmpeg -c copy`) keeps every sample and hides the
    leading ones with an edit list, so its sample indices are not its frame indices."""
    source = _encode(tmp_path / "source.mp4", _frames(68, seed=1))
    a = _remux(source, tmp_path / "a.mp4", "-ss", "0.3")
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2))
    a_frames = _decode(a)
    out, context = _concat(tmp_path, [a, b], "out", transition="cut", transition_frames=0)
    assert "Stream-copied" not in _logged(context)
    assert len(_decode(out)) == len(a_frames) + 40


def test_rotated_square_first_clip_keeps_its_orientation(tmp_path):
    """Rotation metadata leaves a square clip's encoding unchanged, but decoding applies it:
    a copied head would carry the rotation onto every re-encoded (already upright) frame."""
    square = _frames(70, seed=1)
    square = [f[:, : f.shape[0]].copy() for f in square]
    source = tmp_path / "sq.mp4"
    writer = make_mp4_writer(source, FPS)
    for frame in square:
        writer.append_data(frame)
    writer.close()
    a = _remux(source, tmp_path / "a.mp4", "-display_rotation", "90")
    b_frames = [f[:, : f.shape[0]].copy() for f in _frames(20, seed=2)]
    b = tmp_path / "b.mp4"
    writer = make_mp4_writer(b, FPS)
    for frame in b_frames:
        writer.append_data(frame)
    writer.close()

    out, context = _concat(tmp_path, [a, b], "out", transition="cut", transition_frames=0)
    assert "Stream-copied" not in _logged(context)
    out_frames, a_frames = _decode(out), _decode(a)
    assert len(out_frames) == 90
    # Every output frame matches what its input decodes to (A rotated upright, B as-is).
    for got, want in [(out_frames[0], a_frames[0]), (out_frames[-1], _decode(b)[-1])]:
        assert np.abs(got.astype(int) - want.astype(int)).mean() < 8


def test_short_remainder_keeps_one_timeline(tmp_path):
    """A remainder of 1-2 frames would be encoded with a shorter B-frame delay than the
    copied head, distorting the joined timestamps."""
    b = _encode(tmp_path / "b.mp4", _frames(1, seed=2))
    interval = round(FPS * KEYFRAME_INTERVAL_SECONDS)
    keyframe = 2 * interval
    # The short final clip is joined to the preceding tail. Move that tail back only when
    # needed to reach the encoder's minimum bridge length.
    for n_a in range(keyframe + 1, keyframe + MIN_ENCODED_FRAMES + 1):
        a = _encode(tmp_path / f"a{n_a}.mp4", _frames(n_a, seed=1))
        layout = read_h264_layout(a)
        assert layout is not None and layout.sync_samples == (0, interval, keyframe), f"first clip of {n_a} frames"
        out, context = _concat(tmp_path, [a, b], f"out{n_a}", transition="cut", transition_frames=0)
        copied = keyframe if n_a - keyframe + 1 >= MIN_ENCODED_FRAMES else interval
        assert f"Stream-copied the first {copied} frames" in _logged(context), f"first clip of {n_a} frames"
        _, _, duration, fps = probe_video(out)
        assert fps == pytest.approx(FPS, rel=1e-3), f"first clip of {n_a} frames"
        assert duration == pytest.approx((n_a + 1) / FPS, abs=0.5 / FPS)
        assert len(_decode(out)) == n_a + 1
        if n_a == keyframe + 1:
            assert _sample_payload_hashes(a, 0, copied) == _sample_payload_hashes(out, 0, copied)
            assert np.abs(_decode(out)[-1].astype(int) - _decode(b)[-1].astype(int)).mean() < 15


def test_boundary_decode_failure_falls_back_without_copy_success_log(tmp_path, monkeypatch):
    from invokeai.app.util.h264_splice import SpliceError

    a = _encode(tmp_path / "a.mp4", _frames(64, seed=1))
    b = _encode(tmp_path / "b.mp4", _frames(64, seed=2))
    commands: list[list[str]] = []

    def fail_boundary_decode(args, description, is_canceled=None):
        del description, is_canceled
        commands.append(list(args))
        raise SpliceError("simulated boundary decode failure")

    monkeypatch.setattr("invokeai.app.invocations.video_concat.run_ffmpeg", fail_boundary_decode)
    out, context = _concat(tmp_path, [a, b], "boundary_decode_failure", transition="cut", transition_frames=0)

    assert len(commands) == 1
    assert "-xerror" in commands[0]
    assert len(_decode(out)) == 128
    assert "falling back to full video encoding: simulated boundary decode failure" in _logged(context)
    assert "Stream-copied" not in _logged(context)


def test_short_intermediate_clip_falls_back_without_overlapping_bridges(tmp_path):
    clips = [
        _encode(tmp_path / "a.mp4", _frames(100, seed=1)),
        _encode(tmp_path / "b.mp4", _frames(1, seed=2)),
        _encode(tmp_path / "c.mp4", _frames(100, seed=3)),
    ]
    out, context = _concat(tmp_path, clips, "short_intermediate", transition="cut", transition_frames=0)

    assert len(_decode(out)) == 201
    assert "short intermediate clip" in _logged(context)
    assert "Stream-copied" not in _logged(context)


def test_short_edge_clips_keep_long_cut_seam_as_copy(tmp_path):
    clips = [
        _encode(tmp_path / "a.mp4", _frames(1, seed=1)),
        _encode(tmp_path / "b.mp4", _frames(100, seed=2)),
        _encode(tmp_path / "c.mp4", _frames(100, seed=3)),
        _encode(tmp_path / "d.mp4", _frames(1, seed=4)),
    ]
    layouts = [read_h264_layout(path) for path in clips]
    assert all(layout is not None for layout in layouts)
    first_end = min(k for k in layouts[1].sync_samples if k >= MIN_ENCODED_FRAMES - 1)
    last_start = max(k for k in layouts[2].sync_samples if k <= layouts[2].sample_count - MIN_ENCODED_FRAMES)
    first_middle_copy = _sample_payload_hashes(clips[1], first_end, 100)
    second_middle_copy = _sample_payload_hashes(clips[2], 0, last_start)

    out, context = _concat(tmp_path, clips, "short_edges", transition="cut", transition_frames=0)

    output_frames = _decode(out)
    assert len(output_frames) == 202
    first_bridge_frames = 1 + first_end
    assert first_middle_copy == _sample_payload_hashes(
        out, first_bridge_frames, first_bridge_frames + len(first_middle_copy)
    )
    second_copy_start = first_bridge_frames + len(first_middle_copy)
    assert second_middle_copy == _sample_payload_hashes(
        out, second_copy_start, second_copy_start + len(second_middle_copy)
    )
    assert "Smart rendering selected" in _logged(context)
    assert f"{len(first_middle_copy)} frames from clip 2" in _logged(context)
    assert f"{len(second_middle_copy)} frames from clip 3" in _logged(context)
    assert "copying" in _logged(context) and "encoding" in _logged(context)
    assert "2 bridge(s)" in _logged(context)


@pytest.mark.parametrize("transition", ["crossfade", "fade_through_black"])
def test_zero_frame_transition_uses_cut_smart_plan(tmp_path, transition):
    clips = [
        _encode(tmp_path / "a.mp4", _frames(2, seed=5)),
        _encode(tmp_path / "b.mp4", _frames(100, seed=6)),
        _encode(tmp_path / "c.mp4", _frames(100, seed=7)),
    ]
    out, context = _concat(tmp_path, clips, "zero_transition", transition=transition, transition_frames=0)

    assert len(_decode(out)) == 202
    assert "Smart rendering selected" in _logged(context)
    assert "Smart rendering failed" not in _logged(context)


def test_foreign_sample_entry_falls_back_before_bridge_work(tmp_path, monkeypatch):
    clips = [
        _encode(tmp_path / "a.mp4", _frames(64, seed=1), legacy=True),
        _encode(tmp_path / "b.mp4", _frames(64, seed=2), legacy=True),
    ]
    copied = False

    def fail_copy(*args, **kwargs):
        nonlocal copied
        copied = True
        raise AssertionError("bridge work started before sample-entry fallback")

    monkeypatch.setattr("invokeai.app.invocations.video_concat.copy_frames", fail_copy)
    out, context = _concat(tmp_path, clips, "foreign_entry", transition="crossfade", transition_frames=2)

    assert not copied
    assert len(_decode(out)) == 126
    assert "Smart rendering unavailable" in _logged(context)
    assert "Stream-copied" not in _logged(context)


def test_cut_copy_plan_accepts_foreign_sample_entry(tmp_path):
    clips = [
        _encode(tmp_path / "a.mp4", _frames(64, seed=1), legacy=True),
        _encode(tmp_path / "b.mp4", _frames(64, seed=2), legacy=True),
    ]
    out, context = _concat(tmp_path, clips, "foreign_cut", transition="cut", transition_frames=0)

    assert len(_decode(out)) == 128
    assert "Stream-copied" in _logged(context)


def test_trimmed_source_extends(tmp_path):
    """The Extend graph: trim the source, then join a continuation onto the trim."""
    from invokeai.app.invocations.video_frame_extract_range import ExtractVideoRangeInvocation

    source = _encode(tmp_path / "source.mp4", _frames(100, seed=1))
    trimmed = tmp_path / "trimmed.mp4"

    def _save(source_path: Path, **_: object) -> MagicMock:
        shutil.copyfile(source_path, trimmed)
        return MagicMock()

    context = MagicMock()
    context.videos.get_path.return_value = source
    context.util.is_canceled.return_value = False
    context.videos.save.side_effect = _save
    node = ExtractVideoRangeInvocation(video=VideoField(video_name="source"), start_frame=31, end_frame=66)
    base = MagicMock(video=VideoField(video_name="trimmed"), width=WIDTH, height=HEIGHT, num_frames=36, fps=FPS)
    with patch("invokeai.app.invocations.video_frame_extract_range.VideoOutput.build", return_value=base):
        node.invoke(context)

    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2))
    out, _ = _concat(tmp_path, [trimmed, b], "out", transition="crossfade", transition_frames=2)
    assert len(_decode(out)) == 36 + 40 - 2


def test_repeated_extension_leaves_the_original_segment_and_soundtrack_untouched(tmp_path):
    """Every join used to re-encode the whole first clip, video and soundtrack, compounding
    lossy generations in the earliest footage and audio."""
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1), tone=440.0)
    original_frames = _decode(a)
    original = extract_audio_pcm(a)
    assert original is not None
    current = a
    for round_ in range(3):
        nxt = _encode(tmp_path / f"n{round_}.mp4", _frames(40, seed=10 + round_), tone=660.0 + 110 * round_)
        current, context = _concat(tmp_path, [current, nxt], f"r{round_}", transition="crossfade", transition_frames=2)
        assert "audio samples of the first clip without re-encoding" in _logged(context)
        frames = 100 + (round_ + 1) * (40 - 2)
        # The soundtrack ends with the video, not a padded AAC frame later (a frozen last frame).
        assert _movie_duration(current) == pytest.approx(frames / FPS, abs=0.002)
        # Decoding yields whole AAC frames: the timeline's samples, padded to less than one more.
        decoded = extract_audio_pcm(current)
        assert decoded is not None
        assert 0 <= decoded[0].shape[1] - round(frames * RATE / FPS) < 1024

    final_frames = _decode(current)
    assert len(final_frames) == 100 + 3 * (40 - 2)
    # Each round re-encodes only from the last keyframe of its first clip, which lies in the
    # newest continuation from the second round on: A's own head is copied every time.
    keyframe = _last_usable_keyframe(a, 2)
    assert all(np.array_equal(x, y) for x, y in zip(original_frames[:keyframe], final_frames[:keyframe], strict=True))

    final = extract_audio_pcm(current)
    assert final is not None and final[1] == original[1] == RATE
    # A's audio is copied every round up to a whole AAC frame at least PREROLL_FRAMES clear of
    # its outgoing crossfade.
    unchanged = (100 - 2) * RATE // int(FPS) - (PREROLL_FRAMES + 1) * 1024
    assert np.array_equal(final[0][:, :unchanged], original[0][:, :unchanged])
    # The new audio is where it belongs: the last round's tone closes the soundtrack.
    assert _dominant_freq(final[0][:, -RATE // 2 :], RATE) == pytest.approx(880.0, abs=20.0)
