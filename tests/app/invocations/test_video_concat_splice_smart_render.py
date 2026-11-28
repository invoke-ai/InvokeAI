"""Verify GOP-aware smart rendering: copied picture data bypasses the video encoder, bridges fall back
safely, and fractional frame rates keep their sample timing."""

import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_concat import VideoConcatInvocation
from invokeai.app.util.h264_splice import read_h264_layout
from invokeai.app.util.video_encoding import (
    make_mp4_writer,
    writer_sample_entry,
)
from invokeai.app.util.video_thumbnails import probe_video
from tests.app.invocations.video_concat_splice_helpers import (
    HEIGHT,
    WIDTH,
    _concat,
    _decode,
    _encode,
    _encode_fractional_rate,
    _encode_open_gop,
    _frames,
    _logged,
    _movie_duration,
    _sample_payload_hashes,
)

pytestmark = pytest.mark.ffmpeg


def test_smart_render_false_disables_video_packet_copy(tmp_path):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1))
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2))

    out, context = _concat(tmp_path, [a, b], "full", transition="cut", transition_frames=0, smart_render=False)

    assert len(_decode(out)) == 140
    assert "Stream-copied" not in _logged(context)


def test_smart_render_defaults_to_true():
    node = VideoConcatInvocation(videos=[VideoField(video_name="a"), VideoField(video_name="b")])
    assert node.smart_render is True


def test_smart_render_preserves_safe_regions_on_both_sides_of_transition(tmp_path, monkeypatch):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=21))
    b = _encode(tmp_path / "b.mp4", _frames(100, seed=22))
    a_layout, b_layout = read_h264_layout(a), read_h264_layout(b)
    assert a_layout is not None and b_layout is not None
    a_boundary = max(k for k in a_layout.sync_samples if k <= a_layout.sample_count - 2)
    b_boundary = min(k for k in b_layout.sync_samples if k >= 2)
    bridge_frames = a_layout.sample_count - a_boundary + b_boundary - 2
    encoded_counts: list[int] = []
    video_concat = __import__("invokeai.app.invocations.video_concat", fromlist=["encode_rgb_frames"])
    encode_frames = video_concat.encode_rgb_frames

    def count_encoded_frames(frames, *args, **kwargs):
        count = 0

        def counted_frames():
            nonlocal count
            for frame in frames:
                count += 1
                yield frame

        encoded = encode_frames(counted_frames(), *args, **kwargs)
        encoded_counts.append(count)
        return encoded

    monkeypatch.setattr(video_concat, "encode_rgb_frames", count_encoded_frames)
    out, context = _concat(tmp_path, [a, b], "smart", transition="crossfade", transition_frames=2)

    a_frames, b_frames, output_frames = _decode(a), _decode(b), _decode(out)
    expected_frames = len(a_frames) + len(b_frames) - 2
    assert len(output_frames) == expected_frames
    assert _sample_payload_hashes(a, 0, a_boundary) == _sample_payload_hashes(out, 0, a_boundary)
    b_copy_frames = len(b_frames) - b_boundary
    assert _sample_payload_hashes(b, b_boundary, len(b_frames)) == _sample_payload_hashes(
        out, expected_frames - b_copy_frames, expected_frames
    )
    assert all(np.array_equal(x, y) for x, y in zip(a_frames[:a_boundary], output_frames[:a_boundary], strict=True))
    assert all(
        np.array_equal(x, y)
        for x, y in zip(b_frames[b_boundary:], output_frames[-(len(b_frames) - b_boundary) :], strict=True)
    )
    copied_frames = a_boundary + len(b_frames) - b_boundary
    assert encoded_counts == [bridge_frames]
    assert f"copying {copied_frames}, encoding {bridge_frames} across 1 bridge" in _logged(context)


@pytest.mark.parametrize(
    "transition,transition_frames",
    [("crossfade", 8), ("fade_through_black", 4), ("fade_through_black", 5)],
    ids=["crossfade", "fade-even", "fade-odd"],
)
def test_smart_render_transition_values_match_full_encoding(tmp_path, transition, transition_frames):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=71))
    b = _encode(tmp_path / "b.mp4", _frames(100, seed=72))

    smart, smart_context = _concat(
        tmp_path, [a, b], "smart", transition=transition, transition_frames=transition_frames
    )
    full, _ = _concat(
        tmp_path,
        [a, b],
        "full",
        transition=transition,
        transition_frames=transition_frames,
        smart_render=False,
    )

    smart_log = _logged(smart_context)
    assert "Smart rendering selected" in smart_log
    assert "falling back to full video encoding" not in smart_log
    assert "Stream-copied" in smart_log
    a_frames, b_frames = _decode(a), _decode(b)
    smart_frames, full_frames = _decode(smart), _decode(full)
    tail_frames = transition_frames if transition == "crossfade" else transition_frames // 2
    head_frames = transition_frames if transition == "crossfade" else transition_frames - tail_frames
    transition_start = 100 - tail_frames
    transition_end = transition_start + transition_frames
    assert len(smart_frames) == len(full_frames)

    if transition == "crossfade":
        expected_frames = [
            np.clip(
                a_frames[100 - tail_frames + i].astype(np.float32) * (1 - (i + 1) / (transition_frames + 1))
                + b_frames[i].astype(np.float32) * ((i + 1) / (transition_frames + 1)),
                0,
                255,
            ).astype(np.uint8)
            for i in range(transition_frames)
        ]
    else:
        expected_frames = [
            np.clip(
                a_frames[100 - tail_frames + i].astype(np.float32) * (1 - (i + 1) / (tail_frames + 1)), 0, 255
            ).astype(np.uint8)
            for i in range(tail_frames)
        ] + [
            np.clip(
                b_frames[i].astype(np.float32) * ((i + 1) / (head_frames + 1)),
                0,
                255,
            ).astype(np.uint8)
            for i in range(head_frames)
        ]

    # The smart bridge has different GOP context, so allow codec noise while requiring the
    # decoded transition to match both the existing full-encode path and independent math.
    full_encode_errors = [
        float(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).mean())
        for actual, expected in zip(
            full_frames[transition_start:transition_end],
            expected_frames,
            strict=True,
        )
    ]
    smart_render_errors = [
        float(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).mean())
        for actual, expected in zip(
            smart_frames[transition_start:transition_end],
            full_frames[transition_start:transition_end],
            strict=True,
        )
    ]
    assert len(full_encode_errors) == len(smart_render_errors) == transition_frames
    assert max(full_encode_errors) < 8.0
    assert max(smart_render_errors) < 8.0


def test_smart_render_copies_intermediate_clip_gap_between_bridges(tmp_path):
    sources = [
        _encode(tmp_path / f"{name}.mp4", _frames(100, seed=seed))
        for name, seed in zip("abc", (31, 32, 33), strict=True)
    ]
    layouts = [read_h264_layout(path) for path in sources]
    assert all(layout is not None for layout in layouts)
    a0 = max(k for k in layouts[0].sync_samples if k <= 98)
    b1 = min(k for k in layouts[1].sync_samples if k >= 2)
    a1 = max(k for k in layouts[1].sync_samples if k <= 98)
    b2 = min(k for k in layouts[2].sync_samples if k >= 2)
    out, context = _concat(tmp_path, sources, "three", transition="crossfade", transition_frames=2)

    output_frames = _decode(out)
    first_bridge_frames = 100 - a0 + b1 - 2
    copied_middle_start = a0 + first_bridge_frames
    assert len(output_frames) == 3 * 100 - 2 * 2
    middle_copy_frames = a1 - b1
    third_copy_frames = 100 - b2
    assert _sample_payload_hashes(sources[0], 0, a0) == _sample_payload_hashes(out, 0, a0)
    assert _sample_payload_hashes(sources[1], b1, a1) == _sample_payload_hashes(
        out, copied_middle_start, copied_middle_start + middle_copy_frames
    )
    assert _sample_payload_hashes(sources[2], b2, 100) == _sample_payload_hashes(
        out, len(output_frames) - third_copy_frames, len(output_frames)
    )
    assert all(
        np.array_equal(source, output)
        for source, output in zip(
            _decode(sources[1])[b1:a1],
            output_frames[copied_middle_start : copied_middle_start + a1 - b1],
            strict=True,
        )
    )
    assert f"{a1 - b1} frames from clip 2" in _logged(context)
    assert all(
        np.array_equal(source, output)
        for source, output in zip(_decode(sources[2])[b2:], output_frames[-third_copy_frames:], strict=True)
    )
    assert f"{third_copy_frames} frames from clip 3" in _logged(context)


@pytest.mark.parametrize("transition_frames", [4, 5], ids=["even", "odd"])
def test_smart_render_fade_through_black_keeps_asymmetric_transition_counts(tmp_path, transition_frames):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=41))
    b = _encode(tmp_path / "b.mp4", _frames(100, seed=42))
    a_layout, b_layout = read_h264_layout(a), read_h264_layout(b)
    assert a_layout is not None and b_layout is not None
    tail = transition_frames // 2
    head = transition_frames - tail
    a_boundary = max(k for k in a_layout.sync_samples if k <= 100 - tail)
    b_boundary = min(k for k in b_layout.sync_samples if k >= head)
    encoded = 100 - a_boundary + b_boundary
    out, context = _concat(
        tmp_path,
        [a, b],
        f"fade_{transition_frames}",
        transition="fade_through_black",
        transition_frames=transition_frames,
    )

    output_frames = _decode(out)
    assert len(output_frames) == 200
    assert all(np.array_equal(x, y) for x, y in zip(_decode(a)[:a_boundary], output_frames[:a_boundary], strict=True))
    assert all(
        np.array_equal(x, y) for x, y in zip(_decode(b)[b_boundary:], output_frames[-(100 - b_boundary) :], strict=True)
    )
    assert f"encoding {encoded} across 1 bridge" in _logged(context)


@pytest.mark.parametrize(
    "transition,transition_frames",
    [("cut", 8), ("crossfade", 0), ("fade_through_black", 0)],
)
def test_smart_render_copy_only_joins_do_not_encode_frames(tmp_path, monkeypatch, transition, transition_frames):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=51))
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=52))

    def unexpected_encode(*args, **kwargs):
        raise AssertionError("copy-only smart rendering must not invoke a video encoder")

    monkeypatch.setattr("invokeai.app.invocations.video_concat.encode_rgb_frames", unexpected_encode)
    out, context = _concat(
        tmp_path,
        [a, b],
        "copy_only",
        transition=transition,
        transition_frames=transition_frames,
    )

    output_frames = _decode(out)
    assert len(output_frames) == 140
    assert all(np.array_equal(x, y) for x, y in zip(_decode(a), output_frames[:100], strict=True))
    assert all(np.array_equal(x, y) for x, y in zip(_decode(b), output_frames[100:], strict=True))
    assert "copying 140, encoding 0 across 0 bridge" in _logged(context)


def test_failed_smart_output_validation_falls_back_before_saving(tmp_path, monkeypatch):
    from invokeai.app.util.h264_splice import SpliceError

    a = _encode(tmp_path / "a.mp4", _frames(100, seed=61))
    b = _encode(tmp_path / "b.mp4", _frames(100, seed=62))
    validate = VideoConcatInvocation._validate_smart_part

    def fail_joined_output(self, path, *args, **kwargs):
        if path.name == "smart_joined.mp4":
            raise SpliceError("test rejected assembled timestamps")
        return validate(self, path, *args, **kwargs)

    monkeypatch.setattr(VideoConcatInvocation, "_validate_smart_part", fail_joined_output)
    out, context = _concat(tmp_path, [a, b], "validated", transition="crossfade", transition_frames=2)

    assert len(_decode(out)) == 198
    assert "falling back to full video encoding: test rejected assembled timestamps" in _logged(context)
    assert context.videos.save.call_count == 1


@pytest.mark.parametrize("open_side", ["left", "right"])
def test_smart_render_extends_bridge_to_source_edge_when_no_interior_idr(tmp_path, open_side):
    normal = _encode(tmp_path / "normal.mp4", _frames(100, seed=71))
    open_gop = _encode_open_gop(tmp_path / "open.mp4", _frames(100, seed=72))
    assert read_h264_layout(open_gop).sync_samples == (0,)
    a, b = (open_gop, normal) if open_side == "left" else (normal, open_gop)
    a_layout, b_layout = read_h264_layout(a), read_h264_layout(b)
    assert a_layout is not None and b_layout is not None
    a_boundary = 0 if open_side == "left" else max(k for k in a_layout.sync_samples if k <= 98)
    b_boundary = min(k for k in b_layout.sync_samples if k >= 2) if open_side == "left" else 100
    encoded = 100 - a_boundary + b_boundary - 2
    copied = a_boundary + 100 - b_boundary

    out, context = _concat(tmp_path, [a, b], "edge", transition="crossfade", transition_frames=2)

    output_frames = _decode(out)
    assert len(output_frames) == 198
    if a_boundary:
        assert all(
            np.array_equal(x, y) for x, y in zip(_decode(a)[:a_boundary], output_frames[:a_boundary], strict=True)
        )
    if b_boundary < 100:
        assert all(
            np.array_equal(x, y)
            for x, y in zip(_decode(b)[b_boundary:], output_frames[-(100 - b_boundary) :], strict=True)
        )
    assert f"copying {copied}, encoding {encoded} across 1 bridge" in _logged(context)


def test_overlapping_intermediate_bridges_fall_back_to_full_encoding(tmp_path):
    clips = [
        _encode(tmp_path / "a.mp4", _frames(100, seed=81)),
        _encode(tmp_path / "b.mp4", _frames(40, seed=82)),
        _encode(tmp_path / "c.mp4", _frames(100, seed=83)),
    ]
    out, context = _concat(tmp_path, clips, "overlap", transition="crossfade", transition_frames=16)

    assert len(_decode(out)) == 100 + 40 + 100 - 2 * 16
    assert "overlap around clip 2" in _logged(context)
    assert "falling back to full video encoding" in _logged(context)


def test_bridge_ffmpeg_failure_falls_back_to_full_encoding(tmp_path, monkeypatch):
    from invokeai.app.util.h264_splice import SpliceError

    a = _encode(tmp_path / "a.mp4", _frames(100, seed=91))
    b = _encode(tmp_path / "b.mp4", _frames(100, seed=92))

    def fail_bridge(*args, **kwargs):
        raise SpliceError("test bridge encoder failure")

    monkeypatch.setattr("invokeai.app.invocations.video_concat.encode_rgb_frames", fail_bridge)
    out, context = _concat(tmp_path, [a, b], "ffmpeg_failure", transition="crossfade", transition_frames=2)

    assert len(_decode(out)) == 198
    assert "falling back to full video encoding: test bridge encoder failure" in _logged(context)


def test_bridge_timeout_falls_back_to_full_encoding(tmp_path, monkeypatch):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=101))
    b = _encode(tmp_path / "b.mp4", _frames(100, seed=102))

    def timeout_bridge(*args, **kwargs):
        raise TimeoutError("test bridge decoder timeout")

    monkeypatch.setattr("invokeai.app.invocations.video_concat.encode_rgb_frames", timeout_bridge)
    out, context = _concat(tmp_path, [a, b], "bridge_timeout", transition="crossfade", transition_frames=2)

    assert len(_decode(out)) == 198
    assert "falling back to full video encoding: test bridge decoder timeout" in _logged(context)


def test_internal_bridge_abort_is_not_reported_as_user_cancellation(tmp_path, monkeypatch):
    from invokeai.app.util.h264_splice import SpliceError

    a = _encode(tmp_path / "a.mp4", _frames(100, seed=111))
    b = _encode(tmp_path / "b.mp4", _frames(100, seed=112))

    def fail_after_internal_abort(frames, output_path, canvas, fps, expected_frames, is_canceled, abort_event):
        del frames, output_path, canvas, fps, expected_frames
        abort_event.set()
        assert not is_canceled(), "an internal encoder abort must not appear as user cancellation"
        raise SpliceError("simulated bridge encoder timeout")

    monkeypatch.setattr("invokeai.app.invocations.video_concat.encode_rgb_frames", fail_after_internal_abort)
    out, context = _concat(tmp_path, [a, b], "internal_abort", transition="crossfade", transition_frames=2)

    assert len(_decode(out)) == 198
    assert "falling back to full video encoding: simulated bridge encoder timeout" in _logged(context)


def test_bridge_source_decode_failure_falls_back_to_full_encoding(tmp_path, monkeypatch):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=121))
    b = _encode(tmp_path / "b.mp4", _frames(100, seed=122))
    module = __import__("invokeai.app.invocations.video_concat", fromlist=["encode_rgb_frames"])
    encode = module.encode_rgb_frames

    def decode_failure(frames, output_path, canvas, fps, expected_frames, **kwargs):
        def broken_frames():
            yield next(iter(frames))
            raise ValueError("simulated bridge decoder failure")

        return encode(broken_frames(), output_path, canvas, fps, expected_frames, **kwargs)

    monkeypatch.setattr(module, "encode_rgb_frames", decode_failure)
    out, context = _concat(tmp_path, [a, b], "decoder_failure", transition="crossfade", transition_frames=2)

    assert len(_decode(out)) == 198
    assert "falling back to full video encoding" in _logged(context)
    assert "simulated bridge decoder failure" in _logged(context)


@pytest.mark.parametrize(
    "decoder_error,should_fallback",
    [
        (RuntimeError("Unable to open video decoder output stream"), True),
        (RuntimeError("unexpected decoder programming error"), False),
    ],
)
def test_decoder_startup_runtime_error_fallback_is_narrow(tmp_path, monkeypatch, decoder_error, should_fallback):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=131))
    b = _encode(tmp_path / "b.mp4", _frames(100, seed=132))
    original = __import__("invokeai.app.invocations.video_concat", fromlist=["iter_video_frames"]).iter_video_frames
    calls = 0

    def fail_once(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:

            def fail_during_iteration():
                raise decoder_error
                yield np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)

            return fail_during_iteration()
        return original(path, *args, **kwargs)

    monkeypatch.setattr("invokeai.app.invocations.video_concat.iter_video_frames", fail_once)
    if should_fallback:
        out, context = _concat(tmp_path, [a, b], "decoder_startup", transition="crossfade", transition_frames=2)
        assert len(_decode(out)) == 198
        assert "Unable to open video decoder output stream" in _logged(context)
        assert "falling back to full video encoding" in _logged(context)
    else:
        with pytest.raises(RuntimeError, match="unexpected decoder programming error"):
            _concat(tmp_path, [a, b], "decoder_programming_error", transition="crossfade", transition_frames=2)


def test_exact_24000_1001_sources_smart_render_without_retiming(tmp_path, monkeypatch):
    fps = 24000 / 1001
    sources = [_encode_fractional_rate(tmp_path / f"{name}.mp4", seed) for name, seed in (("a", 141), ("b", 142))]
    rates = [probe_video(path)[3] for path in sources]
    assert rates == pytest.approx([fps, fps], rel=1e-3)

    def unexpected_encode(*args, **kwargs):
        raise AssertionError("exact-rate compatible sources must smart-render")

    monkeypatch.setattr("invokeai.app.invocations.video_concat.encode_rgb_frames", unexpected_encode)
    build_audio = VideoConcatInvocation._build_audio_track
    audio_rates = []

    def capture_audio_rates(self, *args, **kwargs):
        audio_rates.append((kwargs["native_rates"], kwargs["output_fps"]))
        return build_audio(self, *args, **kwargs)

    monkeypatch.setattr(VideoConcatInvocation, "_build_audio_track", capture_audio_rates)
    out, context = _concat(tmp_path, sources, "fractional", transition="cut", transition_frames=0)

    assert audio_rates == [([fps, fps], fps)]
    assert "copying 80, encoding 0" in _logged(context)
    assert len(_decode(out)) == 80
    assert _movie_duration(out) == pytest.approx(80 / fps, abs=0.001)
    out_layout = read_h264_layout(out)
    assert out_layout is not None
    assert out_layout.time_scale / out_layout.frame_duration == pytest.approx(fps, rel=1e-9)
    assert context.videos.save.call_args.kwargs["duration"] == pytest.approx(80 / fps)


def test_exact_24000_1001_smart_bridge_preserves_sample_timing(tmp_path):
    fps = 24000 / 1001
    sources = [_encode_fractional_rate(tmp_path / f"{name}.mp4", seed, 100) for name, seed in (("a", 161), ("b", 162))]

    out, context = _concat(tmp_path, sources, "fractional_bridge", transition="crossfade", transition_frames=2)

    layout = read_h264_layout(out)
    assert layout is not None
    assert layout.sample_count == 198
    assert layout.time_scale / layout.frame_duration == pytest.approx(fps, rel=1e-9)
    assert "Smart rendering selected" in _logged(context)
    assert "Smart rendering failed" not in _logged(context)
    assert context.videos.save.call_args.kwargs["duration"] == pytest.approx(198 / fps)


def test_fractional_bridge_entry_mismatch_falls_back_before_bridge_work(tmp_path, monkeypatch):
    fps = 24000 / 1001
    sources = []
    for name, seed in (("a", 163), ("b", 164)):
        path = tmp_path / f"{name}.mp4"
        writer = make_mp4_writer(path, fps)
        for frame in _frames(100, seed):
            writer.append_data(frame)
        writer.close()
        sources.append(path)
    from invokeai.app.util.video_smart_concat import _bridge_sample_entry

    monkeypatch.setattr("invokeai.app.util.video_smart_concat._bridge_sample_entry_cache", {})
    source_layout = read_h264_layout(sources[0])
    assert source_layout is not None
    assert source_layout.sample_entry == writer_sample_entry(WIDTH, HEIGHT, fps)
    assert source_layout.sample_entry != _bridge_sample_entry((WIDTH, HEIGHT), fps)
    plan_fps = source_layout.time_scale / source_layout.frame_duration
    assert plan_fps != pytest.approx(fps, rel=1e-9)
    bridge_work = []

    def fail_bridge(*args, **kwargs):
        bridge_work.append((args, kwargs))
        raise AssertionError("fractional bridge work started before fallback")

    monkeypatch.setattr("invokeai.app.invocations.video_concat.copy_frames", fail_bridge)
    monkeypatch.setattr("invokeai.app.invocations.video_concat.encode_rgb_frames", fail_bridge)
    monkeypatch.setattr(
        "invokeai.app.util.video_smart_concat._bridge_sample_entry",
        lambda canvas, probe_fps, is_canceled=None: source_layout.sample_entry + b"mismatch",
    )
    out, context = _concat(tmp_path, sources, "fractional_entry_fallback", transition="crossfade", transition_frames=2)

    assert not bridge_work
    layout = read_h264_layout(out)
    assert layout is not None and layout.sample_count == 198
    assert "Smart rendering unavailable" in _logged(context)
    assert "Smart rendering selected" not in _logged(context)


def test_rounded_probe_does_not_override_exact_sample_rate_for_explicit_fps(tmp_path):
    sources = [
        _encode_fractional_rate(tmp_path / f"{name}.mp4", seed, rate=(23999, 1000))
        for name, seed in (("a", 181), ("b", 182))
    ]
    assert [probe_video(path)[3] for path in sources] == [24.0, 24.0]

    out, context = _concat(tmp_path, sources, "rounded_override", transition="cut", transition_frames=0, fps=24)

    assert "Smart rendering selected" not in _logged(context)
    assert "Stream-copied" not in _logged(context)
    assert len(_decode(out)) == 80
    assert context.videos.save.call_args.kwargs["duration"] == pytest.approx(80 / 24)


def test_exact_24000_1001_source_with_changed_fps_override_reencodes(tmp_path):
    sources = [_encode_fractional_rate(tmp_path / f"{name}.mp4", seed) for name, seed in (("a", 151), ("b", 152))]
    out, context = _concat(tmp_path, sources, "fractional_override", transition="cut", transition_frames=0, fps=24)
    assert "Stream-copied" not in _logged(context)
    assert len(_decode(out)) == 80
    assert context.videos.save.call_args.kwargs["duration"] == pytest.approx(80 / 24)


def test_multi_boundary_validation_batches_ffmpeg_launches(tmp_path, monkeypatch):
    sources = [
        _encode(tmp_path / f"{name}.mp4", _frames(40, seed)) for name, seed in zip("abcd", range(161, 165), strict=True)
    ]
    launches = []
    original = __import__("invokeai.app.invocations.video_concat", fromlist=["run_ffmpeg"]).run_ffmpeg

    def count_launch(args, *rest, **kwargs):
        launches.append(list(args))
        return original(args, *rest, **kwargs)

    monkeypatch.setattr("invokeai.app.invocations.video_concat.run_ffmpeg", count_launch)
    out, context = _concat(tmp_path, sources, "batched_boundaries", transition="cut", transition_frames=0)

    assert len(_decode(out)) == 160
    assert len(sources) - 1 == 3
    assert len(launches) == 1
    assert launches
    assert all("-xerror" in command for command in launches)
    assert "copying 160, encoding 0" in _logged(context)


def test_multi_boundary_interior_decode_error_falls_back(tmp_path, monkeypatch):
    from invokeai.app.util.h264_splice import SpliceError

    sources = [
        _encode(tmp_path / f"{name}.mp4", _frames(40, seed)) for name, seed in zip("abcd", range(171, 175), strict=True)
    ]
    calls = []

    def fail_boundary(args, description, is_canceled=None):
        del is_canceled
        args = list(args)
        calls.append((args, description))
        seeks = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "-ss"]
        assert seeks == ["2.5", "5", "7.5"], "the failed batch must include every boundary, including the interior one"
        raise SpliceError("simulated interior boundary decode failure")

    monkeypatch.setattr("invokeai.app.invocations.video_concat.run_ffmpeg", fail_boundary)
    out, context = _concat(tmp_path, sources, "interior_boundary_failure", transition="cut", transition_frames=0)

    assert len(_decode(out)) == 160
    assert len(calls) == 1
    assert "simulated interior boundary decode failure" in _logged(context)
    assert "falling back to full video encoding" in _logged(context)
    assert "Stream-copied" not in _logged(context)
