"""Verify GOP-aware smart concatenation and the inherited H.264/AAC splice contracts.

The smart-render cases assert that copied picture data bypasses the video encoder, while
the bridge cases use the existing full-timeline blending and audio behavior. Real FFmpeg
(the imageio-ffmpeg bundled binary) encodes and decodes throughout.
"""

import hashlib
import importlib
import shutil
import struct
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import imageio.v2 as iio2
import imageio_ffmpeg
import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_concat import VideoConcatInvocation
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.util.aac_splice import PREROLL_FRAMES
from invokeai.app.util.h264_splice import MIN_ENCODED_FRAMES, read_h264_layout
from invokeai.app.util.mp4_metadata import find_box, write_mp4_tags
from invokeai.app.util.video_audio import extract_audio_pcm
from invokeai.app.util.video_encoding import (
    KEYFRAME_INTERVAL_SECONDS,
    make_mp4_writer,
    write_stereo_wav,
    writer_sample_entry,
)
from invokeai.app.util.video_thumbnails import probe_video

FPS = 16.0
WIDTH, HEIGHT = 128, 96
RATE = 32000


def _frames(n: int, seed: int) -> list[np.ndarray]:
    # Textured, moving content: flat frames encode near-losslessly and would hide a re-encode.
    rng = np.random.default_rng(seed)
    texture = rng.integers(0, 256, (HEIGHT + n, WIDTH + n, 3), dtype=np.uint8)
    return [texture[i : i + HEIGHT, i : i + WIDTH].copy() for i in range(n)]


def _tone(freq: float, n_frames: int) -> np.ndarray:
    t = np.arange(round(n_frames / FPS * RATE)) / RATE
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono])


def _encode(path: Path, frames: list[np.ndarray], tone: float | None = None, legacy: bool = False) -> Path:
    audio_path = None
    if tone is not None:
        audio_path = path.with_suffix(".wav")
        write_stereo_wav(audio_path, _tone(tone, len(frames)), RATE)
    if legacy:
        # A file from before the current encoder settings (CRF 25), with keyframes as frequent as
        # ours so a splice point exists and only the decoder configuration rules the copy out.
        writer = iio2.get_writer(
            str(path),
            format="FFMPEG",
            mode="I",
            fps=FPS,
            codec="libx264",
            macro_block_size=1,
            quality=None,
            output_params=["-crf", "25", "-g", "32"],
        )
    else:
        writer = make_mp4_writer(path, FPS, audio_path=audio_path)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return path


def _encode_open_gop(path: Path, frames: list[np.ndarray]) -> Path:
    writer = iio2.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=FPS,
        codec="libx264",
        macro_block_size=1,
        quality=None,
        output_params=["-crf", "18", "-g", "32", "-x264-params", "open-gop=1"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return path


def _decode(path: Path) -> list[np.ndarray]:
    reader = iio2.get_reader(str(path), format="FFMPEG")
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


def _movie_duration(path: Path) -> float:
    """The container's own duration (``mvhd``), at its full precision."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        moov = find_box(fh, 0, fh.tell(), b"moov")
        assert moov is not None
        mvhd = find_box(fh, *moov, b"mvhd")
        assert mvhd is not None
        fh.seek(mvhd[0])
        wide = fh.read(1)[0] == 1
        fh.seek(mvhd[0] + (20 if wide else 12))
        timescale, duration = struct.unpack(">IQ" if wide else ">II", fh.read(12 if wide else 8))
    return duration / timescale


def _dominant_freq(pcm: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(pcm[0]))
    spectrum[0] = 0.0
    return float(np.fft.rfftfreq(pcm.shape[1], 1.0 / rate)[int(np.argmax(spectrum))])


def _sample_payload_hashes(path: Path, start: int, end: int) -> list[str]:
    """Hash coded picture NALs, ignoring legal parameter-set and framing changes."""
    from invokeai.app.util import h264_splice
    from invokeai.app.util.mp4_metadata import find_box, iter_boxes

    with path.open("rb") as stream:
        stream.seek(0, 2)
        moov = find_box(stream, 0, stream.tell(), b"moov")
        assert moov is not None
        stbl = None
        for box_type, trak_start, trak_end in iter_boxes(stream, *moov):
            if box_type != b"trak":
                continue
            mdia = find_box(stream, trak_start, trak_end, b"mdia")
            if mdia is None:
                continue
            hdlr = find_box(stream, *mdia, b"hdlr")
            if hdlr is None:
                continue
            stream.seek(hdlr[0] + 8)
            if stream.read(4) != b"vide":
                continue
            minf = find_box(stream, *mdia, b"minf")
            stbl = find_box(stream, *minf, b"stbl") if minf is not None else None
            break
        assert stbl is not None
        stsz = find_box(stream, *stbl, b"stsz")
        assert stsz is not None
        stream.seek(stsz[0] + 8)
        sample_count = int.from_bytes(stream.read(4), "big")
        stream.seek(0, 2)
        file_size = stream.tell()
        sizes = h264_splice._sample_sizes(stream, stsz, sample_count, file_size)
        assert sizes is not None
        locations = h264_splice._sample_locations(stream, stbl, sizes, tuple(range(start, end)), file_size)
        assert locations is not None
        layout = read_h264_layout(path)
        assert layout is not None
        length_size = (layout.decoder_config[4] & 0x03) + 1
        hashes = []
        for offset, size in locations:
            stream.seek(offset)
            sample = stream.read(size)
            digest = hashlib.sha256()
            position = 0
            has_picture = False
            while position + length_size < len(sample):
                nal_size = int.from_bytes(sample[position : position + length_size], "big")
                position += length_size
                nal = sample[position : position + nal_size]
                assert len(nal) == nal_size
                if nal and (nal[0] & 0x1F) in range(1, 6):
                    digest.update(nal)
                    has_picture = True
                position += nal_size
            assert position == len(sample) and has_picture
            hashes.append(digest.hexdigest())
        return hashes


def _concat(tmp_path: Path, inputs: list[Path], out_name: str, **fields: object) -> tuple[Path, MagicMock]:
    """Run the node on real files; returns the saved MP4 and the mocked context."""
    by_name = {p.stem: p for p in inputs}
    saved = tmp_path / f"{out_name}.mp4"

    def _save(source_path: Path, **_: object) -> MagicMock:
        # The stream-copy remux save() runs to embed the record, so chained joins read what
        # production stores.
        write_mp4_tags(source_path, saved, {"invokeai_metadata": "{}"})
        return MagicMock()

    context = MagicMock()
    context.videos.get_path.side_effect = lambda name: by_name[name]
    context.util.is_canceled.return_value = False
    context.videos.save.side_effect = _save
    node = VideoConcatInvocation(videos=[VideoField(video_name=p.stem) for p in inputs], **fields)  # type: ignore[arg-type]
    with patch("invokeai.app.invocations.video_concat.VideoOutput.build", return_value=MagicMock()):
        node.invoke(context)
    return saved, context


def _logged(context: MagicMock) -> str:
    return "\n".join(str(call.args[0]) for call in context.logger.info.call_args_list)


def _last_usable_keyframe(path: Path, tail: int) -> int:
    layout = read_h264_layout(path)
    assert layout is not None
    return max(k for k in layout.sync_samples if 0 < k <= layout.sample_count - max(tail, 1))


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


def _remux(src: Path, dst: Path, *args: str) -> Path:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *args, "-i", str(src), "-c", "copy", str(dst)],
        check=True,
        capture_output=True,
    )
    return dst


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


def test_soundtrack_from_another_encoder_is_re_encoded(tmp_path):
    a_video = _encode(tmp_path / "av.mp4", _frames(64, seed=1))
    a_wav = tmp_path / "a.wav"
    write_stereo_wav(a_wav, _tone(440.0, 64), RATE)
    a = tmp_path / "a.mp4"
    # Mono AAC: a different AudioSpecificConfig from the stereo tracks this app writes.
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(a_video), "-i", str(a_wav)]
        + ["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-ac", "1", str(a)],
        check=True,
    )
    b = _encode(tmp_path / "b.mp4", _frames(64, seed=2), tone=880.0)
    out, context = _concat(tmp_path, [a, b], "out", transition="cut", transition_frames=0)
    assert "audio samples of the first clip" not in _logged(context)
    extracted = extract_audio_pcm(out)
    assert extracted is not None
    assert extracted[0].shape[1] / RATE == pytest.approx(128 / FPS, abs=0.1)


@pytest.mark.parametrize("b_tone", [660.0, None], ids=["b-audible", "b-silent"])
@pytest.mark.parametrize(
    "transition,transition_frames", [("cut", 0), ("crossfade", 2), ("crossfade", 8), ("fade_through_black", 3)]
)
def test_spliced_soundtrack_matches_a_full_re_encode(tmp_path, monkeypatch, transition, transition_frames, b_tone):
    """Copying the head must neither move what follows nor glitch the seam: the spliced
    soundtrack matches a full re-encode of the same timeline everywhere, seam included, to
    within the coding noise of two lossy encodes. A silent continuation (a Wan clip) must
    splice too: its silence codes with no error and is no measure of a seam."""
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1), tone=440.0)
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2), tone=b_tone)
    fields = {"transition": transition, "transition_frames": transition_frames}
    spliced, context = _concat(tmp_path, [a, b], "spliced", **fields)
    assert "audio samples of the first clip without re-encoding" in _logged(context)

    monkeypatch.setattr(VideoConcatInvocation, "_mux_spliced_audio", lambda *args, **kwargs: False)
    full, _ = _concat(tmp_path, [a, b], "full", **fields)

    got, want = extract_audio_pcm(spliced), extract_audio_pcm(full)
    assert got is not None and want is not None
    n = min(got[0].shape[1], want[0].shape[1])
    assert float(np.abs(got[0][:, :n] - want[0][:, :n]).max()) < 0.1


@pytest.mark.parametrize(
    "tag",
    [("-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc"), ("-aspect", "16:9")],
    ids=["colr", "pasp"],
)
def test_first_clip_with_presentation_tags_is_re_encoded(tmp_path, tag):
    """The joined track would present the re-encoded continuation with the copied head's colour
    or aspect tags, which were never applied to it."""
    _encode(tmp_path / "plain.mp4", _frames(100, seed=1))
    a = tmp_path / "a.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp_path / "plain.mp4")]
        + ["-c", "copy", *tag, str(a)],
        check=True,
    )
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2))
    out, context = _concat(tmp_path, [a, b], "out", transition="crossfade", transition_frames=2)
    assert "Stream-copied" not in _logged(context)
    layout = read_h264_layout(out)
    assert layout is not None and b"colr" not in layout.sample_entry and b"pasp" not in layout.sample_entry
    assert len(_decode(out)) == 138


@pytest.mark.parametrize(
    "stage",
    [
        ("invokeai.app.invocations.video_concat.copy_frames", "crossfade"),
        ("invokeai.app.invocations.video_concat.concat_stream_copy", "cut"),
        ("invokeai.app.invocations.video_concat.splice_soundtrack", "cut"),
        # The audio decodes: an input's soundtrack, and the splice's own validation decodes.
        ("invokeai.app.invocations.video_concat.extract_audio_pcm", "cut"),
        ("invokeai.app.util.aac_splice.extract_audio_pcm", "cut"),
    ],
)
def test_cancellation_during_a_splice_saves_nothing(tmp_path, stage):
    """Cancelling while a splice stage runs stops it — the stage itself raises, not just the
    pre-save check — and nothing is saved."""
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1), tone=440.0)
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2), tone=660.0)
    canceled = [False]
    raised: list[BaseException] = []
    stage, transition = stage
    module, name = stage.rsplit(".", 1)
    real = getattr(importlib.import_module(module), name)

    def cancel_then_run(*args, **kwargs):
        canceled[0] = True
        try:
            return real(*args, **kwargs)
        except BaseException as e:
            raised.append(e)
            raise

    context = MagicMock()
    context.videos.get_path.side_effect = lambda name: {"a": a, "b": b}[name]
    context.util.is_canceled.side_effect = lambda: canceled[0]
    node = VideoConcatInvocation(
        videos=[VideoField(video_name="a"), VideoField(video_name="b")],
        transition=transition,
        transition_frames=2 if transition == "crossfade" else 0,
    )
    with patch(stage, side_effect=cancel_then_run):
        with pytest.raises(CanceledException):
            node.invoke(context)
    assert len(raised) == 1 and isinstance(raised[0], CanceledException)
    context.videos.save.assert_not_called()


def test_tkhd_display_size_does_not_leak_onto_the_join(tmp_path):
    """ffmpeg turns a track header display size that disagrees with the coded size into a
    pasp when it copies the track: the copied head would then stretch the new clip too."""
    source = _encode(tmp_path / "plain.mp4", _frames(100, seed=1))
    data = bytearray(source.read_bytes())
    tkhd = data.find(b"tkhd")
    assert data[tkhd + 4] == 0  # version 0: width at payload offset 76, as 16.16 fixed point
    (width,) = struct.unpack_from(">I", data, tkhd + 4 + 76)
    struct.pack_into(">I", data, tkhd + 4 + 76, width * 2)
    a = tmp_path / "a.mp4"
    a.write_bytes(bytes(data))
    b = _encode(tmp_path / "b.mp4", _frames(40, seed=2))

    out, context = _concat(tmp_path, [a, b], "out", transition="cut", transition_frames=0)
    layout = read_h264_layout(out)
    assert layout is not None and b"pasp" not in layout.sample_entry
    assert len(_decode(out)) == 140


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


def _encode_fractional_rate(
    path: Path, seed: int, frame_count: int = 40, rate: tuple[int, int] = (24000, 1001)
) -> Path:
    frames = np.stack(_frames(frame_count, seed))
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{WIDTH}x{HEIGHT}",
            "-framerate",
            f"{rate[0]}/{rate[1]}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-g",
            "48",
            "-video_track_timescale",
            str(rate[0]),
            "-frames:v",
            str(len(frames)),
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
        capture_output=True,
    )
    layout = read_h264_layout(path)
    assert layout is not None
    assert layout.time_scale / layout.frame_duration == pytest.approx(rate[0] / rate[1])
    return path


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
