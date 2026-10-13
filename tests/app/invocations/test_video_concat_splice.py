"""Joining onto a clip this app encoded must not re-encode that clip's head.

Extend Video joins each continuation onto the video it extends, so a video extended N
times used to carry its first segment through N lossy re-encodes, with artifacts
compounding in the earliest footage. The concat node now stream-copies the first clip up
to its last usable keyframe and encodes only the rest. These tests pin that the copied
frames come through bit-exact (also across repeated extensions), that the joined
timeline has the same length and audio as a full re-encode would, and that inputs which
cannot be spliced losslessly still join correctly by re-encoding.

Real ffmpeg (the imageio-ffmpeg bundled binary) encodes and decodes throughout.
"""

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
from invokeai.app.util.aac_splice import PREROLL_FRAMES
from invokeai.app.util.h264_splice import read_h264_layout
from invokeai.app.util.mp4_metadata import find_box, write_mp4_tags
from invokeai.app.util.video_audio import extract_audio_pcm
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav
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
    keyframe = _last_usable_keyframe(a, tail)
    a_frames, out_frames = _decode(a), _decode(out)
    assert len(out_frames) == expected_len
    assert all(np.array_equal(x, y) for x, y in zip(a_frames[:keyframe], out_frames[:keyframe], strict=True))
    assert f"Stream-copied the first {keyframe} frames" in _logged(context)
    saved = context.videos.save.call_args.kwargs
    assert saved["duration"] == pytest.approx(expected_len / FPS)


def test_repeated_extension_leaves_the_original_segment_untouched(tmp_path):
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1))
    original = _decode(a)
    current = a
    for round_ in range(3):
        nxt = _encode(tmp_path / f"n{round_}.mp4", _frames(40, seed=10 + round_))
        current, _ = _concat(tmp_path, [current, nxt], f"round{round_}", transition="crossfade", transition_frames=2)
    final = _decode(current)
    assert len(final) == 100 + 3 * (40 - 2)
    # Each round re-encodes only from the last keyframe of its first clip, which lies in the
    # newest continuation from the second round on: A's own head is copied every time.
    keyframe = _last_usable_keyframe(a, 2)
    assert all(np.array_equal(x, y) for x, y in zip(original[:keyframe], final[:keyframe], strict=True))


def test_spliced_join_keeps_the_audio_timeline(tmp_path):
    a = _encode(tmp_path / "a.mp4", _frames(64, seed=1), tone=440.0)
    b = _encode(tmp_path / "b.mp4", _frames(64, seed=2), tone=880.0)
    out, context = _concat(tmp_path, [a, b], "out", transition="cut", transition_frames=0)
    assert "Stream-copied" in _logged(context)

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
    # Every first-clip length just past a keyframe, so one leaves a 1-2 frame remainder.
    for n_a in range(65, 70):
        a = _encode(tmp_path / f"a{n_a}.mp4", _frames(n_a, seed=1))
        out, _ = _concat(tmp_path, [a, b], f"out{n_a}", transition="cut", transition_frames=0)
        _, _, duration, fps = probe_video(out)
        assert fps == pytest.approx(FPS, rel=1e-3), f"first clip of {n_a} frames"
        assert duration == pytest.approx((n_a + 1) / FPS, abs=0.5 / FPS)
        assert len(_decode(out)) == n_a + 1


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


def test_repeated_extension_leaves_the_original_soundtrack_untouched(tmp_path):
    """Every join used to decode and re-encode the whole soundtrack, compounding AAC loss
    in the earliest audio just as in its video."""
    a = _encode(tmp_path / "a.mp4", _frames(100, seed=1), tone=440.0)
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
