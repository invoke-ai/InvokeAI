"""A trimmed source's soundtrack must not be re-encoded where the trim keeps it.

Extend Video trims its source before joining a continuation onto it. The trim used to
decode and re-encode the whole kept soundtrack, so every extension of a trimmed source
cost the earlier audio one more AAC generation even though its video was stream-copied.
The node now copies the source's AAC packets for every output frame wholly inside the
range and encodes only a lead-in and a tail. These tests pin that the copied audio comes
through bit-exact, that the seams are no worse than a full re-encode, that the soundtrack
ends with the video, and that a source the splice cannot vouch for is re-encoded instead.

Real ffmpeg (the imageio-ffmpeg bundled binary) encodes and decodes throughout.
"""

import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import imageio.v2 as iio2
import imageio_ffmpeg
import numpy as np
import pytest

from invokeai.app.invocations.fields import VideoField
from invokeai.app.invocations.video_concat import VideoConcatInvocation
from invokeai.app.invocations.video_frame_extract_range import ExtractVideoRangeInvocation
from invokeai.app.util import video_thumbnails
from invokeai.app.util.aac_splice import EDGE_FRAMES, PREROLL_FRAMES, splice_soundtrack, spliceable_aac_layout
from invokeai.app.util.mp4_metadata import find_box, write_mp4_tags
from invokeai.app.util.video_audio import extract_audio_pcm
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav

FPS = 16.0
RATE = 32000
SIZE = 64
# How far into a trimmed soundtrack its encoded lead-in and its seam reach, and how far from its
# end the tail's: the copied run starts within a frame after EDGE_FRAMES of lead-in, and ends at
# least EDGE_FRAMES (and less than one more) before the end.
LEAD_IN_LIMIT = (EDGE_FRAMES + 2) * 1024
TAIL_LIMIT = (EDGE_FRAMES + 1) * 1024


def _signal(kind: str, n_frames: int, seed: int) -> np.ndarray:
    n = round(n_frames / FPS * RATE)
    t = np.arange(n) / RATE
    if kind == "tone":
        mono = 0.4 * np.sin(2 * np.pi * (440 + 110 * seed) * t)
    else:
        # Broadband noise is where the encoder would use perceptual noise substitution.
        mono = 0.25 * np.random.default_rng(seed).standard_normal(n)
    return np.stack([mono, 0.9 * mono]).astype(np.float32)


def _clip(dirpath: Path, name: str, n_frames: int, kind: str, seed: int) -> Path:
    path, wav = dirpath / f"{name}.mp4", dirpath / f"{name}.wav"
    write_stereo_wav(wav, _signal(kind, n_frames, seed), RATE)
    writer = make_mp4_writer(path, FPS, audio_path=wav)
    rng = np.random.default_rng(seed)
    for _ in range(n_frames):
        writer.append_data(rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8))
    writer.close()
    return path


def _uploaded_clip_with_audio(
    tmp_path: Path,
    *,
    audio_args: list[str],
    audio_frames: int,
    second_audio_track: bool = False,
) -> Path:
    video = tmp_path / "video.mp4"
    writer = make_mp4_writer(video, FPS)
    rng = np.random.default_rng(0)
    for _ in range(40):
        writer.append_data(rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8))
    writer.close()

    audio = tmp_path / "audio.wav"
    write_stereo_wav(audio, _signal("tone", audio_frames, 0), RATE)
    inputs = ["-i", str(video), *audio_args, "-i", str(audio)]
    maps = ["-map", "0:v", "-map", "1:a"]
    if second_audio_track:
        inputs += ["-i", str(audio)]
        maps += ["-map", "2:a"]
    source = tmp_path / "source.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *inputs, *maps]
        + ["-c:v", "copy", "-c:a", "aac", "-aac_pns", "0", str(source)],
        check=True,
    )
    return source


def _save_to(saved: Path):
    def _save(source_path: Path, **_: object) -> MagicMock:
        # The stream-copy remux save() runs to embed the record.
        write_mp4_tags(source_path, saved, {"invokeai_metadata": "{}"})
        return MagicMock()

    return _save


def _trim(dirpath: Path, source: Path, start: int, end: int, name: str = "trimmed") -> tuple[Path, MagicMock]:
    saved = dirpath / f"{name}.mp4"
    context = MagicMock()
    context.videos.get_path.return_value = source
    context.util.is_canceled.return_value = False
    context.videos.save.side_effect = _save_to(saved)
    node = ExtractVideoRangeInvocation(video=VideoField(video_name=source.stem), start_frame=start, end_frame=end)
    base = MagicMock(video=VideoField(video_name=name), width=SIZE, height=SIZE, num_frames=1, fps=FPS, duration=1.0)
    with patch("invokeai.app.invocations.video_frame_extract_range.VideoOutput.build", return_value=base):
        node.invoke(context)
    return saved, context


def _logged(context: MagicMock) -> str:
    return "\n".join(str(call.args[0]) for call in context.logger.info.call_args_list)


def _movie_duration(path: Path) -> float:
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


def _window(source: Path, start: int, end: int) -> np.ndarray:
    decoded = extract_audio_pcm(source)
    assert decoded is not None
    return decoded[0][:, round(start * RATE / FPS) : round((end + 1) * RATE / FPS)]


@pytest.mark.parametrize("kind", ["tone", "noise"])
@pytest.mark.parametrize("start,end", [(20, 90), (0, 80), (33, 99)])
def test_trim_copies_the_source_audio_it_keeps(tmp_path, monkeypatch, kind, start, end):
    source = _clip(tmp_path, "source", 100, kind, seed=1)
    trimmed, context = _trim(tmp_path, source, start, end)
    assert "audio packets" in _logged(context)

    want = _window(source, start, end)
    got = extract_audio_pcm(trimmed)
    assert got is not None
    n = end - start + 1
    # The soundtrack ends with the video, and decodes to whole frames past it.
    assert _movie_duration(trimmed) == pytest.approx(n / FPS, abs=0.002)
    assert 0 <= got[0].shape[1] - round(n * RATE / FPS) < 1024
    # Every frame away from the lead-in and tail seams is the source's own audio, bit for bit.
    interior = slice(LEAD_IN_LIMIT, want.shape[1] - TAIL_LIMIT)
    assert np.array_equal(got[0][:, interior], want[:, interior])

    # The seams are no worse than re-encoding the whole range.
    monkeypatch.setattr(ExtractVideoRangeInvocation, "_mux_spliced_audio", lambda *args, **kwargs: False)
    full, _ = _trim(tmp_path, source, start, end, name="full")
    reference = extract_audio_pcm(full)
    assert reference is not None
    m = min(got[0].shape[1], reference[0].shape[1], want.shape[1])
    spliced_error = np.abs(got[0][:, :m] - want[:, :m])
    full_error = np.abs(reference[0][:, :m] - want[:, :m])
    for seam in (slice(0, LEAD_IN_LIMIT), slice(want.shape[1] - TAIL_LIMIT, m)):
        assert (
            float(np.sqrt(np.mean(spliced_error[:, seam] ** 2)))
            <= 2 * float(np.sqrt(np.mean(full_error[:, seam] ** 2))) + 1e-3
        )


def test_copied_audio_that_does_not_decode_back_is_rejected(tmp_path):
    """The splice trusts nothing it has not decoded: if the source's packets do not reproduce
    the soundtrack's samples where they are copied (here, a timeline nudged by one LSB), the
    splice is refused and the caller re-encodes."""
    source = _clip(tmp_path, "source", 100, "tone", seed=1)
    video = tmp_path / "video.mp4"
    writer = make_mp4_writer(video, FPS)
    for _ in range(71):
        writer.append_data(np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    writer.close()
    layout = spliceable_aac_layout(source, RATE)
    assert layout is not None
    timeline = _window(source, 20, 90)
    window_start = round(20 * RATE / FPS)
    first = -(-(window_start + layout.priming + EDGE_FRAMES * 1024) // 1024)
    copied_at = first * 1024 - layout.priming - window_start
    end = (round(91 * RATE / FPS) - EDGE_FRAMES * 1024 + layout.priming) // 1024
    args = (video, source, layout.priming, first, end, copied_at)

    assert splice_soundtrack(*args, timeline, RATE, tmp_path / "ok.mp4") is None
    nudged = timeline.copy()
    nudged[:, timeline.shape[1] // 2] += 1 / 32768
    assert (
        splice_soundtrack(*args, nudged, RATE, tmp_path / "bad.mp4") == "the copied audio did not decode back exactly"
    )


def test_extending_a_trimmed_source_keeps_its_audio(tmp_path):
    """The Extend graph: trim the source, then join a continuation onto the trim."""
    source = _clip(tmp_path, "source", 100, "tone", seed=1)
    trimmed, trim_context = _trim(tmp_path, source, 20, 95)
    assert "audio packets" in _logged(trim_context)
    continuation = _clip(tmp_path, "continuation", 40, "tone", seed=3)

    joined = tmp_path / "joined.mp4"
    by_name = {"trimmed": trimmed, "continuation": continuation}
    context = MagicMock()
    context.videos.get_path.side_effect = lambda name: by_name[name]
    context.util.is_canceled.return_value = False
    context.videos.save.side_effect = _save_to(joined)
    node = VideoConcatInvocation(
        videos=[VideoField(video_name="trimmed"), VideoField(video_name="continuation")],
        transition="crossfade",
        transition_frames=2,
    )
    with patch("invokeai.app.invocations.video_concat.VideoOutput.build", return_value=MagicMock()):
        node.invoke(context)
    assert "audio samples of the first clip without re-encoding" in _logged(context)

    want = _window(source, 20, 95)
    got = extract_audio_pcm(joined)
    assert got is not None
    # The source's audio inside the trim, clear of the trim's lead-in and the join, is unchanged.
    interior = slice(LEAD_IN_LIMIT, want.shape[1] - round(2 * RATE / FPS) - (PREROLL_FRAMES + 2) * 1024)
    assert np.array_equal(got[0][:, interior], want[:, interior])
    reader = iio2.get_reader(str(joined), format="FFMPEG")
    try:
        assert sum(1 for _ in reader) == 76 + 40 - 2
    finally:
        reader.close()


def test_trim_splices_when_frames_hold_fractional_samples(tmp_path):
    """At 44.1 kHz and 24 fps a frame is 1837.5 samples: rounding the trimmed span on its own
    came out one off the window, which interpolated the whole soundtrack and ruled out a copy."""
    rate, fps = 44100, 24.0
    n = round(110 / fps * rate)
    tone = (0.4 * np.sin(2 * np.pi * 523.0 * np.arange(n) / rate)).astype(np.float32)
    wav, source = tmp_path / "source.wav", tmp_path / "source.mp4"
    write_stereo_wav(wav, np.stack([tone, tone]), rate)
    writer = make_mp4_writer(source, fps, audio_path=wav)
    for i in range(110):
        writer.append_data(np.full((SIZE, SIZE, 3), i, dtype=np.uint8))
    writer.close()

    trimmed, context = _trim(tmp_path, source, 7, 100)
    assert "audio packets" in _logged(context)
    decoded, got = extract_audio_pcm(source), extract_audio_pcm(trimmed)
    assert decoded is not None and got is not None
    want = decoded[0][:, round(7 * rate / fps) : round(101 * rate / fps)]
    interior = slice(LEAD_IN_LIMIT, want.shape[1] - TAIL_LIMIT)
    assert np.array_equal(got[0][:, interior], want[:, interior])


@pytest.mark.parametrize("n_frames,fps,rate", [(40, 16.0, 32000), (97, 24.0, 32000), (81, 16.0, 16000)])
def test_whole_clip_copy_keeps_a_soundtrack_that_ends_with_the_video(tmp_path, n_frames, fps, rate):
    """The app's own soundtracks decode up to a packet past their end (the decoder does not
    apply the end trim); that must not cost them the copy. 97 frames at 24 fps and 32 kHz is
    a MiniMax H3 clip, 81 at 16 fps and 16 kHz a 16 kHz Wan-length one."""
    n = round(n_frames / fps * rate)
    wav, source = tmp_path / "source.wav", tmp_path / "source.mp4"
    write_stereo_wav(wav, np.stack([np.sin(np.arange(n) / 7.0)] * 2).astype(np.float32) * 0.4, rate)
    writer = make_mp4_writer(source, fps, audio_path=wav)
    for i in range(n_frames):
        writer.append_data(np.full((SIZE, SIZE, 3), i, dtype=np.uint8))
    writer.close()

    saved, context = _trim(tmp_path, source, 0, -1)
    assert "the whole of the source, copied" in _logged(context)
    want, got = extract_audio_pcm(source), extract_audio_pcm(saved)
    assert want is not None and got is not None
    assert np.array_equal(got[0], want[0])


@pytest.mark.parametrize(
    "audio_args,audio_frames,copied,second_audio_track,video_only_probe",
    [
        ([], 80, "Stream-copied frames 0-39", False, False),
        (["-itsoffset", "1.0"], 39, "Stream-copied frames 0-39", False, False),
        ([], 40, "the whole of the source, copied", True, False),
        ([], 80, "Stream-copied frames 0-39", False, True),
    ],
    ids=["longer-audio", "delayed-audio", "second-audio-track", "longer-audio-video-only-probe-result"],
)
def test_whole_clip_copy_does_not_trust_a_video_only_probe_duration(
    tmp_path, monkeypatch, audio_args, audio_frames, copied, second_audio_track, video_only_probe
):
    """The invocation must trim out-of-range audio even when its probe reports only video duration."""
    source = _uploaded_clip_with_audio(
        tmp_path,
        audio_args=audio_args,
        audio_frames=audio_frames,
        second_audio_track=second_audio_track,
    )

    if video_only_probe:
        # Simulate the duration returned by video_decode_worker._probe()'s OpenCV fallback.
        monkeypatch.setattr(
            "invokeai.app.invocations.video_frame_extract_range.probe_video",
            lambda _path: (SIZE, SIZE, 40 / FPS, FPS),
        )

    saved, context = _trim(tmp_path, source, 0, -1)
    # Nothing plays past the last frame (the probe reports to the hundredth of a second).
    assert _movie_duration(saved) <= 40 / FPS + 0.5 / FPS
    banner = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(saved)], capture_output=True, text=True
    ).stderr
    assert banner.count("Audio:") == 1
    # The video is still copied, not re-encoded.
    assert copied in _logged(context)


def test_whole_clip_longer_audio_trims_when_opencv_probe_fallback_runs(tmp_path, monkeypatch):
    """Force imageio's probe to fail in the worker and exercise its real OpenCV fallback."""
    source = _uploaded_clip_with_audio(tmp_path, audio_args=[], audio_frames=80)
    worker_wrapper = tmp_path / "force_opencv_probe.py"
    worker_wrapper.write_text(
        "import imageio.v3 as iio\n"
        "import runpy\n"
        "import sys\n"
        "\n"
        "def fail_immeta(*_args, **_kwargs):\n"
        "    raise RuntimeError('force the OpenCV fallback')\n"
        "\n"
        "iio.immeta = fail_immeta\n"
        "worker_path = sys.argv[1]\n"
        "sys.argv = [worker_path, *sys.argv[2:]]\n"
        "runpy.run_path(worker_path, run_name='__main__')\n"
    )
    original_worker_command = video_thumbnails._worker_command
    worker_path = original_worker_command("probe", str(source))[1]

    def worker_command(*args: str) -> list[str]:
        if args[0] == "probe":
            return [sys.executable, str(worker_wrapper), worker_path, *args]
        return original_worker_command(*args)

    monkeypatch.setattr(video_thumbnails, "_worker_command", worker_command)

    width, height, probed_duration, fps = video_thumbnails.probe_video(source)
    assert (width, height) == (SIZE, SIZE)
    assert probed_duration == pytest.approx(40 / FPS, abs=0.1)
    assert fps == pytest.approx(FPS)
    assert _movie_duration(source) > 4.9

    saved, context = _trim(tmp_path, source, 0, -1)

    assert _movie_duration(saved) <= 40 / FPS + 0.5 / FPS
    assert "Stream-copied frames 0-39" in _logged(context)
