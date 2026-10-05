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
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav


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


def _ffmpeg_has_encoder(name: str) -> bool:
    """True if the ffmpeg binary imageio will actually invoke can encode with ``name``.

    imageio-ffmpeg's bundled binaries carry libmp3lame, but ``IMAGEIO_FFMPEG_EXE``
    can point at a distro build that does not.
    """
    import subprocess

    import imageio_ffmpeg

    try:
        encoders = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.split()[1:2] == [name] for line in encoders.splitlines() if line.strip())


def _write_sine_wav(path: Path, duration_s: float, sample_rate: int = 32_000) -> None:
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    samples = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)])
    write_stereo_wav(path, samples, sample_rate)


def test_audio_path_muxes_an_aac_stream(tmp_path: Path) -> None:
    fps, num_frames = 8.0, 8
    wav_path = tmp_path / "audio.wav"
    _write_sine_wav(wav_path, duration_s=num_frames / fps)

    path = tmp_path / "out.mp4"
    writer = make_mp4_writer(path, fps=fps, audio_path=wav_path)
    try:
        for i in range(num_frames):
            writer.append_data(np.full((84, 120, 3), i * 10, dtype=np.uint8))
    finally:
        writer.close()

    reader = iio2.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
    finally:
        reader.close()
    assert meta["audio_codec"] == "aac"
    # The audio was pre-trimmed to the video duration, so the container must not
    # be stretched past it (ffmpeg gets no -shortest; see module docstring).
    assert meta["duration"] == pytest.approx(num_frames / fps, abs=0.1)
    assert meta["size"] == (120, 84)


@pytest.mark.skipif(not _ffmpeg_has_encoder("libmp3lame"), reason="ffmpeg build lacks the libmp3lame encoder")
def test_audio_codec_is_forwarded_not_defaulted(tmp_path: Path) -> None:
    # ffmpeg's default MP4 audio encoder is aac, so the aac test above cannot
    # distinguish "audio_codec forwarded" from "audio_codec dropped". A
    # non-default codec can.
    wav_path = tmp_path / "audio.wav"
    _write_sine_wav(wav_path, duration_s=0.5)

    path = tmp_path / "out.mp4"
    writer = make_mp4_writer(path, fps=8.0, audio_path=wav_path, audio_codec="libmp3lame")
    try:
        for _ in range(4):
            writer.append_data(np.zeros((84, 120, 3), dtype=np.uint8))
    finally:
        writer.close()

    reader = iio2.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
    finally:
        reader.close()
    assert meta["audio_codec"] == "mp3"


def test_missing_audio_file_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does-not-exist.wav"):
        make_mp4_writer(tmp_path / "out.mp4", fps=8.0, audio_path=tmp_path / "does-not-exist.wav")


def test_no_audio_path_produces_no_audio_stream(tmp_path: Path) -> None:
    path = tmp_path / "out.mp4"
    writer = make_mp4_writer(path, fps=8.0)
    try:
        for _ in range(4):
            writer.append_data(np.zeros((84, 120, 3), dtype=np.uint8))
    finally:
        writer.close()

    reader = iio2.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
    finally:
        reader.close()
    assert meta.get("audio_codec") in (None, "")


def test_write_stereo_wav_format_and_clipping(tmp_path: Path) -> None:
    import wave

    sample_rate = 32_000
    samples = np.stack([np.full(100, 2.0), np.full(100, -2.0)])  # out of range -> clipped
    path = tmp_path / "a.wav"
    write_stereo_wav(path, samples, sample_rate)

    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == sample_rate
        assert wav.getnframes() == 100
        frames = np.frombuffer(wav.readframes(100), dtype=np.int16).reshape(-1, 2)
    assert frames[:, 0].max() == 32767
    assert frames[:, 1].min() == -32767

    with pytest.raises(ValueError, match=r"\(2, n_samples\)"):
        write_stereo_wav(tmp_path / "b.wav", np.zeros((100,)), sample_rate)


def test_write_stereo_wav_rounds_rather_than_truncates(tmp_path: Path) -> None:
    import wave

    # In-range values only — the clipping test above pins full scale via clip(),
    # so it cannot see whether the scaled value is rounded or truncated toward
    # zero. Truncation would give 32766 and 1 here.
    samples = np.stack([np.full(4, 0.99999), np.full(4, 1.9 / 32767.0)])
    path = tmp_path / "round.wav"
    write_stereo_wav(path, samples, 32_000)

    with wave.open(str(path), "rb") as wav:
        frames = np.frombuffer(wav.readframes(4), dtype=np.int16).reshape(-1, 2)
    assert frames[0, 0] == 32767
    assert frames[0, 1] == 2


def test_write_stereo_wav_rejects_empty_samples(tmp_path: Path) -> None:
    # A zero-frame WAV is not an ffmpeg error: it muxes into an MP4 with no
    # audio stream and no diagnostic, which is the silent failure the writer's
    # existence check exists to prevent.
    with pytest.raises(ValueError, match="empty WAV"):
        write_stereo_wav(tmp_path / "empty.wav", np.zeros((2, 0)), 32_000)


def test_write_stereo_wav_float16_and_nan_are_safe(tmp_path: Path) -> None:
    import wave

    # float16: 1.0 * 32767.0 rounds to 32770 in fp16 and would wrap to -32768
    # if the conversion ran in the input dtype.
    fp16 = np.ones((2, 8), dtype=np.float16)
    path = tmp_path / "fp16.wav"
    write_stereo_wav(path, fp16, 32_000)
    with wave.open(str(path), "rb") as wav:
        frames = np.frombuffer(wav.readframes(8), dtype=np.int16)
    assert frames.min() == 32767

    # NaN becomes silence, not undefined int16 garbage.
    nan = np.full((2, 8), np.nan)
    path = tmp_path / "nan.wav"
    write_stereo_wav(path, nan, 32_000)
    with wave.open(str(path), "rb") as wav:
        frames = np.frombuffer(wav.readframes(8), dtype=np.int16)
    assert np.all(frames == 0)


def test_validate_even_dimensions_accepts_even_and_rejects_odd() -> None:
    _validate_even_dimensions(1920, 1080, "ok.mp4")
    with pytest.raises(ValueError, match="even dimensions"):
        _validate_even_dimensions(833, 480, "odd-width.mp4")
    with pytest.raises(ValueError, match="even dimensions"):
        _validate_even_dimensions(832, 481, "odd-height.mp4")
