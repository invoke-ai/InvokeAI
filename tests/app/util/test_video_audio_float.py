"""PCM extraction uses the first audio stream for both decode formats.

The float branch pins ``-map 0:a:0`` and reads the rate off the container banner; the two
must describe the same stream, including on multi-track containers where ffmpeg's default
selection would pick a different one.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from invokeai.app.util.video_audio import _ffmpeg_exe, extract_audio_pcm
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav


def _sine(rate: int, seconds: float, freq: float) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.5
    return np.stack([wave, wave])


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _make_video(tmp_dir: Path, wavs: list[tuple[np.ndarray, int]]) -> Path:
    """A tiny h264 MP4 carrying the given audio tracks, in order."""
    silent = tmp_dir / "silent.mp4"
    writer = make_mp4_writer(silent, fps=8)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(16):
        writer.append_data(frame)
    writer.close()

    wav_paths = []
    for index, (samples, rate) in enumerate(wavs):
        wav_path = tmp_dir / f"a{index}.wav"
        write_stereo_wav(wav_path, samples, rate)
        wav_paths.append(wav_path)

    out = tmp_dir / "multi.mp4"
    args = [_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(silent)]
    for wav_path in wav_paths:
        args += ["-i", str(wav_path)]
    args += ["-map", "0:v:0"]
    for index in range(len(wav_paths)):
        args += ["-map", f"{index + 1}:a:0"]
    args += ["-c:v", "copy", "-c:a", "aac"]
    if len(wav_paths) > 1:
        # Default stream 1 forces ffmpeg's automatic selection away from stream 0.
        args += ["-disposition:a:0", "0", "-disposition:a:1", "default"]
    args += [str(out)]
    subprocess.run(args, check=True, capture_output=True, timeout=120)
    return out


def test_float_branch_returns_float_samples_at_the_probed_rate(tmp_dir):
    video = _make_video(tmp_dir, [(_sine(44100, 2.0, 440.0), 44100)])
    extracted = extract_audio_pcm(video, float_pcm=True)

    assert extracted is not None
    pcm, rate = extracted
    assert rate == 44100
    assert pcm.dtype == np.float32
    assert pcm.shape[0] == 2
    # Genuinely float samples, not an int16 re-quantization.
    assert np.mean((pcm * 32768.0) % 1.0 > 1e-6) > 0.5


@pytest.mark.parametrize("float_pcm", [False, True])
def test_multi_track_container_decodes_the_first_audio_stream(tmp_dir, float_pcm):
    # Track 0 at 44100 Hz, track 1 at 48000 Hz: without the -map pin, ffmpeg's default
    # selection picks the wrong soundtrack and, for float PCM, disagrees with the probe.
    video = _make_video(tmp_dir, [(_sine(44100, 2.0, 440.0), 44100), (_sine(48000, 2.0, 220.0), 48000)])
    extracted = extract_audio_pcm(video, float_pcm=float_pcm)

    assert extracted is not None
    pcm, rate = extracted
    assert rate == 44100
    # The sample COUNT must match the probed rate's timeline (2 s +/- codec padding).
    assert pcm.shape[1] == pytest.approx(2.0 * rate, rel=0.05)
    spectrum = np.abs(np.fft.rfft(pcm[0]))
    frequency = np.fft.rfftfreq(pcm.shape[1], 1.0 / rate)[np.argmax(spectrum)]
    assert frequency == pytest.approx(440.0, abs=5.0)


@pytest.mark.parametrize("float_pcm", [False, True])
def test_container_without_audio_returns_none(tmp_dir, float_pcm):
    video = _make_video(tmp_dir, [])

    assert extract_audio_pcm(video, float_pcm=float_pcm) is None


def test_int16_branch_decodes_single_audio_stream(tmp_dir):
    video = _make_video(tmp_dir, [(_sine(44100, 1.0, 440.0), 44100)])
    extracted = extract_audio_pcm(video)

    assert extracted is not None
    pcm, rate = extracted
    assert rate == 44100
    assert pcm.shape[0] == 2
