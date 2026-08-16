"""Audio extraction and remuxing helpers for video-editing invocations.

The decode side of the app's video plumbing (imageio's FFMPEG reader) exposes only video
frames, so audio work shells out to the same bundled ffmpeg binary imageio uses. Two
operations are provided:

- :func:`extract_audio_pcm` — decode a container's audio track to float stereo PCM
  (returns ``None`` for silent containers, which is how Wan clips present).
- :func:`mux_audio_into_video` — attach a WAV to an already-encoded MP4 by stream-copying
  the video (no re-encode) and encoding the audio to AAC-LC, the browser-safe codec used
  everywhere else in the app.

Plus :func:`resample_linear`, a dependency-free linear resampler used both to unify
sample rates across inputs and to retime audio when a clip's video is retimed by an
output-fps override (a deliberate speed/pitch change, matching what retiming does to the
video).
"""

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class AudioExtractionError(RuntimeError):
    """ffmpeg failed to decode an audio track for a reason other than 'no audio stream'."""


def extract_audio_pcm(video_path: Path) -> tuple[np.ndarray, int] | None:
    """Decode ``video_path``'s audio track to float32 stereo PCM at its native sample rate.

    Returns ``(samples, sample_rate)`` with ``samples`` shaped ``(2, n)`` in [-1, 1], or
    ``None`` when the container has no audio stream. Mono sources are upmixed to stereo by
    ffmpeg; multi-channel sources are downmixed.
    """
    with tempfile.NamedTemporaryFile(prefix="invokeai_audio_extract_", suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        try:
            proc = subprocess.run(
                [
                    _ffmpeg_exe(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-ac",
                    "2",
                    "-acodec",
                    "pcm_s16le",
                    str(wav_path),
                ],
                capture_output=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired as e:
            raise AudioExtractionError(f"ffmpeg timed out extracting audio from {video_path.name}") from e
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            # ffmpeg's phrasing for an input with no audio track ("Output file #0 does not
            # contain any stream" / newer "does not contain any stream" variants).
            if "does not contain any stream" in stderr:
                return None
            raise AudioExtractionError(
                f"ffmpeg could not extract audio from {video_path.name}: {stderr.strip()[-500:]}"
            )
        with wave.open(str(wav_path), "rb") as wav:
            rate = wav.getframerate()
            n = wav.getnframes()
            raw = wav.readframes(n)
        if n == 0:
            return None
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T
        return data.astype(np.float32) / 32768.0, rate
    finally:
        wav_path.unlink(missing_ok=True)


def resample_linear(samples: np.ndarray, num_output_samples: int) -> np.ndarray:
    """Linearly resample ``(2, n)`` PCM to ``(2, num_output_samples)``.

    Used for sample-rate unification and for retiming (where the accompanying video is
    being speed-changed, so the pitch shift this introduces is the correct behavior, not
    an artifact). An empty input yields silence.
    """
    if samples.ndim != 2 or samples.shape[0] != 2:
        raise ValueError(f"Expected samples shaped (2, n), got {samples.shape}")
    n_in = samples.shape[1]
    if num_output_samples <= 0:
        return np.zeros((2, 0), dtype=np.float32)
    if n_in == 0:
        return np.zeros((2, num_output_samples), dtype=np.float32)
    if n_in == num_output_samples:
        return samples.astype(np.float32, copy=False)
    positions = np.linspace(0.0, n_in - 1, num_output_samples)
    grid = np.arange(n_in, dtype=np.float64)
    return np.stack([np.interp(positions, grid, samples[c]) for c in range(2)]).astype(np.float32)


def mux_audio_into_video(video_path: Path, wav_path: Path, output_path: Path) -> None:
    """Produce ``output_path`` = ``video_path``'s streams with ``wav_path`` as an AAC track.

    The video stream is copied bit-exactly (no re-encode); only the audio is encoded.
    """
    try:
        proc = _run_mux(video_path, wav_path, output_path)
    except subprocess.TimeoutExpired as e:
        raise AudioExtractionError(f"ffmpeg timed out muxing audio into {output_path.name}") from e
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise AudioExtractionError(f"ffmpeg could not mux audio into {output_path.name}: {stderr.strip()[-500:]}")


def _run_mux(video_path: Path, wav_path: Path, output_path: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            _ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(wav_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(output_path),
        ],
        capture_output=True,
        timeout=600,
    )
