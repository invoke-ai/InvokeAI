"""Shared MP4 writer construction for video-producing invocations.

All video nodes encode through this helper so the encoder settings stay in one
place. libx264 + yuv420p (imageio's defaults for the FFMPEG plugin) give
broadly-compatible browser playback. ``macro_block_size=1`` is load-bearing:
imageio's default of 16 makes ffmpeg silently *rescale* frames to the next
multiple of 16 (e.g. 1920x1080 -> 1920x1088), which desynchronizes the encoded
file from the dimensions recorded in the video DTO and breaks same-dimension
checks downstream (e.g. concatenating a trimmed clip with its source).

yuv420p requires even dimensions; callers validate that before encoding.

Audio muxing: pass ``audio_path`` to include an audio track. Two constraints:

- imageio-ffmpeg does not pass ``-shortest``, so the container duration is the
  *max* of the stream durations. Callers must trim the audio to the video
  duration (num_frames / fps seconds) before handing it over, or the player
  shows trailing frozen video.
- ``audio_codec`` must name a real encoder. The default AAC-LC plays in every
  browser; never pass ``"copy"`` for a PCM WAV — PCM-in-MP4 does not play in
  browsers.
"""

import wave
from pathlib import Path

import imageio.v2 as iio2
import numpy as np


def make_mp4_writer(path: Path | str, fps: float, audio_path: Path | str | None = None, audio_codec: str = "aac"):
    """Returns an imageio FFMPEG writer that preserves frame dimensions exactly.

    If ``audio_path`` is given, that file's audio is encoded with ``audio_codec``
    and muxed into the output (see module docstring for the trim requirement).
    The audio file must exist until at least the first ``append_data`` call has
    completed: ffmpeg is spawned lazily on the first append, and a file deleted
    before then fails as an unexplained BrokenPipeError later in the encode (or,
    for tiny clips, silently produces no output at all). The existence check
    here converts the common failure into an immediate, named error.
    """
    kwargs = {}
    if audio_path is not None:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file for muxing does not exist: {audio_path}")
        kwargs["audio_path"] = str(audio_path)
        kwargs["audio_codec"] = audio_codec
    return iio2.get_writer(str(path), format="FFMPEG", mode="I", fps=fps, codec="libx264", macro_block_size=1, **kwargs)


def write_stereo_wav(path: Path | str, samples: np.ndarray, sample_rate: int) -> None:
    """Write float PCM as a 16-bit stereo WAV suitable for ``make_mp4_writer``'s ``audio_path``.

    ``samples`` must be shaped ``(2, n_samples)`` (channels first), values in
    [-1, 1]; anything outside is clipped rather than wrapped, and NaN becomes
    silence (0) rather than undefined int16 garbage. The conversion is done in
    float64 regardless of input dtype: in float16, ``1.0 * 32767.0`` rounds up
    to 32770 and would wrap full-scale peaks to -32768.
    """
    if samples.ndim != 2 or samples.shape[0] != 2:
        raise ValueError(f"Expected samples shaped (2, n_samples), got {samples.shape}")
    widened = np.nan_to_num(samples.astype(np.float64), nan=0.0)
    int16 = (np.clip(widened, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(int16.T.reshape(-1).tobytes())
