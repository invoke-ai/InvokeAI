"""Shared MP4 writer construction for video-producing invocations.

All video nodes encode through this helper so the encoder settings stay in one
place. libx264 + yuv420p (imageio's defaults for the FFMPEG plugin) give
broadly-compatible browser playback. ``macro_block_size=1`` is load-bearing:
imageio's default of 16 makes ffmpeg silently *rescale* frames to the next
multiple of 16 (e.g. 1920x1080 -> 1920x1088), which desynchronizes the encoded
file from the dimensions recorded in the video DTO and breaks same-dimension
checks downstream (e.g. concatenating a trimmed clip with its source).

yuv420p requires even dimensions; callers validate that before encoding.

Quality and keyframes: frames are encoded at CRF 18 (imageio's default quality maps to
CRF 25), because derived videos — trims, joins, extensions — are routinely re-encoded
from earlier outputs and the losses compound. An IDR keyframe is forced every
``KEYFRAME_INTERVAL_SECONDS`` so a later join can stream-copy the head of this file up to
a keyframe near its end instead of re-encoding it (see ``h264_splice``). Both settings
are part of the SPS/PPS, so changing either makes files written before and after the
change unspliceable with each other: a join between them falls back to re-encoding.

Audio muxing: pass ``audio_path`` to include an audio track. Two constraints:

- imageio-ffmpeg does not pass ``-shortest``, so the container duration is the
  *max* of the stream durations. Callers must trim the audio to the video
  duration (num_frames / fps seconds) before handing it over, or the player
  shows trailing frozen video.
- ``audio_codec`` must name a real encoder. The default AAC-LC plays in every
  browser; never pass ``"copy"`` for a PCM WAV — PCM-in-MP4 does not play in
  browsers.
"""

import tempfile
import wave
from pathlib import Path

import imageio.v2 as iio2
import numpy as np

from invokeai.app.util.video_audio import AAC_ENCODER_OPTIONS

CRF = 18
KEYFRAME_INTERVAL_SECONDS = 2.0


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
    # Soundtracks share the WAV-mux path's encoder options so either can be spliced onto the other.
    audio_options = AAC_ENCODER_OPTIONS if audio_path is not None and audio_codec == "aac" else []
    keyframe_interval = max(1, round(fps * KEYFRAME_INTERVAL_SECONDS))
    return iio2.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        quality=None,
        output_params=["-crf", str(CRF), "-g", str(keyframe_interval), *audio_options],
        **kwargs,
    )


_writer_sample_entrys: dict[tuple[int, int, float], bytes] = {}


def writer_sample_entry(width: int, height: int, fps: float) -> bytes | None:
    """The sample entry ``make_mp4_writer`` writes for this canvas and rate (see
    ``H264Layout.sample_entry``).

    They depend only on the encoder settings, canvas and rate, never on the content, so one
    encoded frame tells a splice up front whether a file can share a track with new output
    — before a long encode is spent finding out. None if the probe cannot be read; only a
    successful probe is remembered.
    """
    key = (width, height, fps)
    if key not in _writer_sample_entrys:
        config = _probe_sample_entry(width, height, fps)
        if config is None:
            return None
        _writer_sample_entrys[key] = config
    return _writer_sample_entrys[key]


def _probe_sample_entry(width: int, height: int, fps: float) -> bytes | None:
    from invokeai.app.util.h264_splice import read_h264_layout

    with tempfile.TemporaryDirectory(prefix="invokeai_encoder_probe_") as work:
        path = Path(work) / "probe.mp4"
        writer = make_mp4_writer(path, fps)
        try:
            writer.append_data(np.zeros((height, width, 3), dtype=np.uint8))
        finally:
            writer.close()
        layout = read_h264_layout(path)
    return layout.sample_entry if layout is not None else None


def write_stereo_wav(path: Path | str, samples: np.ndarray, sample_rate: int) -> None:
    """Write float PCM as a 16-bit stereo WAV suitable for ``make_mp4_writer``'s ``audio_path``.

    ``samples`` must be shaped ``(2, n_samples)`` (channels first), values in
    [-1, 1]; anything outside is clipped rather than wrapped, and NaN becomes
    silence (0) rather than undefined int16 garbage. The conversion is done in
    float64 regardless of input dtype: in float16, ``1.0 * 32767.0`` rounds up
    to 32770 and would wrap full-scale peaks to -32768. Scaled values are
    rounded rather than truncated — ``astype`` alone truncates toward zero,
    which costs up to a full LSB and biases the signal toward silence.

    An empty ``samples`` is rejected: a zero-frame WAV is not an ffmpeg error,
    so it would mux as an MP4 with no audio stream at all, which is exactly the
    silent failure ``make_mp4_writer``'s existence check exists to prevent.
    """
    if samples.ndim != 2 or samples.shape[0] != 2:
        raise ValueError(f"Expected samples shaped (2, n_samples), got {samples.shape}")
    if samples.shape[1] == 0:
        raise ValueError("Refusing to write an empty WAV: it would mux as an MP4 with no audio stream")
    widened = np.nan_to_num(samples.astype(np.float64), nan=0.0)
    int16 = np.round(np.clip(widened, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(int16.T.reshape(-1).tobytes())
