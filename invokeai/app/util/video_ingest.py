"""Upload-time normalization of video and audio files into browser-safe H.264/AAC MP4.

Uploaded videos are stored under a ``{uuid}.mp4`` name and served with the ``video/mp4``
MIME type unconditionally, so everything in the store must genuinely be H.264/AAC MP4 —
anything else silently breaks browser playback (the reason uploads were historically
restricted to already-compliant files). This module widens what users may *upload* by
converting at ingest time instead of widening what is *stored*:

- H.264 video in a foreign container (iPhone ``.mov`` in "Most Compatible" mode, ``.m4v``,
  screen recordings): lossless **remux** — stream-copy the video, seconds even for large
  files.
- Any other decodable video codec (HEVC from iPhone "High Efficiency" mode, ProRes, VP9):
  **transcode** to H.264. 10-bit/HDR sources are converted to 8-bit yuv420p without
  tonemapping — HDR footage will look flatter than the original; the bundled ffmpeg has
  no zscale/tonemap support.
- Audio-only files (voice memos, music): **wrap** into a video whose frames are a rendered
  waveform. The result flows through every existing video path — gallery, thumbnails,
  trim, and audio-only reference conditioning — without audio becoming a first-class
  media type. The waveform (rather than black frames) keeps the clip recognizable in a
  gallery of thumbnails and makes trim positions legible.

Audio is normalized to AAC in every branch except when the source track is already AAC,
which is stream-copied to avoid a needless lossy generation.

The ffmpeg child gets the same hostile-input discipline as the decode worker: a hard
wall-clock timeout (``subprocess.run`` kills the child on expiry) and a post-hoc output
size cap. The bundled binary ships without ffprobe, so stream layout is read from the
banner ffmpeg prints for a probe-only invocation — the established fallback used by
``video_audio._probe_audio_sample_rate``.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

# Hard wall-clock bound for a single ffmpeg child (probe or convert). Matches the bound
# used for audio extraction: generous enough for a worst-case ~1 GB 4K transcode on a
# modest CPU, small enough that a hostile file cannot camp an upload slot forever.
INGEST_TIMEOUT_SECONDS = 600
PROBE_TIMEOUT_SECONDS = 60

# The waveform canvas for wrapped audio uploads. 24 fps matches the video models' native
# rate so frame-based trims map to 1/24 s of audio.
AUDIO_WRAP_SIZE = "640x360"
AUDIO_WRAP_FPS = 24

IngestAction = Literal["remux", "transcode", "audio_wrap"]


class VideoIngestError(RuntimeError):
    """The upload could not be converted to a browser-safe MP4. Message is user-safe."""


@dataclass(frozen=True)
class MediaProbe:
    """Stream layout parsed from ffmpeg's probe banner."""

    video_codec: Optional[str]
    audio_codec: Optional[str]

    @property
    def has_video(self) -> bool:
        return self.video_codec is not None

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


# ffmpeg banner lines look like:
#   Stream #0:0[0x1](und): Video: hevc (Main 10) (hvc1 / 0x31637668), yuv420p10le(tv, ...
#   Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp
# The codec token is the first word after the stream-type label. "Attached pictures"
# (album art in audio files) are video streams flagged "(attached pic)" — they are cover
# art, not video content, and must not make an audio file look like a video.
_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: (Video|Audio): (\w+)")
_ATTACHED_PIC_RE = re.compile(r"\(attached pic\)")


def probe_media_streams(path: Path) -> MediaProbe:
    """Returns the first real video stream's codec and the first audio stream's codec.

    Raises VideoIngestError when ffmpeg cannot read the file at all.
    """
    try:
        proc = subprocess.run(
            [_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
            capture_output=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise VideoIngestError("Timed out reading the uploaded file") from e
    # A probe-only invocation always exits non-zero; the banner is on stderr.
    banner = proc.stderr.decode("utf-8", errors="replace")
    if "Invalid data found" in banner or "Stream #" not in banner:
        raise VideoIngestError("The uploaded file is not a readable video or audio file")

    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    for line in banner.splitlines():
        match = _STREAM_RE.search(line)
        if match is None:
            continue
        kind, codec = match.group(1), match.group(2).lower()
        if kind == "Video" and video_codec is None and not _ATTACHED_PIC_RE.search(line):
            video_codec = codec
        elif kind == "Audio" and audio_codec is None:
            audio_codec = codec
    return MediaProbe(video_codec=video_codec, audio_codec=audio_codec)


def _audio_args(probe: MediaProbe) -> list[str]:
    """Copy an already-AAC track; encode anything else (PCM, MP3, ALAC, …) to AAC."""
    if not probe.has_audio:
        return []
    if probe.audio_codec == "aac":
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", "192k"]


def plan_ingest(probe: MediaProbe) -> IngestAction:
    """Decides how ``ingest_media_to_mp4`` will convert a probed upload."""
    if probe.has_video:
        return "remux" if probe.video_codec == "h264" else "transcode"
    if probe.has_audio:
        return "audio_wrap"
    raise VideoIngestError("The uploaded file contains no video or audio stream")


def ingest_media_to_mp4(src: Path, dst: Path, *, max_output_bytes: int) -> IngestAction:
    """Converts ``src`` into a browser-safe H.264/AAC MP4 at ``dst``.

    Returns the action taken. Raises VideoIngestError (user-safe message) on any
    failure; ``dst`` is left absent or partial on failure and the caller owns cleanup.
    """
    probe = probe_media_streams(src)
    action = plan_ingest(probe)

    # `-fs` bounds the output *during* the encode, so a long audio wrap or a
    # high-entropy transcode cannot grow the temp file past the upload cap before the
    # post-hoc check runs. ffmpeg stops once the size is exceeded (the final file lands
    # at or slightly above the cap) and exits 0, so the >= check below is what turns a
    # truncated-at-cap file into an error.
    common_output = ["-fs", str(max_output_bytes), "-movflags", "+faststart", "-f", "mp4", str(dst)]
    if action == "remux":
        args = [
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "copy",
            *_audio_args(probe),
            *common_output,
        ]
    elif action == "transcode":
        args = [
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            # x264 yuv420p needs even dimensions; also folds 10-bit/HDR sources to 8-bit.
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            *_audio_args(probe),
            *common_output,
        ]
    else:  # audio_wrap
        args = [
            "-i",
            str(src),
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x101418:s={AUDIO_WRAP_SIZE}:r={AUDIO_WRAP_FPS}",
            "-filter_complex",
            # Waveform over a dark background. `shortest=1` ends the wrap at the
            # waveform stream's end, i.e. the audio duration — the color source is
            # infinite. aformat guarantees showwaves a stereo layout for odd sources;
            # colors covers both channels (an uncovered channel falls back to ffmpeg's
            # default green) and sqrt scaling keeps quiet recordings visible.
            (
                f"[0:a]aformat=channel_layouts=stereo,"
                f"showwaves=s={AUDIO_WRAP_SIZE}:mode=cline:rate={AUDIO_WRAP_FPS}"
                f":colors=0x4FA8FF|0x2E7CD6:scale=sqrt[w];"
                f"[1:v][w]overlay=shortest=1,format=yuv420p[v]"
            ),
            "-map",
            "[v]",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            *_audio_args(probe),
            *common_output,
        ]

    try:
        proc = subprocess.run(
            [_ffmpeg_exe(), "-y", "-loglevel", "error", *args],
            capture_output=True,
            timeout=INGEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise VideoIngestError("Timed out converting the uploaded file to MP4") from e
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        # The message reaches API clients; server temp paths are not theirs to see.
        stderr = stderr.replace(str(src), src.name).replace(str(dst), dst.name)
        raise VideoIngestError(f"Could not convert the uploaded file to MP4: {stderr[-500:]}")
    try:
        output_bytes = dst.stat().st_size
    except OSError as e:
        raise VideoIngestError("Could not convert the uploaded file to MP4") from e
    if output_bytes >= max_output_bytes:
        raise VideoIngestError("The converted video exceeds the maximum upload size")
    return action
