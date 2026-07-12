"""Extract a contiguous range of frames from a video and re-encode as MP4.

Companion to ``video_frame_extract`` (single frame → image) and
``video_concat`` (many videos → one). This node takes a slice of an input
video and emits a new MP4, so the output can be fed straight into
Concatenate Videos to splice clips together — e.g. trim a generated clip
to a usable middle section before chaining it to another shot.
"""

import tempfile
from pathlib import Path
from typing import Iterator, Protocol

import imageio.v2 as iio2
import imageio.v3 as iio
import numpy as np

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    InputField,
    OutputField,
    UIComponent,
    VideoField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_thumbnails import decoder_frame_count, probe_video


class _FrameWriter(Protocol):
    def append_data(self, frame: np.ndarray) -> None: ...


def _write_frame_range(frames: Iterator[np.ndarray], writer: _FrameWriter, start: int, end: int) -> int:
    """Streams frames[start..end] (inclusive) from a lazy decoder into the writer.

    Frames are appended one at a time as they stream past — the full range is never
    materialized in RAM. The upload cap admits files whose decoded frames would run to
    tens of gigabytes, so peak memory here must stay one-frame-sized regardless of the
    requested range. Decoding stops as soon as ``end`` has been written. Returns the
    number of frames written.
    """
    written = 0
    for idx, frame in enumerate(frames):
        if idx < start:
            continue
        if idx > end:
            break
        writer.append_data(np.ascontiguousarray(frame))
        written += 1
    return written


@invocation_output("extract_video_range_output")
class ExtractVideoRangeOutput(BaseInvocationOutput):
    """Output of ``extract_video_range``: a trimmed video plus the resolved frame indices.

    Mirrors ``VideoOutput`` so the video can be piped directly into Concatenate Videos or
    any other ``VideoField``-consuming node, and additionally exposes the resolved
    (positive, clamped) start and end indices so chained workflows can feed them back in
    — e.g. drive a downstream Frame from Video to pull the same boundary frame.
    """

    video: VideoField = OutputField(description="The trimmed video")
    width: int = OutputField(description="The width of the video in pixels")
    height: int = OutputField(description="The height of the video in pixels")
    num_frames: int = OutputField(description="The number of frames in the trimmed video")
    fps: float = OutputField(description="The frames-per-second of the trimmed video")
    duration: float = OutputField(description="The duration of the trimmed video in seconds")
    start_frame: int = OutputField(description="The resolved (positive, 0-based) start frame index in the source video")
    end_frame: int = OutputField(description="The resolved (positive, 0-based) end frame index in the source video")


@invocation(
    "extract_video_range",
    title="Frame Range from Video",
    tags=["video", "trim", "range", "frames"],
    category="video",
    version="1.1.0",
    classification=Classification.Prototype,
)
class ExtractVideoRangeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Trim a video to a contiguous frame range and re-encode as MP4.

    Both bounds are inclusive and 0-based — ``start_frame=10, end_frame=50``
    emits 41 frames. Negative indices count from the end (``end_frame=-1``
    is the final frame), matching ``video_frame_extract``. The output frame
    rate defaults to the source video's frame rate; set ``fps=0`` to inherit
    it (or 16 fps if the source rate can't be probed).

    The resolved (positive) ``start_frame`` and ``end_frame`` are also emitted as
    outputs, so chained workflows can re-use the boundary indices — e.g. feeding
    them into a downstream Frame from Video to extract the same boundary frame.
    """

    video: VideoField = InputField(description="The video to extract a frame range from.")
    start_frame: int = InputField(
        default=0,
        description=("First frame to keep, inclusive. 0 = first frame. Negative indices count from the end."),
        ui_component=UIComponent.VideoFrameIndex,
    )
    end_frame: int = InputField(
        default=-1,
        description=("Last frame to keep, inclusive. -1 = last frame. Negative indices count from the end."),
        ui_component=UIComponent.VideoFrameIndex,
    )
    fps: int = InputField(
        default=0,
        ge=0,
        le=120,
        description="Output frame rate. 0 = match the source video's frame rate "
        "(falls back to 16 fps if the source rate can't be probed).",
    )

    def invoke(self, context: InvocationContext) -> ExtractVideoRangeOutput:
        video_path = context.videos.get_path(self.video.video_name)
        width, height, duration, source_fps = probe_video(video_path)

        n_frames = decoder_frame_count(video_path)
        if n_frames is None:
            if not source_fps or duration <= 0:
                raise ValueError(
                    f"Cannot determine frame count for {self.video.video_name}: "
                    f"probe returned duration={duration}, fps={source_fps}."
                )
            n_frames = int(round(duration * source_fps))
        if n_frames <= 0:
            raise ValueError(f"Video {self.video.video_name} has no decodable frames (probed {n_frames}).")

        start = self._resolve_index(self.start_frame, n_frames, "start_frame")
        end = self._resolve_index(self.end_frame, n_frames, "end_frame")
        if end < start:
            raise ValueError(
                f"end_frame ({self.end_frame} → {end}) must be >= start_frame "
                f"({self.start_frame} → {start}) after resolving negative indices."
            )

        # Derive the output frame rate from the source video when ``fps`` is 0
        # (the default), so a trimmed clip plays back at the same speed as its
        # source. Fall back to 16 fps (the Wan video default) when the source
        # rate couldn't be probed.
        if self.fps > 0:
            output_fps = float(self.fps)
        elif source_fps and source_fps > 0:
            output_fps = float(source_fps)
        else:
            output_fps = 16.0

        context.util.signal_progress(f"Transcoding frames {start}-{end} of {n_frames} @ {output_fps:.2f} fps")

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_video_range_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            # imageio's iter_index isn't exposed by iio.imiter, so we enumerate and skip.
            # Frames stream straight from the decoder into the encoder; see _write_frame_range.
            writer = iio2.get_writer(str(tmp_path), format="FFMPEG", mode="I", fps=output_fps, codec="libx264")
            try:
                num_frames = _write_frame_range(iio.imiter(video_path, plugin="FFMPEG"), writer, start, end)
            finally:
                writer.close()

            if num_frames == 0:
                raise ValueError(
                    f"Decoded zero frames for range {start}-{end} of {self.video.video_name} "
                    f"(probed {n_frames} frames). The container's metadata may be inaccurate."
                )

            out_duration = num_frames / output_fps
            context.logger.info(
                f"Encoded trimmed MP4: {num_frames} frames @ {output_fps:.2f} fps "
                f"({out_duration:.2f}s) at {width}x{height}"
            )
            video_dto = context.videos.save(
                source_path=tmp_path,
                width=width,
                height=height,
                duration=out_duration,
                fps=output_fps,
            )
            context.logger.info(f"Saved trimmed video: {video_dto.video_name}")
            base = VideoOutput.build(video_dto)
            return ExtractVideoRangeOutput(
                video=base.video,
                width=base.width,
                height=base.height,
                num_frames=base.num_frames,
                fps=base.fps,
                duration=base.duration,
                start_frame=start,
                end_frame=end,
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _resolve_index(value: int, n_frames: int, field_name: str) -> int:
        resolved = value + n_frames if value < 0 else value
        if resolved < 0 or resolved >= n_frames:
            raise ValueError(f"{field_name}={value} is out of range for a {n_frames}-frame video.")
        return resolved
