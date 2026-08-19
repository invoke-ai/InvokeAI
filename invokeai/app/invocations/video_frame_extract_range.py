"""Extract a contiguous range of frames from a video and re-encode as MP4.

Companion to ``video_frame_extract`` (single frame → image) and
``video_concat`` (many videos → one). This node takes a slice of an input
video and emits a new MP4, so the output can be fed straight into
Concatenate Videos to splice clips together — e.g. trim a generated clip
to a usable middle section before chaining it to another shot.

If the source carries an audio track, the matching slice of it is carried
through (and retimed alongside the video when the output fps differs from
the source's). Like ``video_concat``, the audio path buffers the source
soundtrack in full — float32 stereo PCM, ~3 orders of magnitude smaller
per second than decoded frames — and audio failures fail the node loudly
rather than emit a silently-stripped clip.
"""

import tempfile
from itertools import islice
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

import numpy as np
from PIL import Image

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
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_audio import extract_audio_pcm, mux_audio_into_video, resample_linear
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav
from invokeai.app.util.video_thumbnails import decoder_frame_count, iter_video_frames, probe_video


class _FrameWriter(Protocol):
    def append_data(self, frame: np.ndarray) -> None: ...


def _validate_even_dimensions(width: int, height: int, video_name: str) -> None:
    """Raises if the source dimensions can't be encoded as-is with libx264 + yuv420p.

    We encode with ``macro_block_size=1`` to preserve the source dimensions exactly
    (imageio's default of 16 silently rescales), which requires even width and height.
    """
    if width % 2 or height % 2:
        raise ValueError(
            f"Video {video_name} is {width}x{height}; H.264 encoding requires even dimensions. "
            "Re-encode or crop the source to even width and height first."
        )


def _write_frame_range(
    frames: Iterator[np.ndarray],
    writer: _FrameWriter,
    start: int,
    end: int,
    is_canceled: Optional[Callable[[], bool]] = None,
    resize_to: Optional[tuple[int, int]] = None,
) -> int:
    """Streams frames[start..end] (inclusive) from a lazy decoder into the writer.

    Frames are appended one at a time as they stream past — the full range is never
    materialized in RAM. The upload cap admits files whose decoded frames would run to
    tens of gigabytes, so peak memory here must stay one-frame-sized regardless of the
    requested range. Decoding stops as soon as ``end`` has been written. Returns the
    number of frames written.

    ``resize_to`` is a ``(width, height)`` pair; each frame is resampled to it on the way
    past, one frame at a time, so the streaming memory profile is unchanged.
    """
    written = 0
    for idx, frame in enumerate(islice(frames, end + 1)):
        if is_canceled is not None and is_canceled():
            raise CanceledException
        if idx < start:
            continue
        if resize_to is not None and (frame.shape[1], frame.shape[0]) != resize_to:
            frame = np.asarray(Image.fromarray(frame).resize(resize_to, Image.LANCZOS))
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
    version="1.3.0",
    classification=Classification.Prototype,
)
class ExtractVideoRangeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Trim a video to a contiguous frame range and re-encode as MP4.

    Both bounds are inclusive and 0-based — ``start_frame=10, end_frame=50``
    emits 41 frames. Negative indices count from the end (``end_frame=-1``
    is the final frame), matching ``video_frame_extract``. The output frame
    rate defaults to the source video's frame rate; set ``fps=0`` to inherit
    it (or 16 fps if the source rate can't be probed). If the source has an
    audio track, the same range of it is carried into the output (retimed
    with the video when the output fps changes playback speed); silent
    sources stay silent.

    Setting ``width``/``height`` conforms the trimmed clip to that canvas (0 on an
    axis keeps the source's). This is what lets a trimmed source be concatenated
    with a generated clip: Concatenate Videos rejects mismatched dimensions, and
    video models render onto their own fixed canvas family, so the source segment
    has to be brought onto the same canvas first.

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
    width: int = InputField(
        default=0,
        ge=0,
        description="Output width in pixels. 0 = keep the source width. Set both width and height "
        "to conform the trimmed clip to a specific canvas — e.g. to match a generated clip before "
        "concatenating, since Concatenate Videos rejects mismatched dimensions.",
    )
    height: int = InputField(
        default=0,
        ge=0,
        description="Output height in pixels. 0 = keep the source height. Aspect ratio is not "
        "preserved: the frame is resampled to exactly the requested canvas.",
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

        # Conform to the requested canvas when either axis is set; 0 on an axis keeps the
        # source's. Everything downstream (the encoder, the even-dimension check, the saved
        # video record) uses the output dimensions, not the source's.
        out_width = self.width or width
        out_height = self.height or height
        resize_to = (out_width, out_height) if (out_width, out_height) != (width, height) else None

        # libx264 + yuv420p needs even dimensions. Reject odd sizes up front with a clear
        # message instead of surfacing an opaque ffmpeg error mid-encode.
        if resize_to is None:
            _validate_even_dimensions(out_width, out_height, self.video.video_name)
        elif out_width % 2 or out_height % 2:
            raise ValueError(
                f"Requested output size is {out_width}x{out_height}; H.264 encoding requires even "
                "dimensions. Choose an even width and height."
            )

        context.util.signal_progress(f"Transcoding frames {start}-{end} of {n_frames} @ {output_fps:.2f} fps")

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_video_range_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        wav_path: Optional[Path] = None
        muxed_path: Optional[Path] = None
        try:
            # imageio's iter_index isn't exposed by iio.imiter, so we enumerate and skip.
            # Frames stream straight from the decoder into the encoder; see _write_frame_range.
            writer = make_mp4_writer(tmp_path, output_fps)
            try:
                num_frames = _write_frame_range(
                    iter_video_frames(video_path, is_canceled=context.util.is_canceled),
                    writer,
                    start,
                    end,
                    is_canceled=context.util.is_canceled,
                    resize_to=resize_to,
                )
            finally:
                writer.close()

            expected_frames = end - start + 1
            if num_frames != expected_frames:
                raise ValueError(
                    f"Decoded only {num_frames} of {expected_frames} requested frames for range {start}-{end} "
                    f"of {self.video.video_name} "
                    f"(probed {n_frames} frames). The container's metadata may be inaccurate."
                )

            # Carry the source's audio (if any) through: slice the matching span of
            # its soundtrack and map it onto the output timeline. Silent sources
            # keep the old behavior (no audio stream).
            source_path = tmp_path
            audio = self._build_audio_track(
                context=context,
                video_path=video_path,
                start=start,
                end=end,
                n_frames=n_frames,
                source_fps=source_fps,
                source_duration=duration,
                output_fps=output_fps,
            )
            if audio is not None:
                pcm, rate = audio
                context.util.signal_progress("Muxing audio")
                wav_tmp = tempfile.NamedTemporaryFile(prefix="invokeai_video_range_", suffix=".wav", delete=False)
                wav_tmp.close()
                wav_path = Path(wav_tmp.name)
                write_stereo_wav(wav_path, pcm, rate)
                mux_tmp = tempfile.NamedTemporaryFile(prefix="invokeai_video_range_", suffix=".mp4", delete=False)
                mux_tmp.close()
                muxed_path = Path(mux_tmp.name)
                mux_audio_into_video(tmp_path, wav_path, muxed_path)
                source_path = muxed_path

            out_duration = num_frames / output_fps
            context.logger.info(
                f"Encoded trimmed MP4: {num_frames} frames @ {output_fps:.2f} fps "
                f"({out_duration:.2f}s) at {out_width}x{out_height}"
                + (", with AAC audio" if audio is not None else ", silent")
            )
            video_dto = context.videos.save(
                source_path=source_path,
                width=out_width,
                height=out_height,
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
            for cleanup in (tmp_path, wav_path, muxed_path):
                if cleanup is None:
                    continue
                try:
                    cleanup.unlink(missing_ok=True)
                except Exception:
                    pass

    def _build_audio_track(
        self,
        context: InvocationContext,
        video_path: Path,
        start: int,
        end: int,
        n_frames: int,
        source_fps: Optional[float],
        source_duration: Optional[float],
        output_fps: float,
    ) -> Optional[tuple[np.ndarray, int]]:
        """Slice the source soundtrack to frames [start, end] and map it onto the output timeline.

        Returns ``(pcm, sample_rate)`` — ``pcm`` shaped ``(2, n)``, float32 in [-1, 1] —
        or ``None`` when the source has no audio track.

        Frame-to-sample mapping needs the source's effective frame rate. Preference
        order (mirroring ``video_concat``): the probed fps; the probed duration (fps can
        be None for VFR-flagged containers); the extracted length itself, which
        approximates the native span to within the codec padding. The final resample
        maps the sliced window onto the span the trimmed video occupies in the output —
        a no-op at matching fps, a deliberate speed/pitch retime under an fps override
        (matching what the retime does to the video).
        """
        context.util.signal_progress("Extracting audio")
        extracted = extract_audio_pcm(video_path)
        if extracted is None:
            return None
        if context.util.is_canceled():
            raise CanceledException
        pcm, rate = extracted
        if source_fps is not None and source_fps > 0:
            eff_fps = float(source_fps)
        elif source_duration is not None and source_duration > 0:
            eff_fps = n_frames / float(source_duration)
        else:
            # extract_audio_pcm returns None for zero-sample tracks, so pcm is non-empty.
            eff_fps = n_frames * rate / pcm.shape[1]
        window_start = round(start * rate / eff_fps)
        window_end = round((end + 1) * rate / eff_fps)
        window = pcm[:, window_start:window_end]
        if window.shape[1] < window_end - window_start:
            # Audio track shorter than the video (or the range lies past its end):
            # keep temporal alignment by padding the shortfall with silence.
            window = np.pad(window, ((0, 0), (0, (window_end - window_start) - window.shape[1])))
        target = round((end - start + 1) / output_fps * rate)
        return resample_linear(window, target), rate

    @staticmethod
    def _resolve_index(value: int, n_frames: int, field_name: str) -> int:
        resolved = value + n_frames if value < 0 else value
        if resolved < 0 or resolved >= n_frames:
            raise ValueError(f"{field_name}={value} is out of range for a {n_frames}-frame video.")
        return resolved
