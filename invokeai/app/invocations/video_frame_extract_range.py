"""Extract a contiguous range of frames from a video and re-encode as MP4.

Companion to ``video_frame_extract`` (single frame → image) and
``video_concat`` (many videos → one). This node takes a slice of an input
video and emits a new MP4, so the output can be fed straight into
Concatenate Videos to splice clips together — e.g. trim a generated clip
to a usable middle section before chaining it to another shot.

Frames are re-encoded only where they have to be. A range covering the whole
source at its own frame rate is a copy of the source file, so passing a clip
through this node (as video extension does with an untrimmed source) costs no
quality. A trim of a source this app encoded stream-copies the keyframe-aligned
run of frames inside the range and encodes only the partial keyframe intervals
at either end (see ``h264_splice``), and likewise copies the source's AAC
packets for the audio inside the range, encoding only a short lead-in and tail
(see ``aac_splice``). Anything else is re-encoded in full.

If the source carries an audio track, the matching slice of it is carried
through (and retimed alongside the video when the output fps differs from
the source's). Like ``video_concat``, the audio path buffers the source
soundtrack in full — float32 stereo PCM, ~3 orders of magnitude smaller
per second than decoded frames — and audio failures fail the node loudly
rather than emit a silently-stripped clip.
"""

import math
import tempfile
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

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
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.aac_splice import (
    AAC_FRAME_SAMPLES,
    EDGE_FRAMES,
    splice_soundtrack,
    spliceable_aac_layout,
)
from invokeai.app.util.h264_splice import (
    MIN_ENCODED_FRAMES,
    H264Layout,
    SpliceError,
    concat_stream_copy,
    copy_frames,
    probes_as,
    read_h264_layout,
)
from invokeai.app.util.mp4_metadata import read_mp4_movie_duration, write_mp4_tags
from invokeai.app.util.video_audio import extract_audio_pcm, mux_audio_into_video, resample_linear
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav, writer_sample_entry
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
) -> int:
    """Streams frames[start..end] (inclusive) from a lazy decoder into the writer.

    Frames are appended one at a time as they stream past — the full range is never
    materialized in RAM. The upload cap admits files whose decoded frames would run to
    tens of gigabytes, so peak memory here must stay one-frame-sized regardless of the
    requested range. Decoding stops as soon as ``end`` has been written. Returns the
    number of frames written.
    """
    written = 0
    for idx, frame in enumerate(islice(frames, end + 1)):
        if is_canceled is not None and is_canceled():
            raise CanceledException
        if idx < start:
            continue
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
    # Echoed back so a graph can record what the user actually picked: the `video` output is a
    # freshly-encoded intermediate, so naming it as a generation's provenance would point at a
    # clip that never appears in the gallery.
    source_video: VideoField = OutputField(description="The video the range was taken from", title="Source Video")
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

    The resolved (positive) ``start_frame`` and ``end_frame`` are also emitted as
    outputs, so chained workflows can re-use the boundary indices — e.g. feeding
    them into a downstream Frame from Video to extract the same boundary frame, and the
    input video is echoed back on ``source_video`` so a graph can record the clip the user
    chose rather than the freshly-encoded trim.
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

        exact_frame_count = decoder_frame_count(video_path)
        n_frames = exact_frame_count
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

        # libx264 + yuv420p needs even dimensions. Reject odd sources up front with a
        # clear message instead of surfacing an opaque ffmpeg error mid-encode.
        _validate_even_dimensions(width, height, self.video.video_name)

        # Frames can be copied rather than re-encoded only at the source's own rate (a stream copy
        # cannot retime), and only from a track whose sample indices are the decoder's frame indices:
        # read_h264_layout refuses edit-list trims and rotations, and the count must agree.
        layout = None
        if source_fps is not None and math.isclose(output_fps, source_fps, rel_tol=1e-6):
            layout = read_h264_layout(video_path)
            if layout is not None and layout.sample_count != exact_frame_count:
                layout = None
        # The whole clip is the source itself; copying it avoids another lossy generation. The
        # OpenCV probe fallback knows only the video duration, so also require the MP4 movie header
        # to fit the selected frames before bypassing the ordinary path that trims the soundtrack.
        whole_range = layout is not None and start == 0 and end == layout.sample_count - 1
        movie_duration = read_mp4_movie_duration(video_path) if whole_range else None
        if (
            whole_range
            and movie_duration is not None
            and max(duration, movie_duration) <= n_frames / output_fps + 0.5 / output_fps + 0.01
        ):
            return self._save_unchanged(context, video_path, width, height, n_frames, output_fps)

        context.util.signal_progress(f"Transcoding frames {start}-{end} of {n_frames} @ {output_fps:.2f} fps")

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_video_range_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        wav_path: Optional[Path] = None
        muxed_path: Optional[Path] = None
        try:
            num_frames: Optional[int] = None
            copy_span = self._plan_copy_span(layout, start, end, width, height, output_fps) if layout else None
            if copy_span is not None:
                num_frames = self._write_spliced_range(
                    context, video_path, start, end, *copy_span, output_fps, tmp_path
                )
            if num_frames is None:
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
                mux_tmp = tempfile.NamedTemporaryFile(prefix="invokeai_video_range_", suffix=".mp4", delete=False)
                mux_tmp.close()
                muxed_path = Path(mux_tmp.name)
                if not self._mux_spliced_audio(
                    context, video_path, start, end, source_fps, output_fps, pcm, rate, tmp_path, muxed_path
                ):
                    wav_tmp = tempfile.NamedTemporaryFile(prefix="invokeai_video_range_", suffix=".wav", delete=False)
                    wav_tmp.close()
                    wav_path = Path(wav_tmp.name)
                    write_stereo_wav(wav_path, pcm, rate)
                    mux_audio_into_video(tmp_path, wav_path, muxed_path, context.util.is_canceled)
                source_path = muxed_path

            out_duration = num_frames / output_fps
            context.logger.info(
                f"Encoded trimmed MP4: {num_frames} frames @ {output_fps:.2f} fps "
                f"({out_duration:.2f}s) at {width}x{height}" + (", with AAC audio" if audio is not None else ", silent")
            )
            # A cancellation that arrived during an uninterruptible stage must not be saved.
            if context.util.is_canceled():
                raise CanceledException
            video_dto = context.videos.save(
                source_path=source_path,
                width=width,
                height=height,
                duration=out_duration,
                fps=output_fps,
            )
            context.logger.info(f"Saved trimmed video: {video_dto.video_name}")
            base = VideoOutput.build(video_dto)
            return ExtractVideoRangeOutput(
                video=base.video,
                source_video=self.video,
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

    @staticmethod
    def _plan_copy_span(
        layout: H264Layout, start: int, end: int, width: int, height: int, fps: float
    ) -> Optional[tuple[int, int, H264Layout]]:
        """The keyframe-aligned frames ``[k1, k2)`` inside ``[start, end]`` that can be
        stream-copied rather than re-encoded, with the source's layout; None if there are none.

        ``k1`` is the first keyframe at or after ``start``, ``k2`` the last keyframe at or
        before ``end + 1`` — or the end of the track when the range runs to the last frame —
        each moved one keyframe inward when it would leave an encoded end shorter than
        ``MIN_ENCODED_FRAMES``. The source must also carry the decoder configuration this
        node's encoder writes and a constant frame duration, or the re-encoded ends could not
        share its track.
        """
        if layout.frame_duration is None or layout.sample_entry != writer_sample_entry(width, height, fps):
            return None
        boundaries = [*layout.sync_samples, layout.sample_count]
        inner_starts = [k for k in layout.sync_samples if k >= start and not 0 < k - start < MIN_ENCODED_FRAMES]
        inner_ends = [k for k in boundaries if k <= end + 1 and not 0 < end + 1 - k < MIN_ENCODED_FRAMES]
        if not inner_starts or not inner_ends or inner_starts[0] >= inner_ends[-1]:
            return None
        return inner_starts[0], inner_ends[-1], layout

    def _write_spliced_range(
        self,
        context: InvocationContext,
        video_path: Path,
        start: int,
        end: int,
        k1: int,
        k2: int,
        layout: H264Layout,
        output_fps: float,
        out_path: Path,
    ) -> Optional[int]:
        """Write frames ``[start, end]`` to ``out_path``, stream-copying ``[k1, k2)``.

        Only the partial GOPs on either side — ``[start, k1)`` and ``[k2, end]`` — are
        encoded. Returns the number of frames written, which the caller checks against the
        range like a full encode's, or None (leaving the caller to re-encode the whole range)
        when an encoded part's decoder configuration differs from the source's, so the parts
        could not share one track.
        """
        with tempfile.TemporaryDirectory(prefix="invokeai_video_range_") as work:
            work_dir = Path(work)
            head = (start, k1, work_dir / "head.mp4")
            tail = (k2, end + 1, work_dir / "tail.mp4")
            encoded = [span for span in (head, tail) if span[0] < span[1]]
            writers = [make_mp4_writer(path, output_fps) for _, _, path in encoded]
            written = 0
            # Closed explicitly: an open decoder holds the process-wide stream slot, which the
            # copy and join below (and a fallback re-encode) must not wait behind.
            with closing(iter_video_frames(video_path, is_canceled=context.util.is_canceled)) as frames:
                try:
                    for idx, frame in enumerate(islice(frames, end + 1)):
                        if context.util.is_canceled():
                            raise CanceledException
                        for (lo, hi, _), writer in zip(encoded, writers, strict=True):
                            if lo <= idx < hi:
                                writer.append_data(np.ascontiguousarray(frame))
                                written += 1
                finally:
                    for writer in writers:
                        writer.close()

            middle_path = work_dir / "middle.mp4"
            reason: Optional[str] = None
            if any(
                (part := read_h264_layout(path)) is None or part.sample_entry != layout.sample_entry
                for _, _, path in encoded
            ):
                reason = "its encoding differs from this node's"
            else:
                try:
                    copy_frames(video_path, k1, k2, layout.sample_count, middle_path, context.util.is_canceled)
                    middle = read_h264_layout(middle_path)
                    if middle is None or middle.sample_count != k2 - k1 or middle.sample_entry != layout.sample_entry:
                        reason = "copying its frames did not yield the expected count"
                    else:
                        parts = [head[2]] if head in encoded else []
                        parts += [middle_path] + ([tail[2]] if tail in encoded else [])
                        concat_stream_copy(parts, out_path, context.util.is_canceled)
                        joined = read_h264_layout(out_path)
                        if (
                            joined is None
                            or joined.sample_entry != layout.sample_entry
                            or joined.sample_count != written + (k2 - k1)
                            or joined.frame_duration != layout.frame_duration
                            or not probes_as(out_path, output_fps, joined.sample_count)
                        ):
                            reason = "the joined file does not hold the expected frames on one timeline"
                except SpliceError as e:
                    reason = str(e)
            if reason is not None:
                context.logger.info(f"Re-encoding the whole range: the source cannot be stream-copied ({reason})")
                return None
        context.logger.info(f"Stream-copied frames {k1}-{k2 - 1} of the source without re-encoding")
        return written + (k2 - k1)

    def _save_unchanged(
        self,
        context: InvocationContext,
        video_path: Path,
        width: int,
        height: int,
        n_frames: int,
        fps: float,
    ) -> ExtractVideoRangeOutput:
        """Save a copy of the source as the output: the requested range is all of it."""
        with tempfile.TemporaryDirectory(prefix="invokeai_video_range_") as work:
            copy_path = Path(work) / "source.mp4"
            # A stream-copy remux that drops the source's keyed metadata: save() writes this
            # node's record, and if that rewrite fails the source's must not be read back as it.
            # Only the first soundtrack: the ordinary path carries no other.
            write_mp4_tags(video_path, copy_path, {}, context.util.is_canceled, first_audio_only=True)
            # A cancellation that arrived during an uninterruptible stage must not be saved.
            if context.util.is_canceled():
                raise CanceledException
            video_dto = context.videos.save(
                source_path=copy_path,
                width=width,
                height=height,
                duration=n_frames / fps,
                fps=fps,
            )
        context.logger.info(f"Saved {video_dto.video_name}: the whole of the source, copied without re-encoding")
        base = VideoOutput.build(video_dto)
        return ExtractVideoRangeOutput(
            video=base.video,
            source_video=self.video,
            width=base.width,
            height=base.height,
            num_frames=base.num_frames,
            fps=base.fps,
            duration=base.duration,
            start_frame=0,
            end_frame=n_frames - 1,
        )

    def _mux_spliced_audio(
        self,
        context: InvocationContext,
        video_path: Path,
        start: int,
        end: int,
        source_fps: Optional[float],
        output_fps: float,
        pcm: np.ndarray,
        rate: int,
        trimmed_video: Path,
        out_path: Path,
    ) -> bool:
        """Mux ``pcm`` onto ``trimmed_video``, stream-copying the source's own AAC packets
        for the part of the range they hold whole (see ``aac_splice``).

        ``pcm`` is the source's decoded audio over the range, sample for sample, when the
        range plays at the source's own rate and its span maps one-to-one onto the output
        timeline; the source's packets are then copied for every output frame that lies
        inside both the range and the source's real audio, and only the partial frames at
        each end are encoded. Returns False, leaving the caller to encode the whole
        soundtrack, when the source's track is not one this app's encoder writes, the range
        is too short, or the splice does not verify.
        """
        if source_fps is None or source_fps != output_fps:
            return False
        window_start = round(start * rate / source_fps)
        window_end = round((end + 1) * rate / source_fps)
        if window_end - window_start != pcm.shape[1]:
            return False
        layout = spliceable_aac_layout(video_path, rate)
        if layout is None:
            return False
        frame, priming = AAC_FRAME_SAMPLES, layout.priming
        # Packet k's output frame is decoded samples [k * frame - priming, (k + 1) * frame - priming).
        if window_start == 0:
            first, copied_at = 0, -priming
        else:
            # The first frame wholly inside the range that leaves a full lead-in before it.
            first = -(-(window_start + priming + EDGE_FRAMES * frame) // frame)
            copied_at = first * frame - priming - window_start
        # The last packet that leaves a full encoded tail after the run (see aac_splice); past the
        # source's own audio the tail is the silence the timeline was padded with.
        end_packet = min(layout.packet_count, (window_end - EDGE_FRAMES * frame + priming) // frame)
        reason = splice_soundtrack(
            trimmed_video,
            video_path,
            priming,
            first,
            end_packet,
            copied_at,
            pcm,
            rate,
            out_path,
            context.util.is_canceled,
        )
        if reason is not None:
            context.logger.info(f"Re-encoding the whole soundtrack: {reason}")
            return False
        context.logger.info(f"Stream-copied audio packets {first}-{end_packet - 1} of the source without re-encoding")
        return True

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
        extracted = extract_audio_pcm(video_path, is_canceled=context.util.is_canceled)
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
        # At the source's own rate the window maps sample for sample; rounding the span on its
        # own could come out one off the window and interpolate the whole of it.
        target = window.shape[1] if output_fps == eff_fps else round((end - start + 1) / output_fps * rate)
        return resample_linear(window, target), rate

    @staticmethod
    def _resolve_index(value: int, n_frames: int, field_name: str) -> int:
        resolved = value + n_frames if value < 0 else value
        if resolved < 0 or resolved >= n_frames:
            raise ValueError(f"{field_name}={value} is out of range for a {n_frames}-frame video.")
        return resolved
