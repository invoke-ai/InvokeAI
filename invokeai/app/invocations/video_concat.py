"""Concatenate two or more videos with an optional transition.

Pairs naturally with the I2V chaining workflow: feed several Wan-generated
clips into this node to glue them into one longer video. The transition
options hide the seam between independently-denoised clips.

The full-encode fallback uses the existing imageio frame pipeline; smart bridges stream
RGB frames to the bundled FFmpeg encoder. Frames stream one at a time, buffering only
the transition windows, so peak memory stays
O(transition_frames) even when the inputs are long uploads (the upload cap
admits files whose decoded frames would run to tens of gigabytes).

When ``smart_render`` is enabled, compatible clips keep their compressed H.264
samples outside transition bridges. Each bridge spans from a verified IDR boundary in
the outgoing clip to one in the incoming clip, and the complete timeline is assembled
from copied samples and continuously encoded bridges (see ``video_smart_concat`` and
``h264_splice``). Unsupported streams fall back to the full-encoding path. The
soundtrack is treated as before: compatible AAC packets from the first clip are kept
up to its outgoing transition and only the rest is encoded (see ``aac_splice``).

The AUDIO path is a deliberate exception to that bound: soundtracks buffer in
full (float32 stereo PCM of the emitted timeline, roughly 23 MB per minute at
48 kHz). Audio is ~3 orders of magnitude smaller per second than decoded
frames, so even upload-cap-length inputs stay within a few hundred MB — but it
is O(total duration), not O(transition_frames). Audio problems fail the node
loudly rather than emit another silent video: a silent output masquerading as
success is the exact bug this path exists to fix.
"""

import math
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Optional

import numpy as np
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    InputField,
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
    PREROLL_FRAMES,
    splice_soundtrack,
    spliceable_aac_layout,
)
from invokeai.app.util.h264_splice import (
    SpliceError,
    concat_stream_copy,
    copy_frames,
    probes_as,
    read_h264_layout,
    run_ffmpeg,
)
from invokeai.app.util.video_audio import (
    extract_audio_pcm,
    mux_audio_into_video,
    resample_linear,
)
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav
from invokeai.app.util.video_smart_concat import (
    CopySegment,
    SmartRenderPlan,
    encode_rgb_frames,
    try_build_smart_render_plan,
)
from invokeai.app.util.video_thumbnails import iter_video_frames, probe_video

TransitionMode = Literal["cut", "crossfade", "fade_through_black"]
SizeMismatchMode = Literal["error", "match_first"]


def _iter_frames_on_canvas(frames: Iterator[np.ndarray], canvas: tuple[int, int]) -> Iterator[np.ndarray]:
    """Yields each frame at ``canvas`` (a ``(width, height)`` pair) as it streams past.

    A frame already on the canvas is passed through untouched, so this is free for the
    common case where every clip matches. Resizing happens one frame at a time, so the
    O(transition_frames) memory bound of the join is preserved.
    """
    for frame in frames:
        if (frame.shape[1], frame.shape[0]) == canvas:
            yield frame
        else:
            yield np.asarray(Image.fromarray(frame).resize(canvas, Image.LANCZOS))


MAX_TRANSITION_MEMORY_BYTES = 512 * 1024 * 1024
_BLEND_WORKING_FRAMES = 13


def _crossfade(a_tail: list[np.ndarray], b_head: list[np.ndarray]) -> Iterator[np.ndarray]:
    """Yields a linear A to B cross-dissolve without retaining the blended frames."""
    n = len(a_tail)
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        blended = a_tail[i].astype(np.float32) * (1.0 - alpha) + b_head[i].astype(np.float32) * alpha
        yield np.clip(blended, 0, 255).astype(np.uint8)


def _fade_through_black(a_tail: list[np.ndarray], b_head: list[np.ndarray]) -> Iterator[np.ndarray]:
    """A fades to black, then black fades to B. Consumes N/2 frames from each side and returns N output frames.

    Asymmetric framing: the first ``len(a_tail)`` output frames are the trailing A frames scaled
    toward zero brightness; the next ``len(b_head)`` are the leading B frames scaled up from zero.
    """
    n_a = len(a_tail)
    for i, fa in enumerate(a_tail):
        # 1.0 at i=0 (fully visible) → near 0 at i=n_a-1 (essentially black).
        alpha = 1.0 - (i + 1) / (n_a + 1)
        yield np.clip(fa.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    n_b = len(b_head)
    for j, fb in enumerate(b_head):
        alpha = (j + 1) / (n_b + 1)
        yield np.clip(fb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


@invocation(
    "video_concat",
    title="Concatenate Videos",
    tags=["video", "concat", "transition"],
    category="video",
    version="1.3.0",
    classification=Classification.Prototype,
)
class VideoConcatInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Join two or more videos into a single MP4.

    Audio: if any input carries an audio track, the output gets an AAC track assembled on
    the emitted timeline — silent inputs contribute silence, `cut` splices the tracks,
    `crossfade` blends them equal-power over the transition window, and
    `fade_through_black` fades the outgoing track to silence and the incoming one up from
    it, mirroring the video. An fps override retimes audio with the video (a deliberate
    speed/pitch change). If no input has audio, the output has no audio stream.

    Transitions:

    * ``cut`` — hard splice, no blending. Fastest; total length is the sum of inputs.
    * ``crossfade`` — linear A→B cross-dissolve over ``transition_frames``. Each boundary
      consumes ``transition_frames`` from both adjacent clips, so total length is
      ``sum(inputs) - transition_frames * (n - 1)``.
    * ``fade_through_black`` — A fades to black, then B fades in from black. Each boundary
      consumes ``transition_frames // 2`` frames from the preceding clip's tail and the
      remainder (``transition_frames - transition_frames // 2``) from the next clip's head,
      so the total emitted is exactly ``transition_frames`` per boundary — even for odd
      ``transition_frames`` — and the overall length equals the sum of inputs.

    Inputs must share the same pixel dimensions. ``size_mismatch`` decides what happens
    when they do not: ``error`` (the default) refuses the join, and ``match_first``
    resamples every later clip onto the first clip's canvas. ``match_first`` is what lets a
    generated clip be appended to source footage — video models render onto their own fixed
    canvas family, so a generated continuation rarely matches the clip it extends.

    Output frame rate defaults to the first input's fps; override with ``fps`` to force a
    specific rate (the frames are not resampled in time, only the container is encoded at
    the new rate).
    """

    videos: list[VideoField] = InputField(
        min_length=2,
        description="Videos to concatenate, in order. At least two are required.",
    )
    transition: TransitionMode = InputField(
        default="cut",
        description="Transition between consecutive clips.",
    )
    transition_frames: int = InputField(
        default=8,
        ge=0,
        le=240,
        description="Length of each transition in frames. Ignored when transition is 'cut'.",
    )
    fps: Optional[int] = InputField(
        default=None,
        ge=1,
        le=120,
        description="Output frame rate. Defaults to the first input's fps.",
    )
    size_mismatch: SizeMismatchMode = InputField(
        default="error",
        description=(
            "What to do when the inputs do not all share the same pixel dimensions. 'error' "
            "refuses the join; 'match_first' resamples every later clip onto the first clip's "
            "canvas, so a generated clip can be appended to source footage it does not match."
        ),
    )

    smart_render: bool = InputField(
        default=True,
        description=(
            "Preserve original video encoding outside transitions when possible. Automatically falls back to full "
            "encoding for incompatible inputs."
        ),
    )

    def invoke(self, context: InvocationContext) -> VideoOutput:
        if len(self.videos) < 2:
            raise ValueError("video_concat requires at least two input videos.")

        paths: list[Path] = [context.videos.get_path(v.video_name) for v in self.videos]

        # Probe inputs up front: settle the output canvas and pick the default output fps.
        probes = [probe_video(p) for p in paths]
        width, height, _, _first_fps = probes[0]
        sizes = {(w, h) for (w, h, _, _) in probes}
        if len(sizes) > 1 and self.size_mismatch == "error":
            raise ValueError(
                f"All inputs must share the same dimensions. Got: {sorted(sizes)}. Re-render at a "
                "single resolution before concatenating, or set Size Mismatch to 'match_first' to "
                "resample the later clips onto the first clip's canvas."
            )
        # libx264 + yuv420p needs even dimensions; we encode with macro_block_size=1 to
        # preserve the source dimensions exactly, so reject odd sources with a clear error.
        if width % 2 or height % 2:
            raise ValueError(
                f"The output canvas is {width}x{height}, taken from the first input; H.264 encoding "
                "requires even dimensions. Re-encode or crop that clip to an even width and height "
                "first."
            )
        self._validate_transition_memory(width, height)
        output_fps = self._resolve_output_fps([probe[3] for probe in probes])
        mismatched = sum(1 for (w, h, _, _) in probes if (w, h) != (width, height))
        if mismatched:
            context.logger.info(
                f"Resampling {mismatched} of {len(paths)} clip(s) onto the first clip's {width}x{height} canvas"
            )

        context.util.signal_progress(f"Joining {len(self.videos)} clip(s) ({self.transition}) @ {output_fps:.2f} fps")

        with tempfile.TemporaryDirectory(prefix="invokeai_video_concat_") as work:
            work_dir = Path(work)
            tmp_path = work_dir / "joined.mp4"
            joined: Optional[tuple[list[int], int]] = None
            fallback_output_fps = output_fps
            fallback_probes = probes.copy()
            if self.smart_render:
                if context.util.is_canceled():
                    raise CanceledException
                tail_need, head_need = self._transition_windows()
                plan, reason = try_build_smart_render_plan(
                    paths=paths,
                    probes=probes,
                    canvas=(width, height),
                    output_fps=output_fps,
                    fps_is_override=self.fps is not None,
                    transition=self.transition,
                    transition_frames=self.transition_frames,
                    tail_need=tail_need,
                    head_need=head_need,
                    is_canceled=context.util.is_canceled,
                )
                if plan is not None:
                    output_fps = plan.output_fps
                    probes = [
                        (width, height, duration, layout.time_scale / layout.frame_duration)
                        for layout, (width, height, duration, _probe_fps) in zip(plan.layouts, probes, strict=True)
                    ]
                    context.logger.info(
                        f"Smart rendering selected: {plan.output_frames} output frames, "
                        f"copying {plan.copied_frames}, encoding {plan.encoded_frames} across "
                        f"{plan.bridge_count} bridge(s)"
                    )
                    context.util.signal_progress(
                        f"Smart rendering: copying {plan.copied_frames} frames, "
                        f"encoding {plan.encoded_frames} in {plan.bridge_count} bridge(s)"
                    )
                    smart_path = work_dir / "smart_joined.mp4"
                    try:
                        self._execute_smart_render(
                            context, paths, (width, height), output_fps, plan, work_dir, smart_path
                        )
                        joined = (list(plan.frame_counts), plan.output_frames)
                        tmp_path = smart_path
                    except CanceledException:
                        raise
                    except (SpliceError, TimeoutError, RuntimeError) as e:
                        if (
                            isinstance(e, RuntimeError)
                            and not isinstance(e, SpliceError)
                            and (type(e) is not RuntimeError or str(e) != "Unable to open video decoder output stream")
                        ):
                            raise
                        if context.util.is_canceled():
                            raise CanceledException from e
                        output_fps = fallback_output_fps
                        probes = fallback_probes
                        smart_path.unlink(missing_ok=True)
                        reason = str(e)
                        context.logger.info(f"Smart rendering failed; falling back to full video encoding: {reason}")
                        context.util.signal_progress(f"Smart rendering failed; using full video encoding: {reason}")
                else:
                    context.logger.info(f"Smart rendering unavailable; falling back to full video encoding: {reason}")
                    context.util.signal_progress(f"Smart rendering unavailable; using full video encoding: {reason}")
            else:
                context.logger.info("Smart rendering disabled; using full video encoding")
            if joined is None:
                joined = self._encode_joined(context, paths, (width, height), output_fps, tmp_path)
            frame_counts, num_frames = joined
            return self._save_joined(context, paths, probes, tmp_path, work_dir, frame_counts, num_frames, output_fps)

    def _save_joined(
        self,
        context: InvocationContext,
        paths: list[Path],
        probes: list[tuple[int, int, Optional[float], Optional[float]]],
        tmp_path: Path,
        work_dir: Path,
        frame_counts: list[int],
        num_frames: int,
        output_fps: float,
    ) -> VideoOutput:
        width, height = probes[0][0], probes[0][1]
        if num_frames == 0:
            raise ValueError("Concatenation produced zero output frames.")

        # Rebuild the soundtrack on the emitted timeline. Inputs without an audio
        # track (e.g. Wan clips) contribute silence; if none carries audio, the
        # output has no audio stream, as before.
        source_path = tmp_path
        audio = self._build_audio_track(
            context=context,
            paths=paths,
            frame_counts=frame_counts,
            native_rates=[probe[3] for probe in probes],
            native_durations=[probe[2] for probe in probes],
            output_fps=output_fps,
            num_output_frames=num_frames,
        )
        if audio is not None:
            pcm, rate = audio
            context.util.signal_progress("Muxing audio")
            muxed_path = work_dir / "muxed.mp4"
            if not self._mux_spliced_audio(
                context, paths[0], probes[0][3], frame_counts[0], pcm, rate, output_fps, tmp_path, muxed_path
            ):
                wav_path = work_dir / "audio.wav"
                write_stereo_wav(wav_path, pcm, rate)
                mux_audio_into_video(tmp_path, wav_path, muxed_path, context.util.is_canceled)
            source_path = muxed_path

        duration = num_frames / output_fps
        context.logger.info(
            f"Encoded concatenated MP4: {num_frames} frames @ {output_fps:.2f} fps "
            f"({duration:.2f}s) at {width}x{height}" + (", with AAC audio" if audio is not None else ", silent")
        )
        # Splice stages run uninterrupted; a cancellation that arrived meanwhile must not be saved.
        if context.util.is_canceled():
            raise CanceledException
        video_dto = context.videos.save(
            source_path=source_path,
            width=width,
            height=height,
            duration=duration,
            fps=output_fps,
        )
        context.logger.info(f"Saved concatenated video: {video_dto.video_name}")
        return VideoOutput.build(video_dto)

    def _transition_windows(self) -> tuple[int, int]:
        """``(tail_need, head_need)``: frames each boundary takes from the outgoing clip's
        tail and the incoming clip's head. The class docstring describes the layout."""
        if self.transition == "crossfade" and self.transition_frames > 0:
            return self.transition_frames, self.transition_frames
        if self.transition == "fade_through_black" and self.transition_frames > 0:
            tail_need = self.transition_frames // 2
            return tail_need, self.transition_frames - tail_need
        return 0, 0

    def _execute_smart_render(
        self,
        context: InvocationContext,
        paths: list[Path],
        canvas: tuple[int, int],
        output_fps: float,
        plan: SmartRenderPlan,
        work_dir: Path,
        out_path: Path,
    ) -> None:
        """Build validated copy and bridge parts, then join them on one MP4 timeline."""
        parts: list[Path] = []
        part_counts: list[int] = []
        target_entry = plan.layouts[0].sample_entry
        bridge_index = 0
        copy_success_logs: list[str] = []
        for segment_index, segment in enumerate(plan.segments):
            if context.util.is_canceled():
                raise CanceledException
            if isinstance(segment, CopySegment):
                expected = segment.end - segment.start
                source = paths[segment.clip]
                part = work_dir / f"copy_{segment_index}.mp4"
                context.util.signal_progress(
                    f"Smart rendering: copying clip {segment.clip + 1}, frames {segment.start}-{segment.end - 1}"
                )
                copy_frames(
                    source,
                    segment.start,
                    segment.end,
                    plan.layouts[segment.clip].sample_count,
                    part,
                    context.util.is_canceled,
                )
                self._validate_smart_part(part, expected, target_entry, output_fps, context.util.is_canceled)
                if segment.clip == 0 and segment.start == 0:
                    copy_success_logs.append(
                        f"Stream-copied the first {expected} frames of the first clip without re-encoding"
                    )
                copy_success_logs.append(
                    f"Stream-copied {expected} frames from clip {segment.clip + 1} "
                    f"(source frames {segment.start}-{segment.end - 1})"
                )
            else:
                bridge_index += 1
                context.util.signal_progress(
                    f"Smart rendering transition {bridge_index}/{plan.bridge_count} "
                    f"between clips {segment.left_clip + 1} and {segment.right_clip + 1}"
                )
                left_path = work_dir / f"bridge_{bridge_index}_left.mp4"
                right_path = work_dir / f"bridge_{bridge_index}_right.mp4"
                copy_frames(
                    paths[segment.left_clip],
                    segment.left_start,
                    plan.layouts[segment.left_clip].sample_count,
                    plan.layouts[segment.left_clip].sample_count,
                    left_path,
                    context.util.is_canceled,
                )
                copy_frames(
                    paths[segment.right_clip],
                    0,
                    segment.right_end,
                    plan.layouts[segment.right_clip].sample_count,
                    right_path,
                    context.util.is_canceled,
                )
                left_expected = plan.layouts[segment.left_clip].sample_count - segment.left_start
                self._validate_smart_part(left_path, left_expected, target_entry, output_fps, context.util.is_canceled)
                self._validate_smart_part(
                    right_path, segment.right_end, target_entry, output_fps, context.util.is_canceled
                )

                bridge_path = work_dir / f"bridge_{bridge_index}.mp4"
                encoded_source_counts: list[int] = []
                abort_event = threading.Event()

                def decoder_cancel(event: threading.Event = abort_event) -> bool:
                    return event.is_set() or context.util.is_canceled()

                left_frames = iter_video_frames(left_path, is_canceled=decoder_cancel)
                right_frames = iter_video_frames(right_path, is_canceled=decoder_cancel)
                try:
                    source_frames = self._iter_joined_frames(
                        [_iter_frames_on_canvas(left_frames, canvas), _iter_frames_on_canvas(right_frames, canvas)],
                        is_canceled=decoder_cancel,
                        frame_counts=encoded_source_counts,
                    )
                    encoded = encode_rgb_frames(
                        source_frames,
                        bridge_path,
                        canvas,
                        output_fps,
                        segment.output_frames,
                        # Only a user cancellation is fatal. Internal encoder aborts also set
                        # ``abort_event`` so the frame decoder can stop, then fall back safely.
                        is_canceled=context.util.is_canceled,
                        abort_event=abort_event,
                    )
                finally:
                    left_frames.close()
                    right_frames.close()
                if encoded != segment.output_frames or encoded_source_counts != [left_expected, segment.right_end]:
                    raise SpliceError(
                        f"transition {bridge_index} emitted {encoded} frames from source counts {encoded_source_counts}; "
                        f"expected {segment.output_frames}"
                    )
                self._validate_smart_part(
                    bridge_path, segment.output_frames, target_entry, output_fps, context.util.is_canceled
                )
                parts.append(bridge_path)
                part_counts.append(segment.output_frames)
                continue
            parts.append(part)
            part_counts.append(expected)

        if not parts or sum(part_counts) != plan.output_frames:
            raise SpliceError("the planned smart-render parts do not cover the expected output timeline")
        context.util.signal_progress("Assembling smart-rendered video")
        concat_stream_copy(parts, out_path, context.util.is_canceled)
        joined = self._validate_smart_part(
            out_path, plan.output_frames, target_entry, output_fps, context.util.is_canceled
        )
        boundaries: set[tuple[int, int]] = set()
        cumulative = 0
        for count in part_counts[:-1]:
            cumulative += count
            boundaries.add((cumulative, cumulative))
        actual_access_points = {(point.sample_index, point.frame_index) for point in joined.access_points}
        if not boundaries.issubset(actual_access_points):
            raise SpliceError("the joined MP4 does not retain independent IDR decoding at every segment boundary")
        boundary_frames = [frame_index for _, frame_index in sorted(boundaries)]
        batch_size = 16
        for start in range(0, len(boundary_frames), batch_size):
            if context.util.is_canceled():
                raise CanceledException
            batch = boundary_frames[start : start + batch_size]
            command = ["-xerror"]
            filters = []
            for index, frame_index in enumerate(batch):
                command.extend(["-ss", format(frame_index / output_fps, ".12g"), "-i", str(out_path)])
                filters.append(f"[{index}:v:0]trim=end_frame=1,setpts=PTS-STARTPTS[v{index}]")
            command.extend(["-filter_complex", ";".join(filters)])
            for index in range(len(batch)):
                command.extend(["-map", f"[v{index}]"])
            command.extend(["-f", "null", "-"])
            run_ffmpeg(
                command,
                f"validating splice boundaries at frames {batch[0]}-{batch[-1]}",
                context.util.is_canceled,
            )
        if context.util.is_canceled():
            raise CanceledException
        for message in copy_success_logs:
            context.logger.info(message)

    def _validate_smart_part(
        self,
        path: Path,
        expected_frames: int,
        sample_entry: bytes,
        fps: float,
        is_canceled: Optional[Callable[[], bool]] = None,
    ):
        """Check the sample table, entry, cadence, start IDR and bounded container probe."""
        if is_canceled is not None and is_canceled():
            raise CanceledException
        try:
            layout = read_h264_layout(path)
        except OSError as e:
            raise SpliceError(f"could not inspect smart-render part {path.name}: {e}") from e
        if is_canceled is not None and is_canceled():
            raise CanceledException
        if (
            layout is None
            or layout.sample_count != expected_frames
            or layout.sample_entry != sample_entry
            or layout.frame_duration is None
            or not math.isclose(layout.time_scale / layout.frame_duration, fps, rel_tol=1e-6)
            or not any(point.sample_index == 0 and point.frame_index == 0 for point in layout.access_points)
        ):
            raise SpliceError(f"smart-render part {path.name} failed frame, codec, cadence, or IDR validation")
        try:
            timeline_valid = probes_as(path, fps, expected_frames)
        except (OSError, TimeoutError) as e:
            if is_canceled is not None and is_canceled():
                raise CanceledException from e
            raise SpliceError(f"could not validate smart-render timing for {path.name}: {e}") from e
        if is_canceled is not None and is_canceled():
            raise CanceledException
        if not timeline_valid:
            raise SpliceError(f"smart-render part {path.name} failed frame, codec, cadence, or IDR validation")
        return layout

    def _encode_joined(
        self,
        context: InvocationContext,
        paths: list[Path],
        canvas: tuple[int, int],
        output_fps: float,
        out_path: Path,
    ) -> tuple[list[int], int]:
        """Encode every joined frame to ``out_path`` as the compatibility fallback."""
        # Frames stream from the decoders straight into the encoder; only the
        # transition windows are buffered. See _iter_joined_frames.
        writer = make_mp4_writer(out_path, output_fps)
        num_frames = 0
        frame_counts: list[int] = []
        decoders = [iter_video_frames(p, is_canceled=context.util.is_canceled) for p in paths]
        try:
            # Every clip goes through the canvas guard, including the first: frames already
            # on it pass straight through, so nothing can reach the writer at a size the
            # file is not being encoded at.
            clip_iters = [_iter_frames_on_canvas(frames, canvas) for frames in decoders]
            for frame in self._iter_joined_frames(
                clip_iters, is_canceled=context.util.is_canceled, frame_counts=frame_counts
            ):
                writer.append_data(frame)
                num_frames += 1
        finally:
            writer.close()
            # An unfinished decoder holds the process-wide stream slot that the splice, or a
            # fallback re-encode, would otherwise wait behind.
            for frames in decoders:
                frames.close()
        return frame_counts, num_frames

    def _mux_spliced_audio(
        self,
        context: InvocationContext,
        first_path: Path,
        first_fps: Optional[float],
        first_frames: int,
        pcm: np.ndarray,
        rate: int,
        output_fps: float,
        video_path: Path,
        out_path: Path,
    ) -> bool:
        """Mux ``pcm`` onto ``video_path``, stream-copying the first clip's own AAC packets
        for as much of the soundtrack's head as they already hold (see ``aac_splice``).

        The rebuilt soundtrack equals the first clip's decoded audio up to its outgoing
        transition — provided that clip plays at the output rate, so its audio is not
        retimed, and sets the soundtrack's sample rate. Returns False, leaving the caller to
        encode the whole soundtrack, when the clip's track is not one this app's encoder
        writes, too little precedes the transition, or the splice does not verify.
        """
        if first_fps is None or first_fps != output_fps:
            return False
        layout = spliceable_aac_layout(first_path, rate)
        if layout is None:
            return False
        # The soundtrack holds the clip's own decoded samples only where its audio was mapped
        # one-to-one: the clip's span at its native rate must be exactly its span on the timeline.
        if round(first_frames / first_fps * rate) != round(first_frames * rate / output_fps):
            return False
        tail_need, _ = self._transition_windows()
        unchanged = min(
            round((first_frames - tail_need) * rate / output_fps),
            layout.packet_count * AAC_FRAME_SAMPLES - layout.priming,
            pcm.shape[1],
        )
        # Keep the cut PREROLL_FRAMES clear of the first changed sample, and a full encoded tail
        # after it (see aac_splice).
        head_packets = min(
            (unchanged + layout.priming) // AAC_FRAME_SAMPLES - PREROLL_FRAMES,
            (pcm.shape[1] - EDGE_FRAMES * AAC_FRAME_SAMPLES + layout.priming) // AAC_FRAME_SAMPLES,
        )
        reason = splice_soundtrack(
            video_path,
            first_path,
            layout.priming,
            0,
            head_packets,
            -layout.priming,
            pcm,
            rate,
            out_path,
            context.util.is_canceled,
        )
        if reason is not None:
            context.logger.info(f"Re-encoding the whole soundtrack: {reason}")
            return False
        cut = head_packets * AAC_FRAME_SAMPLES - layout.priming
        context.logger.info(f"Stream-copied the first {cut} audio samples of the first clip without re-encoding")
        return True

    def _estimate_transition_memory(self, width: int, height: int) -> int:
        if self.transition == "cut" or self.transition_frames == 0:
            return 0
        buffered_frames = self.transition_frames * (2 if self.transition == "crossfade" else 1)
        frame_bytes = width * height * 3
        return frame_bytes * (buffered_frames + _BLEND_WORKING_FRAMES)

    def _resolve_output_fps(self, source_rates: list[Optional[float]]) -> float:
        if self.fps is not None:
            return float(self.fps)
        known_rates = [rate for rate in source_rates if rate is not None and rate > 0]
        if not known_rates:
            return 16.0
        if any(not math.isclose(rate, known_rates[0], rel_tol=1e-3) for rate in known_rates[1:]):
            raise ValueError("Input videos have different frame rates; set Output FPS to retime them.")
        # An unknown rate mixed with agreeing known ones is not an error: probe_video
        # deliberately reports None for metadata-poor containers (VFR flags, missing
        # avg_frame_rate) whose real rate is usually the same as their neighbours'.
        # Erroring here would break previously-working concat workflows over a metadata
        # quirk; disagreement between *known* rates is the case that silently retimes.
        return known_rates[0]

    def _validate_transition_memory(self, width: int, height: int) -> None:
        estimated_bytes = self._estimate_transition_memory(width, height)
        if estimated_bytes > MAX_TRANSITION_MEMORY_BYTES:
            estimated_mib = estimated_bytes / (1024 * 1024)
            limit_mib = MAX_TRANSITION_MEMORY_BYTES / (1024 * 1024)
            raise ValueError(
                f"The requested transition needs an estimated {estimated_mib:.0f} MiB, "
                f"which exceeds the {limit_mib:.0f} MiB transition memory budget. "
                "Lower transition_frames or use lower-resolution clips."
            )

    def _build_audio_track(
        self,
        context: InvocationContext,
        paths: list[Path],
        frame_counts: list[int],
        native_rates: list[Optional[float]],
        native_durations: list[Optional[float]],
        output_fps: float,
        num_output_frames: int,
    ) -> Optional[tuple[np.ndarray, int]]:
        """Assemble the output soundtrack on the emitted timeline.

        Returns ``(pcm, sample_rate)`` — ``pcm`` shaped ``(2, n)``, float in [-1, 1] — or
        ``None`` when no input carries audio. The frame bookkeeping (which frames of each
        clip are emitted directly vs consumed into a boundary) mirrors
        ``_iter_joined_frames`` exactly; ``frame_counts`` must be that pass's per-clip
        decoded frame counts so the audio is cut against the frames actually emitted.

        Every clip's audio is mapped onto the emitted timeline with one linear resample:
        the clip's video spans ``n_i / native_fps`` seconds natively and ``n_i /
        output_fps`` seconds in the output, so this single step covers both sample-rate
        unification (to the first audible clip's rate) and any fps-override retime.
        """
        context.util.signal_progress("Extracting audio")
        extracted: list[Optional[tuple[np.ndarray, int]]] = []
        for path in paths:
            if context.util.is_canceled():
                raise CanceledException
            extracted.append(extract_audio_pcm(path, is_canceled=context.util.is_canceled))
        if all(item is None for item in extracted):
            return None
        if context.util.is_canceled():
            raise CanceledException
        rate = next(item[1] for item in extracted if item is not None)

        def s(frames: int) -> int:
            """Frame count -> sample count on the emitted timeline."""
            return round(frames * rate / output_fps)

        tail_need, head_need = self._transition_windows()

        clips: list[np.ndarray] = []
        for i, (item, n_frames, native_fps) in enumerate(zip(extracted, frame_counts, native_rates, strict=True)):
            want = s(n_frames)
            if item is None:
                clips.append(np.zeros((2, want), dtype=np.float32))
                continue
            pcm, src_rate = item
            # The clip's audio must be sliced to the span its video occupies natively
            # (trimming AAC end-padding) before the resample maps it onto the emitted
            # timeline. Preference order for that native span: the probed fps; the
            # probed duration (fps can be None for VFR-flagged containers, and under an
            # fps override guessing output_fps would skip the retime — the audio would
            # play at the wrong speed and truncate or pad); the extracted length itself,
            # which approximates the native span to within the codec padding.
            if native_fps is not None and native_fps > 0:
                src_want = round(n_frames / native_fps * src_rate)
            elif native_durations[i] is not None and native_durations[i] > 0:
                src_want = round(native_durations[i] * src_rate)
            else:
                src_want = pcm.shape[1]
            if pcm.shape[1] < src_want:
                pcm = np.pad(pcm, ((0, 0), (0, src_want - pcm.shape[1])))
            else:
                pcm = pcm[:, :src_want]
            clips.append(resample_linear(pcm, want))
            extracted[i] = None  # drop the pre-resample copy; peak memory is O(total audio)

        # A boundary exists between consecutive clips whenever the transition consumes
        # any frames — even when only ONE side contributes (fade_through_black with
        # transition_frames=1 has an empty tail window but still fades the head in).
        has_boundary = (head_need + tail_need) > 0

        segments: list[np.ndarray] = []
        prev_tail: Optional[np.ndarray] = None
        for i, pcm in enumerate(clips):
            n_i = frame_counts[i]
            head_want = 0 if i == 0 else head_need
            tail_keep = 0 if i == len(clips) - 1 else tail_need
            head_end = s(head_want)
            tail_start = s(n_i - tail_keep) if tail_keep else pcm.shape[1]
            if prev_tail is not None:
                head = pcm[:, :head_end]
                if self.transition == "crossfade":
                    # Equal-power blend: constant perceived loudness through the overlap.
                    n = min(prev_tail.shape[1], head.shape[1])
                    theta = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
                    segments.append(prev_tail[:, :n] * np.cos(theta) + head[:, :n] * np.sin(theta))
                else:  # fade_through_black: out to silence, then in from silence.
                    segments.append(prev_tail * np.linspace(1.0, 0.0, prev_tail.shape[1], dtype=np.float32))
                    segments.append(head * np.linspace(0.0, 1.0, head.shape[1], dtype=np.float32))
            segments.append(pcm[:, head_end:tail_start])
            # An empty tail window (shape (2, 0)) still marks the boundary as pending so
            # the next clip's head is faded in rather than silently dropped.
            prev_tail = pcm[:, tail_start:] if (has_boundary and i < len(clips) - 1) else None

        pcm_out = np.concatenate(segments, axis=1)
        # Per-clip rounding can drift the total by a sample per boundary; pin the track
        # to the emitted video duration exactly.
        target = s(num_output_frames)
        if pcm_out.shape[1] < target:
            pcm_out = np.pad(pcm_out, ((0, 0), (0, target - pcm_out.shape[1])))
        else:
            pcm_out = pcm_out[:, :target]
        return pcm_out.astype(np.float32, copy=False), rate

    def _iter_joined_frames(
        self,
        clips: list[Iterable[np.ndarray]],
        is_canceled: Optional[Callable[[], bool]] = None,
        frame_counts: Optional[list[int]] = None,
    ) -> Iterator[np.ndarray]:
        """Yields the joined output frames, pulling lazily from each clip's frame iterator.

        A frame is emitted as soon as it can no longer participate in a transition, so at
        most one transition window (the previous clip's tail plus the current clip's head,
        each bounded by ``transition_frames``) is buffered at a time — never a whole clip.

        Transition layout matches the class docstring:

        * ``crossfade`` consumes ``tf`` frames from both sides of each boundary and emits
          ``tf`` blended frames in their place.
        * ``fade_through_black`` splits ``tf`` asymmetrically (``tf // 2`` from the previous
          clip's tail, the remainder from the next clip's head) so an odd ``tf`` still emits
          exactly ``tf`` frames per boundary.

        Raises ValueError if a clip decodes to zero frames or is too short to supply its
        transition windows. Because frames stream straight into the encoder, that error can
        surface mid-encode; the caller discards the partial output file.
        """
        tail_need, head_need = self._transition_windows()
        blend = {"crossfade": _crossfade, "fade_through_black": _fade_through_black}.get(self.transition)
        if head_need + tail_need == 0:
            # "cut", or a zero-length transition: plain concatenation.
            blend = None

        a_tail: list[np.ndarray] = []
        for i, clip in enumerate(clips):
            head_want = 0 if i == 0 else head_need
            tail_keep = 0 if i == len(clips) - 1 else tail_need
            b_head: list[np.ndarray] = []
            head_complete = head_want == 0
            tail_buf: deque[np.ndarray] = deque()
            n_frames = 0
            for frame in clip:
                if is_canceled is not None and is_canceled():
                    raise CanceledException
                frame = np.ascontiguousarray(frame)
                n_frames += 1
                # The clip's first head_want frames are consumed into the boundary blend
                # with the previous clip's tail rather than emitted directly.
                if not head_complete:
                    b_head.append(frame)
                    if len(b_head) == head_want and blend is not None:
                        yield from blend(a_tail, b_head)
                        a_tail = []
                        b_head = []
                        head_complete = True
                    continue
                # Hold back the last tail_keep frames seen so far; anything older is
                # guaranteed not to be part of the next boundary and can be emitted.
                tail_buf.append(frame)
                if len(tail_buf) > tail_keep:
                    yield tail_buf.popleft()
            if frame_counts is not None:
                frame_counts.append(n_frames)
            if n_frames == 0:
                raise ValueError(f"Input video {i} ({self.videos[i].video_name}) decoded to zero frames.")
            if n_frames < head_want + tail_keep:
                raise ValueError(
                    f"Clip {i} has {n_frames} frames but the requested transitions need "
                    f"{head_want} from its head + {tail_keep} from its tail. Lower "
                    f"transition_frames or use longer clips."
                )
            a_tail = list(tail_buf)
