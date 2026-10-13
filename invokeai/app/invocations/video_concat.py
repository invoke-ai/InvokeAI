"""Concatenate two or more videos with an optional transition.

Pairs naturally with the I2V chaining workflow: feed several Wan-generated
clips into this node to glue them into one longer video. The transition
options hide the seam between independently-denoised clips.

Implementation uses imageio (FFMPEG plugin) for both decode and encode, matching
``wan_latents_to_video`` and ``video_thumbnails`` — so we can read our own
output without surprises. Frames stream from the decoders into the encoder one
at a time, buffering only the transition windows, so peak memory stays
O(transition_frames) even when the inputs are long uploads (the upload cap
admits files whose decoded frames would run to tens of gigabytes).

The first clip is not re-encoded when it can be avoided: if it was written by this
app's own encoder at the output canvas and frame rate, everything before its last
usable keyframe is stream-copied, and only the remainder — its tail, the transitions
and the later clips — is encoded and joined on (see ``h264_splice``). Repeatedly
extending a video therefore stops degrading its earlier segments, which otherwise go
through one more lossy generation per extension. Any other first clip is re-encoded
in full, as before. The soundtrack is treated the same way: the first clip's own AAC
packets are kept up to its outgoing transition and only the rest is encoded (see
``aac_splice``).

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
from collections import deque
from itertools import islice
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
    PREROLL_FRAMES,
    encoder_aac_layout,
    mux_spliced_aac,
    read_aac_layout,
)
from invokeai.app.util.h264_splice import (
    MIN_ENCODED_FRAMES,
    H264Layout,
    SpliceError,
    concat_stream_copy,
    copy_frames,
    read_h264_layout,
)
from invokeai.app.util.video_audio import (
    AudioExtractionError,
    extract_audio_pcm,
    mux_audio_into_video,
    resample_linear,
)
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav, writer_decoder_config
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
    version="1.2.0",
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
            joined = None
            splice = self._plan_splice(paths[0], probes[0][3], (width, height), output_fps)
            if splice is not None:
                keyframe, layout = splice
                rest_path = work_dir / "rest.mp4"
                rest = self._encode_joined(context, paths, (width, height), output_fps, rest_path, skip=keyframe)
                if self._splice(context, paths[0], keyframe, layout, rest_path, rest[0][0], work_dir, tmp_path):
                    joined = rest
            if joined is None:
                joined = self._encode_joined(context, paths, (width, height), output_fps, tmp_path, skip=0)
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
                mux_audio_into_video(tmp_path, wav_path, muxed_path)
            source_path = muxed_path

        duration = num_frames / output_fps
        context.logger.info(
            f"Encoded concatenated MP4: {num_frames} frames @ {output_fps:.2f} fps "
            f"({duration:.2f}s) at {width}x{height}" + (", with AAC audio" if audio is not None else ", silent")
        )
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

    def _plan_splice(
        self, first_path: Path, first_fps: Optional[float], canvas: tuple[int, int], output_fps: float
    ) -> Optional[tuple[int, H264Layout]]:
        """Pick the keyframe of the first clip up to which it can be stream-copied, or None.

        The keyframe must leave the clip's transition tail, and at least ``MIN_ENCODED_FRAMES``
        so the re-encoded remainder keeps the copied head's timeline, after it. The clip must
        have a constant frame duration and already play at the
        output rate — a stream copy cannot retime — and carry the decoder configuration this
        node's encoder writes, or the re-encoded remainder could not share its track.
        """
        if first_fps is None or not math.isclose(first_fps, output_fps, rel_tol=1e-6):
            return None
        layout = read_h264_layout(first_path)
        if (
            layout is None
            or layout.frame_duration is None
            or layout.decoder_config != writer_decoder_config(*canvas, output_fps)
        ):
            return None
        tail_need, _ = self._transition_windows()
        last_usable = layout.sample_count - max(tail_need, MIN_ENCODED_FRAMES)
        keyframes = [k for k in layout.sync_samples if 0 < k <= last_usable]
        return (keyframes[-1], layout) if keyframes else None

    def _encode_joined(
        self,
        context: InvocationContext,
        paths: list[Path],
        canvas: tuple[int, int],
        output_fps: float,
        out_path: Path,
        skip: int,
    ) -> tuple[list[int], int]:
        """Encode the joined timeline, minus the first clip's first ``skip`` frames, to ``out_path``.

        Returns ``(frame_counts, num_frames)`` for the WHOLE timeline — the skipped frames
        are counted as part of the first clip and of the output — so the audio is laid out
        the same whether or not the head was stream-copied.
        """
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
            clip_iters = [
                _iter_frames_on_canvas(islice(frames, skip, None) if i == 0 else frames, canvas)
                for i, frames in enumerate(decoders)
            ]
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
        frame_counts[0] += skip
        return frame_counts, num_frames + skip

    def _splice(
        self,
        context: InvocationContext,
        first_path: Path,
        keyframe: int,
        layout: H264Layout,
        rest_path: Path,
        first_clip_frames: int,
        work_dir: Path,
        out_path: Path,
    ) -> bool:
        """Join the first clip's frames before ``keyframe``, stream-copied, onto ``rest_path``.

        Returns False, leaving the caller to re-encode everything, when the two cannot be
        joined losslessly: the remainder was encoded with a different decoder configuration
        than the first clip carries (an upload, or a file from an older encoder setting), or
        the first clip decoded to a different frame count than its sample table lists, which
        would put the copied head and the skipped frames out of step.
        """
        rest = read_h264_layout(rest_path)
        if first_clip_frames != layout.sample_count:
            reason = f"it decoded to {first_clip_frames} frames but lists {layout.sample_count}"
        elif rest is None or rest.decoder_config != layout.decoder_config or rest.sync_samples[:1] != (0,):
            reason = "its encoding differs from this node's"
        else:
            reason = None
        if reason is None:
            head_path = work_dir / "head.mp4"
            try:
                copy_frames(first_path, 0, keyframe, layout.sample_count, head_path)
                head = read_h264_layout(head_path)
                if head is None or head.sample_count != keyframe:
                    reason = "copying its head did not yield the expected frames"
                else:
                    concat_stream_copy([head_path, rest_path], out_path)
                    joined = read_h264_layout(out_path)
                    if (
                        joined is None
                        or joined.sample_count != keyframe + rest.sample_count
                        or joined.frame_duration != layout.frame_duration
                    ):
                        reason = "the joined file does not hold the expected frames on one timeline"
            except SpliceError as e:
                reason = str(e)
        if reason is not None:
            context.logger.info(f"Re-encoding the first clip in full: it cannot be stream-copied ({reason})")
            return False
        context.logger.info(f"Stream-copied the first {keyframe} frames of the first clip without re-encoding")
        return True

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
        writes, nothing whole-frame precedes the transition, or the result's head does not
        decode back to exactly the samples it should hold.
        """
        if first_fps is None or first_fps != output_fps:
            return False
        layout, encoder = read_aac_layout(first_path), encoder_aac_layout(rate)
        if layout is None or encoder is None or layout.sample_rate != rate:
            return False
        if (layout.decoder_config, layout.priming) != (encoder.decoder_config, encoder.priming):
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
        preroll = PREROLL_FRAMES * AAC_FRAME_SAMPLES
        # Keep the cut PREROLL_FRAMES clear of the first changed sample (see aac_splice).
        head_packets = (unchanged + layout.priming) // AAC_FRAME_SAMPLES - PREROLL_FRAMES
        cut = head_packets * AAC_FRAME_SAMPLES - layout.priming
        if cut < preroll:
            return False
        try:
            seam_reference = mux_spliced_aac(
                video_path,
                first_path,
                head_packets,
                layout.priming,
                pcm[:, cut - preroll :],
                rate,
                pcm.shape[1],
                out_path,
            )
            muxed = extract_audio_pcm(out_path)
        except (SpliceError, AudioExtractionError) as e:
            context.logger.info(f"Re-encoding the whole soundtrack: it cannot be stream-copied ({e})")
            return False
        reason = self._spliced_audio_defect(muxed, pcm, rate, cut, seam_reference)
        if reason is not None:
            context.logger.info(f"Re-encoding the whole soundtrack: {reason}")
            return False
        context.logger.info(f"Stream-copied the first {cut} audio samples of the first clip without re-encoding")
        return True

    @staticmethod
    def _spliced_audio_defect(
        muxed: Optional[tuple[np.ndarray, int]],
        pcm: np.ndarray,
        rate: int,
        cut: int,
        seam_reference: np.ndarray,
    ) -> Optional[str]:
        """Why the spliced soundtrack cannot be used, or None.

        The copied head must decode back to exactly the samples it stands for, and the
        seam — the frames where copied and new packets overlap — must be no noisier than an
        ordinary encode of the same samples (``seam_reference``). Whether the overlap
        cancels depends on the encoder's window decisions around the content there, so it is
        measured, not assumed; and it is measured against the seam's own content, since the
        audio after it may be silence that codes with no error at all.
        """
        if muxed is None or muxed[1] != rate or muxed[0].shape[1] < pcm.shape[1]:
            return "the spliced soundtrack did not decode to its full length"
        decoded = muxed[0]
        if not np.array_equal(decoded[:, :cut], pcm[:, :cut]):
            return "the copied audio did not decode back exactly"
        seam = slice(cut, cut + seam_reference.shape[1])
        if seam_reference.shape[1] != PREROLL_FRAMES * AAC_FRAME_SAMPLES or seam.stop > pcm.shape[1]:
            return "the new audio does not cover the seam"
        seam_error = float(np.sqrt(np.mean((decoded[:, seam] - pcm[:, seam]) ** 2)))
        coding_error = float(np.sqrt(np.mean((seam_reference - pcm[:, seam]) ** 2)))
        if seam_error > 2.0 * coding_error + 1e-3:
            return "the seam between copied and new audio is not clean"
        return None

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
            extracted.append(extract_audio_pcm(path))
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
