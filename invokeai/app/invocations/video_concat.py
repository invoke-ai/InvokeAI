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
"""

import math
import tempfile
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Optional

import numpy as np

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
from invokeai.app.util.video_encoding import make_mp4_writer
from invokeai.app.util.video_thumbnails import iter_video_frames, probe_video

TransitionMode = Literal["cut", "crossfade", "fade_through_black"]
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
    version="1.0.0",
    classification=Classification.Prototype,
)
class VideoConcatInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Join two or more videos into a single MP4.

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

    All inputs must share the same pixel dimensions. Output frame rate defaults to the
    first input's fps; override with ``fps`` to force a specific rate (the frames are not
    resampled, only the container is encoded at the new rate).
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

    def invoke(self, context: InvocationContext) -> VideoOutput:
        if len(self.videos) < 2:
            raise ValueError("video_concat requires at least two input videos.")

        paths: list[Path] = [context.videos.get_path(v.video_name) for v in self.videos]

        # Probe inputs up front: enforce matching dims and pick the default output fps.
        probes = [probe_video(p) for p in paths]
        widths = {(w, h) for (w, h, _, _) in probes}
        if len(widths) > 1:
            raise ValueError(
                f"All inputs must share the same dimensions. Got: "
                f"{sorted(widths)}. Re-render at a single resolution before concatenating."
            )
        width, height, _, _first_fps = probes[0]
        # libx264 + yuv420p needs even dimensions; we encode with macro_block_size=1 to
        # preserve the source dimensions exactly, so reject odd sources with a clear error.
        if width % 2 or height % 2:
            raise ValueError(
                f"Input videos are {width}x{height}; H.264 encoding requires even dimensions. "
                "Re-encode or crop the sources to even width and height first."
            )
        self._validate_transition_memory(width, height)
        output_fps = self._resolve_output_fps([probe[3] for probe in probes])

        context.util.signal_progress(f"Joining {len(self.videos)} clip(s) ({self.transition}) @ {output_fps:.2f} fps")

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_video_concat_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            # Frames stream from the decoders straight into the encoder; only the
            # transition windows are buffered. See _iter_joined_frames.
            writer = make_mp4_writer(tmp_path, output_fps)
            num_frames = 0
            try:
                clip_iters = [iter_video_frames(p, is_canceled=context.util.is_canceled) for p in paths]
                for frame in self._iter_joined_frames(clip_iters, is_canceled=context.util.is_canceled):
                    writer.append_data(frame)
                    num_frames += 1
            finally:
                writer.close()

            if num_frames == 0:
                raise ValueError("Concatenation produced zero output frames.")

            duration = num_frames / output_fps
            context.logger.info(
                f"Encoded concatenated MP4: {num_frames} frames @ {output_fps:.2f} fps "
                f"({duration:.2f}s) at {width}x{height}"
            )
            video_dto = context.videos.save(
                source_path=tmp_path,
                width=width,
                height=height,
                duration=duration,
                fps=output_fps,
            )
            context.logger.info(f"Saved concatenated video: {video_dto.video_name}")
            return VideoOutput.build(video_dto)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

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
        if len(known_rates) != len(source_rates) or any(
            not math.isclose(rate, known_rates[0], rel_tol=1e-3) for rate in known_rates[1:]
        ):
            raise ValueError("Input videos have different or unknown frame rates; set Output FPS to retime them.")
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

    def _iter_joined_frames(
        self,
        clips: list[Iterable[np.ndarray]],
        is_canceled: Optional[Callable[[], bool]] = None,
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
        if self.transition == "crossfade" and self.transition_frames > 0:
            tail_need = head_need = self.transition_frames
            blend = _crossfade
        elif self.transition == "fade_through_black" and self.transition_frames > 0:
            tail_need = self.transition_frames // 2
            head_need = self.transition_frames - tail_need
            blend = _fade_through_black
        else:
            # "cut", or a zero-length transition: plain concatenation.
            tail_need = head_need = 0
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
            if n_frames == 0:
                raise ValueError(f"Input video {i} ({self.videos[i].video_name}) decoded to zero frames.")
            if n_frames < head_want + tail_keep:
                raise ValueError(
                    f"Clip {i} has {n_frames} frames but the requested transitions need "
                    f"{head_want} from its head + {tail_keep} from its tail. Lower "
                    f"transition_frames or use longer clips."
                )
            a_tail = list(tail_buf)
