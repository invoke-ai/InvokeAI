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

import tempfile
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

import imageio.v2 as iio2
import imageio.v3 as iio
import numpy as np

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    InputField,
    VideoField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_thumbnails import probe_video

TransitionMode = Literal["cut", "crossfade", "fade_through_black"]


def _crossfade(a_tail: list[np.ndarray], b_head: list[np.ndarray]) -> list[np.ndarray]:
    """Linear A→B cross-dissolve. Consumes N frames from each side, returns N blended frames."""
    n = len(a_tail)
    out: list[np.ndarray] = []
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        blended = a_tail[i].astype(np.float32) * (1.0 - alpha) + b_head[i].astype(np.float32) * alpha
        out.append(np.clip(blended, 0, 255).astype(np.uint8))
    return out


def _fade_through_black(a_tail: list[np.ndarray], b_head: list[np.ndarray]) -> list[np.ndarray]:
    """A fades to black, then black fades to B. Consumes N/2 frames from each side and returns N output frames.

    Asymmetric framing: the first ``len(a_tail)`` output frames are the trailing A frames scaled
    toward zero brightness; the next ``len(b_head)`` are the leading B frames scaled up from zero.
    """
    out: list[np.ndarray] = []
    n_a = len(a_tail)
    for i, fa in enumerate(a_tail):
        # 1.0 at i=0 (fully visible) → near 0 at i=n_a-1 (essentially black).
        alpha = 1.0 - (i + 1) / (n_a + 1)
        out.append(np.clip(fa.astype(np.float32) * alpha, 0, 255).astype(np.uint8))
    n_b = len(b_head)
    for j, fb in enumerate(b_head):
        alpha = (j + 1) / (n_b + 1)
        out.append(np.clip(fb.astype(np.float32) * alpha, 0, 255).astype(np.uint8))
    return out


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
        width, height, _, first_fps = probes[0]
        output_fps = float(self.fps) if self.fps is not None else (first_fps or 16.0)

        context.util.signal_progress(f"Joining {len(self.videos)} clip(s) ({self.transition}) @ {output_fps:.2f} fps")

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_video_concat_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            # Frames stream from the decoders straight into the encoder; only the
            # transition windows are buffered. See _iter_joined_frames.
            writer = iio2.get_writer(str(tmp_path), format="FFMPEG", mode="I", fps=output_fps, codec="libx264")
            num_frames = 0
            try:
                clip_iters = [iio.imiter(p, plugin="FFMPEG") for p in paths]
                for frame in self._iter_joined_frames(clip_iters):
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

    def _iter_joined_frames(self, clips: list[Iterable[np.ndarray]]) -> Iterator[np.ndarray]:
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
            tail_buf: deque[np.ndarray] = deque()
            n_frames = 0
            for frame in clip:
                frame = np.ascontiguousarray(frame)
                n_frames += 1
                # The clip's first head_want frames are consumed into the boundary blend
                # with the previous clip's tail rather than emitted directly.
                if len(b_head) < head_want:
                    b_head.append(frame)
                    if len(b_head) == head_want and blend is not None:
                        yield from blend(a_tail, b_head)
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
