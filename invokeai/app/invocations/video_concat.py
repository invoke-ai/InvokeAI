"""Concatenate two or more videos with an optional transition.

Pairs naturally with the I2V chaining workflow: feed several Wan-generated
clips into this node to glue them into one longer video. The transition
options hide the seam between independently-denoised clips.

Implementation uses imageio (FFMPEG plugin) for both decode and encode, matching
``wan_latents_to_video`` and ``video_thumbnails`` — so we can read our own
output without surprises. All decoded frames live in RAM at once; this is fine
for the short clips the I2V chain produces (a few hundred frames at 832x480),
but be aware before piping in long uploads.
"""

import tempfile
from pathlib import Path
from typing import Literal, Optional

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
      consumes ``transition_frames / 2`` from both adjacent clips and emits
      ``transition_frames`` output frames, so total length equals the sum of inputs.

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

        paths: list[Path] = [
            context.videos.get_path(v.video_name) for v in self.videos
        ]

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

        context.util.signal_progress(f"Decoding {len(self.videos)} clip(s)")
        clip_frames: list[list[np.ndarray]] = []
        for idx, p in enumerate(paths):
            # iio.imiter is a generator — collecting to a list keeps things simple and the
            # downstream blending math straightforward. Memory cost is fine for I2V-length
            # clips; if this ever needs to handle hour-long uploads, switch to streaming.
            frames = [np.ascontiguousarray(f) for f in iio.imiter(p, plugin="FFMPEG")]
            if not frames:
                raise ValueError(f"Input video {idx} ({self.videos[idx].video_name}) decoded to zero frames.")
            clip_frames.append(frames)

        # Validate transition windows fit within the surrounding clips.
        if self.transition != "cut" and self.transition_frames > 0:
            tf = self.transition_frames
            for i, frames in enumerate(clip_frames):
                # Each non-edge clip uses transition_frames from both its head and tail.
                head_need = 0 if i == 0 else (tf if self.transition == "crossfade" else tf // 2)
                tail_need = 0 if i == len(clip_frames) - 1 else (tf if self.transition == "crossfade" else tf // 2)
                if head_need + tail_need > len(frames):
                    raise ValueError(
                        f"Clip {i} has {len(frames)} frames but the requested transitions need "
                        f"{head_need} from its head + {tail_need} from its tail. Lower "
                        f"transition_frames or use longer clips."
                    )

        context.util.signal_progress(f"Joining clips ({self.transition})")
        output_frames = self._assemble(clip_frames)

        if not output_frames:
            raise ValueError("Concatenation produced zero output frames.")

        num_frames = len(output_frames)
        duration = num_frames / output_fps
        context.logger.info(
            f"Encoding concatenated MP4: {num_frames} frames @ {output_fps:.2f} fps "
            f"({duration:.2f}s) at {width}x{height}"
        )
        context.util.signal_progress(f"Encoding MP4 ({num_frames} frames @ {output_fps:.2f} fps)")

        tmp = tempfile.NamedTemporaryFile(
            prefix="invokeai_video_concat_", suffix=".mp4", delete=False
        )
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            iio.imwrite(
                tmp_path,
                output_frames,
                plugin="FFMPEG",
                codec="libx264",
                fps=output_fps,
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

    def _assemble(self, clip_frames: list[list[np.ndarray]]) -> list[np.ndarray]:
        if self.transition == "cut" or self.transition_frames == 0:
            return [f for frames in clip_frames for f in frames]

        tf = self.transition_frames
        if self.transition == "crossfade":
            # Reduction layout: keep clip[i] minus tf from its tail (except the last clip),
            # then insert tf blended frames at each boundary.
            output: list[np.ndarray] = []
            for i, frames in enumerate(clip_frames):
                head_trim = 0 if i == 0 else tf
                tail_trim = 0 if i == len(clip_frames) - 1 else tf
                output.extend(frames[head_trim : len(frames) - tail_trim])
                if i < len(clip_frames) - 1:
                    a_tail = frames[len(frames) - tail_trim :]
                    b_head = clip_frames[i + 1][:tf]
                    output.extend(_crossfade(a_tail, b_head))
            return output

        # fade_through_black: each boundary consumes tf // 2 from each side and emits tf
        # frames of "fade out + fade in" — preserving total length when tf is even.
        half = tf // 2
        if half == 0:
            return [f for frames in clip_frames for f in frames]
        output_ftb: list[np.ndarray] = []
        for i, frames in enumerate(clip_frames):
            head_trim = 0 if i == 0 else half
            tail_trim = 0 if i == len(clip_frames) - 1 else half
            output_ftb.extend(frames[head_trim : len(frames) - tail_trim])
            if i < len(clip_frames) - 1:
                a_tail = frames[len(frames) - tail_trim :]
                b_head = clip_frames[i + 1][:half]
                output_ftb.extend(_fade_through_black(a_tail, b_head))
        return output_ftb
