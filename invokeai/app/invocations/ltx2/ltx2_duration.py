"""Predict how long the shot a prompt describes should run."""

import torch

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import Input, InputField, LTX2ConditioningField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ltx2.constants import (
    LTX2_DEFAULT_FPS,
    LTX2_FRAME_MODULUS,
    LTX2_NUM_FRAMES_MAX,
    LTX2_TEMPORAL_COMPRESSION,
)
from invokeai.backend.ltx2.packing import snap_num_frames_down
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

# The range the head was trained to regress over. Outside it the prediction is extrapolation, so
# these are the widest bounds the node offers rather than merely its defaults.
LTX2_DURATION_MIN_SECONDS: float = 1.0
LTX2_DURATION_MAX_SECONDS: float = 20.0


@invocation_output("ltx2_duration_output")
class LTX2DurationOutput(BaseInvocationOutput):
    """A predicted shot length, as both a frame count and the seconds it came from."""

    num_frames: int = OutputField(
        description="Frame count on LTX-2's 8k+1 grid, including any context frames. Wire into the denoise "
        "node's `num_frames`."
    )
    # The frame count is what a graph consumes, but it has been clamped and snapped, so on its own
    # it cannot tell a user whether the model wanted 3 seconds or 30. This is the raw regression.
    seconds: float = OutputField(description="The head's unclamped prediction, in seconds.")


@invocation(
    "ltx2_duration",
    title="Duration - LTX-2",
    tags=["ltx", "ltx2", "video", "duration", "frames"],
    category="video",
    version="1.1.0",
    classification=Classification.Prototype,
)
class LTX2DurationInvocation(BaseInvocation):
    """Predicts the natural length of the shot a prompt describes, as an LTX-2 frame count.

    The duration head reads the same connector outputs the transformer's prompt cross-attention
    consumes, so it judges the prompt the model will actually see rather than its raw text. The
    prediction is clamped to `min_seconds`/`max_seconds` and then snapped down onto the VAE's
    causal temporal grid (`8k + 1`), which is the only frame count a generation can run at. The
    total is then capped at `max_num_frames` (by default LTX-2's longest clip), which can be
    shorter than `max_seconds` at a high frame rate.

    For an extension, the prompt describes the continuation rather than the frames it opens with,
    so the prediction sizes the new material and `context_frames` is added in front of it.
    Conditioning clips fix the frame count by construction -- the clip decides it -- so this node
    has nothing to say about those graphs.
    """

    conditioning: LTX2ConditioningField = InputField(
        description="Prompt conditioning from the LTX-2 text encoder.",
        input=Input.Connection,
        title="Conditioning",
    )
    duration_head: ModelIdentifierField = InputField(
        description="The LTX-2 duration head.",
        title="Duration Head",
        ui_model_base=BaseModelType.LTX2,
        ui_model_type=ModelType.LTX2DurationHead,
    )
    fps: float = InputField(
        default=LTX2_DEFAULT_FPS,
        ge=1,
        le=120,
        description="Frames per second the clip will play at. The head predicts seconds, so this "
        "is what turns that into a frame count.",
    )
    min_seconds: float = InputField(
        default=LTX2_DURATION_MIN_SECONDS,
        ge=LTX2_DURATION_MIN_SECONDS,
        le=LTX2_DURATION_MAX_SECONDS,
        description="Shortest clip to allow.",
    )
    max_seconds: float = InputField(
        default=LTX2_DURATION_MAX_SECONDS,
        ge=LTX2_DURATION_MIN_SECONDS,
        le=LTX2_DURATION_MAX_SECONDS,
        description="Longest clip to allow.",
    )
    context_frames: int = InputField(
        default=0,
        ge=0,
        description="Source frames the run opens with, as an extension's `context_frames`. The prediction "
        "covers what follows them, so they are added to it. 0 when nothing is held.",
    )
    max_num_frames: int = InputField(
        default=LTX2_NUM_FRAMES_MAX,
        ge=1,
        le=LTX2_NUM_FRAMES_MAX,
        description="Longest total frame count, context included, the run was sized for. The result is "
        "capped at it, so it holds at whatever `fps` the run turns out to have.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LTX2DurationOutput:
        # Equal bounds are refused too, as upstream's pipeline does: `predict_num_frames` then falls
        # back to the nearest grid point, which can land outside the single length asked for.
        if self.min_seconds >= self.max_seconds:
            raise ValueError(
                f"min_seconds ({self.min_seconds}) must be less than max_seconds ({self.max_seconds}) "
                "for the duration head to have a range to choose from."
            )
        # Context comes off the extend node, which only ever holds whole frame groups. Anything else
        # would put the sum below off the grid.
        if self.context_frames and (self.context_frames - 1) % LTX2_FRAME_MODULUS:
            raise ValueError(f"context_frames ({self.context_frames}) must be 0 or on the 8k+1 grid.")
        ceiling = snap_num_frames_down(self.max_num_frames)
        if self.context_frames and ceiling <= self.context_frames:
            raise ValueError(
                f"max_num_frames ({self.max_num_frames}) leaves no room after {self.context_frames} context frames."
            )

        conditioning = context.conditioning.load(self.conditioning.conditioning_name)
        # The text encoder emits one conditioning per prompt; the head reads a single prompt's
        # tokens, and a graph that batched prompts would need one duration per prompt, not a
        # pooled compromise between them.
        if len(conditioning.conditionings) != 1:
            raise ValueError(f"LTX-2 duration expects exactly one conditioning, got {len(conditioning.conditionings)}.")
        info = conditioning.conditionings[0]

        head_info = context.models.load(self.duration_head)
        with head_info.model_on_device() as (_, head):
            video_tokens = info.video_embeds.to(device=head.device)
            audio_tokens = info.audio_embeds.to(device=head.device)
            # Two forward passes, deliberately: `predict_num_frames` runs its own and owns the
            # clamp/snap rules, including the case where the bounds contain no grid point at all.
            # Reproducing those here to save one pass of a 3.6 MB head on ~128 tokens would trade
            # a free operation for a copy of upstream logic that can drift.
            seconds = float(head(video_tokens=video_tokens, audio_tokens=audio_tokens).item())
            num_frames = head.predict_num_frames(
                video_tokens=video_tokens,
                audio_tokens=audio_tokens,
                frame_rate=float(self.fps),
                temporal_compression_ratio=LTX2_TEMPORAL_COMPRESSION,
                min_seconds=float(self.min_seconds),
                max_seconds=float(self.max_seconds),
            )

        if self.context_frames:
            # The prediction is 8k + 1; its k whole groups are the new material, appended to a context
            # that is itself 8k + 1, so the total stays on the grid. At least one group, or the
            # continuation would be all replay.
            num_frames = self.context_frames + max(num_frames - 1, LTX2_FRAME_MODULUS)
        # The ceiling is in frames because only this node knows the rate the run plays at: seconds
        # sized by a caller at a guessed rate would stop short of it or overrun it. Both sides are on
        # the grid, so capping here is exactly the longest prediction that fits.
        num_frames = min(num_frames, ceiling)

        context.logger.info(
            f"LTX-2 duration: predicted {seconds:.2f}s -> {num_frames} frames at {self.fps} fps"
            + (f" (including {self.context_frames} context frames)" if self.context_frames else "")
        )
        return LTX2DurationOutput(num_frames=int(num_frames), seconds=seconds)
