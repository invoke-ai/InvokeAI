"""Compute the MiniMax H3 canvas for a source image's aspect ratio.

H3 was released for exactly one canvas family: a 768 px short edge under a soft area cap
of 768x1344, both axes rounded to the nearest multiple of 32, aspect ratios limited to
1:4 through 4:1. That policy is implemented by
:func:`invokeai.backend.minimax_h3.packing.resolve_canvas_size` (a first-party port of
the released pipeline's canvas resolution). This node wraps it for workflow graphs so
image-to-video workflows can derive the canvas from the uploaded keyframe instead of
asking the user to type matching width/height values, and adds a reduced "768 lowres"
preset (:func:`~invokeai.backend.minimax_h3.packing.resolve_lowres_canvas_size`) that
pins the LONG edge to 768 for cheaper preview/test renders.
"""

from typing import Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.ideal_size import IdealSizeOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.minimax_h3.packing import resolve_canvas_size, resolve_lowres_canvas_size

MiniMaxH3TargetResolution = Literal["768 highres", "768 lowres"]

MINIMAX_H3_TARGET_RESOLUTION_LABELS: dict[str, str] = {
    "768 highres": "768 highres (native: short edge 768, cap 768x1344)",
    "768 lowres": "768 lowres (long edge 768 — smaller, faster)",
}


@invocation(
    "minimax_h3_ideal_dimensions",
    title="MiniMax H3 Ideal Dimensions",
    tags=["minimax", "video", "dimensions", "math"],
    category="video",
    version="1.1.0",
    classification=Classification.Prototype,
)
class MiniMaxH3IdealDimensionsInvocation(BaseInvocation):
    """Ideal dimensions for MiniMax H3 from a source image's aspect ratio.

    "768 highres" applies the released pipeline's canvas policy: short edge 768, soft
    area cap of 768x1344. "768 lowres" pins the LONG edge to 768 instead, giving a
    canvas with roughly half the pixels for non-square ratios (e.g. a 2:3 source becomes
    512x768 instead of 768x1152) — faster and cheaper, at some cost in fidelity since it
    is below the model's training resolution. Both presets round each axis to the
    nearest multiple of 32. Only the aspect ratio of the inputs matters. Aspect ratios
    beyond 1:4 / 4:1 are rejected. Wire from ``Image Primitive``'s width/height outputs
    and into the width/height of ``Prompt - MiniMax H3``, ``Frame Conditioning - MiniMax
    H3`` and ``Denoise - MiniMax H3`` (all three must share the same canvas).
    """

    width: int = InputField(
        default=1024,
        gt=0,
        description="Source image width in pixels.",
    )
    height: int = InputField(
        default=1024,
        gt=0,
        description="Source image height in pixels.",
    )
    target_resolution: MiniMaxH3TargetResolution = InputField(
        default="768 highres",
        description=(
            "Canvas preset. '768 highres' is H3's native policy (short edge 768, soft cap "
            "768x1344). '768 lowres' pins the long edge to 768 for smaller, faster test "
            "renders (e.g. 512x768 from a 2:3 source) at some cost in fidelity."
        ),
        ui_choice_labels=MINIMAX_H3_TARGET_RESOLUTION_LABELS,
    )

    def invoke(self, context: InvocationContext) -> IdealSizeOutput:
        resolver = resolve_lowres_canvas_size if self.target_resolution == "768 lowres" else resolve_canvas_size
        # Both resolvers return (height, width).
        h, w = resolver(float(self.width), float(self.height))
        return IdealSizeOutput(width=w, height=h)
