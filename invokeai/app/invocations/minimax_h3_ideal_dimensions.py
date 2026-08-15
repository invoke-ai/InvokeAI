"""Compute the MiniMax H3 canvas for a source image's aspect ratio.

Unlike Wan (which offers several target resolutions), H3 was released for exactly one
canvas family: a 768 px short edge under a soft area cap of 768x1344, both axes rounded
to the nearest multiple of 32, aspect ratios limited to 1:4 through 4:1. That policy is
implemented by :func:`invokeai.backend.minimax_h3.packing.resolve_canvas_size` (a
first-party port of the released pipeline's canvas resolution); this node is a thin
workflow-graph wrapper so image-to-video workflows can derive the canvas from the
uploaded keyframe instead of asking the user to type matching width/height values.
"""

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.ideal_size import IdealSizeOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.minimax_h3.packing import resolve_canvas_size


@invocation(
    "minimax_h3_ideal_dimensions",
    title="MiniMax H3 Ideal Dimensions",
    tags=["minimax", "video", "dimensions", "math"],
    category="video",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3IdealDimensionsInvocation(BaseInvocation):
    """Ideal dimensions for MiniMax H3 from a source image's aspect ratio.

    Applies the released pipeline's canvas policy: short edge 768, soft area cap of
    768x1344, both axes rounded to the nearest multiple of 32. Only the aspect ratio of
    the inputs matters. Aspect ratios beyond 1:4 / 4:1 are rejected. Wire from ``Image
    Primitive``'s width/height outputs and into the width/height of ``Prompt - MiniMax
    H3``, ``Frame Conditioning - MiniMax H3`` and ``Denoise - MiniMax H3`` (all three
    must share the same canvas).
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

    def invoke(self, context: InvocationContext) -> IdealSizeOutput:
        # resolve_canvas_size returns (height, width).
        h, w = resolve_canvas_size(float(self.width), float(self.height))
        return IdealSizeOutput(width=w, height=h)
