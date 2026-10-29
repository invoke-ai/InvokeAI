"""Canvas resolution for LTX-2 from a source image's aspect ratio."""

from typing import Literal

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ltx2.constants import LTX2_CANVAS_MULTIPLE, LTX2_TWO_STAGE_CANVAS_MULTIPLE
from invokeai.backend.ltx2.packing import base_canvas, resolve_canvas

LTX2TargetResolution = Literal["512p", "704p", "768p", "1024p", "1536p"]

LTX2_TARGET_RESOLUTION_LABELS: dict[str, str] = {
    "512p": "512p (short edge 512 - fastest)",
    "704p": "704p (short edge 704)",
    "768p": "768p (short edge 768 - sharpest, slowest)",
    "1024p": "1024p (short edge 1024, two-stage)",
    "1536p": "1536p (short edge 1536, two-stage - very slow)",
}

_SHORT_EDGES: dict[str, int] = {"512p": 512, "704p": 704, "768p": 768, "1024p": 1024, "1536p": 1536}

# Which presets are a base pass plus a refine pass. Their canvas is chosen on the doubled grid so
# the base pass, which runs at exactly half of it, still lands on the plain one.
LTX2_TWO_STAGE_RESOLUTIONS: frozenset[str] = frozenset({"1024p", "1536p"})


@invocation_output("ltx2_ideal_dimensions_output")
class LTX2IdealDimensionsOutput(BaseInvocationOutput):
    """The canvas a generation runs at, and for a two-stage preset the pass that precedes it."""

    width: int = OutputField(description="Width of the final canvas.")
    height: int = OutputField(description="Height of the final canvas.")
    # A two-stage preset generates at half its canvas and refines an upscaled latent, so a workflow
    # built from the final size alone would be a single pass at a canvas the checkpoints never saw.
    # These are that pass's size; on a single-stage preset they are the same numbers.
    base_width: int = OutputField(description="Width of the base pass; equals `width` on one-stage presets.")
    base_height: int = OutputField(description="Height of the base pass; equals `height` on one-stage presets.")
    two_stage: bool = OutputField(description="Whether this preset runs a base pass and then a refine pass.")


@invocation(
    "ltx2_ideal_dimensions",
    title="LTX-2 Ideal Dimensions",
    tags=["ltx", "ltx2", "video", "dimensions", "math"],
    category="video",
    version="1.1.0",
    classification=Classification.Prototype,
)
class LTX2IdealDimensionsInvocation(BaseInvocation):
    """Ideal LTX-2 dimensions for a source image's aspect ratio.

    The chosen preset pins the canvas's *short* edge and the long edge follows the source's aspect
    ratio. Only the ratio of the inputs matters.

    One-stage presets (512p, 704p, 768p) round each axis to a multiple of 32, the video VAE's
    spatial compression: wire `width`/`height` into both the image conditioning and the denoise
    node, which must share one canvas.

    Two-stage presets (1024p, 1536p) round to 64 instead, because generation happens at half the
    canvas: wire `base_width`/`base_height` into the image conditioning and the *first* denoise
    node, and `width`/`height` into the refine denoise that follows the latent upscaler. Wiring the
    final size into a single pass renders at a canvas the checkpoints were not trained for, and
    nothing will report it.

    Cost grows with the token count, which is the canvas area over 1024: 768p is about 2.2x the
    work of 512p at the same length.
    """

    width: int = InputField(default=1024, gt=0, description="Source image width in pixels.")
    height: int = InputField(default=1024, gt=0, description="Source image height in pixels.")
    target_resolution: LTX2TargetResolution = InputField(
        default="704p",
        description="Which short edge to pin the canvas to.",
        ui_choice_labels=LTX2_TARGET_RESOLUTION_LABELS,
    )

    def invoke(self, context: InvocationContext) -> LTX2IdealDimensionsOutput:
        two_stage = self.target_resolution in LTX2_TWO_STAGE_RESOLUTIONS
        height, width = resolve_canvas(
            float(self.width),
            float(self.height),
            _SHORT_EDGES[self.target_resolution],
            multiple=LTX2_TWO_STAGE_CANVAS_MULTIPLE if two_stage else LTX2_CANVAS_MULTIPLE,
        )
        base_height, base_width = base_canvas(height, width) if two_stage else (height, width)

        return LTX2IdealDimensionsOutput(
            width=width,
            height=height,
            base_width=base_width,
            base_height=base_height,
            two_stage=two_stage,
        )
