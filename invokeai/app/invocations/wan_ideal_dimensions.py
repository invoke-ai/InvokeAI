"""Compute Wan 2.2 I2V-compatible pixel dimensions for a target short-side resolution.

Wan's transformer ``patch_size=(1, 2, 2)`` combined with the VAE's 8x spatial
compression requires pixel dimensions to be multiples of 16 (see
``wan_ref_image_encoder.py``). This node takes a source image's W×H and a
target short-side (e.g. 720 for "720p") and returns the scaled, snapped
(width, height) that can be fed directly into ``wan_ref_image_encoder`` and
the matching ``wan_denoise`` inputs.
"""

import math
from typing import Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.ideal_size import IdealSizeOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "wan_i2v_ideal_dimensions",
    title="Wan 2.2 I2V Ideal Dimensions",
    tags=["wan", "video", "dimensions", "math"],
    category="video",
    version="1.0.0",
)
class WanI2VIdealDimensionsInvocation(BaseInvocation):
    """Compute Wan I2V-compatible (width, height) for a target short-side resolution.

    Scales the input W×H so the shorter side equals ``target_short_side``, then snaps
    each dimension to a multiple of 16 (Wan's pixel-grid constraint). Wire from
    ``Image Primitive``'s width/height outputs and into ``wan_ref_image_encoder`` /
    ``wan_denoise``.
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
    target_short_side: int = InputField(
        default=720,
        ge=16,
        description=(
            "The short side of the output dimensions in pixels. Common Wan values: "
            "480 (Wan 480p) and 720 (Wan 720p)."
        ),
    )
    rounding: Literal["nearest", "floor", "ceiling"] = InputField(
        default="nearest",
        description=(
            "How to snap each dimension to a multiple of 16. 'floor' rounds down — "
            "safest for VRAM, guaranteed not to exceed the unsnapped target. "
            "'ceiling' rounds up. 'nearest' minimizes aspect-ratio drift (default)."
        ),
    )

    def invoke(self, context: InvocationContext) -> IdealSizeOutput:
        short = min(self.width, self.height)
        if short <= 0:
            raise ValueError("Source dimensions must be positive.")

        scale = self.target_short_side / short
        raw_w = self.width * scale
        raw_h = self.height * scale

        if self.rounding == "floor":
            w = int(raw_w // 16) * 16
            h = int(raw_h // 16) * 16
        elif self.rounding == "ceiling":
            w = int(math.ceil(raw_w / 16)) * 16
            h = int(math.ceil(raw_h / 16)) * 16
        else:  # nearest
            w = round(raw_w / 16) * 16
            h = round(raw_h / 16) * 16

        # Guard against zero from extreme inputs (e.g. floor of <16 raw value).
        w = max(w, 16)
        h = max(h, 16)
        return IdealSizeOutput(width=w, height=h)
