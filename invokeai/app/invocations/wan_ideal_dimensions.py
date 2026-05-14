"""Compute Wan 2.2 I2V-compatible pixel dimensions for a target short-side resolution.

Wan's transformer ``patch_size=(1, 2, 2)`` combined with the VAE's 8x spatial
compression requires pixel dimensions to be multiples of 16 (see
``wan_ref_image_encoder.py``). This node takes a source image's W×H and a
target short-side preset (480p / 720p / 1080p) and returns the scaled,
snapped (width, height) that can be fed directly into ``wan_ref_image_encoder``
and the matching ``wan_denoise`` inputs.
"""

import math
from typing import Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.ideal_size import IdealSizeOutput
from invokeai.app.services.shared.invocation_context import InvocationContext

WanTargetResolution = Literal["480p", "720p", "1080p"]

# Short-side pixel count for each preset. "p" notation is by convention the *short*
# dimension in modern video (so a portrait 720p video is 720 wide × 1280 tall).
WAN_TARGET_RESOLUTION_PX: dict[str, int] = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}

WAN_TARGET_RESOLUTION_LABELS: dict[str, str] = {
    "480p": "480p (Wan native)",
    "720p": "720p (Wan native, default)",
    "1080p": "1080p (extrapolated — not a Wan training size)",
}


@invocation(
    "wan_i2v_ideal_dimensions",
    title="Wan 2.2 I2V Ideal Dimensions",
    tags=["wan", "video", "dimensions", "math"],
    category="video",
    version="1.1.0",
)
class WanI2VIdealDimensionsInvocation(BaseInvocation):
    """Compute Wan I2V-compatible (width, height) for a chosen resolution preset.

    Scales the input W×H so the shorter side equals the chosen preset (480 / 720 /
    1080 px), then snaps each dimension to a multiple of 16 (Wan's pixel-grid
    constraint). Wire from ``Image Primitive``'s width/height outputs and into
    ``wan_ref_image_encoder`` / ``wan_denoise``.
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
    target_resolution: WanTargetResolution = InputField(
        default="720p",
        description=(
            "Short-side resolution preset. 480p and 720p are Wan 2.2's native training "
            "resolutions; 1080p works but is extrapolation and costs ~2.25x the memory "
            "of 720p."
        ),
        ui_choice_labels=WAN_TARGET_RESOLUTION_LABELS,
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

        target_short_side = WAN_TARGET_RESOLUTION_PX[self.target_resolution]
        # Reject sources so narrow that the scaled long side is still under one Wan
        # pixel grid (16 px). The downstream clamp to ``max(w, 16)`` would otherwise
        # silently return 16×16, which has no relation to the requested aspect ratio
        # — better to fail fast and have the workflow author fix the inputs.
        long_side = max(self.width, self.height)
        if long_side < 16:
            raise ValueError(
                f"Source longer side ({long_side}px) is smaller than the Wan pixel grid (16px). "
                "Use an input image at least 16px on its longer side."
            )

        scale = target_short_side / short
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

        # Belt-and-suspenders clamp against floor-of-<16 — should be unreachable now
        # that the long_side guard above runs first, but keeps the contract that the
        # returned dimensions are always valid Wan inputs.
        w = max(w, 16)
        h = max(h, 16)
        return IdealSizeOutput(width=w, height=h)
