"""Compute Wan 2.2-compatible pixel dimensions for a target short-side resolution.

Wan's transformer ``patch_size=(1, 2, 2)`` adds a 2x patchify on top of the VAE's
spatial compression, so pixel dimensions must be a multiple of (2 × VAE scale):

- I2V-A14B / T2V (8x VAE)            → multiples of 16
- TI2V-5B (Wan 2.2-VAE, 16x VAE)     → multiples of 32

This module exposes one node per family. Each takes a source image's W×H and a
target short-side preset (480p / 720p / 1080p) and returns the scaled, snapped
(width, height) that can be fed directly into the matching ``wan_ref_image_encoder``
/ ``wan_denoise`` inputs. Both nodes share :func:`_scale_and_snap`; they differ
only in the pixel multiple they snap to.
"""

import math
from typing import Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField
from invokeai.app.invocations.ideal_size import IdealSizeOutput
from invokeai.app.services.shared.invocation_context import InvocationContext

WanTargetResolution = Literal["480p", "720p", "1080p"]
WanRounding = Literal["nearest", "floor", "ceiling"]

# Pixel-grid multiple = 2 (transformer patch) × VAE spatial scale. The 8x-VAE
# I2V/T2V models need multiples of 16; the 16x-VAE TI2V-5B needs multiples of 32.
WAN_I2V_PIXEL_MULTIPLE = 16
WAN_TI2V_PIXEL_MULTIPLE = 32

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


def _scale_and_snap(
    width: int,
    height: int,
    target_short_side: int,
    rounding: WanRounding,
    multiple: int,
) -> tuple[int, int]:
    """Scale a source W×H so its shorter side equals ``target_short_side``, then
    snap each dimension to ``multiple`` using the requested rounding mode.

    ``multiple`` is the Wan pixel-grid constraint (16 for the 8x-VAE I2V/T2V
    models, 32 for the 16x-VAE TI2V-5B). Shared by both ideal-dimensions nodes.
    """
    short = min(width, height)
    if short <= 0:
        raise ValueError("Source dimensions must be positive.")

    # Reject sources so narrow that the scaled long side is still under one Wan
    # pixel grid. The downstream clamp to ``max(w, multiple)`` would otherwise
    # silently return multiple×multiple, which has no relation to the requested
    # aspect ratio — better to fail fast and have the workflow author fix inputs.
    long_side = max(width, height)
    if long_side < multiple:
        raise ValueError(
            f"Source longer side ({long_side}px) is smaller than the Wan pixel grid ({multiple}px). "
            f"Use an input image at least {multiple}px on its longer side."
        )

    scale = target_short_side / short
    raw_w = width * scale
    raw_h = height * scale

    if rounding == "floor":
        w = int(raw_w // multiple) * multiple
        h = int(raw_h // multiple) * multiple
    elif rounding == "ceiling":
        w = int(math.ceil(raw_w / multiple)) * multiple
        h = int(math.ceil(raw_h / multiple)) * multiple
    else:  # nearest
        w = round(raw_w / multiple) * multiple
        h = round(raw_h / multiple) * multiple

    # Belt-and-suspenders clamp against floor-of-<multiple — unreachable now that
    # the long_side guard runs first, but keeps the contract that returned
    # dimensions are always valid Wan inputs.
    w = max(w, multiple)
    h = max(h, multiple)
    return w, h


@invocation(
    "wan_i2v_ideal_dimensions",
    title="Wan 2.2 I2V Ideal Dimensions (A14B)",
    tags=["wan", "video", "dimensions", "math"],
    category="video",
    version="1.1.0",
    classification=Classification.Prototype,
)
class WanI2VIdealDimensionsInvocation(BaseInvocation):
    """Ideal dimensions for the Wan 2.2 A14B models (I2V-A14B and T2V-A14B).

    Use this node for the A14B family. For the TI2V-5B model use "Wan 2.2 TI2V
    Ideal Dimensions" instead — TI2V-5B requires multiples of 32, and feeding it
    these multiples-of-16 dims fails the patchify step.

    Scales the input W×H so the shorter side equals the chosen preset (480 / 720 /
    1080 px), then snaps each dimension to a multiple of 16 (the A14B pixel-grid
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
    rounding: WanRounding = InputField(
        default="nearest",
        description=(
            "How to snap each dimension to a multiple of 16. 'floor' rounds down — "
            "safest for VRAM, guaranteed not to exceed the unsnapped target. "
            "'ceiling' rounds up. 'nearest' minimizes aspect-ratio drift (default)."
        ),
    )

    def invoke(self, context: InvocationContext) -> IdealSizeOutput:
        target_short_side = WAN_TARGET_RESOLUTION_PX[self.target_resolution]
        w, h = _scale_and_snap(
            self.width, self.height, target_short_side, self.rounding, multiple=WAN_I2V_PIXEL_MULTIPLE
        )
        return IdealSizeOutput(width=w, height=h)


@invocation(
    "wan_ti2v_ideal_dimensions",
    title="Wan 2.2 TI2V Ideal Dimensions (5B)",
    tags=["wan", "video", "dimensions", "math"],
    category="video",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanTI2VIdealDimensionsInvocation(BaseInvocation):
    """Ideal dimensions for the Wan 2.2 TI2V-5B model.

    Use this node for TI2V-5B only. For the A14B models (I2V-A14B / T2V-A14B) use
    "Wan 2.2 I2V Ideal Dimensions" instead — those need multiples of 16, and this
    node's multiples-of-32 dims would overshoot their pixel grid.

    Identical to the A14B node but snaps each dimension to a multiple of 32 instead
    of 16: the Wan 2.2-VAE used by TI2V-5B applies 16x spatial compression and the
    transformer adds a 2x patch on top, so pixel dims must divide by 32 for the
    patchify step. Wire from ``Image Primitive``'s width/height outputs and into
    the matching ``wan_denoise`` inputs.
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
    rounding: WanRounding = InputField(
        default="nearest",
        description=(
            "How to snap each dimension to a multiple of 32. 'floor' rounds down — "
            "safest for VRAM, guaranteed not to exceed the unsnapped target. "
            "'ceiling' rounds up. 'nearest' minimizes aspect-ratio drift (default)."
        ),
    )

    def invoke(self, context: InvocationContext) -> IdealSizeOutput:
        target_short_side = WAN_TARGET_RESOLUTION_PX[self.target_resolution]
        w, h = _scale_and_snap(
            self.width, self.height, target_short_side, self.rounding, multiple=WAN_TI2V_PIXEL_MULTIPLE
        )
        return IdealSizeOutput(width=w, height=h)
