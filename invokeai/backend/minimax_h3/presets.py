"""InvokeAI's canvas and frame-count presets for MiniMax H3.

First-party policy that sits on top of the released model's geometry. It lives here rather
than in :mod:`invokeai.backend.minimax_h3.packing` because that module is vendored from the
diffusers MiniMax-H3 branch and is kept byte-identical to upstream (modulo the absolute-import
rewrite) so the eventual re-sync is a clean copy — see the vendoring note in
``invokeai/backend/minimax_h3/__init__.py``. Anything InvokeAI adds belongs on this side of
the line.

Two additions:

- :data:`MINIMAX_H3_VIDEO_FRAME_CHOICES`, the ``17n + 5`` grid points the nodes offer as a
  choice list, so a request can never miss the grid.
- :func:`resolve_lowres_canvas_size`, a reduced-size canvas family for fast preview renders.
"""

from typing import Literal

from invokeai.backend.minimax_h3.packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    MINIMAX_H3_SHORT_EDGE,
)

# The released model targets the 5-15 s window, but shorter grid-aligned clips render and are useful as fast
# test passes. 90 frames (3.75 s) is the floor we accept for video requests; below that only the 5-frame
# still-image block is meaningful.
MINIMAX_H3_MIN_VIDEO_FRAMES = 90

# Every legal video frame count, i.e. the `17n + 5` grid points from the accepted floor up to the 15 s ceiling
# (90 ... 345). The nodes offer exactly these as a choice list, so a request can never miss the grid.
MINIMAX_H3_VIDEO_FRAME_CHOICES = tuple(
    n
    for n in range(MINIMAX_H3_MIN_VIDEO_FRAMES, int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS) + 1)
    if n % MINIMAX_H3_FRAMES_PER_CHUNK == MINIMAX_H3_LATENTS_PER_CHUNK
)


def resolve_lowres_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    r"""
    Resolve a display aspect ratio into a reduced-size MiniMax-H3 canvas with a 768 px LONG edge.

    This is not a released H3 canvas family: the model was trained for a 768 short edge (see
    :func:`~invokeai.backend.minimax_h3.packing.resolve_canvas_size`). Pinning the long edge instead yields a
    canvas with roughly half the pixels for non-square ratios — useful for fast preview/test renders at some
    cost in fidelity. The short edge follows the aspect ratio and both axes are rounded to the nearest multiple
    of 32; the area always sits far below the `768 * 1344` cap, so no cap scaling is applied. Only the ratio of
    the two arguments matters.

    The aspect validation is repeated here rather than shared with ``resolve_canvas_size``: factoring it out
    would mean editing the vendored module.

    Args:
        aspect_width (`float`): Width of the target ratio.
        aspect_height (`float`): Height of the target ratio.

    Returns:
        `tuple[int, int]`: the `(height, width)` of the canvas.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")

    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width, height = float(MINIMAX_H3_SHORT_EDGE), MINIMAX_H3_SHORT_EDGE / ratio
    else:
        width, height = MINIMAX_H3_SHORT_EDGE * ratio, float(MINIMAX_H3_SHORT_EDGE)

    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)


def resolve_reduced_canvas_size(
    aspect_width: float, aspect_height: float, short_edge: Literal[480, 544]
) -> tuple[int, int]:
    """Resolve a draft canvas on the 32-pixel grid, retaining H3's native aspect and area bounds.

    The short edge is a target, not a guarantee: extreme ratios are scaled down to the
    native area cap before rounding. These are below-training-resolution preview presets.
    Returns (height, width), like the native and legacy lowres resolvers.
    """
    if short_edge not in (480, 544):
        raise ValueError("MiniMax-H3 draft short edge must be 480 or 544 pixels.")
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got {aspect_width}:{aspect_height}.")
    # Match the frontend/native operation order at half-grid rounding boundaries.
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    if width * height > MINIMAX_H3_MAX_PIXELS:
        scale = (MINIMAX_H3_MAX_PIXELS / (width * height)) ** 0.5
        width, height = width * scale, height * scale
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return max(multiple, round(height / multiple) * multiple), max(multiple, round(width / multiple) * multiple)
