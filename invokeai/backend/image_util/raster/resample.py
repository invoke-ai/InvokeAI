"""Float resampling that matches Pillow's own filters, without Pillow's 8-bit intermediate."""

import numpy as np
from PIL import Image

from invokeai.backend.image_util.raster.descriptor import (
    AlphaMode,
    RasterChannels,
    RasterConversionError,
    ValueRange,
)
from invokeai.backend.image_util.raster.raster import RasterImage

# Pillow stores resampled alpha in 8 bits, so an alpha below half a step is 0 and its pixel is fully transparent. That
# also keeps filter ringing, which drives alpha near hard edges to ~1e-4, from blowing up the unpremultiply.
MIN_VISIBLE_ALPHA = 0.5 / 255


def _resize_plane(plane: np.ndarray, width: int, height: int, resample: Image.Resampling) -> np.ndarray:
    """One float32 `(H, W)` channel resized as its own Pillow `F` image with the given filter."""
    return np.asarray(Image.fromarray(np.ascontiguousarray(plane), "F").resize((width, height), resample))


def resize_raster(raster: RasterImage, width: int, height: int, resample: Image.Resampling) -> RasterImage:
    """Resizes a float32 sRGB UNIT raster like `Image.resize((width, height), resample)` on the equivalent 8-bit image.

    Each channel goes through Pillow's resampler as an `F` image, so edge handling and filter support are Pillow's, and
    values beyond 0-1 survive. Like Pillow, a straight-alpha raster (LA, RGBA) is resampled premultiplied by alpha
    unless the filter is NEAREST, so color under transparent pixels does not bleed. Source alpha is clamped to [0, 1]
    and so is the result's; a result pixel whose alpha is below half an 8-bit step gets color 0; unpremultiplied color
    is held to the visible source color's range widened by its span on both sides, so ordinary filter ringing passes
    but the quotient of two ringing signals near a soft alpha edge cannot blow up. A same-size resize returns the input.
    """
    descriptor = raster.descriptor
    if raster.dtype != np.float32:
        raise RasterConversionError(f"resizing needs float32 pixels, got {raster.dtype}")
    if descriptor.alpha is AlphaMode.PREMULTIPLIED or descriptor.channels is RasterChannels.DATA:
        raise RasterConversionError("resizing needs straight alpha and color channels")
    if descriptor.value_range is not ValueRange.UNIT:
        raise RasterConversionError("resizing needs a UNIT value range")
    if (width, height) == (raster.width, raster.height):
        return raster  # rasters are immutable

    source = raster.view()
    out = np.empty((height, width, source.shape[2]), np.float32)
    has_alpha = descriptor.alpha is not AlphaMode.NONE
    color_count = source.shape[2] - 1 if has_alpha else source.shape[2]

    if not has_alpha or resample == Image.Resampling.NEAREST or source[:, :, -1].min() >= 1.0:
        # No alpha to weigh by (or alpha that is 1 everywhere, which stays 1).
        for index in range(color_count):
            out[:, :, index] = _resize_plane(source[:, :, index], width, height, resample)
        if has_alpha:
            out[:, :, -1] = (
                _resize_plane(source[:, :, -1], width, height, resample)
                if resample == Image.Resampling.NEAREST
                else 1.0
            )
        return RasterImage._adopt(out, descriptor)

    alpha = np.clip(source[:, :, -1], 0.0, 1.0)
    covered = (alpha > 0)[:, :, np.newaxis]
    color = source[:, :, :color_count]
    if covered.any():
        low = np.min(color, where=covered, initial=np.inf)
        high = np.max(color, where=covered, initial=-np.inf)
        low, high = low - (high - low), high + (high - low)
    else:
        low = high = np.float32(0)
    # Color is divided by the unclamped alpha, which rings together with the premultiplied color, so a flat color stays
    # exactly that color where the alpha overshoots.
    raw_opacity = _resize_plane(alpha, width, height, resample)
    visible = raw_opacity >= MIN_VISIBLE_ALPHA
    divisor = np.where(visible, raw_opacity, np.float32(1))
    for index in range(color_count):
        plane = _resize_plane(color[:, :, index] * alpha, width, height, resample)
        plane = plane / divisor
        np.clip(plane, low, high, out=plane)
        plane *= visible
        out[:, :, index] = plane
    out[:, :, -1] = np.clip(raw_opacity, 0.0, 1.0)
    return RasterImage._adopt(out, descriptor)
