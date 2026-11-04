"""Float counterparts of the Pillow compositing operations the image nodes use (`Image.paste`, `alpha_composite`,
`convert`), on float32 sRGB unit-range pixels with straight alpha.

They reproduce Pillow's per-channel arithmetic without its 8-bit rounding: values beyond 0-1 survive and a mask is a
coverage in [0, 1] (an 8-bit mask divided by 255).
"""

import numpy as np

from invokeai.backend.image_util.raster.convert import srgb_descriptor
from invokeai.backend.image_util.raster.descriptor import RasterChannels
from invokeai.backend.image_util.raster.raster import RasterImage
from invokeai.backend.image_util.raster.resample import MIN_VISIBLE_ALPHA

# Pillow's RGB -> L weights (ITU-R 601-2 luma, 16-bit fixed point).
_LUMA = np.array([19595, 38470, 7471], np.float32) / np.float32(65536)
_CHANNELS_BY_COUNT = {1: RasterChannels.L, 2: RasterChannels.LA, 3: RasterChannels.RGB, 4: RasterChannels.RGBA}
# Compositing runs over row blocks of about this many samples, so temporaries stay small next to the image.
_BLOCK_SAMPLES = 1 << 20


def unit_srgb_raster(pixels: np.ndarray) -> RasterImage:
    """An sRGB unit raster with straight alpha from float32 `(H, W, C)` pixels, C in 1-4.

    Fresh arrays (owning their memory, contiguous, writeable) are adopted without a copy; anything else, such as a view
    of a larger buffer, is copied so the raster never freezes or retains memory it does not own.
    """
    channels = _CHANNELS_BY_COUNT[pixels.shape[2]]
    owned = pixels.dtype == np.float32 and pixels.base is None and pixels.flags.c_contiguous and pixels.flags.writeable
    return RasterImage._adopt(pixels if owned else np.array(pixels, np.float32, order="C"), srgb_descriptor(channels))


def convert_channels(pixels: np.ndarray, source: RasterChannels, target: RasterChannels) -> np.ndarray:
    """float32 `(H, W, C)` pixels in the `target` channels, like `Image.convert` between L, LA, RGB and RGBA.

    Gray is replicated into color, color becomes luma, a missing alpha is opaque, and alpha is dropped without
    compositing. Returns `pixels` itself when the channels already match.
    """
    if source is target:
        return pixels
    has_alpha = source in (RasterChannels.LA, RasterChannels.RGBA)
    color = pixels[:, :, :-1] if has_alpha else pixels
    out = np.empty(pixels.shape[:2] + (len(target.value),), np.float32)
    target_color = out.shape[2] - 1 if target in (RasterChannels.LA, RasterChannels.RGBA) else out.shape[2]
    if target_color == color.shape[2]:
        out[:, :, :target_color] = color
    elif target_color == 3:
        out[:, :, :3] = color  # gray replicated
    else:
        out[:, :, 0] = color @ _LUMA
    if target_color != out.shape[2]:
        out[:, :, -1] = pixels[:, :, -1] if has_alpha else 1.0
    return out


def _clip_to(destination_shape: tuple[int, ...], source_shape: tuple[int, ...], left: int, top: int):
    """(x0, y0, x1, y1) of the part of a source pasted at (left, top) that lies inside the destination."""
    return (
        max(left, 0),
        max(top, 0),
        min(left + source_shape[1], destination_shape[1]),
        min(top + source_shape[0], destination_shape[0]),
    )


def paste_blend(
    destination: np.ndarray,
    source: np.ndarray,
    left: int,
    top: int,
    coverage: np.ndarray | None = None,
    *,
    in_place: bool = False,
) -> np.ndarray:
    """`destination` with `source` pasted at (left, top), like `destination.paste(source, (left, top), mask)`.

    Every channel, alpha included, becomes `destination * (1 - coverage) + source * coverage`; without a coverage the
    source replaces the destination. The part of the source outside the destination is dropped. `coverage` is
    `(h, w)` like the source. The result is a copy unless `in_place`, which writes into `destination` itself.
    """
    out = destination if in_place else destination.copy()
    x0, y0, x1, y1 = _clip_to(out.shape, source.shape, left, top)
    if x0 >= x1 or y0 >= y1:
        return out
    region = source[y0 - top : y1 - top, x0 - left : x1 - left]
    if coverage is None:
        out[y0:y1, x0:x1] = region
        return out
    rows = max(1, _BLOCK_SAMPLES // ((x1 - x0) * out.shape[2]))
    for start in range(y0, y1, rows):
        stop = min(start + rows, y1)
        weight = coverage[start - top : stop - top, x0 - left : x1 - left, np.newaxis]
        view = out[start:stop, x0:x1]
        view[...] = view * (np.float32(1) - weight) + region[start - y0 : stop - y0] * weight
    return out


def alpha_composite(base: np.ndarray, top: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """`top` over `base`, both straight-alpha RGBA of the same size, like `Image.alpha_composite`.

    Where `top` is not visible (alpha below half an 8-bit step, which Pillow's 8-bit alpha rounds to 0) the result is
    `base` unchanged, hidden color included. `out` may be `base` itself.
    """
    top_alpha = top[:, :, 3:4]
    base_alpha = base[:, :, 3:4]
    out_alpha = top_alpha + base_alpha * (np.float32(1) - top_alpha)
    visible = top_alpha >= MIN_VISIBLE_ALPHA
    weight = np.divide(top_alpha, out_alpha, out=np.zeros_like(out_alpha), where=visible)
    if out is None:
        out = np.empty_like(base)
    out[:, :, :3] = base[:, :, :3] * (np.float32(1) - weight) + top[:, :, :3] * weight
    out[:, :, 3:4] = np.where(visible, out_alpha, base_alpha)
    return out


def composite_layer(
    canvas: np.ndarray, layer: np.ndarray, left: int, top: int, coverage: np.ndarray | None = None
) -> None:
    """Alpha-composites `layer`, scaled per channel (alpha included) by `coverage`, over `canvas` at (left, top), in
    place and only over the region the layer covers. Both are straight-alpha RGBA.

    This is a layer pasted with a mask onto a transparent image and then alpha-composited, without building the
    full-size layer.
    """
    x0, y0, x1, y1 = _clip_to(canvas.shape, layer.shape, left, top)
    if x0 >= x1 or y0 >= y1:
        return
    rows = max(1, _BLOCK_SAMPLES // ((x1 - x0) * 4))
    for start in range(y0, y1, rows):
        stop = min(start + rows, y1)
        block = layer[start - top : stop - top, x0 - left : x1 - left]
        if coverage is not None:
            block = block * coverage[start - top : stop - top, x0 - left : x1 - left, np.newaxis]
        view = canvas[start:stop, x0:x1]
        alpha_composite(view, block, out=view)
