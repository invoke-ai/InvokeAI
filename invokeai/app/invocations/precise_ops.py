"""Helpers shared by the image operation nodes that have a float path for images with a precise asset.

The backend math lives in `invokeai/backend/image_util/raster/` (`composite.py`, `resample.py`). These functions
connect it to nodes: routing by asset presence, reading unit-range pixels and saving the result through an
invocation context, plus two small array helpers (`mask_coverage`, `require_same_size`, `writable`) the nodes share.
"""

import numpy
from PIL import Image

from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.raster.composite import convert_channels
from invokeai.backend.image_util.raster.convert import quantize_unit, to_unit_float
from invokeai.backend.image_util.raster.descriptor import RasterChannels
from invokeai.backend.image_util.raster.precise_asset import PreciseAssetError, check_precise_shape_bounds
from invokeai.backend.image_util.raster.raster import RasterImage


def precise_source(context: InvocationContext, image_name: str) -> RasterImage | None:
    """The image's raster when it has a precise asset, else None (the node then keeps its PIL path).

    A precise asset is always a displayable sRGB unit-range raster with straight or no alpha: storage refuses any other.
    """
    if not context.images.has_precise_asset(image_name):
        return None
    return context.images.get_raster(image_name)


def any_precise(context: InvocationContext, *image_names: str) -> bool:
    """Whether any of the images has a precise asset."""
    return any(context.images.has_precise_asset(name) for name in image_names)


def unit_float32(raster: RasterImage) -> RasterImage:
    """The raster as float32 unit values (integers divided by their full scale)."""
    return raster if raster.dtype == numpy.float32 else to_unit_float(raster, numpy.float32)


def save_precise(context: InvocationContext, raster: RasterImage) -> ImageDTO:
    """Saves a float-path result: precise when precise storage is on and the result fits the precise asset bounds,
    else explicitly display-quantized to 8 bits (with a warning when the bounds are the reason)."""
    keep_precise = context.images.precise_storage_enabled
    if keep_precise:
        try:
            check_precise_shape_bounds(raster.height, raster.width, raster.channel_count, raster.dtype.itemsize)
        except PreciseAssetError as error:
            context.logger.warning(f"Precise image storage skipped: {error}; saving the ordinary 8-bit image.")
            keep_precise = False
    if not keep_precise:
        raster = quantize_unit(raster, numpy.uint8)
    return context.images.save_raster(raster)


def unit_pixels(context: InvocationContext, image_name: str) -> tuple[numpy.ndarray, RasterChannels]:
    """Any image as float32 unit pixels and their channels: the precise asset, or the gallery PNG divided by its full
    scale. A 16-bit gray gallery PNG is therefore widened by 65535, where the 8-bit path's `convert("RGBA")` clips it
    to 255; the float path keeps its precision."""
    raster = unit_float32(context.images.get_raster(image_name))
    return raster.view(), raster.descriptor.channels


def rgba_unit(context: InvocationContext, image_name: str) -> numpy.ndarray:
    """Any image as float32 unit RGBA pixels (read-only when it was RGBA already)."""
    pixels, channels = unit_pixels(context, image_name)
    return convert_channels(pixels, channels, RasterChannels.RGBA)


def mask_coverage(mask: Image.Image) -> numpy.ndarray:
    """An 8-bit "L" mask as coverage: the value divided by 255. Masks stay 8-bit data; they are never precise."""
    return numpy.asarray(mask, dtype=numpy.float32) / numpy.float32(255)


def writable(pixels: numpy.ndarray) -> numpy.ndarray:
    """`pixels` if the caller may change it (a fresh array), else a copy: raster views are read-only."""
    return pixels if pixels.flags.writeable else pixels.copy()


def require_same_size(pixels: numpy.ndarray, coverage: numpy.ndarray) -> None:
    """Raises like Pillow when a mask does not have the size of the pixels it applies to."""
    if pixels.shape[:2] != coverage.shape:
        raise ValueError("images do not match")
