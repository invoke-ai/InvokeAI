"""The precise-image policy the model nodes share (VAE decodes and encodes, upscalers, depth): how a model output is
turned into a raster and saved, and how an image is loaded as model input.

A node keeps only its value map (model output to unit range) and its legacy 8-bit quantizer; the policy for the
opt-in precise path lives here once.
"""

import numpy as np
import torch
from PIL.Image import Image as PILImage

from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.raster.convert import decoded_unit_raster
from invokeai.backend.image_util.raster.descriptor import (
    FIXED_CHANNEL_COUNTS,
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    ValueRange,
)
from invokeai.backend.image_util.raster.precise_asset import (
    PreciseAssetError,
    check_precise_asset_bounds,
    check_precise_shape_bounds,
)
from invokeai.backend.image_util.raster.raster import RasterImage
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    image_resized_to_grid_as_tensor,
    raster_as_unit_tensor,
    raster_resized_to_grid_as_tensor,
)


def precise_decode_raster(
    context: InvocationContext, unit_image: torch.Tensor, alpha: bool = False
) -> RasterImage | None:
    """The raster a decode should save precisely, or None for the ordinary 8-bit save.

    None when precise storage is disabled, and (with a warning) when the first sample is not finite: such an output
    (e.g. SDXL fp16 overflow) used to save a black image and must not start failing. Call it before the device tensors
    are released; the finiteness check runs where the tensor lives.
    """
    if not context.images.precise_storage_enabled:
        return None
    raster = decoded_unit_raster(unit_image, alpha=alpha)
    if raster is None:
        context.logger.warning(
            "Precise image storage skipped: the decoder output is not finite; saving the ordinary 8-bit image."
        )
    return raster


def precise_signed_decode_raster(
    context: InvocationContext, decoded: torch.Tensor, alpha: bool = False
) -> RasterImage | None:
    """`precise_decode_raster` for a decoder whose output is in [-1, 1]: the first sample mapped to unit range.

    The sample is copied to the host and widened there, so the precise path allocates no device memory and never
    touches `decoded`, which the legacy quantizer still reads. It is widened before `+ 1.0`: a bf16 or fp16
    sum rounds away the precision this path exists to keep. The result is not clamped.
    """
    if not context.images.precise_storage_enabled:
        return None
    unit = decoded[0].to("cpu", torch.float32, copy=True)
    unit.add_(1.0).div_(2)
    return precise_decode_raster(context, unit, alpha=alpha)


_FLOAT_OUTPUT_CHANNELS = {1: RasterChannels.L, 3: RasterChannels.RGB, 4: RasterChannels.RGBA}
# Rows per block of the finiteness check, so its temporaries stay small next to the image.
_FINITE_CHECK_ROWS = 256


def float_output_fits(context: InvocationContext, height: int, width: int, channels: int) -> bool:
    """Whether a node should keep a float32 HxWxC output for a precise save: precise storage is on and the output is
    within the precise asset bounds. Ask before allocating the buffer; when the bounds refuse, this logs the skip
    warning and the node runs exactly its ordinary 8-bit path."""
    if not context.images.precise_storage_enabled:
        return False
    try:
        check_precise_shape_bounds(height, width, channels, np.dtype(np.float32).itemsize)
    except PreciseAssetError as error:
        context.logger.warning(f"Precise image storage skipped: {error}; saving the ordinary 8-bit image.")
        return False
    return True


def float_output_raster(context: InvocationContext, buffer: np.ndarray) -> RasterImage | None:
    """A zero-copy raster over a float32 HxWxC model output (1, 3 or 4 channels: L, RGB, RGBA with straight alpha),
    or None with a warning when it holds a non-finite value, which the ordinary 8-bit save has always tolerated.

    The caller must not change `buffer` while the raster is in use. The finiteness check runs in row blocks.
    """
    for top in range(0, buffer.shape[0], _FINITE_CHECK_ROWS):
        if not np.isfinite(buffer[top : top + _FINITE_CHECK_ROWS]).all():
            context.logger.warning(
                "Precise image storage skipped: the model output is not finite; saving the ordinary 8-bit image."
            )
            return None
    channels = _FLOAT_OUTPUT_CHANNELS[buffer.shape[2]]
    descriptor = RasterDescriptor(
        channels=channels,
        color=ColorEncoding.SRGB,
        value_range=ValueRange.UNIT,
        alpha=AlphaMode.STRAIGHT if channels is RasterChannels.RGBA else AlphaMode.NONE,
        channel_count=FIXED_CHANNEL_COUNTS[channels],
    )
    return RasterImage.borrow(buffer, descriptor)


def save_decoded_image(context: InvocationContext, legacy: PILImage, raster: RasterImage | None) -> ImageDTO:
    """Saves a decode: `raster` (from `precise_decode_raster`) precisely with `legacy` as its gallery PNG, otherwise
    `legacy` as an ordinary image. A raster outside the precise asset bounds falls back to the ordinary save with a
    warning, as the node saved a PNG there before."""
    if raster is not None:
        try:
            check_precise_asset_bounds(raster)
        except PreciseAssetError as error:
            context.logger.warning(f"Precise image storage skipped: {error}; saving the ordinary 8-bit image.")
        else:
            return context.images.save_raster(raster, display_image=legacy)
    return context.images.save(image=legacy)


def load_precise_unit_tensor(context: InvocationContext, image_name: str, alpha: bool = False) -> torch.Tensor | None:
    """CHW float32 unit-range tensor of an image's precise asset, or None when it has none (the caller then keeps its
    unchanged 8-bit path)."""
    if not context.images.has_precise_asset(image_name):
        return None
    return raster_as_unit_tensor(context.images.get_raster(image_name), alpha=alpha)


def load_image_as_grid_tensor(
    context: InvocationContext, image_name: str, normalize: bool = True, multiple_of: int = 8
) -> torch.Tensor:
    """CHW float32 VAE input for an image. An image without a precise asset takes exactly the 8-bit path (so its
    latents never change); one with an asset encodes its stored values, unclamped."""
    if context.images.has_precise_asset(image_name):
        return raster_resized_to_grid_as_tensor(
            context.images.get_raster(image_name), normalize=normalize, multiple_of=multiple_of
        )
    return image_resized_to_grid_as_tensor(
        context.images.get_pil(image_name).convert("RGB"), normalize=normalize, multiple_of=multiple_of
    )
