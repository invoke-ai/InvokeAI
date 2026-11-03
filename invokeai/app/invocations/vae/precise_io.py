"""The precise-image policy the VAE nodes share: how a decode is saved and how an image is loaded for encoding.

A family keeps only its value map (decoder output to unit range) and its legacy 8-bit quantizer; the policy for the
opt-in precise path lives here once.
"""

import torch
from PIL.Image import Image as PILImage

from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.raster.convert import decoded_unit_raster
from invokeai.backend.image_util.raster.precise_asset import PreciseAssetError, check_precise_asset_bounds
from invokeai.backend.image_util.raster.raster import RasterImage
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    image_resized_to_grid_as_tensor,
    raster_resized_to_grid_as_tensor,
)


def precise_decode_raster(context: InvocationContext, unit_image: torch.Tensor) -> RasterImage | None:
    """The raster a decode should save precisely, or None for the ordinary 8-bit save.

    None when precise storage is disabled, and (with a warning) when the first sample is not finite: such an output
    (e.g. SDXL fp16 overflow) used to save a black image and must not start failing. Call it before the device tensors
    are released; the finiteness check runs on their device.
    """
    if not context.images.precise_storage_enabled:
        return None
    raster = decoded_unit_raster(unit_image)
    if raster is None:
        context.logger.warning(
            "Precise image storage skipped: the VAE decoder output is not finite; saving the ordinary 8-bit image."
        )
    return raster


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
