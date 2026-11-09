"""How the PiD nodes save the decoder's image."""

import torch
from einops import rearrange
from PIL import Image

from invokeai.app.invocations.vae.precise_io import precise_signed_decode_raster, save_decoded_image
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.shared.invocation_context import InvocationContext


def save_pid_image(context: InvocationContext, x0: torch.Tensor) -> ImageDTO:
    """Saves the first sample of `x0`, the PiD decoder's `[B, 3, H, W]` output in [-1, 1].

    `PiDDecoder.decode` has already clamped it, so with precise image storage on the stored float32 image keeps
    finer gradation than the 8-bit PNG but no values beyond 0-1. The gallery PNG is the truncating quantizer the
    nodes always used.
    """
    raster = precise_signed_decode_raster(context, x0)
    img = rearrange(x0[0].clamp(-1, 1), "c h w -> h w c")
    legacy = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())
    return save_decoded_image(context, legacy, raster)
