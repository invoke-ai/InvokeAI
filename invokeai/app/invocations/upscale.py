from typing import Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import ConfigDict

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.invocations.vae.precise_io import (
    float_output_fits,
    float_output_raster,
    load_precise_unit_tensor,
    save_decoded_image,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.basicsr.rrdbnet_arch import RRDBNet
from invokeai.backend.image_util.realesrgan.realesrgan import RealESRGAN
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK

# TODO: Populate this from disk?
# TODO: Use model manager to load?
ESRGAN_MODELS = Literal[
    "RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth",
    "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
    "RealESRGAN_x2plus.pth",
]

ESRGAN_MODEL_URLS: dict[str, str] = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}


def _rgba_8bit(rgb_unit: np.ndarray) -> Image.Image:
    """The 8-bit RGBA image of a float HxWx3 (or wider) RGB array: clamp to [0, 1], `* 255`, round half to even, alpha
    255. The same arithmetic as `RealESRGAN.upscale` plus `.convert("RGBA")`, in row blocks so no full-size float
    temporary is made."""
    height, width = rgb_unit.shape[:2]
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 3] = 255
    rows = max(1, (1 << 22) // (width * 3))
    with np.errstate(invalid="ignore"):  # non-finite output is saved as before: garbage 8-bit, never an error
        for start in range(0, height, rows):
            block = np.clip(rgb_unit[start : start + rows, :, :3], 0, 1)
            out[start : start + rows, :, :3] = (block * 255.0).round().astype(np.uint8)
    return Image.fromarray(out, "RGBA")


@invocation("esrgan", title="Upscale (RealESRGAN)", tags=["esrgan", "upscale"], category="upscale", version="1.3.2")
class ESRGANInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Upscales an image using RealESRGAN."""

    image: ImageField = InputField(description="The input image")
    model_name: ESRGAN_MODELS = InputField(default="RealESRGAN_x4plus.pth", description="The Real-ESRGAN model to use")
    tile_size: int = InputField(
        default=400, ge=0, description="Tile size for tiled ESRGAN upscaling (0=tiling disabled)"
    )

    model_config = ConfigDict(protected_namespaces=())

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # An image with a precise asset feeds its float pixels to the model; one without takes the 8-bit path.
        precise_input = load_precise_unit_tensor(context, self.image.image_name)
        image = context.images.get_pil(self.image.image_name) if precise_input is None else None

        rrdbnet_model = None
        netscale = None

        if self.model_name in [
            "RealESRGAN_x4plus.pth",
            "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
        ]:
            # x4 RRDBNet model
            # RRDBNet is plain torch: it *calls* register_parameter but never installs
            # the accelerate/init_empty_weights patch itself, so it needs the reader
            # role, not the writer one — it must only be excluded from a concurrent
            # construction that would strand these parameters on the meta device. Using
            # the write lock here would additionally block every other worker's VRAM
            # moves for no benefit. See _ModelLoadReadWriteLock.
            with MODEL_LOAD_LOCK.read_lock():
                rrdbnet_model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
            netscale = 4
        elif self.model_name in ["RealESRGAN_x4plus_anime_6B.pth"]:
            # x4 RRDBNet model, 6 blocks
            with MODEL_LOAD_LOCK.read_lock():
                rrdbnet_model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=6,  # 6 blocks
                    num_grow_ch=32,
                    scale=4,
                )
            netscale = 4
        elif self.model_name in ["RealESRGAN_x2plus.pth"]:
            # x2 RRDBNet model
            with MODEL_LOAD_LOCK.read_lock():
                rrdbnet_model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
            netscale = 2
        else:
            msg = f"Invalid RealESRGAN model: {self.model_name}"
            context.logger.error(msg)
            raise ValueError(msg)

        loadnet = context.models.load_remote_model(
            source=ESRGAN_MODEL_URLS[self.model_name],
        )

        with loadnet as loadnet_model:
            upscaler = RealESRGAN(
                scale=netscale,
                loadnet=loadnet_model,
                model=rrdbnet_model,
                half=False,
                tile=self.tile_size,
            )

            if precise_input is None:
                height, width = image.height, image.width
            else:
                height, width = precise_input.shape[1:]
            # No float buffer unless the result is stored precisely and fits the precise asset bounds; otherwise (and
            # with no precise input) the node runs exactly its ordinary 8-bit path.
            store_precise = float_output_fits(context, height * netscale, width * netscale, 4)
            raster = None
            if precise_input is None and not store_precise:
                # prepare image - Real-ESRGAN uses cv2 internally, and cv2 uses BGR vs RGB for PIL
                # TODO: This strips the alpha... is that okay?
                cv2_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
                upscaled_image = upscaler.upscale(cv2_image)

                pil_image = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
            else:
                # Float path: the model gets the stored values (or the 8-bit ones scaled to the same [0, 1] range) as
                # RGB, alpha dropped as above, and its float output is quantized here instead of inside the upscaler.
                if precise_input is not None:
                    rgb_input = precise_input.permute(1, 2, 0).numpy()
                else:
                    rgb_input = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                if store_precise:
                    # The model output goes straight into the RGBA raster buffer (alpha 1.0: the display image is
                    # RGBA and `save_raster` requires the same channels).
                    rgba = np.empty((height * netscale, width * netscale, 4), dtype=np.float32)
                    rgba[:, :, 3] = 1.0
                    output = upscaler.upscale_rgb_unit(rgb_input, out=rgba)
                else:
                    output = upscaler.upscale_rgb_unit(rgb_input)
                del rgb_input, precise_input
                pil_image = _rgba_8bit(output)
                if store_precise:
                    raster = float_output_raster(context, output)

        return ImageOutput.build(save_decoded_image(context, pil_image, raster))
