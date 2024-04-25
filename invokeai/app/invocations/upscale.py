# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) & the InvokeAI Team
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import ConfigDict

from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.basicsr.rrdbnet_arch import RRDBNet
from invokeai.backend.image_util.realesrgan.realesrgan import RealESRGAN

from .baseinvocation import BaseInvocation, invocation
from .fields import InputField, WithBoard, WithMetadata

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


@invocation("esrgan", title="Upscale (RealESRGAN)", tags=["esrgan", "upscale"], category="esrgan", version="1.3.2")
class ESRGANInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Upscales an image using RealESRGAN."""

    image: ImageField = InputField(description="The input image")
    model_name: ESRGAN_MODELS = InputField(default="RealESRGAN_x4plus.pth", description="The Real-ESRGAN model to use")
    tile_size: int = InputField(
        default=400, ge=0, description="Tile size for tiled ESRGAN upscaling (0=tiling disabled)"
    )

    model_config = ConfigDict(protected_namespaces=())

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        rrdbnet_model = None
        netscale = None

        if self.model_name in [
            "RealESRGAN_x4plus.pth",
            "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
        ]:
            # x4 RRDBNet model
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

        loadnet = context.models.load_ckpt_from_url(
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

            # prepare image - Real-ESRGAN uses cv2 internally, and cv2 uses BGR vs RGB for PIL
            # TODO: This strips the alpha... is that okay?
            cv2_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            upscaled_image = upscaler.upscale(cv2_image)

            pil_image = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB)).convert("RGBA")

        image_dto = context.images.save(image=pil_image)

        return ImageOutput.build(image_dto)
