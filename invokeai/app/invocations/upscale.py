# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from email.mime import image
from typing import Literal, Union
import cv2 as cv
import numpy as np

from pydantic import Field
from realesrgan import RealESRGANer
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet

from invokeai.app.models.image import ImageCategory, ImageField, ResourceOrigin
from .baseinvocation import BaseInvocation, InvocationContext, InvocationConfig
from .image import ImageOutput


class UpscaleInvocation(BaseInvocation):
    """Upscales an image."""

    # fmt: off
    type: Literal["upscale"] = "upscale"

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image", default=None)
    strength: float = Field(default=0.75, gt=0, le=1, description="The strength")
    level: Literal[2, 4] = Field(default=2, description="The upscale level")
    # fmt: on

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["upscaling", "image"],
            },
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        results = context.services.restoration.upscale_and_reconstruct(
            image_list=[[image, 0]],
            upscale=(self.level, self.strength),
            strength=0.0,  # GFPGAN strength
            save_original=False,
            image_callback=None,
        )

        # Results are image and seed, unwrap for now
        # TODO: can this return multiple results?
        image_dto = context.services.images.create(
            image=results[0][0],
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )


REALESRGAN_MODELS = Literal[
    "RealESRGAN_x4plus",
    "RealESRGAN_x4plus_anime_6B",
    "ESRGAN_SRx4_DF2KOST_official-ff704c30",
]


class RealESRGANInvocation(BaseInvocation):
    """Upscales an image using Real-ESRGAN."""

    # fmt: off
    type: Literal["realesrgan"] = "realesrgan"
    image: Union[ImageField, None] = Field(default=None, description="The input image" )
    model_name: REALESRGAN_MODELS = Field(default="RealESRGAN_x4plus", description="The Real-ESRGAN model to use")
    scale: Literal[2, 4] = Field(default=4, description="The final upsampling scale")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        model = None
        netscale = None
        model_path = None

        if self.model_name == 'RealESRGAN x4 Plus': # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_path = f'core/upscaling/realesrgan/RealESRGAN_x4plus.pth'
        elif self.model_name == 'RealESRGAN x4 Plus (Anime 6B)':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            model_path = f'core/upscaling/realesrgan/RealESRGAN_x4plus_anime_6B.pth'
        # elif self.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        #     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        #     netscale = 2
        elif self.model_name in ['ESRGAN x4']:  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_path = f'core/upscaling/realesrgan/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'

        if not model or not netscale or not model_path:
            raise Exception(f"Invalid model {self.model_name}")
        
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            half=False,
        )

        # Real-ESRGAN uses cv2 internally, and cv2 uses BGR vs RGB for PIL
        cv_image = cv.cvtColor(np.array(image.convert("RGB")), cv.COLOR_RGB2BGR)
        upscaled_image, img_mode = upsampler.enhance(cv_image, outscale=self.scale)
        pil_image = Image.fromarray(cv.cvtColor(upscaled_image, cv.COLOR_BGR2RGB)).convert('RGBA')

        image_dto = context.services.images.create(
            image=pil_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
