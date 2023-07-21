from typing import Literal, Optional

import numpy
from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageDraw
from pydantic import BaseModel, Field
from typing import Union
import cv2

from ..models.image import ImageCategory, ImageField, ResourceOrigin
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig,
)


class PILInvocationConfig(BaseModel):
    """Helper class to provide all PIL invocations with additional config"""

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["PIL", "image"],
            },
        }


class ImageMaskOutputCPC(BaseInvocationOutput):
    """Base class for invocations that output an image and a mask"""

    # fmt: off
    type: Literal["image_mask_output"] = "image_mask_output"
    image:      ImageField = Field(default=None, description="The output image")
    width:             int = Field(description="The width of the image in pixels")
    height:            int = Field(description="The height of the image in pixels")
    mask:       ImageField = Field(default=None, description="The output mask")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "image", "width", "height", "mask"]}


class CenterPadCropInvocation(BaseInvocation, PILInvocationConfig):
    """Pad or crop an image's sides from the center by specified pixels. Positive values are outside of the image."""

    # fmt: off
    type: Literal["img_pad_crop"] = "img_pad_crop"

    # Inputs
    image:  Optional[ImageField] = Field(default=None, description="The image to crop")
    left:   int = Field(default=0, description="Number of pixels to pad/crop from the left (negative values crop inwards, positive values pad outwards)")
    right:  int = Field(default=0, description="Number of pixels to pad/crop from the right (negative values crop inwards, positive values pad outwards)")
    top:    int = Field(default=0, description="Number of pixels to pad/crop from the top (negative values crop inwards, positive values pad outwards)")
    bottom: int = Field(default=0, description="Number of pixels to pad/crop from the bottom (negative values crop inwards, positive values pad outwards)")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Center Pad Crop",
                "tags": ["image", "center", "pad", "crop"]
            },
        }

    def invoke(self, context: InvocationContext) -> ImageMaskOutputCPC:
        image = context.services.images.get_pil_image(self.image.image_name)
        imgwidth, imgheight = image.size

        # Calculate and create new image dimensions
        new_width = imgwidth + self.right + self.left
        new_height = imgheight + self.top + self.bottom
        image_crop = Image.new(mode="RGBA", size=(new_width, new_height), color=(0, 0, 0, 0))

        # Paste new image onto input
        image_crop.paste(image, (self.left, self.top))

        # Create white mask with size of new image
        white_mask = Image.new("L", image_crop.size, color=255)

        image_dto = context.services.images.create(
            image=image_crop,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )
        white_mask_dto = context.services.images.create(
            image=white_mask,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageMaskOutputCPC(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
            mask=ImageField(image_name=white_mask_dto.image_name),
        )
