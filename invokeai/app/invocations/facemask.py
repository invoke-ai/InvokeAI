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


class ImageMaskOutputFaceMask(BaseInvocationOutput):
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


class FaceMaskInvocation(BaseInvocation, PILInvocationConfig):
    """OpenCV cascade classifier detection to create transparencies in an image"""

    # fmt: off
    type: Literal["img_detect_mask"] = "img_detect_mask"

    # Inputs
    image: Optional[ImageField]  = Field(default=None, description="Image to apply transparency to")
    x_offset: float = Field(default=0.0, description="Offset for the X-axis of the oval mask")
    y_offset: float = Field(default=0.0, description="Offset for the Y-axis of the oval mask")
    invert_mask: bool = Field(default=False, description="Toggle to invert the mask")
    cascade_file_path: Optional[str] = Field(default=None, description="Path to the cascade XML file for detection")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Face Mask",
                "tags": ["image", "face", "mask"]
            },
        }

    def invoke(self, context: InvocationContext) -> ImageMaskOutputFaceMask:
        image = context.services.images.get_pil_image(self.image.image_name)

        # Perform face detection
        cv_image = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        cascade_file_path = self.cascade_file_path
        face_cascade = cv2.CascadeClassifier(cascade_file_path)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Create an elongated oval-shaped transparent mask for the face region
        transparent_mask = Image.new("L", image.size, color=0)
        for (x, y, w, h) in faces:
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            # Adjust the shape of the oval based on the offsets
            x_elongation_factor = 0.8 + (1.2 - 0.8) * self.x_offset
            y_elongation_factor = 1 + (1.2 - 0.8) * self.y_offset
            draw.ellipse((0, 0, w, h), fill=255, outline=255)
            mask = mask.resize((int(w * x_elongation_factor), int(h * y_elongation_factor)), resample=Image.LANCZOS)
            # Calculate the Y-axis offset to ensure symmetry from the middle
            y_offset = int((h - h * y_elongation_factor) / 2)
            transparent_mask.paste(mask, (x + int((w - w * x_elongation_factor) / 2), y + y_offset))

        # Create an RGBA image with transparency
        rgba_image = image.convert("RGBA")

        if self.invert_mask:
            # Apply the transparent mask to the image
            composite_image = Image.composite(rgba_image, Image.new("RGBA", image.size, (0, 0, 0, 0)), transparent_mask)

        else:
            # Invert the transparent mask to mask the rest of the image
            inverted_mask = ImageOps.invert(transparent_mask)
            # Apply the inverted mask to the image
            composite_image = Image.composite(rgba_image, Image.new("RGBA", image.size, (0, 0, 0, 0)), inverted_mask)

        # Create white mask with dimensions as transparency image for use with outpainting
        white_mask = Image.new("L", image.size, color=255)

        image_dto = context.services.images.create(
            image=composite_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
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

        return ImageMaskOutputFaceMask(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
            mask=ImageField(image_name=white_mask_dto.image_name),
        )
