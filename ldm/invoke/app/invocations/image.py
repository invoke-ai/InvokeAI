# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from datetime import datetime, timezone
from typing import Literal, Optional
import numpy
from pydantic import Field, BaseModel
from PIL import Image, ImageOps, ImageFilter
from .baseinvocation import BaseInvocation, BaseInvocationOutput
from ..services.image_storage import ImageType
from ..services.invocation_services import InvocationServices


class ImageField(BaseModel):
    """An image field used for passing image objects between invocations"""
    image_type: ImageType = Field(default=ImageType.RESULT, description="The type of the image")
    image_name: Optional[str] = Field(default=None, description="The name of the image")


class ImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""
    type: Literal['image'] = 'image'

    image: ImageField = Field(default=None, description="The output image")


class MaskOutput(BaseInvocationOutput):
    """Base class for invocations that output a mask"""
    type: Literal['mask'] = 'mask'

    mask: ImageField = Field(default=None, description="The output mask")


# TODO: this isn't really necessary anymore
class LoadImageInvocation(BaseInvocation):
    """Load an image from a filename and provide it as output."""
    type: Literal['load_image'] = 'load_image'

    # Inputs
    image_type: ImageType = Field(description="The type of the image")
    image_name: str = Field(description="The name of the image")

    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput:
        return ImageOutput(
            image = ImageField(image_type = self.image_type, image_name = self.image_name)
        )


class ShowImageInvocation(BaseInvocation):
    """Displays a provided image, and passes it forward in the pipeline."""
    type: Literal['show_image'] = 'show_image'

    # Inputs
    image: ImageField = Field(default=None, description="The image to show")

    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput:
        image = services.images.get(self.image.image_type, self.image.image_name)
        if image:
            image.show()

        # TODO: how to handle failure?

        return ImageOutput(
            image = ImageField(image_type = self.image.image_type, image_name = self.image.image_name)
        )


class CropImageInvocation(BaseInvocation):
    """Crops an image to a specified box. The box can be outside of the image."""
    type: Literal['crop'] = 'crop'

    # Inputs
    image: ImageField = Field(default=None, description="The image to crop")
    x: int      = Field(default=0, description="The left x coordinate of the crop rectangle")
    y: int      = Field(default=0, description="The top y coordinate of the crop rectangle")
    width: int  = Field(default=512, gt=0, description="The width of the crop rectangle")
    height: int = Field(default=512, gt=0, description="The height of the crop rectangle")

    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput:
        image = services.images.get(self.image.image_type, self.image.image_name)

        image_crop = Image.new(mode = 'RGBA', size = (self.width, self.height), color = (0, 0, 0, 0))
        image_crop.paste(image, (-self.x, -self.y))

        image_type = ImageType.INTERMEDIATE
        image_name = f'{session_id}_{self.id}_{str(int(datetime.now(timezone.utc).timestamp()))}.png'
        services.images.save(image_type, image_name, image_crop)
        return ImageOutput(
            image = ImageField(image_type = image_type, image_name = image_name)
        )


class PasteImageInvocation(BaseInvocation):
    """Pastes an image into another image."""
    type: Literal['paste'] = 'paste'

    # Inputs
    base_image: ImageField     = Field(default=None, description="The base image")
    image: ImageField          = Field(default=None, description="The image to paste")
    mask: Optional[ImageField] = Field(default=None, description="The mask to use when pasting")
    x: int                     = Field(default=0, description="The left x coordinate at which to paste the image")
    y: int                     = Field(default=0, description="The top y coordinate at which to paste the image")

    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput:
        base_image = services.images.get(self.base_image.image_type, self.base_image.image_name)
        image = services.images.get(self.image.image_type, self.image.image_name)
        mask = None if self.mask is None else ImageOps.invert(services.images.get(self.mask.image_type, self.mask.image_name))
        # TODO: probably shouldn't invert mask here... should user be required to do it?

        min_x = min(0, self.x)
        min_y = min(0, self.y)
        max_x = max(base_image.width, image.width + self.x)
        max_y = max(base_image.height, image.height + self.y)

        new_image = Image.new(mode = 'RGBA', size = (max_x - min_x, max_y - min_y), color = (0, 0, 0, 0))
        new_image.paste(base_image, (abs(min_x), abs(min_y)))
        new_image.paste(image, (max(0, self.x), max(0, self.y)), mask = mask)

        image_type = ImageType.RESULT
        image_name = f'{session_id}_{self.id}_{str(int(datetime.now(timezone.utc).timestamp()))}.png'
        services.images.save(image_type, image_name, new_image)
        return ImageOutput(
            image = ImageField(image_type = image_type, image_name = image_name)
        )


class MaskFromAlphaInvocation(BaseInvocation):
    """Extracts the alpha channel of an image as a mask."""
    type: Literal['tomask'] = 'tomask'

    # Inputs
    image: ImageField = Field(default=None, description="The image to create the mask from")
    invert: bool = Field(default=False, description="Whether or not to invert the mask")

    def invoke(self, services: InvocationServices, session_id: str) -> MaskOutput:
        image = services.images.get(self.image.image_type, self.image.image_name)

        image_mask = image.split()[-1]
        if self.invert:
            image_mask = ImageOps.invert(image_mask)

        image_type = ImageType.INTERMEDIATE
        image_name = f'{session_id}_{self.id}_{str(int(datetime.now(timezone.utc).timestamp()))}.png'
        services.images.save(image_type, image_name, image_mask)
        return MaskOutput(
            mask = ImageField(image_type = image_type, image_name = image_name)
        )


class BlurInvocation(BaseInvocation):
    """Blurs an image"""
    type: Literal['blur'] = 'blur'

    # Inputs
    image: ImageField = Field(default=None, description="The image to blur")
    radius: float     = Field(default=8.0, ge=0, description="The blur radius")
    blur_type: Literal['gaussian', 'box'] = Field(default='gaussian', description="The type of blur")

    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput:
        image = services.images.get(self.image.image_type, self.image.image_name)

        blur = ImageFilter.GaussianBlur(self.radius) if self.blur_type == 'gaussian' else ImageFilter.BoxBlur(self.radius)
        blur_image = image.filter(blur)

        image_type = ImageType.INTERMEDIATE
        image_name = f'{session_id}_{self.id}_{str(int(datetime.now(timezone.utc).timestamp()))}.png'
        services.images.save(image_type, image_name, blur_image)
        return ImageOutput(
            image = ImageField(image_type = image_type, image_name = image_name)
        )


class LerpInvocation(BaseInvocation):
    """Linear interpolation of all pixels of an image"""
    type: Literal['lerp'] = 'lerp'

    # Inputs
    image: ImageField = Field(default=None, description="The image to lerp")
    min: int = Field(default=0, ge=0, le=255, description="The minimum output value")
    max: int = Field(default=255, ge=0, le=255, description="The maximum output value")

    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput:
        image = services.images.get(self.image.image_type, self.image.image_name)

        image_arr = numpy.asarray(image, dtype=numpy.float32) / 255
        image_arr = image_arr * (self.max - self.min) + self.max

        lerp_image = Image.fromarray(numpy.uint8(image_arr))

        image_type = ImageType.INTERMEDIATE
        image_name = f'{session_id}_{self.id}_{str(int(datetime.now(timezone.utc).timestamp()))}.png'
        services.images.save(image_type, image_name, lerp_image)
        return ImageOutput(
            image = ImageField(image_type = image_type, image_name = image_name)
        )


class InverseLerpInvocation(BaseInvocation):
    """Inverse linear interpolation of all pixels of an image"""
    type: Literal['ilerp'] = 'ilerp'

    # Inputs
    image: ImageField = Field(default=None, description="The image to lerp")
    min: int = Field(default=0, ge=0, le=255, description="The minimum input value")
    max: int = Field(default=255, ge=0, le=255, description="The maximum input value")

    def invoke(self, services: InvocationServices, session_id: str) -> ImageOutput:
        image = services.images.get(self.image.image_type, self.image.image_name)

        image_arr = numpy.asarray(image, dtype=numpy.float32)
        image_arr = numpy.minimum(numpy.maximum(image_arr - self.min, 0) / float(self.max - self.min), 1) * 255

        ilerp_image = Image.fromarray(numpy.uint8(image_arr))

        image_type = ImageType.INTERMEDIATE
        image_name = f'{session_id}_{self.id}_{str(int(datetime.now(timezone.utc).timestamp()))}.png'
        services.images.save(image_type, image_name, ilerp_image)
        return ImageOutput(
            image = ImageField(image_type = image_type, image_name = image_name)
        )
