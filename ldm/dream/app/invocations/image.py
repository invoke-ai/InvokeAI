# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal, Optional
from pydantic import Field, BaseModel
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
