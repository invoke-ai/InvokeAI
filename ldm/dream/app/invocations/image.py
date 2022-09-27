# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal, Optional
from pydantic import Field, BaseModel
from pydantic.dataclasses import dataclass
from PIL import Image
from .baseinvocation import BaseInvocation, BaseInvocationOutput
from ..services.invocation_services import InvocationServices


class ImageField(BaseModel):
    """An image field used for passing image objects between invocations"""
    uri: Optional[str] = Field(default=None, description="The relative path to the image")


class BaseImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""
    image: ImageField = Field(default=None, description="The output image")


# TODO: this isn't really necessary anymore
class LoadImageInvocation(BaseInvocation):
    """Load an image from a filename and provide it as output."""
    type: Literal['load_image']

    # Inputs
    uri: str = Field(description="The URI from which to load the image")

    class Outputs(BaseImageOutput):
        ...

    def invoke(self, services: InvocationServices, context_id: str) -> Outputs:
        return LoadImageInvocation.Outputs.construct(
            image = ImageField.construct(self.uri)
        )


class ShowImageInvocation(BaseInvocation):
    """Displays a provided image, and passes it forward in the pipeline."""
    type: Literal['show_image']

    # Inputs
    image: ImageField = Field(default=None, description="The image to show")

    class Outputs(BaseImageOutput):
        ...

    def invoke(self, services: InvocationServices, context_id: str) -> Outputs:
        image = services.images.get(self.image.uri)
        if image:
            image.show()

        # TODO: how to handle failure?

        return ShowImageInvocation.Outputs.construct(
            image = ImageField.construct(self.image.uri)
        )
