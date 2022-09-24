# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal, Union
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from PIL import Image
from ldm.dream.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext

class ImageFieldConfig:
    arbitrary_types_allowed = True

@dataclass(config=ImageFieldConfig)
class ImageField:
    """An image field used for passing image objects between invocations"""
    image: Union[Image.Image,None]
    # TODO: add lineage/history information to carry to metadata


class BaseImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""
    image: ImageField = Field(default=None, description="The output image")


class LoadImageInvocation(BaseInvocation):
    """Load an image from a filename and provide it as output."""
    type: Literal['load_image']

    # Inputs
    uri: str = Field(description="The URI from which to load the image")

    class Outputs(BaseImageOutput):
        ...

    def invoke(self, context: InvocationContext) -> Outputs:
        output_image = Image.open(self.uri)
        return LoadImageInvocation.Outputs.construct(
            image = ImageField(image = output_image)
        )


class ShowImageInvocation(BaseInvocation):
    """Displays a provided image, and passes it forward in the pipeline."""
    type: Literal['show_image']

    # Inputs
    image: ImageField = Field(default=None, description="The image to show")

    class Outputs(BaseImageOutput):
        ...

    def invoke(self, context: InvocationContext) -> Outputs:
        self.image.image.show()
        return ShowImageInvocation.Outputs.construct(
            image = ImageField(image = self.image.image)
        )
