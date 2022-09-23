# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from marshmallow import fields
from PIL import Image
from ldm.dream.app.services.schemas import ImageField, InvocationSchemaBase
from ldm.dream.app.invocations.invocationabc import InvocationABC

class InvokeLoadImage(InvocationABC):
    """Load an image from a filename and provide it as output."""
    def invoke(self, image: str, **kwargs) -> dict:
        output_image = Image.open(image)
        return dict(
            image = output_image
        )


class LoadImageSchema(InvocationSchemaBase):
  class Meta:
    type = 'load_image'
    invokes = InvokeLoadImage
    outputs = {
      'image': ImageField()
    }

  image = fields.String()


class InvokeShowImage(InvocationABC):
    """Displays a provided image, and passes it forward in the pipeline."""
    def invoke(self, image: Image.Image, **kwargs) -> dict:
        image.show()
        return dict(
            image = image
        )


class LoadImageSchema(InvocationSchemaBase):
  class Meta:
    type = 'show_image'
    invokes = InvokeShowImage
    outputs = {
      'image': ImageField()
    }

  image = ImageField()
