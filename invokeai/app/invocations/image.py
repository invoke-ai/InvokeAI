# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import io
from typing import Literal, Optional

import numpy
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel, Field, validator
from w3lib.url import parse_data_uri

from ..models.image import ImageField, ImageType
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


class ImageOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    # fmt: off
    type: Literal["image"] = "image"
    image:      ImageField = Field(default=None, description="The output image")
    width:   Optional[int] = Field(default=None, description="The width of the image in pixels")
    height:  Optional[int] = Field(default=None, description="The height of the image in pixels")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "image", "width", "height", "mode"]}


def build_image_output(
    image_type: ImageType, image_name: str, image: Image.Image
) -> ImageOutput:
    """Builds an ImageOutput and its ImageField"""
    image_field = ImageField(
        image_name=image_name,
        image_type=image_type,
    )
    return ImageOutput(
        image=image_field,
        width=image.width,
        height=image.height,
        mode=image.mode,
    )


class MaskOutput(BaseInvocationOutput):
    """Base class for invocations that output a mask"""

    # fmt: off
    type: Literal["mask"] = "mask"
    mask:      ImageField = Field(default=None, description="The output mask")
    # fmt: on

    class Config:
        schema_extra = {
            "required": [
                "type",
                "mask",
            ]
        }


class LoadImageInvocation(BaseInvocation):
    """Load an image and provide it as output."""

    # fmt: off
    type: Literal["load_image"] = "load_image"

    # Inputs
    image_type: ImageType = Field(description="The type of the image")
    image_name:       str = Field(description="The name of the image")
    # fmt: on
    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(self.image_type, self.image_name)

        return build_image_output(
            image_type=self.image_type,
            image_name=self.image_name,
            image=image,
        )


class ShowImageInvocation(BaseInvocation):
    """Displays a provided image, and passes it forward in the pipeline."""

    type: Literal["show_image"] = "show_image"

    # Inputs
    image: ImageField = Field(default=None, description="The image to show")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )
        if image:
            image.show()

        # TODO: how to handle failure?

        return build_image_output(
            image_type=self.image.image_type,
            image_name=self.image.image_name,
            image=image,
        )


class DataURLToImageInvocation(BaseInvocation, PILInvocationConfig):
    """Outputs an image from a data URL."""

    type: Literal["dataURL_image"] = "dataURL_image"

    # Inputs
    dataURL: str = Field(description="The b64 data URL")

    @validator("dataURL")
    def must_be_valid_image_dataURL(cls, v):
        try:
            result = parse_data_uri(v)
            img = Image.open(io.BytesIO(result.data))
            img.verify()
        except Exception:
            raise ValueError("Invalid image dataURL")
        return v

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # TODO: Figure out how to use pydantic validator to also transform into a different type,
        # bc this is just the same logic as we use to validate the dataURL.
        result = parse_data_uri(self.dataURL)
        image = Image.open(io.BytesIO(result.data))

        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(ImageType.RESULT, image_name, image, metadata)

        return build_image_output(
            image_type=ImageType.RESULT, image_name=image_name, image=image
        )


class CropImageInvocation(BaseInvocation, PILInvocationConfig):
    """Crops an image to a specified box. The box can be outside of the image."""

    # fmt: off
    type: Literal["crop"] = "crop"

    # Inputs
    image: ImageField = Field(default=None, description="The image to crop")
    x:      int = Field(default=0, description="The left x coordinate of the crop rectangle")
    y:      int = Field(default=0, description="The top y coordinate of the crop rectangle")
    width:  int = Field(default=512, gt=0, description="The width of the crop rectangle")
    height: int = Field(default=512, gt=0, description="The height of the crop rectangle")
    # fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )

        image_crop = Image.new(
            mode="RGBA", size=(self.width, self.height), color=(0, 0, 0, 0)
        )
        image_crop.paste(image, (-self.x, -self.y))

        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(image_type, image_name, image_crop, metadata)
        return build_image_output(
            image_type=image_type,
            image_name=image_name,
            image=image_crop,
        )


class PasteImageInvocation(BaseInvocation, PILInvocationConfig):
    """Pastes an image into another image."""

    # fmt: off
    type: Literal["paste"] = "paste"

    # Inputs
    base_image:     ImageField = Field(default=None, description="The base image")
    image:          ImageField = Field(default=None, description="The image to paste")
    mask: Optional[ImageField] = Field(default=None, description="The mask to use when pasting")
    x:                     int = Field(default=0, description="The left x coordinate at which to paste the image")
    y:                     int = Field(default=0, description="The top y coordinate at which to paste the image")
    # fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        base_image = context.services.images.get(
            self.base_image.image_type, self.base_image.image_name
        )
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )
        mask = (
            None
            if self.mask is None
            else ImageOps.invert(
                context.services.images.get(self.mask.image_type, self.mask.image_name)
            )
        )
        # TODO: probably shouldn't invert mask here... should user be required to do it?

        min_x = min(0, self.x)
        min_y = min(0, self.y)
        max_x = max(base_image.width, image.width + self.x)
        max_y = max(base_image.height, image.height + self.y)

        new_image = Image.new(
            mode="RGBA", size=(max_x - min_x, max_y - min_y), color=(0, 0, 0, 0)
        )
        new_image.paste(base_image, (abs(min_x), abs(min_y)))
        new_image.paste(image, (max(0, self.x), max(0, self.y)), mask=mask)

        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(image_type, image_name, new_image, metadata)
        return build_image_output(
            image_type=image_type,
            image_name=image_name,
            image=new_image,
        )


class MaskFromAlphaInvocation(BaseInvocation, PILInvocationConfig):
    """Extracts the alpha channel of an image as a mask."""

    # fmt: off
    type: Literal["tomask"] = "tomask"

    # Inputs
    image: ImageField = Field(default=None, description="The image to create the mask from")
    invert:      bool = Field(default=False, description="Whether or not to invert the mask")
    # fmt: on

    def invoke(self, context: InvocationContext) -> MaskOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )

        image_mask = image.split()[-1]
        if self.invert:
            image_mask = ImageOps.invert(image_mask)

        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(image_type, image_name, image_mask, metadata)
        return MaskOutput(mask=ImageField(image_type=image_type, image_name=image_name))


class BlurInvocation(BaseInvocation, PILInvocationConfig):
    """Blurs an image"""

    # fmt: off
    type: Literal["blur"] = "blur"

    # Inputs
    image: ImageField = Field(default=None, description="The image to blur")
    radius:     float = Field(default=8.0, ge=0, description="The blur radius")
    blur_type: Literal["gaussian", "box"] = Field(default="gaussian", description="The type of blur")
    # fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )

        blur = (
            ImageFilter.GaussianBlur(self.radius)
            if self.blur_type == "gaussian"
            else ImageFilter.BoxBlur(self.radius)
        )
        blur_image = image.filter(blur)

        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(image_type, image_name, blur_image, metadata)
        return build_image_output(
            image_type=image_type, image_name=image_name, image=blur_image
        )


class LerpInvocation(BaseInvocation, PILInvocationConfig):
    """Linear interpolation of all pixels of an image"""

    # fmt: off
    type: Literal["lerp"] = "lerp"

    # Inputs
    image: ImageField = Field(default=None, description="The image to lerp")
    min: int = Field(default=0, ge=0, le=255, description="The minimum output value")
    max: int = Field(default=255, ge=0, le=255, description="The maximum output value")
    # fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )

        image_arr = numpy.asarray(image, dtype=numpy.float32) / 255
        image_arr = image_arr * (self.max - self.min) + self.max

        lerp_image = Image.fromarray(numpy.uint8(image_arr))

        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(image_type, image_name, lerp_image, metadata)
        return build_image_output(
            image_type=image_type, image_name=image_name, image=lerp_image
        )


class InverseLerpInvocation(BaseInvocation, PILInvocationConfig):
    """Inverse linear interpolation of all pixels of an image"""

    # fmt: off
    type: Literal["ilerp"] = "ilerp"

    # Inputs
    image: ImageField = Field(default=None, description="The image to lerp")
    min: int = Field(default=0, ge=0, le=255, description="The minimum input value")
    max: int = Field(default=255, ge=0, le=255, description="The maximum input value")
    # fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )

        image_arr = numpy.asarray(image, dtype=numpy.float32)
        image_arr = (
            numpy.minimum(
                numpy.maximum(image_arr - self.min, 0) / float(self.max - self.min), 1
            )
            * 255
        )

        ilerp_image = Image.fromarray(numpy.uint8(image_arr))

        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(image_type, image_name, ilerp_image, metadata)
        return build_image_output(
            image_type=image_type, image_name=image_name, image=ilerp_image
        )
