# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal, Optional, Tuple, Union
from anyio import Condition

from pydantic import BaseModel, Field
import torch

from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    UIComponent,
    UIType,
    tags,
    title,
)

"""
Primitives: Boolean, Integer, Float, String, Image, Latents, Conditioning, Color
- primitive nodes
- primitive outputs
- primitive collection outputs
"""

# region Boolean


class BooleanOutput(BaseInvocationOutput):
    """Base class for nodes that output a single boolean"""

    type: Literal["boolean_output"] = "boolean_output"
    a: bool = OutputField(description="The output boolean")


class BooleanCollectionOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of booleans"""

    type: Literal["boolean_collection_output"] = "boolean_collection_output"

    # Outputs
    collection: list[bool] = OutputField(
        default_factory=list, description="The output boolean collection", ui_type=UIType.BooleanCollection
    )


@title("Boolean")
@tags("primitives", "boolean")
class BooleanInvocation(BaseInvocation):
    """A boolean primitive value"""

    type: Literal["boolean"] = "boolean"

    # Inputs
    a: bool = InputField(default=False, description="The boolean value")

    def invoke(self, context: InvocationContext) -> BooleanOutput:
        return BooleanOutput(a=self.a)


@title("Boolean Collection")
@tags("primitives", "boolean", "collection")
class BooleanCollectionInvocation(BaseInvocation):
    """A collection of boolean primitive values"""

    type: Literal["boolean_collection"] = "boolean_collection"

    # Inputs
    collection: list[bool] = InputField(
        default=False, description="The collection of boolean values", ui_type=UIType.BooleanCollection
    )

    def invoke(self, context: InvocationContext) -> BooleanCollectionOutput:
        return BooleanCollectionOutput(collection=self.collection)


# endregion

# region Integer


class IntegerOutput(BaseInvocationOutput):
    """Base class for nodes that output a single integer"""

    type: Literal["integer_output"] = "integer_output"
    a: int = OutputField(description="The output integer")


class IntegerCollectionOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of integers"""

    type: Literal["integer_collection_output"] = "integer_collection_output"

    # Outputs
    collection: list[int] = OutputField(
        default_factory=list, description="The int collection", ui_type=UIType.IntegerCollection
    )


@title("Integer")
@tags("primitives", "integer")
class IntegerInvocation(BaseInvocation):
    """An integer primitive value"""

    type: Literal["integer"] = "integer"

    # Inputs
    a: int = InputField(default=0, description="The integer value")

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        return IntegerOutput(a=self.a)


@title("Integer Collection")
@tags("primitives", "integer", "collection")
class IntegerCollectionInvocation(BaseInvocation):
    """A collection of integer primitive values"""

    type: Literal["integer_collection"] = "integer_collection"

    # Inputs
    collection: list[int] = InputField(
        default=0, description="The collection of integer values", ui_type=UIType.IntegerCollection
    )

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        return IntegerCollectionOutput(collection=self.collection)


# endregion

# region Float


class FloatOutput(BaseInvocationOutput):
    """Base class for nodes that output a single float"""

    type: Literal["float_output"] = "float_output"
    a: float = OutputField(description="The output float")


class FloatCollectionOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of floats"""

    type: Literal["float_collection_output"] = "float_collection_output"

    # Outputs
    collection: list[float] = OutputField(
        default_factory=list, description="The float collection", ui_type=UIType.FloatCollection
    )


@title("Float")
@tags("primitives", "float")
class FloatInvocation(BaseInvocation):
    """A float primitive value"""

    type: Literal["float"] = "float"

    # Inputs
    param: float = InputField(default=0.0, description="The float value")

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(a=self.param)


@title("Float Collection")
@tags("primitives", "float", "collection")
class FloatCollectionInvocation(BaseInvocation):
    """A collection of float primitive values"""

    type: Literal["float_collection"] = "float_collection"

    # Inputs
    collection: list[float] = InputField(
        default=0, description="The collection of float values", ui_type=UIType.FloatCollection
    )

    def invoke(self, context: InvocationContext) -> FloatCollectionOutput:
        return FloatCollectionOutput(collection=self.collection)


# endregion

# region String


class StringOutput(BaseInvocationOutput):
    """Base class for nodes that output a single string"""

    type: Literal["string_output"] = "string_output"
    text: str = OutputField(description="The output string")


class StringCollectionOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of strings"""

    type: Literal["string_collection_output"] = "string_collection_output"

    # Outputs
    collection: list[str] = OutputField(
        default_factory=list, description="The output strings", ui_type=UIType.StringCollection
    )


@title("String")
@tags("primitives", "string")
class StringInvocation(BaseInvocation):
    """A string primitive value"""

    type: Literal["string"] = "string"

    # Inputs
    text: str = InputField(default="", description="The string value", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(text=self.text)


@title("String Collection")
@tags("primitives", "string", "collection")
class StringCollectionInvocation(BaseInvocation):
    """A collection of string primitive values"""

    type: Literal["string_collection"] = "string_collection"

    # Inputs
    collection: list[str] = InputField(
        default=0, description="The collection of string values", ui_type=UIType.StringCollection
    )

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        return StringCollectionOutput(collection=self.collection)


# endregion

# region Image


class ImageField(BaseModel):
    """An image primitive field"""

    image_name: str = Field(description="The name of the image")


class ImageOutput(BaseInvocationOutput):
    """Base class for nodes that output a single image"""

    type: Literal["image_output"] = "image_output"
    image: ImageField = OutputField(description="The output image")
    width: int = OutputField(description="The width of the image in pixels")
    height: int = OutputField(description="The height of the image in pixels")


class ImageCollectionOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of images"""

    type: Literal["image_collection_output"] = "image_collection_output"

    # Outputs
    collection: list[ImageField] = OutputField(
        default_factory=list, description="The output images", ui_type=UIType.ImageCollection
    )


@title("Image Primitive")
@tags("primitives", "image")
class ImageInvocation(BaseInvocation):
    """An image primitive value"""

    # Metadata
    type: Literal["image"] = "image"

    # Inputs
    image: ImageField = InputField(description="The image to load")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        return ImageOutput(
            image=ImageField(image_name=self.image.image_name),
            width=image.width,
            height=image.height,
        )


@title("Image Collection")
@tags("primitives", "image", "collection")
class ImageCollectionInvocation(BaseInvocation):
    """A collection of image primitive values"""

    type: Literal["image_collection"] = "image_collection"

    # Inputs
    collection: list[ImageField] = InputField(
        default=0, description="The collection of image values", ui_type=UIType.ImageCollection
    )

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        return ImageCollectionOutput(collection=self.collection)


# endregion

# region Latents


class LatentsField(BaseModel):
    """A latents tensor primitive field"""

    latents_name: str = Field(description="The name of the latents")
    seed: Optional[int] = Field(default=None, description="Seed used to generate this latents")


class LatentsOutput(BaseInvocationOutput):
    """Base class for nodes that output a single latents tensor"""

    type: Literal["latents_output"] = "latents_output"

    latents: LatentsField = OutputField(
        description=FieldDescriptions.latents,
    )
    width: int = OutputField(description=FieldDescriptions.width)
    height: int = OutputField(description=FieldDescriptions.height)


class LatentsCollectionOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of latents tensors"""

    type: Literal["latents_collection_output"] = "latents_collection_output"

    collection: list[LatentsField] = OutputField(
        default_factory=list,
        description=FieldDescriptions.latents,
        ui_type=UIType.LatentsCollection,
    )


@title("Latents Primitive")
@tags("primitives", "latents")
class LatentsInvocation(BaseInvocation):
    """A latents tensor primitive value"""

    type: Literal["latents"] = "latents"

    # Inputs
    latents: LatentsField = InputField(description="The latents tensor", input=Input.Connection)

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.services.latents.get(self.latents.latents_name)

        return build_latents_output(self.latents.latents_name, latents)


@title("Latents Collection")
@tags("primitives", "latents", "collection")
class LatentsCollectionInvocation(BaseInvocation):
    """A collection of latents tensor primitive values"""

    type: Literal["latents_collection"] = "latents_collection"

    # Inputs
    collection: list[LatentsField] = InputField(
        default=0, description="The collection of latents tensors", ui_type=UIType.LatentsCollection
    )

    def invoke(self, context: InvocationContext) -> LatentsCollectionOutput:
        return LatentsCollectionOutput(collection=self.collection)


def build_latents_output(latents_name: str, latents: torch.Tensor, seed: Optional[int] = None):
    return LatentsOutput(
        latents=LatentsField(latents_name=latents_name, seed=seed),
        width=latents.size()[3] * 8,
        height=latents.size()[2] * 8,
    )


# endregion

# region Color


class ColorField(BaseModel):
    """A color primitive field"""

    r: int = Field(ge=0, le=255, description="The red component")
    g: int = Field(ge=0, le=255, description="The green component")
    b: int = Field(ge=0, le=255, description="The blue component")
    a: int = Field(ge=0, le=255, description="The alpha component")

    def tuple(self) -> Tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)


class ColorOutput(BaseInvocationOutput):
    """Base class for nodes that output a single color"""

    type: Literal["color_output"] = "color_output"
    color: ColorField = OutputField(description="The output color")


class ColorCollectionOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of colors"""

    type: Literal["color_collection_output"] = "color_collection_output"

    # Outputs
    collection: list[ColorField] = OutputField(
        default_factory=list, description="The output colors", ui_type=UIType.ColorCollection
    )


@title("Color Primitive")
@tags("primitives", "color")
class ColorInvocation(BaseInvocation):
    """A color primitive value"""

    type: Literal["color"] = "color"

    # Inputs
    color: ColorField = InputField(default=ColorField(r=0, g=0, b=0, a=255), description="The color value")

    def invoke(self, context: InvocationContext) -> ColorOutput:
        return ColorOutput(color=self.color)


# endregion

# region Conditioning


class ConditioningField(BaseModel):
    """A conditioning tensor primitive value"""

    conditioning_name: str = Field(description="The name of conditioning tensor")


class ConditioningOutput(BaseInvocationOutput):
    """Base class for nodes that output a single conditioning tensor"""

    type: Literal["conditioning_output"] = "conditioning_output"

    conditioning: ConditioningField = OutputField(description=FieldDescriptions.cond)


class ConditioningCollectionOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of conditioning tensors"""

    type: Literal["conditioning_collection_output"] = "conditioning_collection_output"

    # Outputs
    collection: list[ConditioningField] = OutputField(
        default_factory=list,
        description="The output conditioning tensors",
        ui_type=UIType.ConditioningCollection,
    )


@title("Conditioning Primitive")
@tags("primitives", "conditioning")
class ConditioningInvocation(BaseInvocation):
    """A conditioning tensor primitive value"""

    type: Literal["conditioning"] = "conditioning"

    conditioning: ConditioningField = InputField(description=FieldDescriptions.cond, input=Input.Connection)

    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        return ConditioningOutput(conditioning=self.conditioning)


@title("Conditioning Collection")
@tags("primitives", "conditioning", "collection")
class ConditioningCollectionInvocation(BaseInvocation):
    """A collection of conditioning tensor primitive values"""

    type: Literal["conditioning_collection"] = "conditioning_collection"

    # Inputs
    collection: list[ConditioningField] = InputField(
        default=0, description="The collection of conditioning tensors", ui_type=UIType.ConditioningCollection
    )

    def invoke(self, context: InvocationContext) -> ConditioningCollectionOutput:
        return ConditioningCollectionOutput(collection=self.collection)


# endregion
