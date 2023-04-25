# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal
from pydantic import Field
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext, InvocationConfig
from .image import ImageOutput, ImageField, build_image_output

# Pass-through value nodes - used by subgraphs


class IntOutput(BaseInvocationOutput):
    """An integer output"""
    #fmt: off
    type: Literal["int_output"] = "int_output"
    a: int = Field(default=None, description="The output integer value")
    #fmt: on


class IntegerValueInvocation(BaseInvocation):
    """An integer value"""
    #fmt: off
    type: Literal["value_int"] = "value_int"
    a: int = Field(default=0, description="An integer value")
    #fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["values", "integer"],
                "title": "Integer Value"
            }
        }

    def invoke(self, context: InvocationContext) -> IntOutput:
        return IntOutput(a=self.a)


class FloatOutput(BaseInvocationOutput):
    """A float output"""
    #fmt: off
    type: Literal["float_output"] = "float_output"
    a: float = Field(default=None, description="The output float value")
    #fmt: on


class FloatValueInvocation(BaseInvocation):
    """A float value"""
    #fmt: off
    type: Literal["value_float"] = "value_float"
    a: float = Field(default=0, description="A float value")
    #fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["values", "float"],
                "title": "Float Value"
            }
        }

    def invoke(self, context: InvocationContext) -> FloatOutput:
        return FloatOutput(a=self.a)


class StringOutput(BaseInvocationOutput):
    """A string output"""
    #fmt: off
    type: Literal["string_output"] = "string_output"
    a: str = Field(default=None, description="The output string value")
    #fmt: on


class StringValueInvocation(BaseInvocation):
    """A string value"""
    #fmt: off
    type: Literal["value_string"] = "value_string"
    a: str = Field(default='', description="A string value")
    #fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["values", "string"],
                "title": "String Value"
            }
        }

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(a=self.a)


class BooleanOutput(BaseInvocationOutput):
    """A boolean output"""
    #fmt: off
    type: Literal["boolean_output"] = "boolean_output"
    a: bool = Field(default=None, description="The output boolean value")
    #fmt: on


class BooleanValueInvocation(BaseInvocation):
    """A boolean value"""
    #fmt: off
    type: Literal["value_boolean"] = "value_boolean"
    a: bool = Field(default=False, description="A boolean value")
    #fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["values", "boolean"],
                "title": "Boolean Value"
            }
        }

    def invoke(self, context: InvocationContext) -> BooleanOutput:
        return BooleanOutput(a=self.a)


class ImageValueInvocation(BaseInvocation):
    """An image field"""

    # fmt: off
    type: Literal["value_image"] = "value_image"
    # Inputs
    image: ImageField = Field(description="The output image field")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["values", "image"],
                "title": "Image"
            }
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )

        return build_image_output(
            image_type=self.image.image_type,
            image_name=self.image.image_name,
            image=image
        )
