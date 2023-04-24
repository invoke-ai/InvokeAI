# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal
from pydantic import Field
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext, InvocationConfig
from .image import ImageOutput, ImageField, build_image_output

# Pass-through parameter nodes - used by subgraphs


class IntOutput(BaseInvocationOutput):
    """An integer output"""
    #fmt: off
    type: Literal["int_output"] = "int_output"
    a: int = Field(default=None, description="The output integer")
    #fmt: on


class ParamIntInvocation(BaseInvocation):
    """An integer parameter"""
    #fmt: off
    type: Literal["param_int"] = "param_int"
    a: int = Field(default=0, description="An integer value")
    #fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["parameters", "integer"],
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


class ParamFloatInvocation(BaseInvocation):
    """A float parameter"""
    #fmt: off
    type: Literal["param_float"] = "param_float"
    a: float = Field(default=0, description="A float value")
    #fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["parameters", "float"],
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


class ParamStringInvocation(BaseInvocation):
    """A string parameter"""
    #fmt: off
    type: Literal["param_string"] = "param_string"
    a: str = Field(default='', description="A string value")
    #fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["parameters", "string"],
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


class ParamBooleanInvocation(BaseInvocation):
    """A boolean parameter"""
    #fmt: off
    type: Literal["param_boolean"] = "param_boolean"
    a: bool = Field(default=False, description="A boolean value")
    #fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["parameters", "boolean"],
                "title": "Boolean Value"
            }
        }

    def invoke(self, context: InvocationContext) -> BooleanOutput:
        return BooleanOutput(a=self.a)

class ParamImageInvocation(BaseInvocation):
    """Load an image and provide it as output."""

    # fmt: off
    type: Literal["param_image"] = "param_image"
    # Inputs
    image: ImageField = Field(description="The input image")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "tags": ["parameters", "image"],
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