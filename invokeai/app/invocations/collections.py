# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team

from typing import Literal

import numpy as np
from pydantic import Field, validator

from invokeai.app.models.image import ImageField
from invokeai.app.util.misc import SEED_MAX, get_random_seed

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationConfig, InvocationContext, UIConfig


class IntCollectionOutput(BaseInvocationOutput):
    """A collection of integers"""

    type: Literal["int_collection"] = "int_collection"

    # Outputs
    collection: list[int] = Field(default=[], description="The int collection")


class FloatCollectionOutput(BaseInvocationOutput):
    """A collection of floats"""

    type: Literal["float_collection"] = "float_collection"

    # Outputs
    collection: list[float] = Field(default=[], description="The float collection")


class ImageCollectionOutput(BaseInvocationOutput):
    """A collection of images"""

    type: Literal["image_collection"] = "image_collection"

    # Outputs
    collection: list[ImageField] = Field(default=[], description="The output images")

    class Config:
        schema_extra = {"required": ["type", "collection"]}


class RangeInvocation(BaseInvocation):
    """Creates a range of numbers from start to stop with step"""

    type: Literal["range"] = "range"

    # Inputs
    start: int = Field(default=0, description="The start of the range")
    stop: int = Field(default=10, description="The stop of the range")
    step: int = Field(default=1, description="The step of the range")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Range", "tags": ["range", "integer", "collection"]},
        }

    @validator("stop")
    def stop_gt_start(cls, v, values):
        if "start" in values and v <= values["start"]:
            raise ValueError("stop must be greater than start")
        return v

    def invoke(self, context: InvocationContext) -> IntCollectionOutput:
        return IntCollectionOutput(collection=list(range(self.start, self.stop, self.step)))


class RangeOfSizeInvocation(BaseInvocation):
    """Creates a range from start to start + size with step"""

    type: Literal["range_of_size"] = "range_of_size"

    # Inputs
    start: int = Field(default=0, description="The start of the range")
    size: int = Field(default=1, description="The number of values")
    step: int = Field(default=1, description="The step of the range")

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Sized Range", "tags": ["range", "integer", "size", "collection"]},
        }

    def invoke(self, context: InvocationContext) -> IntCollectionOutput:
        return IntCollectionOutput(collection=list(range(self.start, self.start + self.size, self.step)))


class RandomRangeInvocation(BaseInvocation):
    """Creates a collection of random numbers"""

    type: Literal["random_range"] = "random_range"

    # Inputs
    low: int = Field(default=0, description="The inclusive low value")
    high: int = Field(default=np.iinfo(np.int32).max, description="The exclusive high value")
    size: int = Field(default=1, description="The number of values to generate")
    seed: int = Field(
        ge=0,
        le=SEED_MAX,
        description="The seed for the RNG (omit for random)",
        default_factory=get_random_seed,
    )

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Random Range", "tags": ["range", "integer", "random", "collection"]},
        }

    def invoke(self, context: InvocationContext) -> IntCollectionOutput:
        rng = np.random.default_rng(self.seed)
        return IntCollectionOutput(collection=list(rng.integers(low=self.low, high=self.high, size=self.size)))


class ImageCollectionInvocation(BaseInvocation):
    """Load a collection of images and provide it as output."""

    # fmt: off
    type: Literal["image_collection"] = "image_collection"

    # Inputs
    images: list[ImageField] = Field(
        default=[], description="The image collection to load"
    )
    # fmt: on

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        return ImageCollectionOutput(collection=self.images)

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "type_hints": {
                    "title": "Image Collection",
                    "images": "image_collection",
                }
            },
        }
