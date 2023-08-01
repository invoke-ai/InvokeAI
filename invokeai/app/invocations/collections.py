# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team

from typing import Literal

import numpy as np
from pydantic import validator

from invokeai.app.models.image import ImageField
from invokeai.app.util.misc import SEED_MAX, get_random_seed

from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    UITypeHint,
    tags,
    title,
)


class IntCollectionOutput(BaseInvocationOutput):
    """A collection of integers"""

    type: Literal["int_collection_output"] = "int_collection_output"

    # Outputs
    collection: list[int] = OutputField(
        default=[], description="The int collection", ui_type_hint=UITypeHint.IntegerCollection
    )


class FloatCollectionOutput(BaseInvocationOutput):
    """A collection of floats"""

    type: Literal["float_collection_output"] = "float_collection_output"

    # Outputs
    collection: list[float] = OutputField(
        default=[], description="The float collection", ui_type_hint=UITypeHint.FloatCollection
    )


class StringCollectionOutput(BaseInvocationOutput):
    """A collection of strings"""

    type: Literal["string_collection_output"] = "string_collection_output"

    # Outputs
    collection: list[str] = OutputField(
        default=[], description="The output strings", ui_type_hint=UITypeHint.StringCollection
    )


class ImageCollectionOutput(BaseInvocationOutput):
    """A collection of images"""

    type: Literal["image_collection_output"] = "image_collection_output"

    # Outputs
    collection: list[ImageField] = OutputField(
        default=[], description="The output images", ui_type_hint=UITypeHint.ImageCollection
    )


@title("Integer Range")
@tags("collection", "integer", "range")
class RangeInvocation(BaseInvocation):
    """Creates a range of numbers from start to stop with step"""

    type: Literal["range"] = "range"

    # Inputs
    start: int = InputField(default=0, description="The start of the range")
    stop: int = InputField(default=10, description="The stop of the range")
    step: int = InputField(default=1, description="The step of the range")

    @validator("stop")
    def stop_gt_start(cls, v, values):
        if "start" in values and v <= values["start"]:
            raise ValueError("stop must be greater than start")
        return v

    def invoke(self, context: InvocationContext) -> IntCollectionOutput:
        return IntCollectionOutput(collection=list(range(self.start, self.stop, self.step)))


@title("Integer Range of Size")
@tags("range", "integer", "size", "collection")
class RangeOfSizeInvocation(BaseInvocation):
    """Creates a range from start to start + size with step"""

    type: Literal["range_of_size"] = "range_of_size"

    # Inputs
    start: int = InputField(default=0, description="The start of the range")
    size: int = InputField(default=1, description="The number of values")
    step: int = InputField(default=1, description="The step of the range")

    def invoke(self, context: InvocationContext) -> IntCollectionOutput:
        return IntCollectionOutput(collection=list(range(self.start, self.start + self.size, self.step)))


@title("Random Range")
@tags("range", "integer", "random", "collection")
class RandomRangeInvocation(BaseInvocation):
    """Creates a collection of random numbers"""

    type: Literal["random_range"] = "random_range"

    # Inputs
    low: int = InputField(default=0, description="The inclusive low value")
    high: int = InputField(default=np.iinfo(np.int32).max, description="The exclusive high value")
    size: int = InputField(default=1, description="The number of values to generate")
    seed: int = InputField(
        ge=0,
        le=SEED_MAX,
        description="The seed for the RNG (omit for random)",
        default_factory=get_random_seed,
    )

    def invoke(self, context: InvocationContext) -> IntCollectionOutput:
        rng = np.random.default_rng(self.seed)
        return IntCollectionOutput(collection=list(rng.integers(low=self.low, high=self.high, size=self.size)))


@title("Image Collection")
@tags("image", "collection")
class ImageCollectionInvocation(BaseInvocation):
    """Load a collection of images and provide it as output."""

    type: Literal["image_collection"] = "image_collection"

    # Inputs
    images: list[ImageField] = InputField(
        default=[], description="The image collection to load", ui_type_hint=UITypeHint.ImageCollection
    )

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        return ImageCollectionOutput(collection=self.images)
