# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal

import cv2 as cv
import numpy as np
import numpy.random
from PIL import Image, ImageOps
from pydantic import Field

from ..services.image_storage import ImageType
from .baseinvocation import BaseInvocation, InvocationContext, BaseInvocationOutput
from .image import ImageField, ImageOutput


class IntCollectionOutput(BaseInvocationOutput):
    """A collection of integers"""

    type: Literal["int_collection"] = "int_collection"

    # Outputs
    collection: list[int] = Field(default=[], description="The int collection")


class RangeInvocation(BaseInvocation):
    """Creates a range"""

    type: Literal["range"] = "range"

    # Inputs
    start: int = Field(default=0, description="The start of the range")
    stop: int = Field(default=10, description="The stop of the range")
    step: int = Field(default=1, description="The step of the range")

    def invoke(self, context: InvocationContext) -> IntCollectionOutput:
        return IntCollectionOutput(collection=list(range(self.start, self.stop, self.step)))


class RandomRangeInvocation(BaseInvocation):
    """Creates a collection of random numbers"""

    type: Literal["random_range"] = "random_range"

    # Inputs
    low: int = Field(default=0, description="The inclusive low value")
    high: int = Field(default=np.iinfo(np.int32).max, description="The exclusive high value")
    size: int = Field(default=1, description="The number of values to generate")

    def invoke(self, context: InvocationContext) -> IntCollectionOutput:
        return IntCollectionOutput(collection=list(numpy.random.randint(self.low, self.high, size=self.size)))
