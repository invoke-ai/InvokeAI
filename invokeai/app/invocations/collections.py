# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team

from typing import Literal

import numpy as np
from pydantic import validator

from invokeai.app.invocations.primitives import IntegerCollectionOutput
from invokeai.app.util.misc import SEED_MAX, get_random_seed

from .baseinvocation import BaseInvocation, InputField, InvocationContext, node


@node(title="Integer Range", tags=["collection", "integer", "range"], category="collections")
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

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        return IntegerCollectionOutput(collection=list(range(self.start, self.stop, self.step)))


@node(title="Integer Range of Size", tags=["collection", "integer", "size", "range"], category="collections")
class RangeOfSizeInvocation(BaseInvocation):
    """Creates a range from start to start + size with step"""

    type: Literal["range_of_size"] = "range_of_size"

    # Inputs
    start: int = InputField(default=0, description="The start of the range")
    size: int = InputField(default=1, description="The number of values")
    step: int = InputField(default=1, description="The step of the range")

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        return IntegerCollectionOutput(collection=list(range(self.start, self.start + self.size, self.step)))


@node(title="Random Range", tags=["range", "integer", "random", "collection"], category="collections")
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

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        rng = np.random.default_rng(self.seed)
        return IntegerCollectionOutput(collection=list(rng.integers(low=self.low, high=self.high, size=self.size)))
