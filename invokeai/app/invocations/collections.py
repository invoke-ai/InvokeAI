# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team


import numpy as np
from pydantic import ValidationInfo, field_validator

from invokeai.app.invocations.primitives import IntegerCollectionOutput
from invokeai.app.util.misc import SEED_MAX

from .baseinvocation import BaseInvocation, invocation
from .fields import InputField


@invocation(
    "range", title="Integer Range", tags=["collection", "integer", "range"], category="collections", version="1.0.0"
)
class RangeInvocation(BaseInvocation):
    """Creates a range of numbers from start to stop with step"""

    start: int = InputField(default=0, description="The start of the range")
    stop: int = InputField(default=10, description="The stop of the range")
    step: int = InputField(default=1, description="The step of the range")

    @field_validator("stop")
    def stop_gt_start(cls, v: int, info: ValidationInfo):
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("stop must be greater than start")
        return v

    def invoke(self, context) -> IntegerCollectionOutput:
        return IntegerCollectionOutput(collection=list(range(self.start, self.stop, self.step)))


@invocation(
    "range_of_size",
    title="Integer Range of Size",
    tags=["collection", "integer", "size", "range"],
    category="collections",
    version="1.0.0",
)
class RangeOfSizeInvocation(BaseInvocation):
    """Creates a range from start to start + (size * step) incremented by step"""

    start: int = InputField(default=0, description="The start of the range")
    size: int = InputField(default=1, gt=0, description="The number of values")
    step: int = InputField(default=1, description="The step of the range")

    def invoke(self, context) -> IntegerCollectionOutput:
        return IntegerCollectionOutput(
            collection=list(range(self.start, self.start + (self.step * self.size), self.step))
        )


@invocation(
    "random_range",
    title="Random Range",
    tags=["range", "integer", "random", "collection"],
    category="collections",
    version="1.0.1",
    use_cache=False,
)
class RandomRangeInvocation(BaseInvocation):
    """Creates a collection of random numbers"""

    low: int = InputField(default=0, description="The inclusive low value")
    high: int = InputField(default=np.iinfo(np.int32).max, description="The exclusive high value")
    size: int = InputField(default=1, description="The number of values to generate")
    seed: int = InputField(
        default=0,
        ge=0,
        le=SEED_MAX,
        description="The seed for the RNG (omit for random)",
    )

    def invoke(self, context) -> IntegerCollectionOutput:
        rng = np.random.default_rng(self.seed)
        return IntegerCollectionOutput(collection=list(rng.integers(low=self.low, high=self.high, size=self.size)))
