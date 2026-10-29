from typing import Any

import numpy as np
from pydantic import ValidationInfo, field_validator

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import InputField, OutputField, UIType
from invokeai.app.invocations.primitives import IntegerCollectionOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import SEED_MAX

MAX_CARTESIAN_PRODUCT_SIZE = 100_000


@invocation("range", title="Integer Range", tags=["collection", "integer", "range"], category="batch", version="1.0.0")
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

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        return IntegerCollectionOutput(collection=list(range(self.start, self.stop, self.step)))


@invocation(
    "range_of_size",
    title="Integer Range of Size",
    tags=["collection", "integer", "size", "range"],
    category="batch",
    version="1.0.0",
)
class RangeOfSizeInvocation(BaseInvocation):
    """Creates a range from start to start + (size * step) incremented by step"""

    start: int = InputField(default=0, description="The start of the range")
    size: int = InputField(default=1, gt=0, description="The number of values")
    step: int = InputField(default=1, description="The step of the range")

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        return IntegerCollectionOutput(
            collection=list(range(self.start, self.start + (self.step * self.size), self.step))
        )


@invocation(
    "random_range",
    title="Random Range",
    tags=["range", "integer", "random", "collection"],
    category="batch",
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

    def invoke(self, context: InvocationContext) -> IntegerCollectionOutput:
        rng = np.random.default_rng(self.seed)
        return IntegerCollectionOutput(collection=list(rng.integers(low=self.low, high=self.high, size=self.size)))


@invocation_output("collection_concat_output")
class CollectionConcatInvocationOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(description="The concatenated collection", ui_type=UIType._Collection)


@invocation(
    "collection_concat",
    title="Concatenate Collections",
    tags=["collection", "concat", "sequential"],
    category="batch",
    version="1.0.0",
)
class CollectionConcatInvocation(BaseInvocation):
    """Concatenates two collections in left-to-right order."""

    first: list[Any] = InputField(default=[], description="The first collection", ui_type=UIType._Collection)
    second: list[Any] = InputField(default=[], description="The second collection", ui_type=UIType._Collection)

    def invoke(self, context: InvocationContext) -> CollectionConcatInvocationOutput:
        return CollectionConcatInvocationOutput(collection=[*self.first, *self.second])


@invocation_output("collection_zip_output")
class CollectionZipInvocationOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(description="The positional pairs", ui_type=UIType._Collection)


@invocation(
    "collection_zip",
    title="Zip Collections",
    tags=["collection", "zip", "pair"],
    category="batch",
    version="1.0.0",
)
class CollectionZipInvocation(BaseInvocation):
    """Pairs items at matching positions from two equally sized collections."""

    first: list[Any] = InputField(default=[], description="The first collection", ui_type=UIType._Collection)
    second: list[Any] = InputField(default=[], description="The second collection", ui_type=UIType._Collection)

    def invoke(self, context: InvocationContext) -> CollectionZipInvocationOutput:
        if len(self.first) != len(self.second):
            raise ValueError("Zip inputs must have the same length")
        return CollectionZipInvocationOutput(
            collection=[[first, second] for first, second in zip(self.first, self.second, strict=True)]
        )


@invocation_output("collection_cartesian_output")
class CollectionCartesianInvocationOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(description="The Cartesian product pairs", ui_type=UIType._Collection)


@invocation(
    "collection_cartesian",
    title="Cartesian Product of Collections",
    tags=["collection", "cartesian", "product"],
    category="batch",
    version="1.0.0",
)
class CollectionCartesianInvocation(BaseInvocation):
    """Emits every pair formed by one item from each collection, up to 100,000 pairs."""

    first: list[Any] = InputField(default=[], description="The first collection", ui_type=UIType._Collection)
    second: list[Any] = InputField(default=[], description="The second collection", ui_type=UIType._Collection)

    def invoke(self, context: InvocationContext) -> CollectionCartesianInvocationOutput:
        # Divide before multiplying so the comparison cannot overflow for very large input collections.
        if self.first and self.second and len(self.first) > MAX_CARTESIAN_PRODUCT_SIZE // len(self.second):
            raise ValueError(f"Cartesian product exceeds the maximum size of {MAX_CARTESIAN_PRODUCT_SIZE} pairs")
        return CollectionCartesianInvocationOutput(
            collection=[[first, second] for first in self.first for second in self.second]
        )
