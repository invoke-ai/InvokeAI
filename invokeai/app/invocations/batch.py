from typing import Literal

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    Classification,
    invocation,
)
from invokeai.app.invocations.fields import (
    ImageField,
    Input,
    InputField,
)
from invokeai.app.invocations.primitives import FloatOutput, ImageOutput, IntegerOutput, StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext

BATCH_GROUP_IDS = Literal[
    "None",
    "Group 1",
    "Group 2",
    "Group 3",
    "Group 4",
    "Group 5",
]


class BaseBatchInvocation(BaseInvocation):
    batch_group_id: BATCH_GROUP_IDS = InputField(
        default="None",
        description="The ID of this batch node's group. If provided, all batch nodes in with the same ID will be 'zipped' before execution, and all nodes' collections must be of the same size.",
        input=Input.Direct,
        title="Batch Group",
    )

    def __init__(self):
        raise NotImplementedError("This class should never be executed or instantiated directly.")


@invocation(
    "image_batch",
    title="Image Batch",
    tags=["primitives", "image", "batch", "special"],
    category="primitives",
    version="1.0.0",
    classification=Classification.Special,
)
class ImageBatchInvocation(BaseBatchInvocation):
    """Create a batched generation, where the workflow is executed once for each image in the batch."""

    images: list[ImageField] = InputField(
        default=[], min_length=1, description="The images to batch over", input=Input.Direct
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        raise NotImplementedError("This class should never be executed or instantiated directly.")


@invocation(
    "string_batch",
    title="String Batch",
    tags=["primitives", "string", "batch", "special"],
    category="primitives",
    version="1.0.0",
    classification=Classification.Special,
)
class StringBatchInvocation(BaseBatchInvocation):
    """Create a batched generation, where the workflow is executed once for each string in the batch."""

    strings: list[str] = InputField(
        default=[], min_length=1, description="The strings to batch over", input=Input.Direct
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        raise NotImplementedError("This class should never be executed or instantiated directly.")


@invocation(
    "integer_batch",
    title="Integer Batch",
    tags=["primitives", "integer", "number", "batch", "special"],
    category="primitives",
    version="1.0.0",
    classification=Classification.Special,
)
class IntegerBatchInvocation(BaseBatchInvocation):
    """Create a batched generation, where the workflow is executed once for each integer in the batch."""

    integers: list[int] = InputField(
        default=[], min_length=1, description="The integers to batch over", input=Input.Direct
    )

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        raise NotImplementedError("This class should never be executed or instantiated directly.")


@invocation(
    "float_batch",
    title="Float Batch",
    tags=["primitives", "float", "number", "batch", "special"],
    category="primitives",
    version="1.0.0",
    classification=Classification.Special,
)
class FloatBatchInvocation(BaseBatchInvocation):
    """Create a batched generation, where the workflow is executed once for each float in the batch."""

    floats: list[float] = InputField(
        default=[], min_length=1, description="The floats to batch over", input=Input.Direct
    )

    def invoke(self, context: InvocationContext) -> FloatOutput:
        raise NotImplementedError("This class should never be executed or instantiated directly.")
