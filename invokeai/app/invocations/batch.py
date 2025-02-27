from typing import Literal

from pydantic import BaseModel

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    ImageField,
    Input,
    InputField,
    OutputField,
)
from invokeai.app.invocations.primitives import (
    FloatOutput,
    ImageOutput,
    IntegerOutput,
    StringOutput,
)
from invokeai.app.services.shared.invocation_context import InvocationContext

BATCH_GROUP_IDS = Literal[
    "None",
    "Group 1",
    "Group 2",
    "Group 3",
    "Group 4",
    "Group 5",
]


class NotExecutableNodeError(Exception):
    def __init__(self, message: str = "This class should never be executed or instantiated directly."):
        super().__init__(message)

    pass


class BaseBatchInvocation(BaseInvocation):
    batch_group_id: BATCH_GROUP_IDS = InputField(
        default="None",
        description="The ID of this batch node's group. If provided, all batch nodes in with the same ID will be 'zipped' before execution, and all nodes' collections must be of the same size.",
        input=Input.Direct,
        title="Batch Group",
    )

    def __init__(self):
        raise NotExecutableNodeError()


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
        default=[],
        min_length=1,
        description="The images to batch over",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        raise NotExecutableNodeError()


@invocation_output("image_generator_output")
class ImageGeneratorOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of boards"""

    images: list[ImageField] = OutputField(description="The generated images")


class ImageGeneratorField(BaseModel):
    pass


@invocation(
    "image_generator",
    title="Image Generator",
    tags=["primitives", "board", "image", "batch", "special"],
    category="primitives",
    version="1.0.0",
    classification=Classification.Special,
)
class ImageGenerator(BaseInvocation):
    """Generated a collection of images for use in a batched generation"""

    generator: ImageGeneratorField = InputField(
        description="The image generator.",
        input=Input.Direct,
        title="Generator Type",
    )

    def __init__(self):
        raise NotExecutableNodeError()

    def invoke(self, context: InvocationContext) -> ImageGeneratorOutput:
        raise NotExecutableNodeError()


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
        default=[],
        min_length=1,
        description="The strings to batch over",
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        raise NotExecutableNodeError()


@invocation_output("string_generator_output")
class StringGeneratorOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of strings"""

    strings: list[str] = OutputField(description="The generated strings")


class StringGeneratorField(BaseModel):
    pass


@invocation(
    "string_generator",
    title="String Generator",
    tags=["primitives", "string", "number", "batch", "special"],
    category="primitives",
    version="1.0.0",
    classification=Classification.Special,
)
class StringGenerator(BaseInvocation):
    """Generated a range of strings for use in a batched generation"""

    generator: StringGeneratorField = InputField(
        description="The string generator.",
        input=Input.Direct,
        title="Generator Type",
    )

    def __init__(self):
        raise NotExecutableNodeError()

    def invoke(self, context: InvocationContext) -> StringGeneratorOutput:
        raise NotExecutableNodeError()


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
        default=[],
        min_length=1,
        description="The integers to batch over",
    )

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        raise NotExecutableNodeError()


@invocation_output("integer_generator_output")
class IntegerGeneratorOutput(BaseInvocationOutput):
    integers: list[int] = OutputField(description="The generated integers")


class IntegerGeneratorField(BaseModel):
    pass


@invocation(
    "integer_generator",
    title="Integer Generator",
    tags=["primitives", "int", "number", "batch", "special"],
    category="primitives",
    version="1.0.0",
    classification=Classification.Special,
)
class IntegerGenerator(BaseInvocation):
    """Generated a range of integers for use in a batched generation"""

    generator: IntegerGeneratorField = InputField(
        description="The integer generator.",
        input=Input.Direct,
        title="Generator Type",
    )

    def __init__(self):
        raise NotExecutableNodeError()

    def invoke(self, context: InvocationContext) -> IntegerGeneratorOutput:
        raise NotExecutableNodeError()


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
        default=[],
        min_length=1,
        description="The floats to batch over",
    )

    def invoke(self, context: InvocationContext) -> FloatOutput:
        raise NotExecutableNodeError()


@invocation_output("float_generator_output")
class FloatGeneratorOutput(BaseInvocationOutput):
    """Base class for nodes that output a collection of floats"""

    floats: list[float] = OutputField(description="The generated floats")


class FloatGeneratorField(BaseModel):
    pass


@invocation(
    "float_generator",
    title="Float Generator",
    tags=["primitives", "float", "number", "batch", "special"],
    category="primitives",
    version="1.0.0",
    classification=Classification.Special,
)
class FloatGenerator(BaseInvocation):
    """Generated a range of floats for use in a batched generation"""

    generator: FloatGeneratorField = InputField(
        description="The float generator.",
        input=Input.Direct,
        title="Generator Type",
    )

    def __init__(self):
        raise NotExecutableNodeError()

    def invoke(self, context: InvocationContext) -> FloatGeneratorOutput:
        raise NotExecutableNodeError()
