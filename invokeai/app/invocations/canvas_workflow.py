"""Canvas workflow bridge invocations."""

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    Classification,
    invocation,
)
from invokeai.app.invocations.fields import ImageField, Input, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "canvas_composite_raster_input",
    title="Canvas Composite Input",
    tags=["canvas", "workflow", "canvas-workflow-input"],
    category="canvas",
    version="1.0.0",
    classification=Classification.Beta,
)
class CanvasCompositeRasterInputInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Provides the flattened canvas raster layer to a workflow."""

    image: ImageField = InputField(
        description="The flattened canvas raster layer.",
        input=Input.Direct,
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_dto = context.images.get_dto(self.image.image_name)
        return ImageOutput.build(image_dto=image_dto)


@invocation(
    "canvas_workflow_output",
    title="Canvas Workflow Output",
    tags=["canvas", "workflow", "canvas-workflow-output"],
    category="canvas",
    version="1.0.0",
    classification=Classification.Beta,
)
class CanvasWorkflowOutputInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Designates the workflow image output used by the canvas."""

    image: ImageField = InputField(
        description="The workflow's resulting image.",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_dto = context.images.get_dto(self.image.image_name)
        return ImageOutput.build(image_dto=image_dto)
