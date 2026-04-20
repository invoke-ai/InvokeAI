from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "canvas_output",
    title="Canvas Output",
    tags=["canvas", "output", "image"],
    category="canvas",
    version="1.0.0",
    use_cache=False,
)
class CanvasOutputInvocation(BaseInvocation):
    """Outputs an image to the canvas staging area.

    Use this node in workflows intended for canvas workflow integration.
    Connect the final image of your workflow to this node to send it
    to the canvas staging area when run via 'Run Workflow on Canvas'."""

    image: ImageField = InputField(description=FieldDescriptions.image)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)
