from builtins import bool

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.lineart import Generator, LineartEdgeDetector


@invocation(
    "lineart_edge_detection",
    title="Lineart Edge Detection",
    tags=["controlnet", "lineart"],
    category="controlnet",
    version="1.0.0",
)
class LineartEdgeDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an edge map using the Lineart model."""

    image: ImageField = InputField(description="The image to process")
    coarse: bool = InputField(default=False, description="Whether to use coarse mode")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")
        model_url = LineartEdgeDetector.get_model_url(self.coarse)
        loaded_model = context.models.load_remote_model(model_url, LineartEdgeDetector.load_model)

        with loaded_model as model:
            assert isinstance(model, Generator)
            detector = LineartEdgeDetector(model)
            edge_map = detector.run(image=image)

        image_dto = context.images.save(image=edge_map)
        return ImageOutput.build(image_dto)
