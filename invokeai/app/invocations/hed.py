from builtins import bool

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.hed import ControlNetHED_Apache2, HEDEdgeDetector


@invocation(
    "hed_edge_detection",
    title="HED Edge Detection",
    tags=["controlnet", "hed", "softedge"],
    category="controlnet",
    version="1.0.0",
)
class HEDEdgeDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Geneartes an edge map using the HED (softedge) model."""

    image: ImageField = InputField(description="The image to process")
    scribble: bool = InputField(default=False, description=FieldDescriptions.scribble_mode)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")
        loaded_model = context.models.load_remote_model(HEDEdgeDetector.get_model_url(), HEDEdgeDetector.load_model)

        with loaded_model as model:
            assert isinstance(model, ControlNetHED_Apache2)
            hed_processor = HEDEdgeDetector(model)
            edge_map = hed_processor.run(image=image, scribble=self.scribble)

        image_dto = context.images.save(image=edge_map)
        return ImageOutput.build(image_dto)
