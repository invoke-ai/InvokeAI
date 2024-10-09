from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.mlsd import MLSDDetector
from invokeai.backend.image_util.mlsd.models.mbv2_mlsd_large import MobileV2_MLSD_Large


@invocation(
    "mlsd_detection",
    title="MLSD Detection",
    tags=["controlnet", "mlsd", "edge"],
    category="controlnet",
    version="1.0.0",
)
class MLSDDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an line segment map using MLSD."""

    image: ImageField = InputField(description="The image to process")
    score_threshold: float = InputField(
        default=0.1, ge=0, description="The threshold used to score points when determining line segments"
    )
    distance_threshold: float = InputField(
        default=20.0,
        ge=0,
        description="Threshold for including a line segment - lines shorter than this distance will be discarded",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")
        loaded_model = context.models.load_remote_model(MLSDDetector.get_model_url(), MLSDDetector.load_model)

        with loaded_model as model:
            assert isinstance(model, MobileV2_MLSD_Large)
            detector = MLSDDetector(model)
            edge_map = detector.run(image, self.score_threshold, self.distance_threshold)

        image_dto = context.images.save(image=edge_map)
        return ImageOutput.build(image_dto)
