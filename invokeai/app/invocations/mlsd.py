from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.mlsd import MLSDEdgeDetector
from invokeai.backend.image_util.mlsd.models.mbv2_mlsd_large import MobileV2_MLSD_Large


@invocation(
    "mlsd_edge_detection",
    title="MLSD Edge Detection",
    tags=["controlnet", "mlsd", "edge"],
    category="controlnet",
    version="1.0.0",
)
class MLSDEdgeDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an line segment edge map using MLSD."""

    image: ImageField = InputField(description="The image to process")
    thr_v: float = InputField(default=0.1, ge=0, description="MLSD parameter `thr_v`")
    thr_d: float = InputField(default=0.1, ge=0, description="MLSD parameter `thr_d`")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")
        loaded_model = context.models.load_remote_model(MLSDEdgeDetector.get_model_url(), MLSDEdgeDetector.load_model)

        with loaded_model as model:
            assert isinstance(model, MobileV2_MLSD_Large)
            detector = MLSDEdgeDetector(model)
            edge_map = detector.run(image, self.thr_v, self.thr_d)

        image_dto = context.images.save(image=edge_map)
        return ImageOutput.build(image_dto)
