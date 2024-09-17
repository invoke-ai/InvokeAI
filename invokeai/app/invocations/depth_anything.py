from typing import Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.depth_anything.depth_anything_pipeline import DepthAnythingPipeline

DEPTH_ANYTHING_MODEL_SIZES = Literal["large", "base", "small", "small_v2"]
# DepthAnything V2 Small model is licensed under Apache 2.0 but not the base and large models.
DEPTH_ANYTHING_MODELS = {
    "large": "LiheYoung/depth-anything-large-hf",
    "base": "LiheYoung/depth-anything-base-hf",
    "small": "LiheYoung/depth-anything-small-hf",
    "small_v2": "depth-anything/Depth-Anything-V2-Small-hf",
}


@invocation(
    "depth_anything_depth_estimation",
    title="Depth Anything Depth Estimation",
    tags=["controlnet", "depth", "depth anything"],
    category="controlnet",
    version="1.0.0",
)
class DepthAnythingDepthEstimationInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates a depth map using a Depth Anything model."""

    image: ImageField = InputField(description="The image to process")
    model_size: DEPTH_ANYTHING_MODEL_SIZES = InputField(
        default="small_v2", description="The size of the depth model to use"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        model_url = DEPTH_ANYTHING_MODELS[self.model_size]
        image = context.images.get_pil(self.image.image_name, "RGB")

        loaded_model = context.models.load_remote_model(model_url, DepthAnythingPipeline.load_model)

        with loaded_model as depth_anything_detector:
            assert isinstance(depth_anything_detector, DepthAnythingPipeline)
            depth_map = depth_anything_detector.generate_depth(image)

        image_dto = context.images.save(image=depth_map)
        return ImageOutput.build(image_dto)
