from typing import Literal

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.invocations.vae.precise_io import float_output_fits, float_output_raster, save_decoded_image
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
    category="controlnet_preprocessors",
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
        # The image processor resizes and normalizes 8-bit input, so a precise source stays on the 8-bit path.
        image = context.images.get_pil(self.image.image_name, "RGB")
        precise = context.images.precise_storage_enabled

        loaded_model = context.models.load_remote_model(model_url, DepthAnythingPipeline.load_model)

        with loaded_model as depth_anything_detector:
            assert isinstance(depth_anything_detector, DepthAnythingPipeline)
            if precise:
                depth_map, unit = depth_anything_detector.generate_depth_with_unit(image)
            else:
                depth_map, unit = depth_anything_detector.generate_depth(image), None

        raster = None
        if unit is not None:
            if unit.shape != (depth_map.height, depth_map.width):
                context.logger.warning(
                    "Precise image storage skipped: the depth output size differs from the 8-bit depth image; "
                    "saving the ordinary 8-bit image."
                )
            elif float_output_fits(context, unit.shape[0], unit.shape[1], 1):
                raster = float_output_raster(context, unit[:, :, None])

        image_dto = save_decoded_image(context, depth_map, raster)
        return ImageOutput.build(image_dto)
