import cv2

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.util import np_to_pil, pil_to_np


@invocation(
    "color_map",
    title="Color Map",
    tags=["controlnet"],
    category="controlnet",
    version="1.0.0",
)
class ColorMapInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates a color map from the provided image."""

    image: ImageField = InputField(description="The image to process")
    tile_size: int = InputField(default=64, ge=1, description=FieldDescriptions.tile_size)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")

        np_image = pil_to_np(image)
        height, width = np_image.shape[:2]

        width_tile_size = min(self.tile_size, width)
        height_tile_size = min(self.tile_size, height)

        color_map = cv2.resize(
            np_image,
            (width // width_tile_size, height // height_tile_size),
            interpolation=cv2.INTER_CUBIC,
        )
        color_map = cv2.resize(color_map, (width, height), interpolation=cv2.INTER_NEAREST)
        color_map_pil = np_to_pil(color_map)

        image_dto = context.images.save(image=color_map_pil)
        return ImageOutput.build(image_dto)
