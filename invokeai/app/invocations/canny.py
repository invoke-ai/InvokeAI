import cv2

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.util import cv2_to_pil, pil_to_cv2


@invocation(
    "canny_edge_detection",
    title="Canny Edge Detection",
    tags=["controlnet", "canny"],
    category="controlnet",
    version="1.0.0",
)
class CannyEdgeDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Geneartes an edge map using a cv2's Canny algorithm."""

    image: ImageField = InputField(description="The image to process")
    low_threshold: int = InputField(
        default=100, ge=0, le=255, description="The low threshold of the Canny pixel gradient (0-255)"
    )
    high_threshold: int = InputField(
        default=200, ge=0, le=255, description="The high threshold of the Canny pixel gradient (0-255)"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")
        np_img = pil_to_cv2(image)
        edge_map = cv2.Canny(np_img, self.low_threshold, self.high_threshold)
        edge_map_pil = cv2_to_pil(edge_map)
        image_dto = context.images.save(image=edge_map_pil)
        return ImageOutput.build(image_dto)
