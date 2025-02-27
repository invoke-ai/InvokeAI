from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.mediapipe_face import detect_faces


@invocation(
    "mediapipe_face_detection",
    title="MediaPipe Face Detection",
    tags=["controlnet", "face"],
    category="controlnet",
    version="1.0.0",
)
class MediaPipeFaceDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Detects faces using MediaPipe."""

    image: ImageField = InputField(description="The image to process")
    max_faces: int = InputField(default=1, ge=1, description="Maximum number of faces to detect")
    min_confidence: float = InputField(default=0.5, ge=0, le=1, description="Minimum confidence for face detection")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")
        detected_faces = detect_faces(image=image, max_faces=self.max_faces, min_confidence=self.min_confidence)
        image_dto = context.images.save(image=detected_faces)
        return ImageOutput.build(image_dto)
