import onnxruntime as ort

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.dw_openpose import DWOpenposeDetector


@invocation(
    "dw_openpose_detection",
    title="DW Openpose Detection",
    tags=["controlnet", "dwpose", "openpose"],
    category="controlnet",
    version="1.1.1",
)
class DWOpenposeDetectionInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an openpose pose from an image using DWPose"""

    image: ImageField = InputField(description="The image to process")
    draw_body: bool = InputField(default=True)
    draw_face: bool = InputField(default=False)
    draw_hands: bool = InputField(default=False)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")

        onnx_det_path = context.models.download_and_cache_model(DWOpenposeDetector.get_model_url_det())
        onnx_pose_path = context.models.download_and_cache_model(DWOpenposeDetector.get_model_url_pose())

        loaded_session_det = context.models.load_local_model(
            onnx_det_path, DWOpenposeDetector.create_onnx_inference_session
        )
        loaded_session_pose = context.models.load_local_model(
            onnx_pose_path, DWOpenposeDetector.create_onnx_inference_session
        )

        with loaded_session_det as session_det, loaded_session_pose as session_pose:
            assert isinstance(session_det, ort.InferenceSession)
            assert isinstance(session_pose, ort.InferenceSession)
            detector = DWOpenposeDetector(session_det=session_det, session_pose=session_pose)
            detected_image = detector.run(
                image,
                draw_face=self.draw_face,
                draw_hands=self.draw_hands,
                draw_body=self.draw_body,
            )
        image_dto = context.images.save(image=detected_image)

        return ImageOutput.build(image_dto)
