from invokeai.backend.bria.controlnet_bria import BRIA_CONTROL_MODES
from pydantic import BaseModel, Field
from invokeai.invocation_api import ImageOutput
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, OutputField, UIType
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext
import numpy as np
import cv2
from PIL import Image

from invokeai.backend.image_util.depth_anything.depth_anything_pipeline import DepthAnythingPipeline
from invokeai.backend.bria.controlnet_aux.open_pose import OpenposeDetector, Body, Hand, Face

DEPTH_SMALL_V2_URL = "depth-anything/Depth-Anything-V2-Small-hf"
HF_LLLYASVIEL = "https://huggingface.co/lllyasviel/Annotators/resolve/main/"

class BriaControlNetField(BaseModel):
    image: ImageField = Field(description="The control image")
    model: ModelIdentifierField = Field(description="The ControlNet model to use")
    mode: BRIA_CONTROL_MODES = Field(description="The mode of the ControlNet")
    conditioning_scale: float = Field(description="The weight given to the ControlNet")

@invocation_output("flux_controlnet_output")
class BriaControlNetOutput(BaseInvocationOutput):
    """FLUX ControlNet info"""

    control: BriaControlNetField = OutputField(description=FieldDescriptions.control)
    preprocessed_images: ImageField = OutputField(description="The preprocessed control image")


@invocation(
    "bria_controlnet",
    title="ControlNet - Bria",
    tags=["controlnet", "bria"],
    category="controlnet",
    version="1.0.0",
)
class BriaControlNetInvocation(BaseInvocation):
    """Collect Bria ControlNet info to pass to denoiser node."""

    control_image: ImageField = InputField(description="The control image")
    control_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model, ui_type=UIType.BriaControlNetModel
    )
    control_mode: BRIA_CONTROL_MODES = InputField(
        default="depth", description="The mode of the ControlNet"
    )
    control_weight: float = InputField(
        default=1.0, ge=-1, le=2, description="The weight given to the ControlNet"
    )

    def invoke(self, context: InvocationContext) -> BriaControlNetOutput:
        image_in = resize_img(context.images.get_pil(self.control_image.image_name))
        if self.control_mode == "canny":
            control_image = extract_canny(image_in)
        elif self.control_mode == "depth":
            control_image = extract_depth(image_in, context)
        elif self.control_mode == "pose":
            control_image = extract_openpose(image_in, context)
        elif self.control_mode == "colorgrid":
            control_image = tile(64, image_in)
        elif self.control_mode == "recolor":
            control_image = convert_to_grayscale(image_in)
        elif self.control_mode == "tile":
            control_image = tile(16, image_in)
            
        control_image = resize_img(control_image)
        image_dto = context.images.save(image=control_image)
        image_output = ImageOutput.build(image_dto)
        return BriaControlNetOutput(
            preprocessed_images=image_output.image,
            control=BriaControlNetField(
                image=ImageField(image_name=image_dto.image_name),
                model=self.control_model,
                mode=self.control_mode,
                conditioning_scale=self.control_weight,
            ),
        )


RATIO_CONFIGS_1024 = {
    0.6666666666666666: {"width": 832, "height": 1248},
    0.7432432432432432: {"width": 880, "height": 1184},
    0.8028169014084507: {"width": 912, "height": 1136},
    1.0: {"width": 1024, "height": 1024},
    1.2456140350877194: {"width": 1136, "height": 912},
    1.3454545454545455: {"width": 1184, "height": 880},
    1.4339622641509433: {"width": 1216, "height": 848},
    1.5: {"width": 1248, "height": 832},
    1.5490196078431373: {"width": 1264, "height": 816},
    1.62: {"width": 1296, "height": 800},
    1.7708333333333333: {"width": 1360, "height": 768},
}

def extract_depth(image: Image.Image, context: InvocationContext):
    loaded_model = context.models.load_remote_model(DEPTH_SMALL_V2_URL, DepthAnythingPipeline.load_model)

    with loaded_model as depth_anything_detector:
        assert isinstance(depth_anything_detector, DepthAnythingPipeline)
        depth_map = depth_anything_detector.generate_depth(image)
    return depth_map

def extract_openpose(image: Image.Image, context: InvocationContext):
    body_model = context.models.load_remote_model(f"{HF_LLLYASVIEL}body_pose_model.pth", Body)
    hand_model = context.models.load_remote_model(f"{HF_LLLYASVIEL}hand_pose_model.pth", Hand)
    face_model = context.models.load_remote_model(f"{HF_LLLYASVIEL}facenet.pth", Face)

    with body_model as body_model, hand_model as hand_model, face_model as face_model:
        open_pose_model = OpenposeDetector(body_model, hand_model, face_model)
        processed_image_open_pose = open_pose_model(image, hand_and_face=True)
    
    processed_image_open_pose = processed_image_open_pose.resize(image.size)
    return processed_image_open_pose
    

def extract_canny(input_image):
    image = np.array(input_image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def convert_to_grayscale(image):
    gray_image = image.convert('L').convert('RGB')
    return gray_image

def tile(downscale_factor, input_image):
    control_image = input_image.resize((input_image.size[0] // downscale_factor, input_image.size[1] // downscale_factor)).resize(input_image.size, Image.Resampling.NEAREST)
    return control_image
    
def resize_img(control_image):
    image_ratio = control_image.width / control_image.height
    ratio = min(RATIO_CONFIGS_1024.keys(), key=lambda k: abs(k - image_ratio))
    to_height = RATIO_CONFIGS_1024[ratio]["height"]
    to_width = RATIO_CONFIGS_1024[ratio]["width"]
    resized_image = control_image.resize((to_width, to_height), resample=Image.Resampling.LANCZOS)
    return resized_image
