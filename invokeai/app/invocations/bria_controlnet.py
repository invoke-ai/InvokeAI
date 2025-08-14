from PIL import Image
from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    InputField,
    OutputField,
    UIType,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.bria.controlnet_bria import BRIA_CONTROL_MODES
from invokeai.invocation_api import Classification

DEPTH_SMALL_V2_URL = "depth-anything/Depth-Anything-V2-Small-hf"
HF_LLLYASVIEL = "https://huggingface.co/lllyasviel/Annotators/resolve/main/"


class BriaControlNetField(BaseModel):
    image: ImageField = Field(description="The control image")
    model: ModelIdentifierField = Field(description="The ControlNet model to use")
    mode: BRIA_CONTROL_MODES = Field(description="The mode of the ControlNet")
    conditioning_scale: float = Field(description="The weight given to the ControlNet")


@invocation_output("bria_controlnet_output")
class BriaControlNetOutput(BaseInvocationOutput):
    """Bria ControlNet info"""

    control: BriaControlNetField = OutputField(description=FieldDescriptions.control)


@invocation(
    "bria_controlnet",
    title="ControlNet - Bria",
    tags=["controlnet", "bria"],
    category="controlnet",
    version="1.0.0",
    classification=Classification.Prototype,
)
class BriaControlNetInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Collect Bria ControlNet info to pass to denoiser node."""

    control_image: ImageField = InputField(description="The control image")
    control_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.controlnet_model, ui_type=UIType.BriaControlNetModel
    )
    control_mode: BRIA_CONTROL_MODES = InputField(default="depth", description="The mode of the ControlNet")
    control_weight: float = InputField(default=1.0, ge=-1, le=2, description="The weight given to the ControlNet")

    def invoke(self, context: InvocationContext) -> BriaControlNetOutput:
        image_in = resize_img(context.images.get_pil(self.control_image.image_name))
        if self.control_mode == "colorgrid":
            control_image = tile(64, image_in)
        elif self.control_mode == "recolor":
            control_image = convert_to_grayscale(image_in)
        elif self.control_mode == "tile":
            control_image = tile(16, image_in)
        else:
            control_image = image_in

        control_image = resize_img(control_image)
        image_dto = context.images.save(image=control_image)
        return BriaControlNetOutput(
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


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    gray_image = image.convert("L").convert("RGB")
    return gray_image


def tile(downscale_factor: int, input_image: Image.Image) -> Image.Image:
    control_image = input_image.resize(
        (input_image.size[0] // downscale_factor, input_image.size[1] // downscale_factor)
    ).resize(input_image.size, Image.Resampling.NEAREST)
    return control_image


def resize_img(control_image: Image.Image) -> Image.Image:
    image_ratio = control_image.width / control_image.height
    ratio = min(RATIO_CONFIGS_1024.keys(), key=lambda k: abs(k - image_ratio))
    to_height = RATIO_CONFIGS_1024[ratio]["height"]
    to_width = RATIO_CONFIGS_1024[ratio]["width"]
    resized_image = control_image.resize((to_width, to_height), resample=Image.Resampling.LANCZOS)
    return resized_image
