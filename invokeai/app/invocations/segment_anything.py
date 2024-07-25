from typing import Dict, cast

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.grounding_segment_anything.gsa import GroundingSegmentAnythingDetector
from invokeai.backend.util.devices import TorchDevice

GROUNDING_SEGMENT_ANYTHING_MODELS = {
    "groundingdino_swint_ogc": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "segment_anything_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


@invocation(
    "segment_anything",
    title="Segment Anything",
    tags=["grounding_dino", "segment", "anything"],
    category="image",
    version="1.0.0",
)
class SegmentAnythingInvocation(BaseInvocation):
    """Automatically generate masks from an image using GroundingDINO & Segment Anything"""

    image: ImageField = InputField(description="The image to process")
    prompt: str = InputField(default="", description="Keywords to segment", title="Prompt")
    box_threshold: float = InputField(
        default=0.5, ge=0, le=1, description="Threshold of box detection", title="Box Threshold"
    )
    text_threshold: float = InputField(
        default=0.5, ge=0, le=1, description="Threshold of text detection", title="Text Threshold"
    )
    nms_threshold: float = InputField(
        default=0.8, ge=0, le=1, description="Threshold of nms detection", title="NMS Threshold"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        input_image = context.images.get_pil(self.image.image_name)

        grounding_dino_model = context.models.load_remote_model(
            GROUNDING_SEGMENT_ANYTHING_MODELS["groundingdino_swint_ogc"]
        )
        segment_anything_model = context.models.load_remote_model(
            GROUNDING_SEGMENT_ANYTHING_MODELS["segment_anything_vit_h"]
        )

        with (
            grounding_dino_model.model_on_device() as (_, grounding_dino_state_dict),
            segment_anything_model.model_on_device() as (_, segment_anything_state_dict),
        ):
            if not grounding_dino_state_dict or not segment_anything_state_dict:
                raise RuntimeError("Unable to load segmentation models")

            grounding_dino = GroundingSegmentAnythingDetector.build_grounding_dino(
                cast(Dict[str, torch.Tensor], grounding_dino_state_dict), TorchDevice.choose_torch_device()
            )
            segment_anything = GroundingSegmentAnythingDetector.build_segment_anything(
                cast(Dict[str, torch.Tensor], segment_anything_state_dict), TorchDevice.choose_torch_device()
            )
            detector = GroundingSegmentAnythingDetector(grounding_dino, segment_anything)

            mask = detector.predict(
                input_image, self.prompt, self.box_threshold, self.text_threshold, self.nms_threshold
            )
            image_dto = context.images.save(mask)

            """Builds an ImageOutput and its ImageField"""
            processed_image_field = ImageField(image_name=image_dto.image_name)
            return ImageOutput(
                image=processed_image_field,
                width=input_image.width,
                height=input_image.height,
            )
