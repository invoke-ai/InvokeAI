import numpy as np
import torch
from PIL.Image import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ConditioningField, ConditioningOutput, ImageField


@invocation(
    "add_conditioning_mask",
    title="Add Conditioning Mask",
    tags=["conditioning"],
    category="conditioning",
    version="1.0.0",
)
class AddConditioningMaskInvocation(BaseInvocation):
    """Add a mask to an existing conditioning tensor."""

    conditioning: ConditioningField = InputField(description="The conditioning tensor to add a mask to.")
    image: ImageField = InputField(
        description="A mask image to add to the conditioning tensor. Only the first channel of the image is used. "
        "Pixels <128 are excluded from the mask, pixels >=128 are included in the mask."
    )

    @staticmethod
    def convert_image_to_mask(image: Image) -> torch.Tensor:
        """Convert a PIL image to a uint8 mask tensor."""
        np_image = np.array(image)
        torch_image = torch.from_numpy(np_image[0, :, :])
        mask = torch_image >= 128
        return mask.to(dtype=torch.uint8)

    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        mask = self.convert_image_to_mask(image)

        mask_name = f"{context.graph_execution_state_id}__{self.id}_conditioning_mask"
        context.services.latents.save(mask_name, mask)

        self.conditioning.mask_name = mask_name
        return ConditioningOutput(conditioning=self.conditioning)
