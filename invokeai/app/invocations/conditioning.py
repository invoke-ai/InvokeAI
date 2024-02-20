import numpy as np
import torch
from PIL import Image

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    WithMetadata,
    invocation,
)
from invokeai.app.invocations.primitives import ConditioningField, ConditioningOutput, ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


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
    mask: ImageField = InputField(
        description="A mask image to add to the conditioning tensor. Only the first channel of the image is used. "
        "Pixels <128 are excluded from the mask, pixels >=128 are included in the mask."
    )
    mask_strength: float = InputField(
        description="The strength of the mask to apply to the conditioning tensor.", default=1.0
    )

    @staticmethod
    def convert_image_to_mask(image: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a uint8 mask tensor."""
        np_image = np.array(image)
        torch_image = torch.from_numpy(np_image[:, :, 0])
        mask = torch_image >= 128
        return mask.to(dtype=torch.uint8)

    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        image = context.services.images.get_pil_image(self.mask.image_name)
        mask = self.convert_image_to_mask(image)

        mask_name = f"{context.graph_execution_state_id}__{self.id}_conditioning_mask"
        context.services.latents.save(mask_name, mask)

        self.conditioning.mask_name = mask_name
        self.conditioning.mask_strength = self.mask_strength
        return ConditioningOutput(conditioning=self.conditioning)


@invocation(
    "rectangle_mask",
    title="Create Rectangle Mask",
    tags=["conditioning"],
    category="conditioning",
    version="1.0.0",
)
class RectangleMaskInvocation(BaseInvocation, WithMetadata):
    """Create a mask image containing a rectangular mask region."""

    height: int = InputField(description="The height of the image.")
    width: int = InputField(description="The width of the image.")
    y_top: int = InputField(description="The top y-coordinate of the rectangle (inclusive).")
    y_bottom: int = InputField(description="The bottom y-coordinate of the rectangle (exclusive).")
    x_left: int = InputField(description="The left x-coordinate of the rectangle (inclusive).")
    x_right: int = InputField(description="The right x-coordinate of the rectangle (exclusive).")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        mask = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask[self.y_top : self.y_bottom, self.x_left : self.x_right, :] = 255
        mask_image = Image.fromarray(mask)

        image_dto = context.services.images.create(
            image=mask_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
