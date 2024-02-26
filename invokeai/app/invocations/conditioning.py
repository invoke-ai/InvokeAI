import torch

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    WithMetadata,
    invocation,
)
from invokeai.app.invocations.primitives import ConditioningField, ConditioningOutput, MaskField, MaskOutput


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
    mask: MaskField = InputField(description="A mask to add to the conditioning tensor.")

    def invoke(self, context: InvocationContext) -> ConditioningOutput:
        self.conditioning.mask = self.mask
        return ConditioningOutput(conditioning=self.conditioning)


@invocation(
    "rectangle_mask",
    title="Create Rectangle Mask",
    tags=["conditioning"],
    category="conditioning",
    version="1.0.0",
)
class RectangleMaskInvocation(BaseInvocation, WithMetadata):
    """Create a rectangular mask."""

    height: int = InputField(description="The height of the entire mask.")
    width: int = InputField(description="The width of the entire mask.")
    y_top: int = InputField(description="The top y-coordinate of the rectangular masked region (inclusive).")
    x_left: int = InputField(description="The left x-coordinate of the rectangular masked region (inclusive).")
    rectangle_height: int = InputField(description="The height of the rectangular masked region.")
    rectangle_width: int = InputField(description="The width of the rectangular masked region.")

    def invoke(self, context: InvocationContext) -> MaskOutput:
        mask = torch.zeros((1, self.height, self.width), dtype=torch.bool)
        mask[
            :, self.y_top : self.y_top + self.rectangle_height, self.x_left : self.x_left + self.rectangle_width
        ] = True

        mask_name = f"{context.graph_execution_state_id}__{self.id}_mask"
        context.services.latents.save(mask_name, mask)

        return MaskOutput(
            mask=MaskField(mask_name=mask_name),
            width=self.width,
            height=self.height,
        )
