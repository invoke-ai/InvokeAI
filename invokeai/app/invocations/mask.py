import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationContext, invocation
from invokeai.app.invocations.fields import InputField, TensorField, WithMetadata
from invokeai.app.invocations.primitives import MaskOutput


@invocation(
    "rectangle_mask",
    title="Create Rectangle Mask",
    tags=["conditioning"],
    category="conditioning",
    version="1.0.1",
)
class RectangleMaskInvocation(BaseInvocation, WithMetadata):
    """Create a rectangular mask."""

    width: int = InputField(description="The width of the entire mask.")
    height: int = InputField(description="The height of the entire mask.")
    x_left: int = InputField(description="The left x-coordinate of the rectangular masked region (inclusive).")
    y_top: int = InputField(description="The top y-coordinate of the rectangular masked region (inclusive).")
    rectangle_width: int = InputField(description="The width of the rectangular masked region.")
    rectangle_height: int = InputField(description="The height of the rectangular masked region.")

    def invoke(self, context: InvocationContext) -> MaskOutput:
        mask = torch.zeros((1, self.height, self.width), dtype=torch.bool)
        mask[:, self.y_top : self.y_top + self.rectangle_height, self.x_left : self.x_left + self.rectangle_width] = (
            True
        )

        mask_tensor_name = context.tensors.save(mask)
        return MaskOutput(
            mask=TensorField(tensor_name=mask_tensor_name),
            width=self.width,
            height=self.height,
        )
