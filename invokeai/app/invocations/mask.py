import numpy as np
import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationContext, invocation
from invokeai.app.invocations.fields import ImageField, InputField, TensorField, WithMetadata
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


@invocation(
    "alpha_mask_to_tensor",
    title="Alpha Mask to Tensor",
    tags=["conditioning"],
    category="conditioning",
    version="1.0.0",
)
class AlphaMaskToTensorInvocation(BaseInvocation):
    """Convert a mask image to a tensor. Opaque regions are 1 and transparent regions are 0."""

    image: ImageField = InputField(description="The mask image to convert.")

    def invoke(self, context: InvocationContext) -> MaskOutput:
        image = context.images.get_pil(self.image.image_name)
        mask = torch.zeros((1, image.height, image.width), dtype=torch.bool)
        mask[0] = torch.tensor(np.array(image)[:, :, 3] > 0, dtype=torch.bool)

        return MaskOutput(
            mask=TensorField(tensor_name=context.tensors.save(mask)), height=image.height, width=image.width
        )
