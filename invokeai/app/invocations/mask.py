import numpy as np
import torch
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, InvocationContext, invocation
from invokeai.app.invocations.fields import ImageField, InputField, TensorField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput, MaskOutput
from invokeai.backend.image_util.util import pil_to_np


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
    classification=Classification.Beta,
)
class AlphaMaskToTensorInvocation(BaseInvocation):
    """Convert a mask image to a tensor. Opaque regions are 1 and transparent regions are 0."""

    image: ImageField = InputField(description="The mask image to convert.")
    invert: bool = InputField(default=False, description="Whether to invert the mask.")

    def invoke(self, context: InvocationContext) -> MaskOutput:
        image = context.images.get_pil(self.image.image_name)
        mask = torch.zeros((1, image.height, image.width), dtype=torch.bool)
        if self.invert:
            mask[0] = torch.tensor(np.array(image)[:, :, 3] == 0, dtype=torch.bool)
        else:
            mask[0] = torch.tensor(np.array(image)[:, :, 3] > 0, dtype=torch.bool)

        return MaskOutput(
            mask=TensorField(tensor_name=context.tensors.save(mask)),
            height=mask.shape[1],
            width=mask.shape[2],
        )


@invocation(
    "invert_tensor_mask",
    title="Invert Tensor Mask",
    tags=["conditioning"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Beta,
)
class InvertTensorMaskInvocation(BaseInvocation):
    """Inverts a tensor mask."""

    mask: TensorField = InputField(description="The tensor mask to convert.")

    def invoke(self, context: InvocationContext) -> MaskOutput:
        mask = context.tensors.load(self.mask.tensor_name)
        inverted = ~mask

        return MaskOutput(
            mask=TensorField(tensor_name=context.tensors.save(inverted)),
            height=inverted.shape[1],
            width=inverted.shape[2],
        )


@invocation(
    "image_mask_to_tensor",
    title="Image Mask to Tensor",
    tags=["conditioning"],
    category="conditioning",
    version="1.0.0",
)
class ImageMaskToTensorInvocation(BaseInvocation, WithMetadata):
    """Convert a mask image to a tensor. Converts the image to grayscale and uses thresholding at the specified value."""

    image: ImageField = InputField(description="The mask image to convert.")
    cutoff: int = InputField(ge=0, le=255, description="Cutoff (<)", default=128)
    invert: bool = InputField(default=False, description="Whether to invert the mask.")

    def invoke(self, context: InvocationContext) -> MaskOutput:
        image = context.images.get_pil(self.image.image_name, mode="L")

        mask = torch.zeros((1, image.height, image.width), dtype=torch.bool)
        if self.invert:
            mask[0] = torch.tensor(np.array(image)[:, :] >= self.cutoff, dtype=torch.bool)
        else:
            mask[0] = torch.tensor(np.array(image)[:, :] < self.cutoff, dtype=torch.bool)

        return MaskOutput(
            mask=TensorField(tensor_name=context.tensors.save(mask)),
            height=mask.shape[1],
            width=mask.shape[2],
        )


@invocation(
    "tensor_mask_to_image",
    title="Tensor Mask to Image",
    tags=["mask"],
    category="mask",
    version="1.1.0",
)
class MaskTensorToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Convert a mask tensor to an image."""

    mask: TensorField = InputField(description="The mask tensor to convert.")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        mask = context.tensors.load(self.mask.tensor_name)

        # Squeeze the channel dimension if it exists.
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        # Ensure that the mask is binary.
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        mask_np = (mask.float() * 255).byte().cpu().numpy()

        mask_pil = Image.fromarray(mask_np, mode="L")
        image_dto = context.images.save(image=mask_pil)
        return ImageOutput.build(image_dto)


@invocation(
    "apply_tensor_mask_to_image",
    title="Apply Tensor Mask to Image",
    tags=["mask"],
    category="mask",
    version="1.0.0",
)
class ApplyMaskTensorToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Applies a tensor mask to an image.

    The image is converted to RGBA and the mask is applied to the alpha channel."""

    mask: TensorField = InputField(description="The mask tensor to apply.")
    image: ImageField = InputField(description="The image to apply the mask to.")
    invert: bool = InputField(default=False, description="Whether to invert the mask.")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, mode="RGBA")
        mask = context.tensors.load(self.mask.tensor_name)

        # Squeeze the channel dimension if it exists.
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        # Ensure that the mask is binary.
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        mask_np = (mask.float() * 255).byte().cpu().numpy().astype(np.uint8)

        if self.invert:
            mask_np = 255 - mask_np

        # Apply the mask only to the alpha channel where the original alpha is non-zero. This preserves the original
        # image's transparency - else the transparent regions would end up as opaque black.

        # Separate the image into R, G, B, and A channels
        image_np = pil_to_np(image)
        r, g, b, a = np.split(image_np, 4, axis=-1)

        # Apply the mask to the alpha channel
        new_alpha = np.where(a.squeeze() > 0, mask_np, a.squeeze())

        # Stack the RGB channels with the modified alpha
        masked_image_np = np.dstack([r.squeeze(), g.squeeze(), b.squeeze(), new_alpha])

        # Convert back to an image (RGBA)
        masked_image = Image.fromarray(masked_image_np.astype(np.uint8), "RGBA")
        image_dto = context.images.save(image=masked_image)

        return ImageOutput.build(image_dto)
