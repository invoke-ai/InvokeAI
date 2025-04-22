"""Utility functions for ACE++ framework pipelines and different crop/merge operations

1. Create empty image with given size and color
2. Concat images either horizontally or vertically
3. Crop image to given size and position
4. Paste cropped image to given position with resizing
Create empty mask with given size and value

Nodes in Inpaint-Stitch pipeline

Test each node:
Create empty image
- Test create different sizes and colors
- Test create mask - zero mask/ ones mask
Image Resize Advanced
- Resize to fixed size
- Resize with keep proportion
- Condition: downscale if bigger, upscale if smaller
- Gaussian Blur Mask
- Image Concatenate
"""

import math
from typing import List, Literal, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import (
    ImageField,
    Input,
    InputField,
    OutputField,
    TensorField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext

DIRECTION_OPTIONS = Literal["right", "left", "down", "up"]


def concat_images(
    image1: Image.Image, image2: Image.Image, direction: str = "right", match_image_size=True
) -> Image.Image:
    """Concatenate two images either horizontally or vertically."""
    # Ensure that image modes are same
    if image1.mode != image2.mode:
        image2 = image2.convert(image1.mode)

    if direction == "right" or direction == "left":
        if direction == "left":
            image1, image2 = image2, image1
        new_width = image1.width + image2.width
        new_height = max(image1.height, image2.height)
        new_image = Image.new(image1.mode, (new_width, new_height))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))
    elif direction == "down" or direction == "up":
        if direction == "up":
            image1, image2 = image2, image1
        new_width = max(image1.width, image2.width)
        new_height = image1.height + image2.height
        new_image = Image.new(image1.mode, (new_width, new_height))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, image1.height))
    else:
        raise ValueError("Mode must be either 'horizontal' or 'vertical'.")

    return new_image


@invocation(
    "concat_images",
    title="Concatenate Images",
    tags=["image_processing"],
    category="image_processing",
    version="1.0.0",
)
class ConcatImagesInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Concatenate two images either horizontally or vertically."""

    image1: ImageField = InputField(description="The first image to process")
    image2: ImageField = InputField(description="The second image to process")
    mode: DIRECTION_OPTIONS = InputField(
        default="horizontal", description="Mode of concatenation: 'horizontal' or 'vertical'"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image1 = context.images.get_pil(self.image1.image_name)
        image2 = context.images.get_pil(self.image2.image_name)
        concatenated_image = concat_images(image1, image2, self.mode)
        image_dto = context.images.save(image=concatenated_image)
        return ImageOutput.build(image_dto)


@invocation_output("inpaint_crop_output")
class InpaintCropOutput(BaseInvocationOutput):
    """The output of Inpain Crop Invocation."""

    image_crop: ImageField = OutputField(description="Cropped part of image", title="Conditioning")
    stitcher: List[int] = OutputField(description="Parameter for stitching image after inpainting")


@invocation(
    "inpaint_crop",
    title="Inpaint Crop",
    tags=["image_processing"],
    version="1.0.0",
)
class InpaintCropInvocation(BaseInvocation, WithMetadata, WithBoard):
    "Crop from image masked area with resize and expand options"

    image: ImageField = InputField(description="The source image")
    mask: TensorField = InputField(description="Inpaint mask")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name, "RGB")
        mask = context.tensors.load(self.mask.tensor_name)

        # TODO: Finish InpaintCrop implementation
        image_crop = Image.new("RGB", (256, 256))

        image_dto = context.images.save(image=image_crop)
        return ImageOutput.build(image_dto)


@invocation_output("ace_plus_plus_output")
class ACEppProcessorOutput(BaseInvocationOutput):
    """The conditioning output of a FLUX Fill invocation."""

    image: ImageField = OutputField(description="Concatted image", title="Image")
    mask: TensorField = OutputField(description="Inpaint mask")
    crop_pad: int = OutputField(description="Padding to crop result")
    crop_width: int = OutputField(description="Width of output area")
    crop_height: int = OutputField(description="Heihgt of crop area")


@invocation(
    "ace_plus_plus_processor",
    title="ACE++ processor",
    tags=["image_processing"],
    version="1.0.0",
)
class ACEppProcessor(BaseInvocation):
    reference_image: ImageField = InputField(description="Reference Image")
    edit_image: Optional[ImageField] = InputField(description="Edit Image", default=None, input=Input.Connection)
    edit_mask: Optional[TensorField] = InputField(description="Edit Mask", default=None, input=Input.Connection)

    width: int = InputField(default=512, gt=0, description="The width of the crop rectangle")
    height: int = InputField(default=512, gt=0, description="The height of the crop rectangle")

    max_seq_len: int = InputField(default=4096, gt=2048, le=5120, description="The height of the crop rectangle")

    def image_check(self, image_pil: Image.Image) -> torch.Tensor:
        max_aspect_ratio = 4

        image = self.transform_pil_tensor(image_pil)
        image = image.unsqueeze(0)
        # preprocess
        H, W = image.shape[2:]
        if H / W > max_aspect_ratio:
            image[0] = T.CenterCrop([int(max_aspect_ratio * W), W])(image[0])
        elif W / H > max_aspect_ratio:
            image[0] = T.CenterCrop([H, int(max_aspect_ratio * H)])(image[0])
        return image[0]

    def transform_pil_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        transform = T.Compose([T.ToTensor()])
        tensor_image: torch.Tensor = transform(pil_image)
        return tensor_image

    def invoke(self, context: InvocationContext) -> ACEppProcessorOutput:
        d = 16  # Flux pixels per patch rate

        image_pil = context.images.get_pil(self.reference_image.image_name, "RGB")
        image = self.image_check(image_pil) - 0.5

        if self.edit_image is None:
            edit_image = torch.zeros((3, self.height, self.width))
            edit_mask = torch.ones((1, self.height, self.width))
        else:
            # TODO: make variant for editing
            edit_image = context.images.get_pil(self.edit_image.image_name)
            edit_image = self.image_check(edit_image) - 0.5
            if self.edit_mask is None:
                _, eH, eW = edit_image.shape
                edit_mask = torch.ones((eH, eW))
            else:
                edit_mask = context.tensors.load(self.edit_mask.tensor_name)

        out_H, out_W = edit_image.shape[-2:]

        _, H, W = image.shape
        _, eH, eW = edit_image.shape

        # align height with edit_image
        scale = eH / H
        tH, tW = eH, int(W * scale)

        reference_image = T.Resize((tH, tW), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(image)
        edit_image = torch.cat([reference_image, edit_image], dim=-1)
        edit_mask = torch.cat([torch.zeros((1, reference_image.shape[1], reference_image.shape[2])), edit_mask], dim=-1)
        slice_w = reference_image.shape[-1]

        H, W = edit_image.shape[-2:]
        scale = min(1.0, math.sqrt(self.max_seq_len * 2 / ((H / d) * (W / d))))
        rH = int(H * scale) // d * d
        rW = int(W * scale) // d * d
        slice_w = int(slice_w * scale) // d * d

        edit_image = T.Resize((rH, rW), interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=True)(edit_image)
        edit_mask = T.Resize((rH, rW), interpolation=T.InterpolationMode.NEAREST_EXACT, antialias=True)(edit_mask)

        edit_image = edit_image.unsqueeze(0).permute(0, 2, 3, 1)
        slice_w = slice_w if slice_w < 30 else slice_w + 30

        # Manipulations with -0.5/+0.5 needed only for gray color in mask
        # and took from original author's implementation
        # TODO: remove this -0.5/+0.5
        edit_image += 0.5
        # Convert to torch.bool
        edit_mask = edit_mask > 0.5
        image_out = Image.fromarray((edit_image[0].numpy() * 255).astype(np.uint8))

        image_dto = context.images.save(image=image_out)
        mask_name = context.tensors.save(edit_mask)
        return ACEppProcessorOutput(
            image=ImageField(image_name=image_dto.image_name),
            mask=TensorField(tensor_name=mask_name),
            crop_pad=slice_w,
            crop_height=int(out_H * scale),
            crop_width=int(out_W * scale),
        )
