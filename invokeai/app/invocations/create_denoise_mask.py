from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, Input, InputField
from invokeai.app.invocations.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import DenoiseMaskOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor


@invocation(
    "create_denoise_mask",
    title="Create Denoise Mask",
    tags=["mask", "denoise"],
    category="latents",
    version="1.0.2",
)
class CreateDenoiseMaskInvocation(BaseInvocation):
    """Creates mask for denoising model run."""

    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, ui_order=0)
    image: Optional[ImageField] = InputField(default=None, description="Image which will be masked", ui_order=1)
    mask: ImageField = InputField(description="The mask to use when pasting", ui_order=2)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled, ui_order=3)
    fp32: bool = InputField(default=False, description=FieldDescriptions.fp32, ui_order=4)

    def prep_mask_tensor(self, mask_image: Image.Image) -> torch.Tensor:
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_tensor: torch.Tensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        # if shape is not None:
        #    mask_tensor = tv_resize(mask_tensor, shape, T.InterpolationMode.BILINEAR)
        return mask_tensor

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> DenoiseMaskOutput:
        if self.image is not None:
            image = context.images.get_pil(self.image.image_name)
            image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
        else:
            image_tensor = None

        mask = self.prep_mask_tensor(
            context.images.get_pil(self.mask.image_name),
        )

        if image_tensor is not None:
            vae_info = context.models.load(self.vae.vae)

            img_mask = tv_resize(mask, image_tensor.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
            masked_image = image_tensor * torch.where(img_mask < 0.5, 0.0, 1.0)
            # TODO:
            context.util.signal_progress("Running VAE encoder")
            masked_latents = ImageToLatentsInvocation.vae_encode(vae_info, self.fp32, self.tiled, masked_image.clone())

            masked_latents_name = context.tensors.save(tensor=masked_latents)
        else:
            masked_latents_name = None

        mask_name = context.tensors.save(tensor=mask)

        return DenoiseMaskOutput.build(
            mask_name=mask_name,
            masked_latents_name=masked_latents_name,
            gradient=False,
        )
