from typing import Literal, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    OutputField,
)
from invokeai.app.invocations.image_to_latents import ImageToLatentsInvocation
from invokeai.app.invocations.model import UNetField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager import LoadedModel
from invokeai.backend.model_manager.config import MainConfigBase, ModelVariantType
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor


@invocation_output("gradient_mask_output")
class GradientMaskOutput(BaseInvocationOutput):
    """Outputs a denoise mask and an image representing the total gradient of the mask."""

    denoise_mask: DenoiseMaskField = OutputField(
        description="Mask for denoise model run. Values of 0.0 represent the regions to be fully denoised, and 1.0 "
        + "represent the regions to be preserved."
    )
    expanded_mask_area: ImageField = OutputField(
        description="Image representing the total gradient area of the mask. For paste-back purposes."
    )


@invocation(
    "create_gradient_mask",
    title="Create Gradient Mask",
    tags=["mask", "denoise"],
    category="latents",
    version="1.2.0",
)
class CreateGradientMaskInvocation(BaseInvocation):
    """Creates mask for denoising model run."""

    mask: ImageField = InputField(default=None, description="Image which will be masked", ui_order=1)
    edge_radius: int = InputField(
        default=16, ge=0, description="How far to blur/expand the edges of the mask", ui_order=2
    )
    coherence_mode: Literal["Gaussian Blur", "Box Blur", "Staged"] = InputField(default="Gaussian Blur", ui_order=3)
    minimum_denoise: float = InputField(
        default=0.0, ge=0, le=1, description="Minimum denoise level for the coherence region", ui_order=4
    )
    image: Optional[ImageField] = InputField(
        default=None,
        description="OPTIONAL: Only connect for specialized Inpainting models, masked_latents will be generated from the image with the VAE",
        title="[OPTIONAL] Image",
        ui_order=6,
    )
    unet: Optional[UNetField] = InputField(
        description="OPTIONAL: If the Unet is a specialized Inpainting model, masked_latents will be generated from the image with the VAE",
        default=None,
        input=Input.Connection,
        title="[OPTIONAL] UNet",
        ui_order=5,
    )
    vae: Optional[VAEField] = InputField(
        default=None,
        description="OPTIONAL: Only connect for specialized Inpainting models, masked_latents will be generated from the image with the VAE",
        title="[OPTIONAL] VAE",
        input=Input.Connection,
        ui_order=7,
    )
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled, ui_order=8)
    fp32: bool = InputField(default=False, description=FieldDescriptions.fp32, ui_order=9)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> GradientMaskOutput:
        mask_image = context.images.get_pil(self.mask.image_name, mode="L")
        if self.edge_radius > 0:
            if self.coherence_mode == "Box Blur":
                blur_mask = mask_image.filter(ImageFilter.BoxBlur(self.edge_radius))
            else:  # Gaussian Blur OR Staged
                # Gaussian Blur uses standard deviation. 1/2 radius is a good approximation
                blur_mask = mask_image.filter(ImageFilter.GaussianBlur(self.edge_radius / 2))

            blur_tensor: torch.Tensor = image_resized_to_grid_as_tensor(blur_mask, normalize=False)

            # redistribute blur so that the original edges are 0 and blur outwards to 1
            blur_tensor = (blur_tensor - 0.5) * 2
            blur_tensor[blur_tensor < 0] = 0.0

            threshold = 1 - self.minimum_denoise

            if self.coherence_mode == "Staged":
                # wherever the blur_tensor is less than fully masked, convert it to threshold
                blur_tensor = torch.where((blur_tensor < 1) & (blur_tensor > 0), threshold, blur_tensor)
            else:
                # wherever the blur_tensor is above threshold but less than 1, drop it to threshold
                blur_tensor = torch.where((blur_tensor > threshold) & (blur_tensor < 1), threshold, blur_tensor)

        else:
            blur_tensor: torch.Tensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)

        mask_name = context.tensors.save(tensor=blur_tensor.unsqueeze(1))

        # compute a [0, 1] mask from the blur_tensor
        expanded_mask = torch.where((blur_tensor < 1), 0, 1)
        expanded_mask_image = Image.fromarray((expanded_mask.squeeze(0).numpy() * 255).astype(np.uint8), mode="L")
        expanded_image_dto = context.images.save(expanded_mask_image)

        masked_latents_name = None
        if self.unet is not None and self.vae is not None and self.image is not None:
            # all three fields must be present at the same time
            main_model_config = context.models.get_config(self.unet.unet.key)
            assert isinstance(main_model_config, MainConfigBase)
            if main_model_config.variant is ModelVariantType.Inpaint:
                mask = blur_tensor
                vae_info: LoadedModel = context.models.load(self.vae.vae)
                image = context.images.get_pil(self.image.image_name)
                image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                img_mask = tv_resize(mask, image_tensor.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
                masked_image = image_tensor * torch.where(img_mask < 0.5, 0.0, 1.0)
                context.util.signal_progress("Running VAE encoder")
                masked_latents = ImageToLatentsInvocation.vae_encode(
                    vae_info, self.fp32, self.tiled, masked_image.clone()
                )
                masked_latents_name = context.tensors.save(tensor=masked_latents)

        return GradientMaskOutput(
            denoise_mask=DenoiseMaskField(mask_name=mask_name, masked_latents_name=masked_latents_name, gradient=True),
            expanded_mask_area=ImageField(image_name=expanded_image_dto.image_name),
        )
