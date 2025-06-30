from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
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
from invokeai.backend.model_manager.config import MainConfigBase
from invokeai.backend.model_manager.taxonomy import ModelVariantType
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
    version="1.3.0",
)
class CreateGradientMaskInvocation(BaseInvocation):
    """Creates mask for denoising."""

    mask: ImageField = InputField(description="Image which will be masked", ui_order=1)
    edge_radius: int = InputField(default=16, ge=0, description="How far to expand the edges of the mask", ui_order=2)
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

        # Resize the mask_image. Makes the filter 64x faster and doesn't hurt quality in latent scale anyway
        mask_image = mask_image.resize(
            (
                mask_image.width // LATENT_SCALE_FACTOR,
                mask_image.height // LATENT_SCALE_FACTOR,
            ),
            resample=Image.Resampling.BILINEAR,
        )

        mask_np_orig = np.array(mask_image, dtype=np.float32)

        self.edge_radius = self.edge_radius // LATENT_SCALE_FACTOR  # scale the edge radius to match the mask size

        if self.edge_radius > 0:
            mask_np = 255 - mask_np_orig  # invert so 0 is unmasked (higher values = higher denoise strength)
            dilated_mask = mask_np.copy()

            # Create kernel based on coherence mode
            if self.coherence_mode == "Box Blur":
                # Create a circular distance kernel that fades from center outward
                kernel_size = self.edge_radius * 2 + 1
                center = self.edge_radius
                kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                        if dist <= self.edge_radius:
                            kernel[i, j] = 1.0 - (dist / self.edge_radius)
            else:  # Gaussian Blur or Staged
                # Create a Gaussian kernel
                kernel_size = self.edge_radius * 2 + 1
                kernel = cv2.getGaussianKernel(
                    kernel_size, self.edge_radius / 2.5
                )  # 2.5 is a magic number (standard deviation capturing)
                kernel = kernel * kernel.T  # Make 2D gaussian kernel
                kernel = kernel / np.max(kernel)  # Normalize center to 1.0

                # Ensure values outside radius are 0
                center = self.edge_radius
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                        if dist > self.edge_radius:
                            kernel[i, j] = 0

            # 2D max filter
            mask_tensor = torch.tensor(mask_np)
            kernel_tensor = torch.tensor(kernel)
            dilated_mask = 255 - self.max_filter2D_torch(mask_tensor, kernel_tensor).cpu()
            dilated_mask = dilated_mask.numpy()

            threshold = (1 - self.minimum_denoise) * 255

            if self.coherence_mode == "Staged":
                # wherever expanded mask is darker than the original mask but original was above threshhold, set it to the threshold
                # makes any expansion areas drop to threshhold. Raising minimum across the image happen outside of this if
                threshold_mask = (dilated_mask < mask_np_orig) & (mask_np_orig > threshold)
                dilated_mask = np.where(threshold_mask, threshold, mask_np_orig)

            # wherever expanded mask is less than 255 but greater than threshold, drop it to threshold (minimum denoise)
            threshold_mask = (dilated_mask > threshold) & (dilated_mask < 255)
            dilated_mask = np.where(threshold_mask, threshold, dilated_mask)

        else:
            dilated_mask = mask_np_orig.copy()

        # convert to tensor
        dilated_mask = np.clip(dilated_mask, 0, 255).astype(np.uint8)
        mask_tensor = torch.tensor(dilated_mask, device=torch.device("cpu"))

        # binary mask for compositing
        expanded_mask = np.where((dilated_mask < 255), 0, 255)
        expanded_mask_image = Image.fromarray(expanded_mask.astype(np.uint8), mode="L")
        expanded_mask_image = expanded_mask_image.resize(
            (
                mask_image.width * LATENT_SCALE_FACTOR,
                mask_image.height * LATENT_SCALE_FACTOR,
            ),
            resample=Image.Resampling.NEAREST,
        )
        expanded_image_dto = context.images.save(expanded_mask_image)

        # restore the original mask size
        dilated_mask = Image.fromarray(dilated_mask.astype(np.uint8))
        dilated_mask = dilated_mask.resize(
            (
                mask_image.width * LATENT_SCALE_FACTOR,
                mask_image.height * LATENT_SCALE_FACTOR,
            ),
            resample=Image.Resampling.NEAREST,
        )

        # stack the mask as a tensor, repeating 4 times on dimmension 1
        dilated_mask_tensor = image_resized_to_grid_as_tensor(dilated_mask, normalize=False)
        mask_name = context.tensors.save(tensor=dilated_mask_tensor.unsqueeze(0))

        masked_latents_name = None
        if self.unet is not None and self.vae is not None and self.image is not None:
            # all three fields must be present at the same time
            main_model_config = context.models.get_config(self.unet.unet.key)
            assert isinstance(main_model_config, MainConfigBase)
            if main_model_config.variant is ModelVariantType.Inpaint:
                mask = dilated_mask_tensor
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

    def max_filter2D_torch(self, image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        This morphological operation is much faster in torch than numpy or opencv
        For reasonable kernel sizes, the overhead of copying the data to the GPU is not worth it.
        """
        h, w = kernel.shape
        pad_h, pad_w = h // 2, w // 2

        padded = torch.nn.functional.pad(image, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)
        result = torch.zeros_like(image)

        # This looks like it's inside out, but it does the same thing and is more efficient
        for i in range(h):
            for j in range(w):
                weight = kernel[i, j]
                if weight <= 0:
                    continue

                # Extract the region from padded tensor
                region = padded[i : i + image.shape[0], j : j + image.shape[1]]

                # Apply weight and update max
                result = torch.maximum(result, region * weight)

        return result
