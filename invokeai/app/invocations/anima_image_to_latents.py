"""Anima image-to-latents invocation.

Encodes an image to latent space using the Anima VAE (AutoencoderKLWan or FLUX VAE).

For Wan VAE (AutoencoderKLWan):
- Input image is converted to 5D tensor [B, C, T, H, W] with T=1
- After encoding, latents are normalized: (latents - mean) / std
  (inverse of the denormalization in anima_latents_to_image.py)

For FLUX VAE (AutoEncoder):
- Encoding is handled internally by the FLUX VAE
"""

from typing import Union

import einops
import torch
from diffusers.models.autoencoders import AutoencoderKLWan

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.modules.autoencoder import AutoEncoder as FluxAutoEncoder
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux

AnimaVAE = Union[AutoencoderKLWan, FluxAutoEncoder]


@invocation(
    "anima_i2l",
    title="Image to Latents - Anima",
    tags=["image", "latents", "vae", "i2l", "anima"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class AnimaImageToLatentsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates latents from an image using the Anima VAE (supports Wan 2.1 and FLUX VAE)."""

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @staticmethod
    def vae_encode(vae_info: LoadedModel, image_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(vae_info.model, (AutoencoderKLWan, FluxAutoEncoder)):
            raise TypeError(
                f"Expected AutoencoderKLWan or FluxAutoEncoder for Anima VAE, got {type(vae_info.model).__name__}."
            )

        estimated_working_memory = estimate_vae_working_memory_flux(
            operation="encode",
            image_tensor=image_tensor,
            vae=vae_info.model,
        )

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            if not isinstance(vae, (AutoencoderKLWan, FluxAutoEncoder)):
                raise TypeError(f"Expected AutoencoderKLWan or FluxAutoEncoder, got {type(vae).__name__}.")

            vae_dtype = next(iter(vae.parameters())).dtype
            image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

            with torch.inference_mode():
                if isinstance(vae, FluxAutoEncoder):
                    # FLUX VAE handles scaling internally
                    generator = torch.Generator(device=TorchDevice.choose_torch_device()).manual_seed(0)
                    latents = vae.encode(image_tensor, sample=True, generator=generator)
                else:
                    # AutoencoderKLWan expects 5D input [B, C, T, H, W]
                    if image_tensor.ndim == 4:
                        image_tensor = image_tensor.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

                    encoded = vae.encode(image_tensor, return_dict=False)[0]
                    latents = encoded.sample().to(dtype=vae_dtype)

                    # Normalize to denoiser space: (latents - mean) / std
                    # This is the inverse of the denormalization in anima_latents_to_image.py
                    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
                    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
                    latents = (latents - latents_mean) / latents_std

                    # Remove temporal dimension: [B, C, 1, H, W] -> [B, C, H, W]
                    if latents.ndim == 5:
                        latents = latents.squeeze(2)

        return latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, (AutoencoderKLWan, FluxAutoEncoder)):
            raise TypeError(
                f"Expected AutoencoderKLWan or FluxAutoEncoder for Anima VAE, got {type(vae_info.model).__name__}."
            )

        context.util.signal_progress("Running Anima VAE encode")
        latents = self.vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
