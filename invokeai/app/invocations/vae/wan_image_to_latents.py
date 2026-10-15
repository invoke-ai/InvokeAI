"""Wan 2.2 image-to-latents invocation.

Encodes an image to latent space using the Wan VAE (AutoencoderKLWan). The Wan
VAE expects 5D ``[B, C, T, H, W]`` input with ``T=1`` for single images. After
encoding, latents are normalised against the per-channel ``latents_mean`` and
``latents_std`` stored in the VAE config — this matches the Diffusers
``WanPipeline`` reference and is the inverse of the denormalisation in
``wan_latents_to_image.py``.
"""

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
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_wan


@invocation(
    "wan_i2l",
    title="Image to Latents - Wan 2.2",
    tags=["image", "latents", "vae", "i2l", "wan"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanImageToLatentsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Encodes an image with the Wan VAE (AutoencoderKLWan).

    The output latents have the temporal dimension squeezed out, so downstream
    nodes see 4D ``[B, C, H, W]``. The denoise loop re-adds ``T=1`` before
    feeding the transformer.
    """

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @staticmethod
    def vae_encode(vae_info: LoadedModel, image_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(vae_info.model, AutoencoderKLWan):
            raise TypeError(f"Expected AutoencoderKLWan for Wan VAE, got {type(vae_info.model).__name__}.")

        estimated_working_memory = estimate_vae_working_memory_wan(
            operation="encode",
            vae=vae_info.model,
            pixel_height=image_tensor.shape[-2],
            pixel_width=image_tensor.shape[-1],
            pixel_frames=image_tensor.shape[2] if image_tensor.ndim == 5 else 1,
        )

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            assert isinstance(vae, AutoencoderKLWan)

            vae_dtype = next(iter(vae.parameters())).dtype
            image_tensor = image_tensor.to(device=get_effective_device(vae), dtype=vae_dtype)

            with torch.inference_mode():
                # Wan VAE expects 5D [B, C, T, H, W].
                if image_tensor.ndim == 4:
                    image_tensor = image_tensor.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

                encoded = vae.encode(image_tensor, return_dict=False)[0]
                latents = encoded.sample().to(dtype=vae_dtype)

                # Normalise to the denoiser's expected zero-centred space:
                #   (latents - mean) / std
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
                latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
                latents = (latents - latents_mean) / latents_std

                # Drop the temporal dim to keep the rest of the InvokeAI pipeline 4D.
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

        context.util.signal_progress("Running Wan VAE encode")
        latents = self.vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
