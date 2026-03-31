"""Anima latents-to-image invocation.

Decodes Anima latents using the QwenImage VAE (AutoencoderKLWan) or
compatible FLUX VAE as fallback.

Latents from the denoiser are in normalized space (zero-centered). Before
VAE decode, they must be denormalized using the Wan 2.1 per-channel
mean/std: latents = latents * std + mean (matching diffusers WanPipeline).

The VAE expects 5D latents [B, C, T, H, W] — for single images, T=1.
"""

import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from einops import rearrange
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.modules.autoencoder import AutoEncoder as FluxAutoEncoder
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux


@invocation(
    "anima_l2i",
    title="Latents to Image - Anima",
    tags=["latents", "image", "vae", "l2i", "anima"],
    category="latents",
    version="1.0.2",
    classification=Classification.Prototype,
)
class AnimaLatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents using the Anima VAE.

    Supports the Wan 2.1 QwenImage VAE (AutoencoderKLWan) with explicit
    latent denormalization, and FLUX VAE as fallback.
    """

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, (AutoencoderKLWan, FluxAutoEncoder)):
            raise TypeError(
                f"Expected AutoencoderKLWan or FluxAutoEncoder for Anima VAE, got {type(vae_info.model).__name__}."
            )

        estimated_working_memory = estimate_vae_working_memory_flux(
            operation="decode",
            image_tensor=latents,
            vae=vae_info.model,
        )

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            context.util.signal_progress("Running Anima VAE decode")
            if not isinstance(vae, (AutoencoderKLWan, FluxAutoEncoder)):
                raise TypeError(f"Expected AutoencoderKLWan or FluxAutoEncoder, got {type(vae).__name__}.")

            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

            TorchDevice.empty_cache()

            with torch.inference_mode():
                if isinstance(vae, FluxAutoEncoder):
                    # FLUX VAE handles scaling internally, expects 4D [B, C, H, W]
                    img = vae.decode(latents)
                else:
                    # Expects 5D latents [B, C, T, H, W]
                    if latents.ndim == 4:
                        latents = latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

                    # Denormalize from denoiser space to raw VAE space
                    # (same as diffusers WanPipeline and ComfyUI Wan21.process_out)
                    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
                    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
                    latents = latents * latents_std + latents_mean

                    decoded = vae.decode(latents, return_dict=False)[0]

                    # Output is 5D [B, C, T, H, W] — squeeze temporal dim
                    if decoded.ndim == 5:
                        decoded = decoded.squeeze(2)
                    img = decoded

            img = img.clamp(-1, 1)
            img = rearrange(img[0], "c h w -> h w c")
            img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        TorchDevice.empty_cache()

        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
