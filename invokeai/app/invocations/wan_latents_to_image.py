"""Wan 2.2 latents-to-image invocation.

Decodes Wan latents using the Wan VAE (AutoencoderKLWan).

Latents from the denoise loop are in normalised space (zero-centred). Before
VAE decode they are denormalised using the VAE config's per-channel
``latents_mean`` / ``latents_std`` (matching Diffusers ``WanPipeline``).

The VAE expects 5D ``[B, C, T, H, W]``; downstream nodes work with 4D, so this
node re-adds ``T=1`` before decode and squeezes it back out afterwards.
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
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_wan


@invocation(
    "wan_l2i",
    title="Latents to Image - Wan 2.2",
    tags=["latents", "image", "vae", "l2i", "wan"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanLatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decodes Wan latents back to RGB."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        if latents.ndim not in (4, 5):
            raise ValueError(
                f"Wan latents-to-image expects a 4D or 5D latent tensor [B, C, (T), H, W]; got {tuple(latents.shape)}."
            )

        # This node decodes exactly one image. Multi-frame video latents would otherwise
        # run the full (expensive) multi-frame VAE decode — under a working-memory
        # estimate that assumed one frame — and then die in an opaque einops rank error
        # at the final rearrange. Checked before the VAE is even loaded.
        if latents.ndim == 5 and latents.shape[2] != 1:
            raise ValueError(
                f"These latents hold {latents.shape[2]} frames of video; this node decodes a single "
                "image. Use 'Latents to Video - Wan 2.2' (wan_l2v) for video latents."
            )

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, AutoencoderKLWan):
            raise TypeError(f"Expected AutoencoderKLWan for Wan VAE, got {type(vae_info.model).__name__}.")

        spatial_scale = getattr(vae_info.model.config, "scale_factor_spatial", None) or 8
        estimated_working_memory = estimate_vae_working_memory_wan(
            operation="decode",
            vae=vae_info.model,
            pixel_height=latents.shape[-2] * spatial_scale,
            pixel_width=latents.shape[-1] * spatial_scale,
            pixel_frames=1,
        )

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            context.util.signal_progress("Running Wan VAE decode")
            assert isinstance(vae, AutoencoderKLWan)

            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=get_effective_device(vae), dtype=vae_dtype)

            TorchDevice.empty_cache()

            with torch.inference_mode():
                # Re-add the temporal dim if upstream squeezed it out.
                if latents.ndim == 4:
                    latents = latents.unsqueeze(2)

                if latents.shape[1] != vae.config.z_dim:
                    raise ValueError(
                        f"Latent channel mismatch: these latents have {latents.shape[1]} channels but the "
                        f"selected VAE expects {vae.config.z_dim}. A14B models need the 16-channel Wan 2.1 "
                        "VAE; TI2V-5B needs the 48-channel Wan 2.2 VAE."
                    )

                # Denormalise from denoiser space back to raw VAE space.
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
                latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
                latents = latents * latents_std + latents_mean

                decoded = vae.decode(latents, return_dict=False)[0]

                if decoded.ndim == 5:
                    decoded = decoded.squeeze(2)

            img = decoded.clamp(-1, 1)
            img = rearrange(img[0], "c h w -> h w c")
            img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        TorchDevice.empty_cache()

        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
