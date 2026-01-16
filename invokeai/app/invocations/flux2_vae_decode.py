"""Flux2 Klein VAE Decode Invocation.

Decodes latents to images using the FLUX.2 32-channel VAE (AutoencoderKLFlux2).
"""

import torch
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
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux2_vae_decode",
    title="Latents to Image - FLUX2",
    tags=["latents", "image", "vae", "l2i", "flux2", "klein"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2VaeDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents using FLUX.2 Klein's 32-channel VAE."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )

    def _patchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert from (B, 32, H, W) to patched format (B, 128, H/2, W/2).

        This groups 2x2 spatial patches into channels for BN processing.
        """
        batch_size, num_channels, height, width = latents.shape
        # 32 channels * 4 (2x2 patch) = 128 channels, spatial dims halved
        latents = latents.reshape(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    def _unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert from patched format (B, 128, H, W) to (B, 32, H*2, W*2).

        This reverses _patchify_latents.
        """
        batch_size, num_channels, height, width = latents.shape
        # 128 channels / 4 = 32 channels, spatial dims doubled
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    def _vae_decode(self, vae_info: LoadedModel, latents: torch.Tensor) -> Image.Image:
        # FLUX.2 uses AutoencoderKLFlux2 from diffusers
        # Input latents shape: (B, 32, H, W) from denoise step
        with vae_info.model_on_device() as (_, vae):
            vae_dtype = next(iter(vae.parameters())).dtype
            device = TorchDevice.choose_torch_device()
            latents = latents.to(device=device, dtype=vae_dtype)

            # FLUX.2 VAE uses Batch Normalization in the patchified space (128 channels)
            # We need to: patchify -> BN denorm -> unpatchify
            if hasattr(vae, "bn") and vae.bn is not None:
                # Patchify to 128 channels for BN
                latents = self._patchify_latents(latents)

                # Apply BN denormalization
                bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=vae_dtype)
                bn_var = vae.bn.running_var.view(1, -1, 1, 1).to(device=device, dtype=vae_dtype)
                bn_eps = vae.config.batch_norm_eps if hasattr(vae.config, "batch_norm_eps") else 1e-4
                bn_std = torch.sqrt(bn_var + bn_eps)
                latents = latents * bn_std + bn_mean

                # Unpatchify back to 32 channels
                latents = self._unpatchify_latents(latents)

            # AutoencoderKLFlux2 expects latents in BCHW format with 32 channels
            # Decode using diffusers API
            decoded = vae.decode(latents, return_dict=False)[0]

        # Convert from [-1, 1] to [0, 1] then to [0, 255] PIL image
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = rearrange(img[0], "c h w -> h w c")
        img_pil = Image.fromarray((img * 255).byte().cpu().numpy())
        return img_pil

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)
        vae_info = context.models.load(self.vae.vae)
        context.util.signal_progress("Running VAE")
        image = self._vae_decode(vae_info=vae_info, latents=latents)

        TorchDevice.empty_cache()
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)
