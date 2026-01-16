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

    def _vae_decode(self, vae_info: LoadedModel, latents: torch.Tensor) -> Image.Image:
        # FLUX.2 uses AutoencoderKLFlux2 from diffusers
        with vae_info.model_on_device() as (_, vae):
            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

            # AutoencoderKLFlux2 expects latents in BCHW format
            # The latents are already in the correct format from the denoise step
            # Decode using diffusers API
            decoded = vae.decode(latents, return_dict=False)[0]

        # Convert from [-1, 1] to [0, 255] PIL image
        img = decoded.clamp(-1, 1)
        img = rearrange(img[0], "c h w -> h w c")
        img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())
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
