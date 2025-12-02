from contextlib import nullcontext
from typing import Union

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
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
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.util.devices import TorchDevice

# Z-Image can use either the Diffusers AutoencoderKL or the FLUX AutoEncoder
ZImageVAE = Union[AutoencoderKL, FluxAutoEncoder]


@invocation(
    "z_image_l2i",
    title="Latents to Image - Z-Image",
    tags=["latents", "image", "vae", "l2i", "z-image"],
    category="latents",
    version="1.1.0",
    classification=Classification.Prototype,
)
class ZImageLatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents using Z-Image VAE (supports both Diffusers and FLUX VAE)."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        vae_info = context.models.load(self.vae.vae)
        assert isinstance(vae_info.model, (AutoencoderKL, FluxAutoEncoder))

        is_flux_vae = isinstance(vae_info.model, FluxAutoEncoder)

        # FLUX VAE doesn't support seamless, so only apply for AutoencoderKL
        seamless_context = (
            nullcontext()
            if is_flux_vae
            else SeamlessExt.static_patch_model(vae_info.model, self.vae.seamless_axes)
        )

        with seamless_context, vae_info.model_on_device() as (_, vae):
            context.util.signal_progress("Running VAE")
            assert isinstance(vae, (AutoencoderKL, FluxAutoEncoder))

            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

            # Disable tiling for AutoencoderKL
            if isinstance(vae, AutoencoderKL):
                vae.disable_tiling()

            # Clear memory as VAE decode can request a lot
            TorchDevice.empty_cache()

            with torch.inference_mode():
                if isinstance(vae, FluxAutoEncoder):
                    # FLUX VAE handles scaling internally
                    img = vae.decode(latents)
                else:
                    # AutoencoderKL - Apply scaling_factor and shift_factor from VAE config
                    # Z-Image uses: latents = latents / scaling_factor + shift_factor
                    scaling_factor = vae.config.scaling_factor
                    shift_factor = getattr(vae.config, "shift_factor", None)

                    latents = latents / scaling_factor
                    if shift_factor is not None:
                        latents = latents + shift_factor

                    img = vae.decode(latents, return_dict=False)[0]

            img = img.clamp(-1, 1)
            img = rearrange(img[0], "c h w -> h w c")
            img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        TorchDevice.empty_cache()

        image_dto = context.images.save(image=img_pil)

        return ImageOutput.build(image_dto)
