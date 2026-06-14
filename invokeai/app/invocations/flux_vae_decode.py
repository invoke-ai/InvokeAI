import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from einops import rearrange
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
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
from invokeai.backend.flux.modules.autoencoder import AutoEncoder
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux


@invocation(
    "flux_vae_decode",
    title="Latents to Image - FLUX",
    tags=["latents", "image", "vae", "l2i", "flux"],
    category="latents",
    version="1.0.2",
)
class FluxVaeDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )

    def _vae_decode(self, vae_info: LoadedModel, latents: torch.Tensor) -> Image.Image:
        assert isinstance(vae_info.model, (AutoEncoder, AutoencoderKL))

        # Only estimate working memory for BFL AutoEncoder (diffusers VAE handles this internally)
        if isinstance(vae_info.model, AutoEncoder):
            estimated_working_memory = estimate_vae_working_memory_flux(
                operation="decode", image_tensor=latents, vae=vae_info.model
            )
        else:
            estimated_working_memory = 0

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            assert isinstance(vae, (AutoEncoder, AutoencoderKL))
            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

            if isinstance(vae, AutoEncoder):
                # BFL AutoEncoder returns tensor directly
                img = vae.decode(latents)
            else:
                # Diffusers AutoencoderKL returns DecoderOutput with .sample attribute
                # Scale latents for diffusers VAE (FLUX uses shift_factor and scale_factor)
                latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
                img = vae.decode(latents, return_dict=False)[0]

        img = img.clamp(-1, 1)
        img = rearrange(img[0], "c h w -> h w c")  # noqa: F821
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
