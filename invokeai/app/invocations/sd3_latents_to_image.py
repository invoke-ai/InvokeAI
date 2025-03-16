from contextlib import nullcontext

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from einops import rearrange
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
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
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "sd3_l2i",
    title="Latents to Image - SD3",
    tags=["latents", "image", "vae", "l2i", "sd3"],
    category="latents",
    version="1.3.2",
)
class SD3LatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )

    def _estimate_working_memory(self, latents: torch.Tensor, vae: AutoencoderKL) -> int:
        """Estimate the working memory required by the invocation in bytes."""
        out_h = LATENT_SCALE_FACTOR * latents.shape[-2]
        out_w = LATENT_SCALE_FACTOR * latents.shape[-1]
        element_size = next(vae.parameters()).element_size()
        scaling_constant = 2200  # Determined experimentally.
        working_memory = out_h * out_w * element_size * scaling_constant
        return int(working_memory)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        vae_info = context.models.load(self.vae.vae)
        assert isinstance(vae_info.model, (AutoencoderKL))
        estimated_working_memory = self._estimate_working_memory(latents, vae_info.model)
        with (
            SeamlessExt.static_patch_model(vae_info.model, self.vae.seamless_axes),
            vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae),
        ):
            context.util.signal_progress("Running VAE")
            assert isinstance(vae, (AutoencoderKL))
            latents = latents.to(TorchDevice.choose_torch_device())

            vae.disable_tiling()

            tiling_context = nullcontext()

            # clear memory as vae decode can request a lot
            TorchDevice.empty_cache()

            with torch.inference_mode(), tiling_context:
                # copied from diffusers pipeline
                latents = latents / vae.config.scaling_factor
                img = vae.decode(latents, return_dict=False)[0]

            img = img.clamp(-1, 1)
            img = rearrange(img[0], "c h w -> h w c")  # noqa: F821
            img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        TorchDevice.empty_cache()

        image_dto = context.images.save(image=img_pil)

        return ImageOutput.build(image_dto)
