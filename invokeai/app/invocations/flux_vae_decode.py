import torch
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
from invokeai.backend.flux.modules.autoencoder import AutoEncoder
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux_vae_decode",
    title="FLUX Latents to Image",
    tags=["latents", "image", "vae", "l2i", "flux"],
    category="latents",
    version="1.0.1",
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

    def _estimate_working_memory(self, latents: torch.Tensor, vae: AutoEncoder) -> int:
        """Estimate the working memory required by the invocation in bytes."""
        out_h = LATENT_SCALE_FACTOR * latents.shape[-2]
        out_w = LATENT_SCALE_FACTOR * latents.shape[-1]
        element_size = next(vae.parameters()).element_size()
        scaling_constant = 2200  # Determined experimentally.
        working_memory = out_h * out_w * element_size * scaling_constant
        return int(working_memory)

    def _vae_decode(self, vae_info: LoadedModel, latents: torch.Tensor) -> Image.Image:
        estimated_working_memory = self._estimate_working_memory(latents, vae_info.model)
        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            assert isinstance(vae, AutoEncoder)
            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)
            img = vae.decode(latents)

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
