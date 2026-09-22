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
from invokeai.backend.ideogram4.autoencoder import AutoEncoder
from invokeai.backend.ideogram4.latent_norm import get_latent_norm
from invokeai.backend.ideogram4.sampling_utils import unpatchify_and_denormalize
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "ideogram4_l2i",
    title="Latents to Image - Ideogram 4",
    tags=["latents", "image", "vae", "l2i", "ideogram4"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4LatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decodes Ideogram 4 packed latents to an image with the FLUX.2-style VAE."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Packed latents from denoise: (1, 128, grid_h, grid_w).
        latents = context.tensors.load(self.latents.latents_name)
        device = TorchDevice.choose_torch_device()

        vae_info = context.models.load(self.vae.vae)
        latent_shift, latent_scale = get_latent_norm()

        with vae_info.model_on_device() as (_, vae):
            assert isinstance(vae, AutoEncoder), f"Expected Ideogram 4 AutoEncoder, got {type(vae).__name__}."
            context.util.signal_progress("Running VAE")
            vae_dtype = next(vae.parameters()).dtype

            # Denormalize + unpatchify to a standard (1, 32, H/8, W/8) latent.
            z = unpatchify_and_denormalize(latents.float().to(device), latent_shift.to(device), latent_scale.to(device))
            TorchDevice.empty_cache()
            decoded = vae.decoder(z.to(vae_dtype))

            img = decoded.float().clamp(-1.0, 1.0)
            img = rearrange(img[0], "c h w -> h w c")
            img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        TorchDevice.empty_cache()
        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
