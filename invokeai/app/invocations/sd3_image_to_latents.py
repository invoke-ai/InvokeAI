import einops
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

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
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "sd3_i2l",
    title="SD3 Image to Latents",
    tags=["image", "latents", "vae", "i2l", "sd3"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class SD3ImageToLatentsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates latents from an image."""

    image: ImageField = InputField(description="The image to encode")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @staticmethod
    def vae_encode(vae_info: LoadedModel, image_tensor: torch.Tensor) -> torch.Tensor:
        with vae_info as vae:
            assert isinstance(vae, AutoencoderKL)

            vae.disable_tiling()

            image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae.dtype)
            with torch.inference_mode():
                image_tensor_dist = vae.encode(image_tensor).latent_dist
                # TODO: Use seed to make sampling reproducible.
                latents: torch.Tensor = image_tensor_dist.sample().to(dtype=vae.dtype)

            latents = vae.config.scaling_factor * latents

        return latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        vae_info = context.models.load(self.vae.vae)
        latents = self.vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
