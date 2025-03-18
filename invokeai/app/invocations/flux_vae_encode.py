import einops
import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.modules.autoencoder import AutoEncoder
from invokeai.backend.model_manager import LoadedModel
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux_vae_encode",
    title="Image to Latents - FLUX",
    tags=["latents", "image", "vae", "i2l", "flux"],
    category="latents",
    version="1.0.1",
)
class FluxVaeEncodeInvocation(BaseInvocation):
    """Encodes an image into latents."""

    image: ImageField = InputField(
        description="The image to encode.",
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )

    @staticmethod
    def vae_encode(vae_info: LoadedModel, image_tensor: torch.Tensor) -> torch.Tensor:
        # TODO(ryand): Expose seed parameter at the invocation level.
        # TODO(ryand): Write a util function for generating random tensors that is consistent across devices / dtypes.
        # There's a starting point in get_noise(...), but it needs to be extracted and generalized. This function
        # should be used for VAE encode sampling.
        generator = torch.Generator(device=TorchDevice.choose_torch_device()).manual_seed(0)
        with vae_info as vae:
            assert isinstance(vae, AutoEncoder)
            vae_dtype = next(iter(vae.parameters())).dtype
            image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)
            latents = vae.encode(image_tensor, sample=True, generator=generator)
            return latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        vae_info = context.models.load(self.vae.vae)

        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        context.util.signal_progress("Running VAE")
        latents = self.vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
