"""Flux2 Klein VAE Encode Invocation.

Encodes images to latents using the FLUX.2 32-channel VAE (AutoencoderKLFlux2).
"""

import einops
import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux2_vae_encode",
    title="Image to Latents - FLUX2",
    tags=["latents", "image", "vae", "i2l", "flux2", "klein"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2VaeEncodeInvocation(BaseInvocation):
    """Encodes an image into latents using FLUX.2 Klein's 32-channel VAE."""

    image: ImageField = InputField(
        description="The image to encode.",
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )

    def _vae_encode(self, vae_info: LoadedModel, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode image to latents using FLUX.2 VAE.

        The VAE encodes to 32-channel latent space.
        Output latents shape: (B, 32, H/8, W/8).
        """
        with vae_info.model_on_device() as (_, vae):
            vae_dtype = next(iter(vae.parameters())).dtype
            device = TorchDevice.choose_torch_device()
            image_tensor = image_tensor.to(device=device, dtype=vae_dtype)

            # Encode using diffusers API
            # The VAE.encode() returns a DiagonalGaussianDistribution-like object
            latent_dist = vae.encode(image_tensor, return_dict=False)[0]

            # Sample from the distribution (or use mode for deterministic output)
            # Using mode() for deterministic encoding
            if hasattr(latent_dist, "mode"):
                latents = latent_dist.mode()
            elif hasattr(latent_dist, "sample"):
                # Fall back to sampling if mode is not available
                generator = torch.Generator(device=device).manual_seed(0)
                latents = latent_dist.sample(generator=generator)
            else:
                # Direct tensor output (some VAE implementations)
                latents = latent_dist

        return latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        vae_info = context.models.load(self.vae.vae)

        # Convert image to tensor (HWC -> CHW, normalize to [-1, 1])
        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        context.util.signal_progress("Running VAE Encode")
        latents = self._vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
