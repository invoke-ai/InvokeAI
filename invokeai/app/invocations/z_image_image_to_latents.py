from typing import Union

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
from invokeai.backend.flux.modules.autoencoder import AutoEncoder as FluxAutoEncoder
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice

# Z-Image can use either the Diffusers AutoencoderKL or the FLUX AutoEncoder
ZImageVAE = Union[AutoencoderKL, FluxAutoEncoder]


@invocation(
    "z_image_i2l",
    title="Image to Latents - Z-Image",
    tags=["image", "latents", "vae", "i2l", "z-image"],
    category="image",
    version="1.1.0",
    classification=Classification.Prototype,
)
class ZImageImageToLatentsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates latents from an image using Z-Image VAE (supports both Diffusers and FLUX VAE)."""

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @staticmethod
    def vae_encode(vae_info: LoadedModel, image_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(vae_info.model, (AutoencoderKL, FluxAutoEncoder)):
            raise TypeError(
                f"Expected AutoencoderKL or FluxAutoEncoder for Z-Image VAE, got {type(vae_info.model).__name__}. "
                "Ensure you are using a compatible VAE model."
            )

        with vae_info.model_on_device() as (_, vae):
            if not isinstance(vae, (AutoencoderKL, FluxAutoEncoder)):
                raise TypeError(
                    f"Expected AutoencoderKL or FluxAutoEncoder, got {type(vae).__name__}. "
                    "VAE model type changed unexpectedly after loading."
                )

            vae_dtype = next(iter(vae.parameters())).dtype
            image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

            with torch.inference_mode():
                if isinstance(vae, FluxAutoEncoder):
                    # FLUX VAE handles scaling internally
                    generator = torch.Generator(device=TorchDevice.choose_torch_device()).manual_seed(0)
                    latents = vae.encode(image_tensor, sample=True, generator=generator)
                else:
                    # AutoencoderKL - needs manual scaling
                    vae.disable_tiling()
                    image_tensor_dist = vae.encode(image_tensor).latent_dist
                    latents: torch.Tensor = image_tensor_dist.sample().to(dtype=vae.dtype)

                    # Apply scaling_factor and shift_factor from VAE config
                    # Z-Image uses: latents = (latents - shift_factor) * scaling_factor
                    scaling_factor = vae.config.scaling_factor
                    shift_factor = getattr(vae.config, "shift_factor", None)

                    if shift_factor is not None:
                        latents = latents - shift_factor
                    latents = latents * scaling_factor

        return latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, (AutoencoderKL, FluxAutoEncoder)):
            raise TypeError(
                f"Expected AutoencoderKL or FluxAutoEncoder for Z-Image VAE, got {type(vae_info.model).__name__}. "
                "Ensure you are using a compatible VAE model."
            )

        context.util.signal_progress("Running VAE")
        latents = self.vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
