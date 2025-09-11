"""Qwen-Image: Image to Latents (VAE encode)."""

import torch
import einops

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
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
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "qwen_image_i2l",
    title="Image to Latents - Qwen-Image",
    tags=["image", "latents", "vae", "i2l", "qwen"],
    category="image",
    version="1.0.0",
)
class QwenImageImageToLatentsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Encodes an image to VAE latents for Qwen-Image."""

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        pil = context.images.get_pil(self.image.image_name).convert("RGB")

        # Convert image to tensor in [-1, 1] and BCHW
        image_tensor = image_resized_to_grid_as_tensor(pil)
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        # Load VAE and encode
        vae_info = context.models.load(self.vae.vae)
        device = TorchDevice.choose_torch_device()
        with vae_info.model_on_device() as (_, vae):
            image_tensor = image_tensor.to(device=device, dtype=vae.dtype)
            latents_dist = vae.encode(image_tensor).latent_dist  # type: ignore[attr-defined]
            latents = latents_dist.sample().to(dtype=vae.dtype)

        # Store latents un-packed; the denoise node will handle packing
        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
