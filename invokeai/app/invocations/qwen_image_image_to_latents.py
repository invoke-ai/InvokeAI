import einops
import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from PIL import Image as PILImage

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
    "qwen_image_i2l",
    title="Image to Latents - Qwen Image Edit",
    tags=["image", "latents", "vae", "i2l", "qwen_image"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImageImageToLatentsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates latents from an image using the Qwen Image Edit VAE."""

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)
    width: int | None = InputField(
        default=None,
        description="Resize the image to this width before encoding. If not set, encodes at the image's original size.",
    )
    height: int | None = InputField(
        default=None,
        description="Resize the image to this height before encoding. If not set, encodes at the image's original size.",
    )

    @staticmethod
    def vae_encode(vae_info: LoadedModel, image_tensor: torch.Tensor) -> torch.Tensor:
        with vae_info.model_on_device() as (_, vae):
            assert isinstance(vae, AutoencoderKLQwenImage)

            vae.disable_tiling()

            image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae.dtype)
            with torch.inference_mode():
                # The Qwen Image Edit VAE expects 5D input: (B, C, num_frames, H, W)
                if image_tensor.dim() == 4:
                    image_tensor = image_tensor.unsqueeze(2)

                posterior = vae.encode(image_tensor).latent_dist
                # Use mode (argmax) for deterministic encoding, matching diffusers
                latents: torch.Tensor = posterior.mode().to(dtype=vae.dtype)

            # Normalize with per-channel latents_mean / latents_std
            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(vae.config.latents_std)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = (latents - latents_mean) / latents_std

        return latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        # If target dimensions are specified, resize the image BEFORE encoding
        # (matching the diffusers pipeline which resizes in pixel space, not latent space).
        if self.width is not None and self.height is not None:
            image = image.convert("RGB").resize((self.width, self.height), resample=PILImage.LANCZOS)

        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        vae_info = context.models.load(self.vae.vae)

        latents = self.vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
