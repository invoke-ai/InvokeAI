import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image

from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import FieldDescriptions, Input, InputField, LatentsField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.invocation_api import BaseInvocation, Classification, ImageOutput, invocation


@invocation(
    "bria_latents_to_image",
    title="Latents to Image - Bria",
    tags=["image", "bria"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class BriaLatentsToImageInvocation(BaseInvocation):
    """
    Decode Bria latents to an image.
    """

    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    height: int = InputField(
        title="Height",
        description="The height of the output image",
    )
    width: int = InputField(
        title="Width",
        description="The width of the output image",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)
        latents = _unpack_latents(latents, self.height, self.width)

        with context.models.load(self.vae.vae) as vae:
            assert isinstance(vae, AutoencoderKL)
            latents = latents / vae.config.scaling_factor
            latents = latents.to(device=vae.device, dtype=vae.dtype)

            decoded_output = vae.decode(latents)
            image = decoded_output.sample

        # Convert to numpy with proper gradient handling
        image = ((image.clamp(-1, 1) + 1) / 2 * 255).cpu().detach().permute(0, 2, 3, 1).numpy().astype("uint8")[0]
        img = Image.fromarray(image)
        image_dto = context.images.save(image=img)
        return ImageOutput.build(image_dto)


def _unpack_latents(latents, height, width, vae_scale_factor=16):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

    return latents