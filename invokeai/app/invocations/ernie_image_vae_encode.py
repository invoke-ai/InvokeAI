"""ERNIE-Image VAE encode invocation.

Encodes an input image into the latent space the ERNIE-Image transformer consumes:
  image -> VAE encode -> 32-channel latents -> BN normalize -> 2x2 patchify -> [B, 128, H/16, W/16].
"""

import torch
from einops import rearrange

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ernie_image.sampling_utils import patchify_latents, vae_normalize
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "ernie_image_vae_encode",
    title="Image to Latents - ERNIE-Image",
    tags=["latents", "image", "vae", "i2l", "ernie-image"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ErnieImageVaeEncodeInvocation(BaseInvocation, WithMetadata):
    """Encode an image into ERNIE-Image-ready packed latents."""

    image: ImageField = InputField(description="Input image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image_pil = context.images.get_pil(self.image.image_name).convert("RGB")
        # [-1, 1] tensor with shape [1, 3, H, W]
        img_np = torch.tensor(list(image_pil.tobytes()), dtype=torch.uint8).reshape(
            image_pil.height, image_pil.width, 3
        )
        img = rearrange(img_np, "h w c -> c h w").float() / 255.0
        img = (img * 2.0 - 1.0).unsqueeze(0)

        vae_info = context.models.load(self.vae.vae)
        with vae_info.model_on_device() as (_, vae):
            device = TorchDevice.choose_torch_device()
            vae_dtype = next(iter(vae.parameters())).dtype
            img = img.to(device=device, dtype=vae_dtype)

            if not hasattr(vae, "bn"):
                raise RuntimeError(
                    "Connected VAE has no `.bn` (BatchNorm) layer. ERNIE-Image expects an "
                    "AutoencoderKLFlux2-style VAE that wraps a BatchNorm normalization stage."
                )

            posterior = vae.encode(img).latent_dist
            latents = posterior.mode()  # deterministic
            latents = vae_normalize(latents, vae.bn)
            latents = patchify_latents(latents)

        latents = latents.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents)
