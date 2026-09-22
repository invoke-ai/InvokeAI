"""ERNIE-Image VAE decode invocation.

The denoiser emits patched latents [B, 128, H/2, W/2]. Before the VAE can decode them
we have to (a) apply the VAE's BatchNorm denormalization, and (b) reverse the 2x2
patchify back to [B, 32, H, W]. This is the same sequence the upstream pipeline
performs at the end of `__call__`.
"""

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
from invokeai.backend.ernie_image.sampling_utils import unpatchify_latents, vae_denormalize
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "ernie_image_vae_decode",
    title="Latents to Image - ERNIE-Image",
    tags=["latents", "image", "vae", "l2i", "ernie-image"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ErnieImageVaeDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode ERNIE-Image patched latents to an image."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        vae_info = context.models.load(self.vae.vae)
        with vae_info.model_on_device() as (_, vae):
            device = TorchDevice.choose_torch_device()
            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=device, dtype=vae_dtype)

            if not hasattr(vae, "bn"):
                raise RuntimeError(
                    "Connected VAE has no `.bn` (BatchNorm) layer. ERNIE-Image expects an "
                    "AutoencoderKLFlux2-style VAE that wraps a BatchNorm denormalization stage."
                )

            latents = vae_denormalize(latents, vae.bn)
            latents = unpatchify_latents(latents)
            decoded = vae.decode(latents, return_dict=False)[0]

        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = rearrange(img[0], "c h w -> h w c")
        img_np = (img * 255).byte().cpu().numpy()
        img_pil = Image.fromarray(img_np, mode="RGB")

        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
