import torch
from diffusers import AutoencoderKLFlux2
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
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux2


def _decode(vae: object, z: torch.Tensor) -> torch.Tensor:
    """Decode a (1, 32, H/8, W/8) latent with either shape the VAE reaches this node in.

    A diffusers pipeline bundles the autoencoder in the vendored BFL layout; a standalone install
    of the same weights (the file Comfy-Org ships next to the single-file transformers) is the
    diffusers `AutoencoderKLFlux2`. Both compute the same thing — `AutoEncoder.decoder` runs the
    `post_quant_conv` that `AutoencoderKLFlux2.decode` applies before its own decoder — so this
    only picks the entry point.
    """
    if isinstance(vae, AutoEncoder):
        return vae.decoder(z)
    if isinstance(vae, AutoencoderKLFlux2):
        return vae.decode(z, return_dict=False)[0]
    raise ValueError(
        f"Expected a 32-channel Ideogram 4 / FLUX.2 VAE, got {type(vae).__name__}. Select the VAE that "
        "ships with the model."
    )


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

        vae_info = context.models.load(self.vae.vae)
        # The VAE's intended device, not the globally chosen one: a VAE record may be marked
        # `cpu_only`, and inferring the device from current parameter residency would follow a
        # partially offloaded model onto the CPU (see #9373).
        device = vae_info.compute_device
        latent_shift, latent_scale = get_latent_norm()

        # Denormalize + unpatchify to a standard (1, 32, H/8, W/8) latent. The shift/scale are
        # Ideogram's own statistics over the packed latent space, not the VAE's `bn` — that stage
        # belongs to FLUX.2's pipeline and Ideogram's does not run it. Done before the model is
        # locked so the estimate below sees the latent the decoder will actually be handed; the
        # packed one is half as wide per side and would under-reserve fourfold.
        z = unpatchify_and_denormalize(latents.float().to(device), latent_shift.to(device), latent_scale.to(device))

        # This is the 32-channel autoencoder FLUX.2 decodes with, at the same resolutions: ~4.3 GB
        # of activations at 1024px, far above what the cache reserves by default. It matters more
        # here than anywhere else, because both Ideogram transformers may still be resident.
        estimated_working_memory = estimate_vae_working_memory_flux2(
            operation="decode", image_tensor=z, vae=vae_info.model, device=device
        )

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            context.util.signal_progress("Running VAE")
            vae_dtype = next(vae.parameters()).dtype
            TorchDevice.empty_cache()
            decoded = _decode(vae, z.to(vae_dtype))

            img = decoded.float().clamp(-1.0, 1.0)
            img = rearrange(img[0], "c h w -> h w c")
            img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        TorchDevice.empty_cache()
        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
