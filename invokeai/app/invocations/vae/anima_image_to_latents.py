"""Anima image-to-latents invocation.

Encodes an image to latent space with the Wan 2.1 VAE, in either layout: the original-layout file loads as
AutoencoderKLWan, the diffusers-layout Qwen-Image export as AutoencoderKLQwenImage (same weights, same
latent statistics). Any other encoder is refused, and so is a Wan VAE of another geometry (Wan 2.2's
48-channel VAE).

- Input image is converted to 5D tensor [B, C, T, H, W] with T=1
- After encoding, latents are normalized: (latents - mean) / std
  (inverse of the denormalization in anima_latents_to_image.py)
"""

import einops
import torch

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
from invokeai.backend.util.qwen_image_vae import (
    QWEN_IMAGE_VAE_MIN_TILE_SIZE,
    as_qwen_image_vae,
    patch_qwen_image_vae_tiling,
    resolve_qwen_image_vae_tile_size,
)
from invokeai.backend.util.vae_working_memory import (
    estimate_vae_working_memory_anima,
)


@invocation(
    "anima_i2l",
    title="Image to Latents - Anima",
    tags=["image", "latents", "vae", "i2l", "anima"],
    category="image",
    version="1.1.0",
    classification=Classification.Prototype,
)
class AnimaImageToLatentsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates latents from an image using the Anima VAE (the Wan 2.1 VAE, in either layout)."""

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # NOTE: tile_size = 0 is a special value meaning "use the model's default", matching the
    # Qwen-Image and SD i2l nodes. `int | None` is avoided because the workflow UI cannot send None.
    tile_size: int = InputField(
        default=0,
        multiple_of=8,
        description=f"{FieldDescriptions.vae_tile_size} Values between 1 and "
        f"{QWEN_IMAGE_VAE_MIN_TILE_SIZE} are raised to {QWEN_IMAGE_VAE_MIN_TILE_SIZE}.",
    )

    @staticmethod
    def vae_encode(
        vae_info: LoadedModel, image_tensor: torch.Tensor, tiled: bool = False, tile_size: int = 0
    ) -> torch.Tensor:
        try:
            # Also raises ValueError for a Wan VAE of another geometry (Wan 2.2's 48-channel VAE).
            as_qwen_image_vae(vae_info.model)
        except TypeError as e:
            # The encode side of the same mismatch the decode refuses: a FLUX VAE has Anima's
            # channel count and compression but a different basis, so it would hand the denoiser
            # a latent that means something else. See `anima_latents_to_image` for the measurement.
            raise TypeError(
                "Anima encodes into the 16-channel Wan 2.1 latent space, and "
                f"{type(vae_info.model).__name__} is not that encoder. Choose a VAE registered "
                "under 'anima', 'qwen-image', or a 16-channel 'wan' VAE."
            ) from e

        # Resolved before estimating, so the reservation matches the tiles the VAE will use, and
        # against a module constant rather than the cached module's current geometry.
        effective_tile_size = resolve_qwen_image_vae_tile_size(tile_size) if tiled else None
        estimated_working_memory = estimate_vae_working_memory_anima(
            operation="encode",
            image_tensor=image_tensor,
            vae=vae_info.model,
            tile_size=effective_tile_size,
        )
        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            vae = as_qwen_image_vae(vae)

            vae_dtype = next(iter(vae.parameters())).dtype
            image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

            # The cached VAE instance is shared with the decode invocation, which may have enabled
            # tiling. That used to be handled by switching tiling off here and leaving it off, which
            # also meant this node could never tile; the scope sets exactly what this encode asked
            # for and restores what was there, so "untiled for exactness" is still the default
            # without being the only option.
            with torch.inference_mode(), patch_qwen_image_vae_tiling(vae, effective_tile_size):
                # Both VAE classes expect 5D input [B, C, T, H, W]
                if image_tensor.ndim == 4:
                    image_tensor = image_tensor.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

                encoded = vae.encode(image_tensor, return_dict=False)[0]
                latents = encoded.sample().to(dtype=vae_dtype)

                # Normalize to denoiser space: (latents - mean) / std
                # This is the inverse of the denormalization in anima_latents_to_image.py
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
                latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
                latents = (latents - latents_mean) / latents_std

                # Remove temporal dimension: [B, C, 1, H, W] -> [B, C, H, W]
                if latents.ndim == 5:
                    latents = latents.squeeze(2)

        return latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        # `vae_encode` refuses a foreign encoder with a message that says which VAE to pick instead.
        vae_info = context.models.load(self.vae.vae)
        context.util.signal_progress("Running Anima VAE encode")
        latents = self.vae_encode(
            vae_info=vae_info,
            image_tensor=image_tensor,
            tiled=self.tiled or context.config.get().force_tiled_decode,
            tile_size=self.tile_size,
        )

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
