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
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_tiling_scope import MIN_TILE_SAMPLE_SIZE, scoped_vae_tiling
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux


@invocation(
    "z_image_i2l",
    title="Image to Latents - Z-Image",
    tags=["image", "latents", "vae", "i2l", "z-image"],
    category="latents",
    version="1.2.0",
    classification=Classification.Prototype,
)
class ZImageImageToLatentsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates latents from an image using the Z-Image VAE."""

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # NOTE: tile_size = 0 is a special value. We use this rather than `int | None`, because the workflow UI does not
    # offer a way to directly set None values. It is snapped down to a tile this VAE's own tiled encode can
    # assemble -- see `scoped_vae_tiling`.
    tile_size: int = InputField(
        default=0,
        multiple_of=8,
        description=f"{FieldDescriptions.vae_tile_size} Values between 1 and "
        f"{MIN_TILE_SAMPLE_SIZE} are raised to {MIN_TILE_SAMPLE_SIZE}.",
    )

    @staticmethod
    def vae_encode(
        vae_info: LoadedModel,
        image_tensor: torch.Tensor,
        tiled: bool = False,
        tile_size: int = 0,
    ) -> torch.Tensor:
        """Encode an image to a Z-Image latent.

        `tiled` defaults to off so `z_image_denoise`'s control-latent call keeps the behaviour it
        has: a control image is already at generation resolution, not the upscaled frame tiling
        exists for.
        """
        if not isinstance(vae_info.model, AutoencoderKL):
            raise TypeError(
                f"Expected AutoencoderKL for Z-Image VAE, got {type(vae_info.model).__name__}. "
                "Ensure you are using a compatible VAE model."
            )

        effective_tile_size = tile_size if tiled else None
        estimated_working_memory = estimate_vae_working_memory_flux(
            operation="encode",
            image_tensor=image_tensor,
            vae=vae_info.model,
            tile_size=effective_tile_size,
        )

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            if not isinstance(vae, AutoencoderKL):
                raise TypeError(
                    f"Expected AutoencoderKL, got {type(vae).__name__}. VAE model type changed unexpectedly after loading."
                )

            vae_dtype = next(iter(vae.parameters())).dtype
            image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

            # The VAE belongs to the model cache and is shared with the decode node and with every
            # other node that reaches this class. Tiling is a property of this one encode, so the
            # state is scoped and restored rather than switched off by hand and left behind.
            with torch.inference_mode(), scoped_vae_tiling(vae, effective_tile_size):
                latents: torch.Tensor = vae.encode(image_tensor).latent_dist.sample().to(dtype=vae.dtype)

            # Apply scaling_factor and shift_factor from VAE config.
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

        context.util.signal_progress("Running VAE")
        latents = self.vae_encode(
            vae_info=vae_info,
            image_tensor=image_tensor,
            tiled=self.tiled or context.config.get().force_tiled_decode,
            tile_size=self.tile_size,
        )

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
