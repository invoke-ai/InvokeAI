import einops
import torch
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
from invokeai.backend.krea2.vae_compat import (
    QWEN_IMAGE_VAE_MIN_TILE_SIZE,
    as_qwen_image_vae,
    patch_qwen_image_vae_tiling,
    resolve_qwen_image_vae_tile_size,
)
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_qwen_image


@invocation(
    "qwen_image_i2l",
    title="Image to Latents - Qwen Image",
    tags=["image", "latents", "vae", "i2l", "qwen_image"],
    category="image",
    version="1.1.0",
    classification=Classification.Prototype,
)
class QwenImageImageToLatentsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates latents from an image using the Qwen Image VAE."""

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # NOTE: tile_size = 0 is a special value meaning "use the model's default", matching the
    # SD/SDXL i2l node. `int | None` is avoided because the workflow UI does not handle it well.
    tile_size: int = InputField(
        default=0,
        multiple_of=8,
        description=f"{FieldDescriptions.vae_tile_size} Values between 1 and "
        f"{QWEN_IMAGE_VAE_MIN_TILE_SIZE} are raised to {QWEN_IMAGE_VAE_MIN_TILE_SIZE}.",
    )
    width: int | None = InputField(
        default=None,
        description="Resize the image to this width before encoding. If not set, encodes at the image's original size.",
    )
    height: int | None = InputField(
        default=None,
        description="Resize the image to this height before encoding. If not set, encodes at the image's original size.",
    )

    @staticmethod
    def vae_encode(
        vae_info: LoadedModel, image_tensor: torch.Tensor, tiled: bool = False, tile_size: int = 0
    ) -> torch.Tensor:
        # NOTE: vae_info.model may be an AutoencoderKLWan (a native-layout qwen_image_vae single file is
        # classified with the Anima base); it is reinterpreted as AutoencoderKLQwenImage inside the
        # model_on_device context below. The working-memory estimate only reads tensor shape + element
        # size, so it is safe to run on either class here.
        # Resolve tile_size=0 ("model default") before estimating, so the reserved working memory
        # matches the tiles the VAE will actually use. Resolved against a constant rather than the
        # module's current tile_sample_min_height, which a previous invocation may have overwritten.
        effective_tile_size = resolve_qwen_image_vae_tile_size(tile_size) if tiled else None

        estimated_working_memory = estimate_vae_working_memory_qwen_image(
            operation="encode",
            image_tensor=image_tensor,
            vae=vae_info.model,
            tile_size=effective_tile_size,
        )
        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            # Reinterpret an Anima-classified Wan VAE as AutoencoderKLQwenImage (identical weights).
            vae = as_qwen_image_vae(vae)

            image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae.dtype)

            # Tiling bounds the encode's peak memory to a single tile, which is what makes large
            # inputs (e.g. a 2560x1440 upscale round-trip) encodable while a multi-GB transformer
            # is still resident. Off by default: full-frame is faster and avoids tile blending.
            # The tiling state is scoped to this block: the VAE module belongs to the model cache
            # and is shared with later invocations (and with the Anima decode node).
            with torch.inference_mode(), patch_qwen_image_vae_tiling(vae, effective_tile_size):
                # The Qwen Image VAE expects 5D input: (B, C, num_frames, H, W)
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
        #
        # `width`/`height` are `int | None`, but the workflow UI cannot represent None in a number
        # input and sends 0 for "unset" — which `is not None`, so a naive check reached
        # `resize((0, 0))` and raised "height and width must be > 0". Treat any non-positive value
        # as unset, which is also how `tile_size` uses 0. Note this means a half-filled pair (e.g.
        # width=1024, height=0) encodes at the original size rather than raising.
        if (self.width or 0) > 0 and (self.height or 0) > 0:
            image = image.convert("RGB").resize((self.width, self.height), resample=PILImage.LANCZOS)

        # multiple_of=16 ensures the post-VAE latents (vae_scale_factor=8) have even
        # spatial dims, which the transformer's 2x2 patch packing requires.
        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"), multiple_of=16)
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        vae_info = context.models.load(self.vae.vae)

        latents = self.vae_encode(
            vae_info=vae_info,
            image_tensor=image_tensor,
            tiled=self.tiled or context.config.get().force_tiled_decode,
            tile_size=self.tile_size,
        )

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
