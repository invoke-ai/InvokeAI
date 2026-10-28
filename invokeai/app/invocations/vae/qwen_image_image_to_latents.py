import math

import einops
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_functional
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
from invokeai.backend.util.qwen_image_vae import (
    QWEN_IMAGE_VAE_MIN_TILE_SIZE,
    as_qwen_image_vae,
    patch_qwen_image_vae_tiling,
    resolve_qwen_image_vae_tile_size,
)
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_qwen_image

# Krea-2 edit-reference sizing, as ai-toolkit trains Krea-2 edit LoRAs (krea2 `_encode_ref_latents`):
# references above 1 MP are scaled to that area, and every side rounds to the VAE (8) x patch (2) grid.
REFERENCE_MAX_PIXELS = 1024 * 1024
REFERENCE_GRID = 16


def reference_image_size(height: int, width: int) -> tuple[int, int]:
    """The (height, width) a Krea-2 edit reference is encoded at, exactly as ai-toolkit sizes it.

    An image above 1 MP is scaled to about 1 MP, keeping its aspect ratio; a smaller one keeps its size.
    Each side then rounds to the nearest multiple of 16 (at least 16), so the result can land slightly
    above 1 MP and a side under 16 pixels grows to 16.
    """
    new_height, new_width = float(height), float(width)
    if height * width > REFERENCE_MAX_PIXELS:
        ratio = height / width
        new_height = math.sqrt(REFERENCE_MAX_PIXELS * ratio)
        new_width = new_height / ratio
    return (
        max(REFERENCE_GRID, int(round(new_height / REFERENCE_GRID)) * REFERENCE_GRID),
        max(REFERENCE_GRID, int(round(new_width / REFERENCE_GRID)) * REFERENCE_GRID),
    )


def reference_image_tensor(image: PILImage.Image) -> torch.Tensor:
    """An edit reference as a (1, 3, H, W) tensor in [-1, 1].

    RGB with any alpha dropped (not composited), resized bilinearly without antialiasing, and only when
    the size changes -- the exact preprocessing the edit LoRAs saw in training.
    """
    tensor = tv_functional.to_tensor(image.convert("RGB")).unsqueeze(0)
    height, width = tensor.shape[-2:]
    size = reference_image_size(height, width)
    if size != (height, width):
        tensor = F.interpolate(tensor, size=size, mode="bilinear")
    return tensor * 2.0 - 1.0


@invocation(
    "qwen_image_i2l",
    title="Image to Latents - Qwen Image",
    tags=["image", "latents", "vae", "i2l", "qwen_image"],
    category="image",
    version="1.2.0",
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
    reference: bool = InputField(
        default=False,
        description="Encode as a Krea-2 edit reference (for 'Denoise - Krea-2' reference latents), sized as "
        "ai-toolkit trains Krea-2 edit LoRAs: RGB with alpha dropped; an image above 1 megapixel is scaled to "
        "about 1 megapixel keeping its aspect ratio; each side is rounded to a multiple of 16; resized "
        "bilinearly. Not the Qwen-Image Edit path. Width and height must be left unset.",
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
            device=vae_info.compute_device,
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
                # Keep the per-channel normalization in fp32 even when the VAE runs at reduced precision.
                latents: torch.Tensor = posterior.mode().to(dtype=torch.float32)

            # Normalize with per-channel latents_mean / latents_std
            latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=torch.float32).view(
                1, vae.config.z_dim, 1, 1, 1
            )
            latents_std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=torch.float32).view(
                1, vae.config.z_dim, 1, 1, 1
            )
            latents = (latents - latents_mean) / latents_std

        return latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        # `width`/`height` are `int | None`, but the workflow UI cannot represent None in a number
        # input and sends 0 for "unset" — which `is not None`, so a naive check reached
        # `resize((0, 0))` and raised "height and width must be > 0". Treat any non-positive value
        # as unset, which is also how `tile_size` uses 0. Note this means a half-filled pair (e.g.
        # width=1024, height=0) encodes at the original size rather than raising.
        has_target_size = (self.width or 0) > 0 and (self.height or 0) > 0

        if self.reference:
            if has_target_size:
                raise ValueError("An edit reference is sized automatically; leave width and height unset.")
            image_tensor = reference_image_tensor(image)
        else:
            # If target dimensions are specified, resize the image BEFORE encoding
            # (matching the diffusers pipeline which resizes in pixel space, not latent space).
            if has_target_size:
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
