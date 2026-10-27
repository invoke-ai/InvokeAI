import torch
from PIL import Image as PILImage

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.qwen_image_2_1.vae import (
    SPATIAL_SCALE,
    choose_tile_size,
    to_reference,
    to_vae_input,
    working_memory_bytes,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.qwen_image_vae import patch_qwen_image_vae_tiling

# The VAE downscales 16x and the transformer reads latents in 2x2 slots.
_GRID = 32


@invocation(
    "qwen_image_2_1_i2l",
    title="Image to Latents - Qwen-Image-2.1",
    tags=["image", "latents", "vae", "i2l", "qwen_image_2_1"],
    category="image",
    version="1.1.0",
    classification=Classification.Prototype,
)
class QwenImage21ImageToLatentsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Encodes an image into Qwen-Image-2.1 latents.

    For image-to-image and the canvas, the image is encoded opaque at its own size; its alpha is not read. As a
    reference for an edit, it is resized to the area the text encoder reads it at and keeps its alpha.
    """

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    tile_size: int = InputField(default=0, ge=0, multiple_of=16, description=FieldDescriptions.vae_tile_size)
    reference: bool = InputField(
        default=False,
        description="Encode as a reference image for the denoise node's reference latents: resized to ~1 megapixel, "
        "as the text encoder reads it, with its alpha kept.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)
        if self.reference:
            image = to_reference(image)
            width, height = image.size
        else:
            # Crop-free resize onto the 32px grid the transformer needs, as the canvas sizes its own images.
            width, height = image.width // _GRID * _GRID, image.height // _GRID * _GRID
            if width == 0 or height == 0:
                raise ValueError(
                    f"Qwen-Image-2.1 encodes images of at least {_GRID}x{_GRID}; this one is {image.size}."
                )
            if (width, height) != image.size:
                image = image.resize((width, height), resample=PILImage.LANCZOS)

        vae_info = context.models.load(self.vae.vae)
        device = vae_info.compute_device
        config = context.config.get()
        element_size = next(vae_info.model.parameters()).element_size()
        # Encodes are tiled up front like decodes: priced like the decoder, an untiled 2048x2048 canvas box would
        # ask for ~32 GiB.
        tile_size = choose_tile_size(
            height,
            width,
            element_size,
            device,
            tiled=self.tiled or config.force_tiled_decode,
            tile_size=self.tile_size,
            auto_tile=config.auto_tiled_decode,
        )
        working_memory = working_memory_bytes(height, width, tile_size, element_size)

        with vae_info.model_on_device(working_mem_bytes=working_memory) as (_, vae):
            pixels = to_vae_input(image, keep_alpha=self.reference).to(device=device, dtype=vae.dtype).unsqueeze(2)
            with torch.inference_mode(), patch_qwen_image_vae_tiling(vae, tile_size, SPATIAL_SCALE):
                z = vae.encode(pixels).latent_dist.mode()[:, :, 0]
            # A reference is normalized in the VAE's dtype, as the pipeline normalizes it, so the transformer reads
            # the same values; the canvas's latents only seed a sampler, and keep fp32.
            z = z if self.reference else z.float()
            # float32 first, then the latents' dtype: straight from Python floats to bf16 can round differently.
            mean = torch.tensor(vae.config.latents_mean).to(z.device, z.dtype).view(1, -1, 1, 1)
            std = torch.tensor(vae.config.latents_std).to(z.device, z.dtype).view(1, -1, 1, 1)
            latents = ((z - mean) / std).float().cpu()

        TorchDevice.empty_cache()
        name = context.tensors.save(tensor=latents)
        return LatentsOutput(latents=LatentsField(latents_name=name, seed=None), width=width, height=height)
