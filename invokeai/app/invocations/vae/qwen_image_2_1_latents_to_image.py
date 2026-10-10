import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, LatentsField, WithBoard, WithMetadata
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.qwen_image_2_1.vae import SPATIAL_SCALE, choose_tile_size, to_image, working_memory_bytes
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.qwen_image_vae import patch_qwen_image_vae_tiling


@invocation(
    "qwen_image_2_1_l2i",
    title="Latents to Image - Qwen-Image-2.1",
    tags=["latents", "image", "vae", "l2i", "qwen_image_2_1"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImage21LatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decodes Qwen-Image-2.1 latents into an image. Opaque results are saved as RGB, transparent ones as RGBA."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # 0 means "pick for the device": the largest of 1024/768/512 that fits. Small tiles leave visible seams.
    tile_size: int = InputField(default=0, ge=0, multiple_of=16, description=FieldDescriptions.vae_tile_size)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)
        if latents.dim() == 5:
            latents = latents.squeeze(2)
        height, width = latents.shape[-2] * SPATIAL_SCALE, latents.shape[-1] * SPATIAL_SCALE

        vae_info = context.models.load(self.vae.vae)
        device = vae_info.compute_device
        config = context.config.get()
        element_size = next(vae_info.model.parameters()).element_size()
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
            context.util.signal_progress("Running VAE")
            # Denormalize in fp32, then cast to the VAE's dtype. The device is the VAE's configured one, not the
            # residency of its weights, which partial loading may have moved to the CPU.
            mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
            z = (latents.to(device=device, dtype=torch.float32).unsqueeze(2) * std + mean).to(vae.dtype)
            TorchDevice.empty_cache()
            with torch.inference_mode(), patch_qwen_image_vae_tiling(vae, tile_size, SPATIAL_SCALE):
                decoded = vae.decode(z, return_dict=False)[0][0, :, 0]
            image = to_image(decoded)

        TorchDevice.empty_cache()
        return ImageOutput.build(context.images.save(image=image))
