from contextlib import nullcontext

import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
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
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_qwen_image


@invocation(
    "qwen_image_l2i",
    title="Latents to Image - Qwen Image",
    tags=["latents", "image", "vae", "l2i", "qwen_image"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class QwenImageLatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents using the Qwen Image VAE."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        vae_info = context.models.load(self.vae.vae)
        assert isinstance(vae_info.model, AutoencoderKLQwenImage)
        estimated_working_memory = estimate_vae_working_memory_qwen_image(
            operation="decode",
            image_tensor=latents,
            vae=vae_info.model,
        )
        with (
            SeamlessExt.static_patch_model(vae_info.model, self.vae.seamless_axes),
            vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae),
        ):
            context.util.signal_progress("Running VAE")
            assert isinstance(vae, AutoencoderKLQwenImage)
            # Use the VAE's actual device (may be CPU if the model is configured cpu_only).
            latents = latents.to(device=get_effective_device(vae), dtype=vae.dtype)

            # Honor the global force_tiled_decode setting, like the SD/SDXL l2i node. Tiling bounds the
            # VAE's per-tile memory, which is the scalable way to decode very large outputs that would
            # exceed VRAM even after offloading the transformer/text encoder. For normal sizes, leave
            # it off (faster, no tile blending) — the reserved working memory offloads other models so
            # the full-frame decode fits.
            if context.config.get().force_tiled_decode:
                vae.enable_tiling()
            else:
                vae.disable_tiling()

            tiling_context = nullcontext()

            TorchDevice.empty_cache()

            with torch.inference_mode(), tiling_context:
                # The Qwen Image VAE uses per-channel latents_mean / latents_std
                # instead of a single scaling_factor.
                # Latents are 5D: (B, C, num_frames, H, W) — the unpack from the
                # denoise step already produces this shape.
                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean

                img = vae.decode(latents, return_dict=False)[0]
                # Drop the temporal frame dimension: (B, C, 1, H, W) -> (B, C, H, W)
                img = img[:, :, 0]

            img = img.clamp(-1, 1)
            img = rearrange(img[0], "c h w -> h w c")
            img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        TorchDevice.empty_cache()

        image_dto = context.images.save(image=img_pil)

        return ImageOutput.build(image_dto)
