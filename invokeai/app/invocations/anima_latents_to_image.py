"""Anima latents-to-image invocation.

Decodes Anima latents using the QwenImage VAE (AutoencoderKLWan) or
compatible FLUX VAE as fallback.

Latents from the denoiser are in normalized space (zero-centered). Before
VAE decode, they must be denormalized using the Wan 2.1 per-channel
mean/std: latents = latents * std + mean (matching diffusers WanPipeline).

The VAE expects 5D latents [B, C, T, H, W] — for single images, T=1.
"""

import torch
from diffusers.models.autoencoders import AutoencoderKLWan
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
from invokeai.backend.flux.modules.autoencoder import AutoEncoder as FluxAutoEncoder
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import (
    estimate_vae_working_memory_anima,
    estimate_vae_working_memory_flux,
)

# Tile geometry for tiled Wan VAE decode. 512px tiles with a 384px stride (128px blended
# overlap) cap peak decode working memory at ~1.7GB regardless of image size, while images
# <=512px still decode in a single pass.
ANIMA_VAE_TILE_SIZE = 512
ANIMA_VAE_TILE_STRIDE = 384


@invocation(
    "anima_l2i",
    title="Latents to Image - Anima",
    tags=["latents", "image", "vae", "l2i", "anima"],
    category="latents",
    version="1.0.2",
    classification=Classification.Prototype,
)
class AnimaLatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents using the Anima VAE.

    Supports the Wan 2.1 QwenImage VAE (AutoencoderKLWan) with explicit
    latent denormalization, and FLUX VAE as fallback.
    """

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @staticmethod
    def _use_tiled_decode(device: torch.device, full_decode_working_memory: int) -> bool:
        """Decide whether to decode in tiles.

        A full 1024x1024 Wan VAE decode reserves ~6GB of working memory. On small-VRAM
        GPUs this evicts the (~4GB) Anima transformer from the model cache and thrashes
        the allocator near the VRAM ceiling (decode times of 7s+ observed on 8GB, vs
        ~1s tiled with the transformer left resident). Tile when the full-decode working
        memory would consume most of the device, otherwise a single-pass decode is
        faster (~0.65s vs ~1.05s at 1024x1024) and exact.
        """
        if device.type != "cuda":
            return False
        total_vram = torch.cuda.get_device_properties(device).total_memory
        return full_decode_working_memory > 0.7 * total_vram

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, (AutoencoderKLWan, FluxAutoEncoder)):
            raise TypeError(
                f"Expected AutoencoderKLWan or FluxAutoEncoder for Anima VAE, got {type(vae_info.model).__name__}."
            )

        use_tiling = False
        if isinstance(vae_info.model, AutoencoderKLWan):
            full_decode_working_memory = estimate_vae_working_memory_anima(
                operation="decode",
                image_tensor=latents,
                vae=vae_info.model,
                tile_size=None,
            )
            use_tiling = self._use_tiled_decode(TorchDevice.choose_torch_device(), full_decode_working_memory)
            estimated_working_memory = estimate_vae_working_memory_anima(
                operation="decode",
                image_tensor=latents,
                vae=vae_info.model,
                tile_size=ANIMA_VAE_TILE_SIZE if use_tiling else None,
            )
        else:
            estimated_working_memory = estimate_vae_working_memory_flux(
                operation="decode",
                image_tensor=latents,
                vae=vae_info.model,
            )

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            context.util.signal_progress("Running Anima VAE decode")
            if not isinstance(vae, (AutoencoderKLWan, FluxAutoEncoder)):
                raise TypeError(f"Expected AutoencoderKLWan or FluxAutoEncoder, got {type(vae).__name__}.")

            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

            TorchDevice.empty_cache()

            with torch.inference_mode():
                if isinstance(vae, FluxAutoEncoder):
                    # FLUX VAE handles scaling internally, expects 4D [B, C, H, W]
                    img = vae.decode(latents)
                else:
                    # The cached VAE instance is shared across invocations, so always set
                    # the tiling state explicitly rather than leaving it as-is.
                    if use_tiling:
                        vae.enable_tiling(
                            tile_sample_min_height=ANIMA_VAE_TILE_SIZE,
                            tile_sample_min_width=ANIMA_VAE_TILE_SIZE,
                            tile_sample_stride_height=ANIMA_VAE_TILE_STRIDE,
                            tile_sample_stride_width=ANIMA_VAE_TILE_STRIDE,
                        )
                    else:
                        vae.disable_tiling()

                    # Expects 5D latents [B, C, T, H, W]
                    if latents.ndim == 4:
                        latents = latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

                    # Denormalize from denoiser space to raw VAE space
                    # (same as diffusers WanPipeline and ComfyUI Wan21.process_out)
                    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
                    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
                    latents = latents * latents_std + latents_mean

                    try:
                        decoded = vae.decode(latents, return_dict=False)[0]
                    except torch.cuda.OutOfMemoryError:
                        if use_tiling:
                            raise
                        # The working-memory estimate was insufficient on this system;
                        # retry once with tiling, which caps the peak allocation.
                        TorchDevice.empty_cache()
                        vae.enable_tiling(
                            tile_sample_min_height=ANIMA_VAE_TILE_SIZE,
                            tile_sample_min_width=ANIMA_VAE_TILE_SIZE,
                            tile_sample_stride_height=ANIMA_VAE_TILE_STRIDE,
                            tile_sample_stride_width=ANIMA_VAE_TILE_STRIDE,
                        )
                        decoded = vae.decode(latents, return_dict=False)[0]

                    # Output is 5D [B, C, T, H, W] — squeeze temporal dim
                    if decoded.ndim == 5:
                        decoded = decoded.squeeze(2)
                    img = decoded

            img = img.clamp(-1, 1)
            img = rearrange(img[0], "c h w -> h w c")
            img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        TorchDevice.empty_cache()

        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
