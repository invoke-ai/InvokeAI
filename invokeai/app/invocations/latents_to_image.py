from contextlib import nullcontext

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
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
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.stable_diffusion.vae_tiling import patch_vae_tiling_params
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_sd15_sdxl


@invocation(
    "l2i",
    title="Latents to Image - SD1.5, SDXL",
    tags=["latents", "image", "vae", "l2i"],
    category="latents",
    version="1.3.2",
)
class LatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # NOTE: tile_size = 0 is a special value. We use this rather than `int | None`, because the workflow UI does not
    # offer a way to directly set None values.
    tile_size: int = InputField(default=0, multiple_of=8, description=FieldDescriptions.vae_tile_size)
    fp32: bool = InputField(default=False, description=FieldDescriptions.fp32)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        use_tiling = self.tiled or context.config.get().force_tiled_decode

        vae_info = context.models.load(self.vae.vae)
        assert isinstance(vae_info.model, (AutoencoderKL, AutoencoderTiny))
        estimated_working_memory = estimate_vae_working_memory_sd15_sdxl(
            operation="decode",
            image_tensor=latents,
            vae=vae_info.model,
            tile_size=self.tile_size if use_tiling else None,
            fp32=self.fp32,
        )
        with (
            SeamlessExt.static_patch_model(vae_info.model, self.vae.seamless_axes),
            vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae),
        ):
            context.util.signal_progress("Running VAE decoder")
            assert isinstance(vae, (AutoencoderKL, AutoencoderTiny))
            latents = latents.to(TorchDevice.choose_torch_device())
            if self.fp32:
                vae.to(dtype=torch.float32)

                use_torch_2_0_or_xformers = hasattr(vae.decoder, "mid_block") and isinstance(
                    vae.decoder.mid_block.attentions[0].processor,
                    (
                        AttnProcessor2_0,
                        XFormersAttnProcessor,
                        LoRAXFormersAttnProcessor,
                        LoRAAttnProcessor2_0,
                    ),
                )
                # if xformers or torch_2_0 is used attention block does not need
                # to be in float32 which can save lots of memory
                if use_torch_2_0_or_xformers:
                    vae.post_quant_conv.to(latents.dtype)
                    vae.decoder.conv_in.to(latents.dtype)
                    vae.decoder.mid_block.to(latents.dtype)
                else:
                    latents = latents.float()

            else:
                vae.to(dtype=torch.float16)
                latents = latents.half()

            if use_tiling:
                vae.enable_tiling()
            else:
                vae.disable_tiling()

            tiling_context = nullcontext()
            if self.tile_size > 0:
                tiling_context = patch_vae_tiling_params(
                    vae,
                    tile_sample_min_size=self.tile_size,
                    tile_latent_min_size=self.tile_size // LATENT_SCALE_FACTOR,
                    tile_overlap_factor=0.25,
                )

            # clear memory as vae decode can request a lot
            TorchDevice.empty_cache()

            with torch.inference_mode(), tiling_context:
                # copied from diffusers pipeline
                latents = latents / vae.config.scaling_factor
                image = vae.decode(latents, return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1)  # denormalize
                # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                np_image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                image = VaeImageProcessor.numpy_to_pil(np_image)[0]

        TorchDevice.empty_cache()

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)
