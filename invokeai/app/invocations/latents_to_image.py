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

    def _estimate_working_memory(
        self, latents: torch.Tensor, use_tiling: bool, vae: AutoencoderKL | AutoencoderTiny
    ) -> int:
        """Estimate the working memory required by the invocation in bytes."""
        # It was found experimentally that the peak working memory scales linearly with the number of pixels and the
        # element size (precision). This estimate is accurate for both SD1 and SDXL.
        element_size = 4 if self.fp32 else 2
        scaling_constant = 2200  # Determined experimentally.

        if use_tiling:
            tile_size = self.tile_size
            if tile_size == 0:
                tile_size = vae.tile_sample_min_size
                assert isinstance(tile_size, int)
            out_h = tile_size
            out_w = tile_size
            working_memory = out_h * out_w * element_size * scaling_constant

            # We add 25% to the working memory estimate when tiling is enabled to account for factors like tile overlap
            # and number of tiles. We could make this more precise in the future, but this should be good enough for
            # most use cases.
            working_memory = working_memory * 1.25
        else:
            out_h = LATENT_SCALE_FACTOR * latents.shape[-2]
            out_w = LATENT_SCALE_FACTOR * latents.shape[-1]
            working_memory = out_h * out_w * element_size * scaling_constant

        if self.fp32:
            # If we are running in FP32, then we should account for the likely increase in model size (~250MB).
            working_memory += 250 * 2**20

        return int(working_memory)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        use_tiling = self.tiled or context.config.get().force_tiled_decode

        vae_info = context.models.load(self.vae.vae)
        assert isinstance(vae_info.model, (AutoencoderKL, AutoencoderTiny))

        estimated_working_memory = self._estimate_working_memory(latents, use_tiling, vae_info.model)
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
