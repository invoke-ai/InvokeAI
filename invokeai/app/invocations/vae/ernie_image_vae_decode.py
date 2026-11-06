"""ERNIE-Image VAE decode invocation.

The denoiser emits patched latents [B, 128, H/2, W/2]. Before the VAE can decode them
we have to (a) apply the VAE's BatchNorm denormalization, and (b) reverse the 2x2
patchify back to [B, 32, H, W]. This is the same sequence the upstream pipeline
performs at the end of `__call__`.
"""

import torch
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
from invokeai.app.invocations.vae.precise_io import precise_signed_decode_raster, save_decoded_image
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ernie_image.sampling_utils import unpatchify_latents, vae_denormalize
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.oom import is_oom_error
from invokeai.backend.util.vae_tiling_scope import scoped_vae_tiling
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux2, should_pretile_vae_decode
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate


@invocation(
    "ernie_image_vae_decode",
    title="Latents to Image - ERNIE-Image",
    tags=["latents", "image", "vae", "l2i", "ernie-image"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ErnieImageVaeDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode ERNIE-Image patched latents to an image."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        vae_info = context.models.load(self.vae.vae)
        config = context.config.get()
        use_tiling = config.force_tiled_decode
        # The estimator reads the decoder's latent grid (32 channels at H x W), which is what the unpatchified latents
        # have; only their shape counts, so the BatchNorm denormalization can wait for the model.
        grid = unpatchify_latents(latents)

        # Like the FLUX.2 node: the cache is told what the decode needs, a decode that would claim most of the device
        # is tiled before it runs, and one that still runs out of memory is retried tiled. The node has no tile
        # fields, so a tiled decode uses the VAE's default tile.
        def estimate(tiled: bool) -> WorkingMemoryEstimate:
            return estimate_vae_working_memory_flux2(
                operation="decode",
                image_tensor=grid,
                vae=vae_info.model,
                tile_size=0 if tiled else None,
                device=vae_info.compute_device,
            )

        estimated_working_memory = estimate(use_tiling)
        if (
            not use_tiling
            and config.auto_tiled_decode
            and should_pretile_vae_decode(vae_info.compute_device, estimated_working_memory.reservation_bytes)
        ):
            use_tiling = True
            estimated_working_memory = estimate(use_tiling)

        with vae_info.model_on_device(working_memory=estimated_working_memory) as (_, vae):
            # The VAE's own compute device, as in the FLUX.2 node (partial loading can leave its weights on the CPU).
            device = vae_info.compute_device
            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=device, dtype=vae_dtype)

            if not hasattr(vae, "bn"):
                raise RuntimeError(
                    "Connected VAE has no `.bn` (BatchNorm) layer. ERNIE-Image expects an "
                    "AutoencoderKLFlux2-style VAE that wraps a BatchNorm denormalization stage."
                )

            latents = unpatchify_latents(vae_denormalize(latents, vae.bn))

            def decode() -> torch.Tensor:
                return vae.decode(latents, return_dict=False)[0]

            try:
                with scoped_vae_tiling(vae, 0 if use_tiling else None):
                    decoded = decode()
            except RuntimeError as e:
                if use_tiling or not is_oom_error(e):
                    raise
                # The estimate was insufficient on this system: retry once tiled, which caps the peak allocation.
                context.util.signal_progress("VAE decode ran out of memory, retrying tiled")
                context.logger.warning(
                    "VAE decode ran out of memory; retrying with tiling. The tiled result is not identical to an "
                    "untiled decode -- the decoder's normalisation and attention are global, so the difference is "
                    "spread over the image rather than confined to the seams."
                )
                e.__traceback__ = None
                TorchDevice.empty_cache()
                with scoped_vae_tiling(vae, 0):
                    decoded = decode()

        raster = precise_signed_decode_raster(context, decoded)

        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = rearrange(img[0], "c h w -> h w c")
        img_np = (img * 255).byte().cpu().numpy()
        img_pil = Image.fromarray(img_np, mode="RGB")

        image_dto = save_decoded_image(context, img_pil, raster)
        return ImageOutput.build(image_dto)
