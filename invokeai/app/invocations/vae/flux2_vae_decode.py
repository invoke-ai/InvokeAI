"""Flux2 Klein VAE Decode Invocation.

Decodes latents to images using the FLUX.2 32-channel VAE (AutoencoderKLFlux2).
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
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.oom import is_oom_error
from invokeai.backend.util.vae_tiling_scope import MIN_TILE_SAMPLE_SIZE, scoped_vae_tiling
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux2, should_pretile_vae_decode
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate


@invocation(
    "flux2_vae_decode",
    title="Latents to Image - FLUX2",
    tags=["latents", "image", "vae", "l2i", "flux2", "klein"],
    category="latents",
    version="1.1.0",
    classification=Classification.Prototype,
)
class Flux2VaeDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents using FLUX.2 Klein's 32-channel VAE."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # 0 is the "use the model's default" sentinel shared with the other VAE nodes: the workflow UI
    # cannot represent None in a number field.
    tile_size: int = InputField(
        default=0,
        multiple_of=8,
        description=f"{FieldDescriptions.vae_tile_size} Values between 1 and "
        f"{MIN_TILE_SAMPLE_SIZE} are raised to {MIN_TILE_SAMPLE_SIZE}.",
    )

    def _vae_decode(self, context: InvocationContext, vae_info: LoadedModel, latents: torch.Tensor) -> Image.Image:
        """Decode latents to image using FLUX.2 VAE.

        Input latents should already be in the correct space after BN denormalization
        was applied in the denoiser. The VAE expects (B, 32, H, W) format.
        """
        config = context.config.get()
        use_tiling = self.tiled or config.force_tiled_decode

        # Decoding at FLUX.2 resolutions costs multiple GB of activations (~4.3GB at 1024x1024),
        # far above the default working memory the cache would otherwise reserve. Tell it up front so
        # it offloads enough of the (possibly still resident) transformer to leave room.
        def estimate(tiled: bool) -> WorkingMemoryEstimate:
            return estimate_vae_working_memory_flux2(
                operation="decode",
                image_tensor=latents,
                vae=vae_info.model,
                tile_size=self.tile_size if tiled else None,
                device=vae_info.compute_device,
            )

        estimated_working_memory = estimate(use_tiling)
        # Where the untiled decode would claim most of what the device keeps resident, tile before it runs: on Windows
        # an allocation that does not fit is paged rather than refused, so the retry below never fires.
        if (
            not use_tiling
            and config.auto_tiled_decode
            and should_pretile_vae_decode(vae_info.compute_device, estimated_working_memory.reservation_bytes)
        ):
            use_tiling = True
            estimated_working_memory = estimate(use_tiling)

        with vae_info.model_on_device(working_memory=estimated_working_memory) as (_, vae):
            vae_dtype = next(iter(vae.parameters())).dtype
            # Use the VAE's intended compute device (CUDA/MPS, or CPU if configured cpu_only). Do NOT infer it from
            # current param residency: partial loading may have temporarily offloaded all weights to RAM, which would
            # wrongly place the latents (and thus the whole decode) on the CPU (see #9373).
            latents = latents.to(device=vae_info.compute_device, dtype=vae_dtype)

            def decode() -> torch.Tensor:
                return vae.decode(latents, return_dict=False)[0]

            # The VAE is the cache's shared instance, so its tiling state is set for this decode only and restored.
            try:
                with scoped_vae_tiling(vae, self.tile_size if use_tiling else None):
                    decoded = decode()
            except RuntimeError as e:
                if use_tiling or not is_oom_error(e):
                    raise
                # The working-memory estimate was insufficient on this system. Retry once with
                # tiling, which caps the peak allocation regardless of resolution.
                context.util.signal_progress("VAE decode ran out of memory, retrying tiled")
                context.logger.warning(
                    "VAE decode ran out of memory; retrying with tiling. The tiled result is not identical to an "
                    "untiled decode -- the decoder's normalisation and attention are global, so the difference is "
                    "spread over the image rather than confined to the seams."
                )
                # The traceback pins the failed attempt's activations; see the FLUX.1 decode node.
                e.__traceback__ = None
                TorchDevice.empty_cache()
                # 0, not `self.tile_size`: the field may hold a size left over from a run where `tiled` was off.
                with scoped_vae_tiling(vae, 0):
                    decoded = decode()

        # Convert from [-1, 1] to [0, 1] then to [0, 255] PIL image
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = rearrange(img[0], "c h w -> h w c")
        img_np = (img * 255).byte().cpu().numpy()
        # Explicitly create RGB image (not grayscale)
        img_pil = Image.fromarray(img_np, mode="RGB")
        return img_pil

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        # Log latent statistics for debugging black image issues
        context.logger.debug(
            f"FLUX.2 VAE decode input: shape={latents.shape}, "
            f"min={latents.min().item():.4f}, max={latents.max().item():.4f}, "
            f"mean={latents.mean().item():.4f}"
        )

        # Warn if input latents are all zeros or very small (would cause black images)
        if latents.abs().max() < 1e-6:
            context.logger.warning(
                "FLUX.2 VAE decode received near-zero latents! This will cause black images. "
                "The latent cache may be corrupted - try clearing the cache."
            )

        vae_info = context.models.load(self.vae.vae)
        context.util.signal_progress("Running VAE")
        image = self._vae_decode(context=context, vae_info=vae_info, latents=latents)

        TorchDevice.empty_cache()
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)
