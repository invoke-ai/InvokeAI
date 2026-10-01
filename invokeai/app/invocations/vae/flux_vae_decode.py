import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from einops import rearrange
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
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
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux


@invocation(
    "flux_vae_decode",
    title="Latents to Image - FLUX",
    tags=["latents", "image", "vae", "l2i", "flux"],
    category="latents",
    version="1.1.0",
)
class FluxVaeDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
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
    # 0 is the "use the model's default" sentinel shared with the SD and Qwen-Image nodes: the
    # workflow UI cannot represent None in a number field.
    tile_size: int = InputField(
        default=0,
        multiple_of=8,
        description=f"{FieldDescriptions.vae_tile_size} Values between 1 and "
        f"{MIN_TILE_SAMPLE_SIZE} are raised to {MIN_TILE_SAMPLE_SIZE}.",
    )

    def _vae_decode(self, context: InvocationContext, vae_info: LoadedModel, latents: torch.Tensor) -> Image.Image:
        # Deliberately still a class check, and deliberately not `is_flux_family_vae`.
        #
        # This node has accepted any `AutoencoderKL` since before the FLUX.1 VAE became one, so the
        # hazard that a foreign 16-channel VAE (SD 3.5's is the same 244 tensors with the same
        # shapes, differing only in `scaling_factor`/`shift_factor`) decodes here and is normalised
        # by the wrong constant is pre-existing, not introduced by that swap. Narrowing it would also
        # refuse a VAE whose config carries no `shift_factor`, which `test_flux_vae_decode.py`
        # exercises on purpose. The encode node *is* guarded, because it used to reject every
        # `AutoencoderKL` and the swap is what widened it.
        assert isinstance(vae_info.model, AutoencoderKL)
        use_tiling = self.tiled or context.config.get().force_tiled_decode
        tile_size = self.tile_size if use_tiling else None

        estimated_working_memory = estimate_vae_working_memory_flux(
            operation="decode", image_tensor=latents, vae=vae_info.model, tile_size=tile_size
        )

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            assert isinstance(vae, AutoencoderKL)
            vae_dtype = next(iter(vae.parameters())).dtype
            # Use the VAE's intended compute device (CUDA/MPS, or CPU if configured cpu_only). Do NOT infer it from
            # current param residency: partial loading may have temporarily offloaded all weights to RAM, which would
            # wrongly place the latents (and thus the whole decode) on the CPU (see #9373).
            latents = latents.to(device=vae_info.compute_device, dtype=vae_dtype)

            # `AutoencoderKL` leaves the scaling to the caller. `shift_factor` is optional on the
            # class: the FLUX VAE sets one, but a plain SD-style config leaves it None, and
            # `tensor + None` raises TypeError. Absent means no shift.
            scaling_factor = vae.config.scaling_factor
            shift_factor = getattr(vae.config, "shift_factor", None)

            latents = latents / scaling_factor
            if shift_factor is not None:
                latents = latents + shift_factor

            def decode() -> torch.Tensor:
                return vae.decode(latents, return_dict=False)[0]

            # The tiling state is set explicitly for this decode rather than inherited from whatever
            # the last node to touch this shared, cached VAE left behind, and restored afterwards.
            try:
                with scoped_vae_tiling(vae, tile_size):
                    img = decode()
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
                # Drop the failed attempt's traceback before retrying. It pins that
                # decode's frames, and their locals hold the full-resolution
                # activations -- exception/traceback/frame is a reference cycle rooted
                # on the stack, so `empty_cache()` frees nothing while `e` is bound and
                # the retry has to fit on top of it. Measured at 1536px: 1.4 GiB held,
                # and a retry that fails with it and succeeds without. Same reasoning as
                # `ModelConfigFactory._detach_traceback`.
                e.__traceback__ = None
                TorchDevice.empty_cache()
                with scoped_vae_tiling(vae, 0):
                    img = decode()

        img = img.clamp(-1, 1)
        img = rearrange(img[0], "c h w -> h w c")  # noqa: F821
        img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())
        return img_pil

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)
        vae_info = context.models.load(self.vae.vae)
        context.util.signal_progress("Running VAE")
        image = self._vae_decode(context=context, vae_info=vae_info, latents=latents)

        TorchDevice.empty_cache()
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)
