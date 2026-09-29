import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
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
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.oom import is_oom_error
from invokeai.backend.util.vae_tiling_scope import MIN_TILE_SAMPLE_SIZE, scoped_vae_tiling
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux


@invocation(
    "z_image_l2i",
    title="Latents to Image - Z-Image",
    tags=["latents", "image", "vae", "l2i", "z-image"],
    category="latents",
    version="1.2.0",
    classification=Classification.Prototype,
)
class ZImageLatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from latents using the Z-Image VAE."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # NOTE: tile_size = 0 is a special value. We use this rather than `int | None`, because the workflow UI does not
    # offer a way to directly set None values. It is snapped down to a tile this VAE's own tiled decode can
    # assemble -- see `scoped_vae_tiling`.
    tile_size: int = InputField(
        default=0,
        multiple_of=8,
        description=f"{FieldDescriptions.vae_tile_size} Values between 1 and "
        f"{MIN_TILE_SAMPLE_SIZE} are raised to {MIN_TILE_SAMPLE_SIZE}.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, AutoencoderKL):
            raise TypeError(
                f"Expected AutoencoderKL for Z-Image VAE, got {type(vae_info.model).__name__}. "
                "Ensure you are using a compatible VAE model."
            )

        use_tiling = self.tiled or context.config.get().force_tiled_decode

        # Estimate working memory needed for VAE decode
        estimated_working_memory = estimate_vae_working_memory_flux(
            operation="decode",
            image_tensor=latents,
            vae=vae_info.model,
            tile_size=self.tile_size if use_tiling else None,
        )

        # Seamless applies to every VAE this node accepts. It used to be skipped for the FLUX
        # autoencoder, which had no circular-padding path -- that class is gone, and a standalone
        # FLUX VAE now honours the axes the UI has always offered for it.
        seamless_context = SeamlessExt.static_patch_model(vae_info.model, self.vae.seamless_axes)

        with seamless_context, vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            context.util.signal_progress("Running VAE")
            if not isinstance(vae, AutoencoderKL):
                raise TypeError(
                    f"Expected AutoencoderKL, got {type(vae).__name__}. "
                    "VAE model type changed unexpectedly after loading."
                )

            vae_dtype = next(iter(vae.parameters())).dtype
            # Use the VAE's intended compute device (CUDA/MPS, or CPU if configured cpu_only). Do NOT infer it from
            # current param residency: partial loading may have temporarily offloaded all weights to RAM, which would
            # wrongly place the latents (and thus the whole decode) on the CPU (see #9373).
            latents = latents.to(device=vae_info.compute_device, dtype=vae_dtype)

            # Clear memory as VAE decode can request a lot
            TorchDevice.empty_cache()

            # `AutoencoderKL` leaves the scaling to the caller: latents / scaling_factor + shift_factor.
            scaling_factor = vae.config.scaling_factor
            shift_factor = getattr(vae.config, "shift_factor", None)

            latents = latents / scaling_factor
            if shift_factor is not None:
                latents = latents + shift_factor

            def decode() -> torch.Tensor:
                return vae.decode(latents, return_dict=False)[0]

            # The VAE belongs to the model cache and is shared with every other node that reaches
            # this class -- FLUX.1 decode and encode, Anima, PiD. Tiling is a property of this one
            # decode, not of the model, so the state is scoped and restored rather than left behind.
            with torch.inference_mode():
                try:
                    with scoped_vae_tiling(vae, self.tile_size if use_tiling else None):
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
                    # 0, not `self.tile_size`: the retry exists because the untiled pass did not
                    # fit, and the field may hold a size left over from a run where `tiled` was off.
                    # The FLUX sibling retries at 0 for the same reason.
                    with scoped_vae_tiling(vae, 0):
                        img = decode()

            img = img.clamp(-1, 1)
            img = rearrange(img[0], "c h w -> h w c")
            img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())

        TorchDevice.empty_cache()

        image_dto = context.images.save(image=img_pil)

        return ImageOutput.build(image_dto)
