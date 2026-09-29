import einops
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.util import is_flux_family_vae
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.oom import is_oom_error
from invokeai.backend.util.vae_tiling_scope import MIN_TILE_SAMPLE_SIZE, scoped_vae_tiling
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux


@invocation(
    "flux_vae_encode",
    title="Image to Latents - FLUX",
    tags=["latents", "image", "vae", "i2l", "flux"],
    category="latents",
    version="1.1.0",
)
class FluxVaeEncodeInvocation(BaseInvocation):
    """Encodes an image into latents."""

    image: ImageField = InputField(
        description="The image to encode.",
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # 0 is the "use the model's default" sentinel shared with the SD and Qwen-Image nodes.
    tile_size: int = InputField(
        default=0,
        multiple_of=8,
        description=f"{FieldDescriptions.vae_tile_size} Values between 1 and "
        f"{MIN_TILE_SAMPLE_SIZE} are raised to {MIN_TILE_SAMPLE_SIZE}.",
    )

    @staticmethod
    def vae_encode(
        vae_info: LoadedModel,
        image_tensor: torch.Tensor,
        tiled: bool = False,
        tile_size: int = 0,
    ) -> torch.Tensor:
        """Encode an image to a FLUX latent.

        `tiled` defaults to off so the callers outside this node -- the Fill conditioning in
        `flux_denoise`, the InstantX ControlNet extension, and PiD Upscale -- keep the behaviour they
        have. Those three resize to the generation size before encoding.

        Not all of them do, and the difference matters: the Kontext reference path encodes the
        user's image at whatever resolution it has, with no resize anywhere. It does not come
        through here -- it calls `vae.encode` directly in `flux/extensions/kontext_extension.py` --
        and it carries its own OOM fallback for that reason.
        """
        # TODO(ryand): Expose seed parameter at the invocation level.
        # TODO(ryand): Write a util function for generating random tensors that is consistent across devices / dtypes.
        # There's a starting point in get_noise(...), but it needs to be extracted and generalized. This function
        # should be used for VAE encode sampling.
        # Not `isinstance(..., AutoencoderKL)`: that used to mean "the FLUX autoencoder" and now
        # admits SD, SD 3.5 and CogView 4 as well, which would hand the denoiser a latent in the
        # wrong space. The normalisation is what separates them.
        if not is_flux_family_vae(vae_info.model):
            raise ValueError(
                "This node encodes into the FLUX.1 latent space (Z-Image uses the same autoencoder). "
                f"The VAE given is a {type(vae_info.model).__name__} with a different latent "
                "normalisation."
            )
        effective_tile_size = tile_size if tiled else None
        estimated_working_memory = estimate_vae_working_memory_flux(
            operation="encode", image_tensor=image_tensor, vae=vae_info.model, tile_size=effective_tile_size
        )
        # Seeded here rather than by the caller on purpose: an OOM retry calls this again, and a
        # generator held outside would already have been drawn from by the attempt that failed, so
        # the retry would produce a different latent than a directly tiled run.
        generator = torch.Generator(device=TorchDevice.choose_torch_device()).manual_seed(0)
        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            assert isinstance(vae, AutoencoderKL)
            vae_dtype = next(iter(vae.parameters())).dtype
            image_tensor = image_tensor.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)
            with scoped_vae_tiling(vae, effective_tile_size):
                latents = vae.encode(image_tensor).latent_dist.sample(generator)
            # `AutoencoderKL` leaves the scaling to the caller; `shift_factor` is None on a
            # plain SD-style config, where absent means no shift.
            shift_factor = getattr(vae.config, "shift_factor", None)
            if shift_factor is not None:
                latents = latents - shift_factor
            return latents * vae.config.scaling_factor

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        vae_info = context.models.load(self.vae.vae)

        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        context.util.signal_progress("Running VAE")
        use_tiling = self.tiled or context.config.get().force_tiled_decode
        try:
            latents = self.vae_encode(
                vae_info=vae_info, image_tensor=image_tensor, tiled=use_tiling, tile_size=self.tile_size
            )
        except RuntimeError as e:
            if use_tiling or not is_oom_error(e):
                raise
            # Same fallback the decode node has, and the same reason: an encode that does not fit
            # should become a slower encode, not a failed generation. The FLUX autoencoder's
            # mid-block attention is one head over the whole latent grid, so an untiled encode grows
            # quadratically -- at 3072px it asks for 18.8 GiB where a tiled one asks for 0.8 GiB.
            context.util.signal_progress("VAE encode ran out of memory, retrying tiled")
            context.logger.warning(
                "VAE encode ran out of memory; retrying with tiling. The tiled result is not identical to an "
                "untiled encode -- the encoder's normalisation and attention are global, so the difference is "
                "spread over the latent rather than confined to the seams."
            )
            e.__traceback__ = None
            TorchDevice.empty_cache()
            latents = self.vae_encode(vae_info=vae_info, image_tensor=image_tensor, tiled=True, tile_size=0)

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
