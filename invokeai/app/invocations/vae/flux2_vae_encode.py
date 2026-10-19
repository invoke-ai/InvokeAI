"""Flux2 Klein VAE Encode Invocation.

Encodes images to latents using the FLUX.2 32-channel VAE (AutoencoderKLFlux2).
"""

import einops
import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.oom import is_oom_error
from invokeai.backend.util.vae_tiling_scope import MIN_TILE_SAMPLE_SIZE, scoped_vae_tiling
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux2


@invocation(
    "flux2_vae_encode",
    title="Image to Latents - FLUX2",
    tags=["latents", "image", "vae", "i2l", "flux2", "klein"],
    category="latents",
    version="1.1.0",
    classification=Classification.Prototype,
)
class Flux2VaeEncodeInvocation(BaseInvocation):
    """Encodes an image into latents using FLUX.2 Klein's 32-channel VAE."""

    image: ImageField = InputField(
        description="The image to encode.",
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    tiled: bool = InputField(default=False, description=FieldDescriptions.tiled)
    # 0 is the "use the model's default" sentinel shared with the other VAE nodes.
    tile_size: int = InputField(
        default=0,
        multiple_of=8,
        description=f"{FieldDescriptions.vae_tile_size} Values between 1 and "
        f"{MIN_TILE_SAMPLE_SIZE} are raised to {MIN_TILE_SAMPLE_SIZE}.",
    )

    def _vae_encode(
        self, vae_info: LoadedModel, image_tensor: torch.Tensor, tiled: bool = False, tile_size: int = 0
    ) -> torch.Tensor:
        """Encode image to latents using FLUX.2 VAE.

        The VAE encodes to 32-channel latent space.
        Output latents shape: (B, 32, H/8, W/8).
        """
        # See the decode node: FLUX.2 VAE activations are multi-GB, so the cache needs the estimate to
        # free room rather than discovering the shortfall as an OOM.
        # Use the VAE's intended compute device for both the probe and the encode, matching the
        # decode node. `choose_torch_device()` disagrees for a cpu_only VAE, which would both ask the
        # dispatch probe about the wrong device and push the tensors onto a device the weights are
        # not on (see #9373, and the same note in the decode node).
        device = vae_info.compute_device
        effective_tile_size = tile_size if tiled else None
        estimated_working_memory = estimate_vae_working_memory_flux2(
            operation="encode",
            image_tensor=image_tensor,
            vae=vae_info.model,
            tile_size=effective_tile_size,
            device=device,
        )

        with vae_info.model_on_device(working_memory=estimated_working_memory) as (_, vae):
            vae_dtype = next(iter(vae.parameters())).dtype
            image_tensor = image_tensor.to(device=device, dtype=vae_dtype)

            # The VAE is the cache's shared instance, so its tiling state is set for this encode only.
            with scoped_vae_tiling(vae, effective_tile_size):
                latent_dist = vae.encode(image_tensor, return_dict=False)[0]

            # Sample from the distribution (or use mode for deterministic output)
            # Using mode() for deterministic encoding
            if hasattr(latent_dist, "mode"):
                latents = latent_dist.mode()
            elif hasattr(latent_dist, "sample"):
                # Fall back to sampling if mode is not available
                generator = torch.Generator(device=device).manual_seed(0)
                latents = latent_dist.sample(generator=generator)
            else:
                # Direct tensor output (some VAE implementations)
                latents = latent_dist

        return latents

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)

        vae_info = context.models.load(self.vae.vae)

        # Convert image to tensor (HWC -> CHW, normalize to [-1, 1])
        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        context.util.signal_progress("Running VAE Encode")
        use_tiling = self.tiled or context.config.get().force_tiled_decode
        try:
            latents = self._vae_encode(
                vae_info=vae_info, image_tensor=image_tensor, tiled=use_tiling, tile_size=self.tile_size
            )
        except RuntimeError as e:
            if use_tiling or not is_oom_error(e):
                raise
            # An encode that does not fit should become a slower encode, not a failed generation.
            context.util.signal_progress("VAE encode ran out of memory, retrying tiled")
            context.logger.warning(
                "VAE encode ran out of memory; retrying with tiling. The tiled result is not identical to an "
                "untiled encode -- the encoder's normalisation and attention are global, so the difference is "
                "spread over the latent rather than confined to the seams."
            )
            # The traceback pins the failed attempt's activations; see the FLUX.1 decode node.
            e.__traceback__ = None
            TorchDevice.empty_cache()
            # 0, not `self.tile_size`: the field may hold a size left over from a run where `tiled` was off.
            latents = self._vae_encode(vae_info=vae_info, image_tensor=image_tensor, tiled=True, tile_size=0)

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
