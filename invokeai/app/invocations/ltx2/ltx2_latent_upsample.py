"""x2 spatial latent upscaling between LTX-2's two generation stages."""

import torch

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, LatentsField, OutputField
from invokeai.app.invocations.model import LTX2LatentUpsamplerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ltx2.constants import LTX2_LATENT_CHANNELS, LTX2_SPATIAL_COMPRESSION, LTX2_TEMPORAL_COMPRESSION
from invokeai.backend.ltx2.packing import denormalize_video_latents, normalize_video_latents
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.util.cancel_hooks import cancel_before_forward
from invokeai.backend.util.devices import TorchDevice


@invocation_output("ltx2_latent_upsample_output")
class LTX2LatentUpsampleOutput(BaseInvocationOutput):
    """The upscaled clip, with the pixel geometry the refine pass runs at."""

    latents: LatentsField = OutputField(description="Upscaled video latents [1, 128, T_lat, 2h, 2w].")
    # Declared rather than taken from `LatentsOutput.build`, which reads `size()[3]` and `size()[2]`
    # as width and height at SD's factor of 8. That is right for a 4D image latent and wrong twice
    # over for a 5D video one, where those axes are the latent height and the frame count: a
    # 1792x1024 clip would report 256x128. `LTX2DenoiseOutput` declares its own for the same reason.
    width: int = OutputField(description="Pixel width of the upscaled latents.")
    height: int = OutputField(description="Pixel height of the upscaled latents.")
    num_frames: int = OutputField(description="Pixel-frame count of the upscaled latents.")


@invocation(
    "ltx2_latent_upsample",
    title="Upscale Latents - LTX-2",
    tags=["ltx", "ltx2", "video", "latents", "upscale"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class LTX2LatentUpsampleInvocation(BaseInvocation):
    """Doubles an LTX-2 clip's latent width and height, between the base and refine passes.

    The upscaler is a small convolutional network over the latent grid, not a denoiser: it produces
    a plausible 2x latent for the refine pass to resolve, and its output decoded on its own is
    softer than either stage. Nothing here touches the audio latents, which have no spatial extent.

    The upscaler was trained on the VAE's own latent scale, while the transformer reads normalized
    latents, so the clip is denormalized on the way in and normalized again on the way out. That is
    what the video VAE is here for -- only its ``latents_mean``/``latents_std`` and scaling factor
    are read, never its encoder or decoder, and it is never moved to the compute device.
    """

    video_latents: LatentsField = InputField(
        description="Video latents [1, 128, T_lat, H/32, W/32] from the base pass.",
        input=Input.Connection,
    )
    latent_upsampler: LTX2LatentUpsamplerField = InputField(
        description=FieldDescriptions.ltx2_latent_upsampler,
        input=Input.Connection,
        title="Latent Upsampler",
    )
    vae: VAEField = InputField(
        description="The video VAE whose latent statistics the upscaler's scale is defined against.",
        input=Input.Connection,
        title="Video VAE",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LTX2LatentUpsampleOutput:
        latents = context.tensors.load(self.video_latents.latents_name)
        if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[1] != LTX2_LATENT_CHANNELS:
            raise ValueError(
                f"LTX-2 latent upscaling expects one 5D clip [1, {LTX2_LATENT_CHANNELS}, T, h, w]; "
                f"got {tuple(latents.shape)}."
            )

        vae_info = context.models.load(self.vae.vae)
        if vae_info.config.base is not BaseModelType.LTX2:
            raise ValueError(f"Expected an LTX-2 video VAE; got a {vae_info.config.base.value} one.")

        # Read off the unlocked model: the statistics are 128-element buffers, and locking the VAE
        # onto the device would stream 1.35 GiB over the bus to read them -- then evict it, and
        # possibly part of the resident transformer, to make room for the upscaler's own
        # reservation. `denormalize_video_latents` moves the two buffers to the working device.
        vae = vae_info.model
        latents_mean = vae.latents_mean.detach().clone()
        latents_std = vae.latents_std.detach().clone()
        scaling_factor = float(vae.config.scaling_factor)

        upsampler_info = context.models.load(self.latent_upsampler.latent_upsampler)
        device = TorchDevice.choose_torch_device()
        working_memory = _estimate_upsample_working_memory(latents.shape)

        with upsampler_info.model_on_device(working_mem_bytes=working_memory) as (_, upsampler):
            context.util.signal_progress("Upscaling LTX-2 latents for the refine pass")
            # One poll per unit of work rather than one at the end: at the largest canvas this is a
            # multi-minute call, and it sits between two denoise passes that are themselves
            # cancellable, so a cancel here would otherwise be the only unresponsive stretch. The
            # resampler is included because it runs at the doubled grid and is a large share of the
            # call -- polling only the residual blocks would leave that stretch uninterruptible.
            blocks = [
                upsampler.initial_conv,
                *upsampler.res_blocks,
                upsampler.upsampler,
                *upsampler.post_upsample_res_blocks,
                upsampler.final_conv,
            ]
            with cancel_before_forward(blocks, context.util.is_canceled, device):
                raw = denormalize_video_latents(
                    latents.to(device=device, dtype=torch.float32), latents_mean, latents_std, scaling_factor
                )
                upsampled = upsampler(raw.to(dtype=next(iter(upsampler.parameters())).dtype))
                del raw
                normalized = normalize_video_latents(
                    upsampled.to(dtype=torch.float32), latents_mean, latents_std, scaling_factor
                )

        TorchDevice.empty_cache()
        _, _, latent_frames, latent_height, latent_width = normalized.shape

        return LTX2LatentUpsampleOutput(
            latents=LatentsField(
                latents_name=context.tensors.save(tensor=normalized.detach().to(device="cpu", dtype=torch.float32))
            ),
            width=latent_width * LTX2_SPATIAL_COMPRESSION,
            height=latent_height * LTX2_SPATIAL_COMPRESSION,
            num_frames=(latent_frames - 1) * LTX2_TEMPORAL_COMPRESSION + 1,
        )


# The network is a fixed stack of 3D residual blocks at `mid_channels` 1024 with no attention, so
# its transient is linear in the latent grid it is handed -- measured on a W7900 as peak reserved
# above the 0.93 GiB resident upscaler, it is 2237-2244 bytes per input element from 344k elements
# (768x512x49) to 1.76M (1248x704x121), a 5x range. Rounded up for the margin a reservation wants.
_UPSAMPLE_BYTES_PER_INPUT_ELEMENT = 2300


def _estimate_upsample_working_memory(shape: torch.Size) -> int:
    elements = 1
    for extent in shape:
        elements *= int(extent)
    return elements * _UPSAMPLE_BYTES_PER_INPUT_ELEMENT
