"""Wan 2.2 latents-to-video invocation.

Decodes multi-frame Wan latents with the Wan VAE and encodes the result to an
MP4 file via :mod:`imageio` (backed by the bundled FFmpeg binary from
``imageio-ffmpeg``). The video is then persisted through ``context.videos.save``,
which moves the temp file into ``outputs/videos/`` and records the DTO.

Latent shape on input is 5D ``[B, C, T_lat, H_lat, W_lat]`` (typically B=1).
The VAE expands the temporal dim by 4× during decode minus the initial offset:
``T_pixel = (T_lat - 1) * 4 + 1`` (e.g. T_lat=21 → 81 pixel frames).
"""

import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from einops import rearrange

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
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "wan_l2v",
    title="Latents to Video - Wan 2.2",
    tags=["latents", "video", "vae", "l2v", "wan"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanLatentsToVideoInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode 5D Wan latents to RGB frames and encode an MP4."""

    latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection)
    fps: int = InputField(
        default=16,
        ge=1,
        le=120,
        description="Frames-per-second for the encoded MP4. Wan 2.2 was trained at 16 FPS.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> VideoOutput:
        latents = context.tensors.load(self.latents.latents_name)
        if latents.ndim == 4:
            # Promote 4D (single-frame) to 5D so this node can also serve as a
            # one-frame "video" encode if someone wires it that way.
            latents = latents.unsqueeze(2)
        if latents.ndim != 5:
            raise ValueError(
                f"Wan latents-to-video expects a 5D latent tensor [B, C, T, H, W]; got {tuple(latents.shape)}."
            )

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, AutoencoderKLWan):
            raise TypeError(
                f"Expected AutoencoderKLWan for Wan VAE, got {type(vae_info.model).__name__}."
            )

        with vae_info.model_on_device() as (_, vae):
            assert isinstance(vae, AutoencoderKLWan)
            context.util.signal_progress("Running Wan VAE decode (video)")

            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=TorchDevice.choose_torch_device(), dtype=vae_dtype)

            TorchDevice.empty_cache()

            with torch.inference_mode():
                # Denormalise from denoiser space back to VAE space.
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
                latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
                latents = latents * latents_std + latents_mean

                # [B, C=3, T_pixel, H, W] in [-1, 1] (roughly).
                decoded = vae.decode(latents, return_dict=False)[0]

            decoded = decoded.clamp(-1, 1)
            # Take batch 0 (we generate one video at a time).
            decoded = decoded[0]  # [C, T, H, W]

        TorchDevice.empty_cache()

        # Convert to a list of numpy uint8 frames [H, W, C].
        decoded = rearrange(decoded, "c t h w -> t h w c")
        # [-1, 1] -> [0, 255]
        frames = (127.5 * (decoded.cpu().float() + 1.0)).round().clamp(0, 255).byte().numpy()
        frames_list = [np.ascontiguousarray(frames[i]) for i in range(frames.shape[0])]

        if not frames_list:
            raise ValueError("Wan VAE decode produced zero frames.")

        height, width = frames_list[0].shape[:2]
        num_frames = len(frames_list)
        duration = num_frames / float(self.fps)

        # Encode to a temporary MP4 via imageio[ffmpeg], then hand the file off
        # to the video service. ``ffmpeg_params=['-pix_fmt', 'yuv420p']`` keeps
        # the output compatible with broadly-supported browser playback.
        tmp = tempfile.NamedTemporaryFile(
            prefix="invokeai_wan_video_", suffix=".mp4", delete=False
        )
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            context.util.signal_progress(f"Encoding MP4 ({num_frames} frames @ {self.fps} fps)")
            iio.imwrite(
                tmp_path,
                frames_list,
                plugin="pyav",
                codec="libx264",
                fps=self.fps,
                pixel_format="yuv420p",
            )
            video_dto = context.videos.save(
                source_path=tmp_path,
                width=width,
                height=height,
                duration=duration,
                fps=float(self.fps),
            )
            return VideoOutput.build(video_dto)
        finally:
            # If save() moved the file this is a no-op; if it failed earlier, we
            # don't want a lingering temp file.
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
