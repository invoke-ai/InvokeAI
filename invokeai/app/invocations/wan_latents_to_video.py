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
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from diffusers.models.autoencoders import AutoencoderKLWan

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
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_encoding import make_mp4_writer
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_wan


class _FrameWriter(Protocol):
    def append_data(self, frame: np.ndarray) -> None: ...


def _iter_decoded_frames(decoded: torch.Tensor) -> Iterator[np.ndarray]:
    for index in range(decoded.shape[1]):
        frame = decoded[:, index]
        frame = frame.clamp(-1, 1).permute(1, 2, 0).cpu().float()
        yield (127.5 * (frame + 1.0)).round().clamp(0, 255).byte().numpy()


def _write_video_frames(writer: _FrameWriter, frames: Iterable[np.ndarray], is_canceled: Callable[[], bool]) -> None:
    frames_iter = iter(frames)
    while True:
        if is_canceled():
            raise CanceledException
        try:
            frame = next(frames_iter)
        except StopIteration:
            return
        writer.append_data(np.ascontiguousarray(frame))


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
            raise TypeError(f"Expected AutoencoderKLWan for Wan VAE, got {type(vae_info.model).__name__}.")

        if latents.shape[1] != vae_info.model.config.z_dim:
            raise ValueError(
                f"Latent channel mismatch: these latents have {latents.shape[1]} channels but the "
                f"selected VAE expects {vae_info.model.config.z_dim}. A14B models need the 16-channel Wan 2.1 VAE; "
                "TI2V-5B needs the 48-channel Wan 2.2 VAE."
            )

        _, _, t_lat, h_lat, w_lat = latents.shape
        spatial_scale = getattr(vae_info.model.config, "scale_factor_spatial", None) or 8
        temporal_scale = getattr(vae_info.model.config, "scale_factor_temporal", None) or 4
        t_pixel = (t_lat - 1) * temporal_scale + 1
        h_pixel, w_pixel = h_lat * spatial_scale, w_lat * spatial_scale

        estimated_working_memory = estimate_vae_working_memory_wan(
            operation="decode",
            vae=vae_info.model,
            pixel_height=h_pixel,
            pixel_width=w_pixel,
            pixel_frames=t_pixel,
        )
        # Long/high-res clips can need a working set no card fits. When the full-frame
        # estimate exceeds the execution device's total VRAM, fall back to spatial tiling
        # and budget for the tiled working set instead. (A cpu_only VAE runs in system
        # RAM, where the working set is not the constraint.)
        use_tiling = False
        if not getattr(vae_info.config, "cpu_only", None):
            exec_device = TorchDevice.choose_torch_device()
            if exec_device.type == "cuda":
                total_vram = torch.cuda.get_device_properties(exec_device).total_memory
                if estimated_working_memory > 0.9 * total_vram:
                    use_tiling = True
                    tile_size = int(getattr(vae_info.model, "tile_sample_min_height", 256))
                    estimated_working_memory = estimate_vae_working_memory_wan(
                        operation="decode",
                        vae=vae_info.model,
                        pixel_height=h_pixel,
                        pixel_width=w_pixel,
                        pixel_frames=t_pixel,
                        tile_size=tile_size,
                    )

        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            assert isinstance(vae, AutoencoderKLWan)
            context.logger.info(
                f"Running Wan VAE decode: {t_lat} latent frames -> {t_pixel} pixel frames at {w_pixel}x{h_pixel}"
                + (" (tiled)" if use_tiling else "")
            )
            context.util.signal_progress("Running Wan VAE decode (video)")

            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(device=get_effective_device(vae), dtype=vae_dtype)

            TorchDevice.empty_cache()

            if use_tiling:
                vae.enable_tiling()
            try:
                with torch.inference_mode():
                    # Denormalise from denoiser space back to VAE space.
                    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
                    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
                    latents = latents * latents_std + latents_mean

                    # [B, C=3, T_pixel, H, W] in [-1, 1] (roughly).
                    decoded = vae.decode(latents, return_dict=False)[0]
                    del latents, latents_mean, latents_std
            finally:
                if use_tiling:
                    # The VAE instance is cached and shared; don't leak tiling into other nodes.
                    vae.disable_tiling()

            # Take batch 0 (we generate one video at a time) and move the clip off the
            # accelerator now — MP4 encoding can take a while, and holding the full
            # decoded clip in VRAM for its duration starves the next node's model load.
            decoded = decoded[0].cpu()  # [C, T, H, W]

        TorchDevice.empty_cache()

        if context.util.is_canceled():
            raise CanceledException

        num_frames = decoded.shape[1]
        if num_frames == 0:
            raise ValueError("Wan VAE decode produced zero frames.")

        height, width = decoded.shape[2:]
        duration = num_frames / float(self.fps)

        # Encode to a temporary MP4 (libx264 + yuv420p, exact frame dimensions —
        # see make_mp4_writer for why macro_block_size matters).
        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_wan_video_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            context.logger.info(
                f"Encoding MP4: {num_frames} frames @ {self.fps} fps ({duration:.2f}s) at {width}x{height} via libx264"
            )
            context.util.signal_progress(f"Encoding MP4 ({num_frames} frames @ {self.fps} fps)")
            writer = make_mp4_writer(tmp_path, self.fps)
            try:
                _write_video_frames(writer, _iter_decoded_frames(decoded), context.util.is_canceled)
            finally:
                writer.close()
            del decoded
            TorchDevice.empty_cache()
            encoded_bytes = tmp_path.stat().st_size
            context.logger.info(f"MP4 encode complete: {encoded_bytes / 1024:.1f} KB")
            video_dto = context.videos.save(
                source_path=tmp_path,
                width=width,
                height=height,
                duration=duration,
                fps=float(self.fps),
            )
            context.logger.info(f"Saved video: {video_dto.video_name}")
            return VideoOutput.build(video_dto)
        finally:
            # If save() moved the file this is a no-op; if it failed earlier, we
            # don't want a lingering temp file.
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
