"""MiniMax H3 latents-to-video invocation.

Decodes the 5D video latents with the H3 video VAE (ImageNet-normalized RGB, reverted here)
and, when audio latents are wired, decodes the stereo waveform with the H3 audio VAE and muxes
it into the MP4 as AAC. The audio is trimmed or zero-padded to exactly the video duration
before muxing: ffmpeg gets no ``-shortest``, so an over-long track would stretch the container.
"""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

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
from invokeai.app.invocations.wan_latents_to_video import _write_video_frames
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_encoding import make_mp4_writer, write_stereo_wav
from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
from invokeai.backend.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
)
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_minimax_h3


def _iter_decoded_frames(decoded: torch.Tensor) -> Iterator[np.ndarray]:
    """Yield uint8 HWC frames from a [C, T, H, W] clip already in [0, 1]."""
    for index in range(decoded.shape[1]):
        frame = decoded[:, index].clamp(0, 1).permute(1, 2, 0).cpu().float()
        yield (255.0 * frame).round().clamp(0, 255).byte().numpy()


def decode_video_latents(
    context: InvocationContext,
    vae_field: VAEField,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Decode 5D H3 video latents to a [C, T, H, W] float clip in [0, 1], on the CPU."""
    if latents.ndim != 5:
        raise ValueError(f"MiniMax H3 video latents must be 5D [B, C, T, H, W]; got {tuple(latents.shape)}.")
    if latents.shape[0] != 1:
        raise ValueError(f"MiniMax H3 latents-to-video requires batch size 1; got {latents.shape[0]}.")

    vae_info = context.models.load(vae_field.vae)
    if not isinstance(vae_info.model, AutoencoderKLMiniMaxH3):
        raise TypeError(
            f"Expected AutoencoderKLMiniMaxH3 for the MiniMax H3 video VAE, got {type(vae_info.model).__name__}."
        )
    if latents.shape[1] != vae_info.model.config.latent_channels:
        raise ValueError(
            f"Latent channel mismatch: these latents have {latents.shape[1]} channels but the "
            f"selected VAE expects {vae_info.model.config.latent_channels}."
        )

    _, _, t_lat, h_lat, w_lat = latents.shape
    spatial = vae_info.model.spatial_compression_ratio
    # 5*n+2 latent frames decode to 17*n+5 pixel frames.
    t_pixel = (t_lat - 2) // 5 * 17 + 5
    h_pixel, w_pixel = h_lat * spatial, w_lat * spatial

    estimated_working_memory = estimate_vae_working_memory_minimax_h3(
        operation="decode",
        vae=vae_info.model,
        pixel_height=h_pixel,
        pixel_width=w_pixel,
        pixel_frames=t_pixel,
    )

    with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
        assert isinstance(vae, AutoencoderKLMiniMaxH3)
        context.logger.info(
            f"Running MiniMax H3 VAE decode: {t_lat} latent frames -> {t_pixel} pixel frames at {w_pixel}x{h_pixel}"
        )
        context.util.signal_progress("Running MiniMax H3 video VAE decode")

        device = get_effective_device(vae)
        vae_dtype = next(iter(vae.parameters())).dtype
        latents = latents.to(device=device, dtype=vae_dtype)

        TorchDevice.empty_cache()

        # The H3 VAE tiles spatially by default (256px tiles / 64px overlap) and the released
        # frames are the blended-tile ones — never toggle tiling on the shared cached instance:
        # it would silently change every later encode/decode using the same cached model.
        with torch.inference_mode():
            latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
            latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
            latents = latents * latents_std + latents_mean

            if device.type == "cuda" and torch.version.hip is None:
                # Upstream's verified decode recipe: float16 autocast over the fp32-pinned weights.
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    decoded = vae.decode(latents, return_dict=False)[0]
            else:
                # ROCm/MPS/CPU: fp16 autocast is unverified there; decode in the weights' dtype.
                decoded = vae.decode(latents, return_dict=False)[0]
            del latents, latents_mean, latents_std

        # Move the clip off-device before the full-clip pixel math to keep VRAM at ~one copy.
        decoded = decoded[0].float().cpu()  # [C, T, H, W]

    TorchDevice.empty_cache()

    # The H3 video VAE emits ImageNet-normalized RGB over a [0, 1] base range; revert it.
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN).view(-1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD).view(-1, 1, 1, 1)
    return (decoded * pixel_std + pixel_mean).clamp_(0, 1)


@invocation(
    "minimax_h3_latents_to_video",
    title="Latents to Video - MiniMax H3",
    tags=["latents", "video", "audio", "vae", "l2v", "minimax"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3LatentsToVideoInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode MiniMax H3 video+audio latents and encode an MP4 with an AAC stereo track."""

    video_latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    audio_latents: LatentsField | None = InputField(
        default=None,
        description="Audio latents [2, 32, T_audio] from the denoise node. Omit for a silent video.",
        input=Input.Connection,
    )
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="Video VAE")
    audio_vae: VAEField | None = InputField(
        default=None,
        description=FieldDescriptions.minimax_h3_audio_vae,
        input=Input.Connection,
        title="Audio VAE",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> VideoOutput:
        latents = context.tensors.load(self.video_latents.latents_name)
        decoded = decode_video_latents(context, self.vae, latents)

        if context.util.is_canceled():
            raise CanceledException

        num_frames = decoded.shape[1]
        if num_frames == 0:
            raise ValueError("MiniMax H3 VAE decode produced zero frames.")
        height, width = decoded.shape[2:]
        fps = MINIMAX_H3_FPS
        duration = num_frames / float(fps)

        # Decode the soundtrack (if wired) before opening the writer: the WAV must exist and
        # be trimmed to the video duration when the muxing writer is constructed.
        wav_path: Path | None = None
        if self.audio_latents is not None:
            if self.audio_vae is None:
                raise ValueError("Audio latents are wired but no Audio VAE is connected.")
            wav_path = self._decode_audio_to_wav(context, duration)

        tmp = tempfile.NamedTemporaryFile(prefix="invokeai_minimax_h3_video_", suffix=".mp4", delete=False)
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            context.logger.info(
                f"Encoding MP4: {num_frames} frames @ {fps} fps ({duration:.2f}s) at {width}x{height}"
                + (" + AAC stereo" if wav_path is not None else "")
            )
            context.util.signal_progress(f"Encoding MP4 ({num_frames} frames @ {fps} fps)")
            writer = make_mp4_writer(tmp_path, float(fps), audio_path=wav_path)
            try:
                _write_video_frames(writer, _iter_decoded_frames(decoded), context.util.is_canceled)
            finally:
                writer.close()
            del decoded
            TorchDevice.empty_cache()
            video_dto = context.videos.save(
                source_path=tmp_path,
                width=int(width),
                height=int(height),
                duration=duration,
                fps=float(fps),
            )
            context.logger.info(f"Saved video: {video_dto.video_name}")
            return VideoOutput.build(video_dto)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if wav_path is not None:
                try:
                    wav_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _decode_audio_to_wav(self, context: InvocationContext, video_duration_s: float) -> Path:
        assert self.audio_latents is not None and self.audio_vae is not None
        audio_latents = context.tensors.load(self.audio_latents.latents_name)
        if audio_latents.ndim != 3 or audio_latents.shape[0] != 2:
            raise ValueError(
                f"MiniMax H3 audio latents must be [2, C, T_audio] (one item per stereo channel); "
                f"got {tuple(audio_latents.shape)}."
            )

        audio_vae_info = context.models.load(self.audio_vae.vae)
        if not isinstance(audio_vae_info.model, AutoencoderKLMiniMaxH3Audio):
            raise TypeError(
                f"Expected AutoencoderKLMiniMaxH3Audio for the MiniMax H3 audio VAE, "
                f"got {type(audio_vae_info.model).__name__}."
            )

        with audio_vae_info.model_on_device() as (_, audio_vae):
            assert isinstance(audio_vae, AutoencoderKLMiniMaxH3Audio)
            context.util.signal_progress("Running MiniMax H3 audio VAE decode")
            device = get_effective_device(audio_vae)
            vae_dtype = next(iter(audio_vae.parameters())).dtype
            audio_latents = audio_latents.to(device=device, dtype=vae_dtype)

            with torch.inference_mode():
                latents_mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
                latents_std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
                audio_latents = audio_latents * latents_std.to(audio_latents) + latents_mean.to(audio_latents)
                # The audio VAE is mono; the two stereo channels are two batch items.
                waveform = audio_vae.decode(audio_latents, return_dict=False)[0]

            sample_rate = int(audio_vae.config.sampling_rate)
            # [2, 1, N] -> [2, N] float on CPU.
            waveform = waveform.float().cpu().reshape(2, -1)

        # Trim (or zero-pad: the 40-latents/s grid can come up ~8 ms short when
        # num_frames % 3 == 2) to exactly the video duration — no -shortest at mux time.
        max_samples = int(round(video_duration_s * sample_rate))
        waveform = waveform[:, :max_samples]
        if waveform.shape[1] < max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.shape[1]))

        wav = tempfile.NamedTemporaryFile(prefix="invokeai_minimax_h3_audio_", suffix=".wav", delete=False)
        wav.close()
        wav_path = Path(wav.name)
        try:
            write_stereo_wav(wav_path, waveform.numpy(), sample_rate)
        except Exception:
            wav_path.unlink(missing_ok=True)
            raise
        return wav_path
