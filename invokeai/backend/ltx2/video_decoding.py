"""Turning LTX-2 latents back into pixels and a waveform.

The video VAE decodes 32x spatially and 8x temporally, so a full clip's activations do not fit on
any card at generation resolutions: both halves of the decode are tiled, and the tiling state --
which lives on the shared cached VAE instance, not on the call -- is restored afterwards.

The audio path is three stages: the audio VAE decodes latents to a log-mel spectrogram, the vocoder
turns that into a 16 kHz stereo waveform, and its bandwidth extender resynthesizes it at 48 kHz.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.ltx2.packing import denormalize_audio_latents, denormalize_video_latents, unpack_audio_latents
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.cancel_hooks import cancel_before_forward

# Attributes of the diffusers LTX-2 video VAE that carry tiling state. `enable_tiling` sets six of
# them; the framewise flags it does not touch are what actually select the temporal tiling, and
# `disable_tiling` clears neither. All of them are snapshotted and put back.
_TILING_ATTRS = (
    "use_tiling",
    "use_framewise_encoding",
    "use_framewise_decoding",
    "tile_sample_min_height",
    "tile_sample_min_width",
    "tile_sample_min_num_frames",
    "tile_sample_stride_height",
    "tile_sample_stride_width",
    "tile_sample_stride_num_frames",
)

# Fraction of a tile that the next one starts after, so a caller chooses a tile size without also
# having to choose a stride that lands on the latent grid. 0.875 is the VAE's own spatial default
# (512 minimum / 448 stride). On the temporal axis the grid is 8 pixel frames, so the ratio floors:
# at the default 16-frame tile the stride becomes 8, i.e. every pixel frame is decoded about twice.
# That is the geometry the working-memory estimate is fitted to, and a larger temporal tile trades
# that redundant work for activation memory (measured: 3.85 GiB at 16 frames, 5.57 GiB at 32).
_TILE_STRIDE_RATIO = 0.875


def _raise_if_canceled(is_canceled: Callable[[], bool] | None) -> None:
    if is_canceled is not None and is_canceled():
        raise CanceledException


@contextmanager
def scoped_ltx2_tiling(vae, *, tile_size: int, temporal_tile: int) -> Iterator[None]:
    """Tile this VAE spatially and temporally for the duration of the block, then restore it.

    Sizes are in output pixels and pixel frames, and are snapped down onto the VAE's compression
    grid: a tile that is not a whole number of latents decodes a band the blend cannot line up.
    """
    spatial = max(vae.spatial_compression_ratio, tile_size - tile_size % vae.spatial_compression_ratio)
    frames = max(vae.temporal_compression_ratio, temporal_tile - temporal_tile % vae.temporal_compression_ratio)
    spatial_stride = max(
        vae.spatial_compression_ratio,
        int(spatial * _TILE_STRIDE_RATIO) // vae.spatial_compression_ratio * vae.spatial_compression_ratio,
    )
    frame_stride = max(
        vae.temporal_compression_ratio,
        int(frames * _TILE_STRIDE_RATIO) // vae.temporal_compression_ratio * vae.temporal_compression_ratio,
    )

    original = {name: getattr(vae, name) for name in _TILING_ATTRS if hasattr(vae, name)}
    try:
        vae.enable_tiling(
            tile_sample_min_height=spatial,
            tile_sample_min_width=spatial,
            tile_sample_min_num_frames=frames,
            tile_sample_stride_height=spatial_stride,
            tile_sample_stride_width=spatial_stride,
            tile_sample_stride_num_frames=frame_stride,
        )
        vae.use_framewise_encoding = True
        vae.use_framewise_decoding = True
        yield
    finally:
        for name, value in original.items():
            setattr(vae, name, value)


@torch.no_grad()
def decode_video_latents(
    vae,
    latents: torch.Tensor,
    *,
    tile_size: int,
    temporal_tile: int,
    is_canceled: Callable[[], bool] | None = None,
) -> torch.Tensor:
    """Normalized 5D video latents -> a ``[3, frames, height, width]`` clip in [0, 1] on the CPU."""
    if latents.ndim != 5 or latents.shape[0] != 1:
        raise ValueError(f"LTX-2 video decode expects one 5D clip [1, C, F, H, W]; got {tuple(latents.shape)}.")
    if latents.shape[1] != vae.config.latent_channels:
        raise ValueError(
            f"Latent channel mismatch: these latents have {latents.shape[1]} channels but the selected "
            f"VAE expects {vae.config.latent_channels}."
        )

    device = get_effective_device(vae)
    vae_dtype = next(iter(vae.parameters())).dtype
    latents = latents.to(device=device, dtype=vae_dtype)

    with scoped_ltx2_tiling(vae, tile_size=tile_size, temporal_tile=temporal_tile):
        # One decoder forward per spatial tile of every temporal tile, so a cancel lands within one
        # tile rather than at the end of a clip-long decode.
        with torch.inference_mode(), cancel_before_forward([vae.decoder], is_canceled, device):
            latents = denormalize_video_latents(
                latents, vae.latents_mean, vae.latents_std, float(vae.config.scaling_factor)
            )
            # `timestep_conditioning` is off on 2.5, so the decoder takes no timestep and the
            # released pipeline's decode-noise branch does not apply.
            decoded = vae.decode(latents, None, return_dict=False)[0]
            del latents

    # Off-device BEFORE widening: the decoder returns bf16, and converting first would allocate a
    # float32 copy of the whole clip on the card beside it (2.5 GiB at 1248x704x241) and then send
    # twice as many bytes over the bus.
    return (decoded[0].cpu().float() / 2 + 0.5).clamp_(0, 1)


@torch.no_grad()
def decode_audio_latents(
    audio_vae,
    vocoder,
    latents: torch.Tensor,
    *,
    is_canceled: Callable[[], bool] | None = None,
) -> torch.Tensor:
    """Normalized *packed* audio latents ``[1, L, 128]`` -> a ``[2, samples]`` waveform on the CPU.

    The waveform is at the vocoder's own output rate; read it from ``vocoder.config``.
    """
    if latents.ndim != 3 or latents.shape[0] != 1:
        raise ValueError(f"LTX-2 audio decode expects packed latents [1, L, 128]; got {tuple(latents.shape)}.")

    device = get_effective_device(audio_vae)
    vae_dtype = next(iter(audio_vae.parameters())).dtype
    latents = latents.to(device=device, dtype=torch.float32)

    # Each of the three stages is a single forward over the whole soundtrack, so a pre-hook on any
    # one of them polls once and never again. Polling between them is all this path can offer, and
    # it is what bounds a cancel to one stage rather than to the whole decode.
    _raise_if_canceled(is_canceled)
    with torch.inference_mode():
        latents = denormalize_audio_latents(latents, audio_vae.latents_mean, audio_vae.latents_std)
        mel = audio_vae.decode(unpack_audio_latents(latents).to(vae_dtype), return_dict=False)[0]
        del latents

    _raise_if_canceled(is_canceled)
    vocoder_device = get_effective_device(vocoder)
    vocoder_dtype = next(iter(vocoder.parameters())).dtype
    with torch.inference_mode():
        waveform = vocoder(mel.to(device=vocoder_device, dtype=vocoder_dtype))
        del mel

    if waveform.ndim != 3 or waveform.shape[0] != 1:
        raise ValueError(f"The LTX-2 vocoder returned {tuple(waveform.shape)}; expected [1, channels, samples].")
    return waveform[0].float().cpu()
