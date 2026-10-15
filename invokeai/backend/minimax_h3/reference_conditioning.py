"""MiniMax H3 Ref2VA reference normalization and VAE conditioning.

First-party port of the reference halves of ``MiniMaxH3Ref2VASetupStep``
(``modular_pipelines/minimax_h3/before_encoder.py``) and
``MiniMaxH3Ref2VAReferenceEncoderStep`` (``encoders.py``) from the merged diffusers
MiniMax-H3 integration (commit recorded in ``__init__``). Every numeric choice below is a
checkpoint contract, matched to upstream operation for operation:

- image references are resized (LANCZOS) to a 2048 px short edge — upscaling included, no
  area cap, a deliberately different rule from the 768 canvas keyframes and targets use;
- video references are resampled onto the 24 fps grid by dropping/duplicating whole frames
  (ffmpeg's ``fps``-filter arithmetic), truncated to the generated frame count, then put on
  the canvas their own aspect ratio resolves to;
- soundtracks are truncated at their native rate to the generated duration and resampled
  ONCE to the audio VAE's 32 kHz with a torchaudio-parity windowed-sinc pass;
- visual references encode with the keyframe recipe (ImageNet norm, seed-42 posterior
  sample, fp16 round-trip, mean/std normalization) — a video through the VAE's temporal
  chunking, its frame count snapped DOWN to ``17 * n + 5``;
- soundtracks take the audio posterior *mean* (never sampled), stereo as two batch items.

The rows returned here are CLEAN: the denoise state noise-augments the visual rows to
``t = 0.999`` with the request generator's leading draws, and audio rows are never noised.
"""

import math

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image, ImageOps

from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
from invokeai.backend.minimax_h3.keyframe_conditioning import vae_encode_autocast
from invokeai.backend.minimax_h3.packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE,
    MINIMAX_H3_TEXT_VIDEO_SAMPLE_FPS,
    patchify_video_latents,
    resolve_canvas_size,
)
from invokeai.backend.minimax_h3.sampling import MINIMAX_H3_PATCH_SIZE
from invokeai.backend.util.audio_resample import resample_sinc

# The audio VAE's sample rate (its config's `sampling_rate`; 800-sample hop -> 40 latents/s).
# `encode_reference_audio` asserts the loaded VAE agrees.
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000


def snap_reference_num_frames(num_frames: int) -> int:
    r"""
    Snap a reference video's frame count DOWN to a `17 * n + 5` count the VAE encodes without padding.

    This only bites when the reference is shorter than the target, whose own frame count already has that form.
    For fewer than 22 frames the result (22) exceeds the input: slicing to it is then a no-op and the VAE pads the
    tail chunk by repeating the last frame, exactly as upstream.
    """
    from invokeai.backend.minimax_h3.packing import MINIMAX_H3_FRAMES_PER_CHUNK, MINIMAX_H3_LATENTS_PER_CHUNK

    return (
        max(1, (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK) * MINIMAX_H3_FRAMES_PER_CHUNK
        + MINIMAX_H3_LATENTS_PER_CHUNK
    )


def resolve_reference_image_short_edge(width: int, height: int, detail: str, target_area: int) -> int:
    r"""
    The short edge a reference image is normalized to.

    ``"max"`` is the reference pipeline's rule: a constant 2048. ``"match"`` mirrors ComfyUI's cheaper option —
    scale the reference to roughly the generation's pixel area (never above the 2048 rule), cutting the rows the
    reference contributes to every denoising step by an order of magnitude.
    """
    if detail == "max":
        return MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE
    if detail == "match":
        scale = math.sqrt(target_area / (width * height))
        matched = max(MINIMAX_H3_CANVAS_MULTIPLE, round(min(width, height) * scale))
        return min(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE, matched)
    raise ValueError(f"A reference image detail must be 'max' or 'match', got {detail!r}.")


def normalize_reference_image(image: Image.Image, short_edge: int) -> Image.Image:
    r"""
    Normalize a reference image: EXIF-oriented RGB, resized (LANCZOS) to `short_edge`.

    Upscaling is included and there is no area cap — the reference pipeline conditions image references at high
    detail. Both axes round to the nearest multiple of 32 with a 32 floor; an image already at its target size
    passes through without a resample.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {image.size}.")
    if width > MINIMAX_H3_MAX_ASPECT_RATIO * height or height > MINIMAX_H3_MAX_ASPECT_RATIO * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    scale = short_edge / min(width, height)
    target_height = max(multiple, round(height * scale / multiple) * multiple)
    target_width = max(multiple, round(width * scale / multiple) * multiple)
    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return image


def resample_video_frame_repeats(
    num_source_frames: int, fps: float, target_fps: float = float(MINIMAX_H3_FPS)
) -> np.ndarray:
    r"""
    Per-source-frame repeat counts that put a stream on the target fps grid.

    ffmpeg's `fps`-filter arithmetic, as upstream reproduces it: every frame is held until the slot of the next
    one, and the last one until the slot the stream's end rounds to. A repeat of 0 drops the frame.
    """
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    scale = target_fps / fps
    slots = np.floor(np.arange(num_source_frames) * scale + 0.5).astype(np.int64)
    return np.diff(slots, append=math.floor(num_source_frames * scale + 0.5))


def normalize_reference_video_frames(frames: np.ndarray, fps: float, num_frames: int) -> np.ndarray:
    r"""
    Normalize a reference video's frames: onto 24 fps, truncated to `num_frames`, on the canvas of its own aspect.

    Mirrors upstream's two passes in the same order: the constant-frame-rate resample first (dropping and
    duplicating whole frames), the LANCZOS rescale second. Frames already at 24 fps and on-canvas flow through
    untouched.

    Args:
        frames (`np.ndarray` of shape `(num_source_frames, height, width, 3)`): `uint8` RGB frames.
        fps (`float`): The frame rate `frames` carries.
        num_frames (`int`): The generated frame count the reference is truncated to.

    Returns:
        `np.ndarray` of shape `(min(num_frames, resampled), canvas_height, canvas_width, 3)`, `uint8`.
    """
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[3] != 3 or frames.dtype != np.uint8:
        raise ValueError(
            f"A reference video must be `(num_frames, height, width, 3)` uint8 RGB frames, got "
            f"{tuple(frames.shape)} {frames.dtype}."
        )
    if frames.shape[0] == 0:
        raise ValueError("A reference video must have at least one frame.")

    if fps != float(MINIMAX_H3_FPS):
        frames = np.repeat(frames, resample_video_frame_repeats(frames.shape[0], fps), axis=0)

    frames = frames[:num_frames]
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1])
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
    )


def normalize_reference_video_frames_streaming(frames, fps: float, num_frames: int) -> np.ndarray:
    r"""
    Streaming variant of :func:`normalize_reference_video_frames` for frame iterators.

    Produces bit-identical output for the kept range: the fps-resample repeat of a source frame depends only on
    the frames before it, per-frame LANCZOS resizing commutes with frame drop/duplication, and truncation to
    `num_frames` makes later source frames irrelevant — so the decode can stop as soon as `num_frames` outputs
    exist, and only canvas-sized frames are ever held.

    Args:
        frames: An iterator (or sequence) of `(height, width, 3)` uint8 RGB frames.
        fps (`float`): The frame rate the source carries.
        num_frames (`int`): The generated frame count the reference is truncated to.
    """
    if fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
    scale = float(MINIMAX_H3_FPS) / fps
    out: list[np.ndarray] = []
    canvas: tuple[int, int] | None = None
    index = 0
    for frame in frames:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"A reference video frame must be `(height, width, 3)` uint8 RGB, got {frame.shape}.")
        if canvas is None:
            canvas = resolve_canvas_size(frame.shape[1], frame.shape[0])
        height, width = canvas
        if frame.shape[:2] != (height, width):
            frame = np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
        if fps != float(MINIMAX_H3_FPS):
            repeats = math.floor((index + 1) * scale + 0.5) - math.floor(index * scale + 0.5)
        else:
            repeats = 1
        for _ in range(repeats):
            if len(out) < num_frames:
                out.append(frame)
        index += 1
        if len(out) >= num_frames:
            break
    if not out:
        raise ValueError("A reference video must have at least one frame.")
    return np.stack(out)


def normalize_reference_audio(waveform: np.ndarray | torch.Tensor, sample_rate: int, num_frames: int) -> torch.Tensor:
    r"""
    Normalize a reference soundtrack: truncated at its NATIVE rate to the generated duration, then resampled once
    to the audio VAE's 32 kHz, stereo.

    The order is part of the contract — truncating after the resample lands on different sample values.
    """
    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError(
            f"A reference soundtrack must be a `(channels, num_samples)` mono or stereo waveform, got "
            f"{tuple(waveform.shape)}."
        )
    max_duration = num_frames / MINIMAX_H3_FPS
    waveform = waveform.to(torch.float32)[:, : int(max_duration * sample_rate)]
    if waveform.shape[0] != 2:
        waveform = waveform.expand(2, -1).contiguous()
    if sample_rate == MINIMAX_H3_AUDIO_SAMPLE_RATE:
        return waveform
    return resample_sinc(waveform, sample_rate, MINIMAX_H3_AUDIO_SAMPLE_RATE)


def _encode_condition_moments(
    vae: AutoencoderKLMiniMaxH3, moments: torch.Tensor
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Sample, round and normalize one visual condition's posterior moments into packed rows."""
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    posterior = DiagonalGaussianDistribution(moments)
    latents = posterior.sample(generator=torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED))
    # The fp16 rounding before normalization is a checkpoint contract (~11 bits of every
    # conditioning latent); without it the released model's conditioning is not reproduced.
    latents = latents.to(torch.float16).float().cpu()
    latents = (latents - latents_mean) / latents_std
    shape = (int(latents.shape[2]), int(latents.shape[3]), int(latents.shape[4]))
    return patchify_video_latents(latents, MINIMAX_H3_PATCH_SIZE), shape


def _normalize_pixels(pixels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 `(1, 3, T, H, W)` pixels onto the VAE's ImageNet convention, float32, on device."""
    pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
    return (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std


@torch.no_grad()
def encode_reference_image(
    vae: AutoencoderKLMiniMaxH3, image: Image.Image, device: torch.device
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Encode a normalized reference image into clean packed rows.

    Returns ``(rows, (num_latent_frames, latent_height, latent_width))`` — rows float32 CPU,
    shape ``(N, 96)``. A single frame goes through the (tiled) spatial encoder alone, exactly
    like a keyframe.
    """
    pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
    with vae_encode_autocast(device):
        moments = vae._encode_clip(_normalize_pixels(pixels, device))
    return _encode_condition_moments(vae, moments.float())


@torch.no_grad()
def encode_reference_video(
    vae: AutoencoderKLMiniMaxH3, frames: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Encode a normalized reference video into clean packed rows.

    Returns ``(rows, (num_latent_frames, latent_height, latent_width))``. The frame count is
    snapped DOWN to ``17 * n + 5`` first, then the stack goes through the VAE's temporal
    chunking. The pixels move to the device one 17-frame chunk at a time — each chunk encodes
    independently, so the result is bit-identical to ``vae._encode`` over the whole stack under
    the same :func:`vae_encode_autocast` precision, while never materializing the full float32
    pixel tensor on the device.
    """
    frames = frames[: snap_reference_num_frames(frames.shape[0])]

    clip_length = int(vae.config.clip_length)
    num_frames = frames.shape[0]
    num_chunks = -(-num_frames // clip_length)

    chunk_moments = []
    for index in range(num_chunks):
        chunk = frames[index * clip_length : (index + 1) * clip_length]
        pixels = torch.from_numpy(np.ascontiguousarray(chunk)).to(device).permute(3, 0, 1, 2)[None]
        if pixels.shape[2] < clip_length:
            # The tail chunk is padded by repeating the last frame, exactly as `_encode` pads
            # the whole stack to a multiple of `clip_length`.
            pad = pixels[:, :, -1:].repeat(1, 1, clip_length - pixels.shape[2], 1, 1)
            pixels = torch.cat([pixels, pad], dim=2)
        with vae_encode_autocast(device):
            moments = vae._encode_clip(_normalize_pixels(pixels, device))
        chunk_moments.append(moments.float().cpu())

    moments = torch.cat(chunk_moments, dim=2)
    if vae.config.token_drop > 0:
        moments = moments[:, :, : -vae.config.token_drop]
    # The posterior arithmetic runs on the device, as upstream's does (the moments tensor is
    # latent-sized — tens of MB, not GB).
    return _encode_condition_moments(vae, moments.to(device))


@torch.no_grad()
def encode_reference_audio(
    audio_vae: AutoencoderKLMiniMaxH3Audio, waveform: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Encode a normalized soundtrack into clean channel-major audio rows, ``(A, 32)`` float32 CPU.

    The posterior *mean* is taken — reference soundtracks are never sampled — and the two
    stereo channels ride as two batch items of the mono audio VAE.
    """
    if int(audio_vae.config.sampling_rate) != MINIMAX_H3_AUDIO_SAMPLE_RATE:
        raise ValueError(
            f"The audio VAE runs at {audio_vae.config.sampling_rate} Hz but reference soundtracks are "
            f"normalized to {MINIMAX_H3_AUDIO_SAMPLE_RATE} Hz."
        )
    audio_latents_mean = torch.tensor(audio_vae.config.latents_mean).view(1, 1, -1)
    audio_latents_std = torch.tensor(audio_vae.config.latents_std).view(1, 1, -1)

    posterior = audio_vae.encode(waveform.to(device)[:, None], return_dict=False)[0]
    latents = posterior.mode().float().cpu().transpose(1, 2)
    normalized = (latents - audio_latents_mean) / audio_latents_std
    return normalized.reshape(-1, normalized.shape[-1])


def sample_text_conditioning_frames(
    frames: np.ndarray, temporal_patch: int, fps: float = float(MINIMAX_H3_FPS)
) -> tuple[list[np.ndarray], list[float]]:
    r"""
    Sample the frames the conditioner sees from a normalized reference video, and label their vision blocks.

    The conditioner reads a reference at 2 fps: every `fps / 2`-th frame, deduplicated. Qwen3-VL then merges the
    sampled frames in groups of `temporal_patch` — repeating the last one when the count does not divide — and a
    merged group is labelled with the mean of its timestamps, which `"<{timestamp:.1f} seconds>"` renders with
    Python's round-half-to-even, so the first block of a 2 fps pair is `"<0.2 seconds>"` rather than
    `"<0.3 seconds>"`.

    Returns:
        `tuple[list[np.ndarray], list[float]]`: the sampled frames and one timestamp per vision block.
    """
    sample_fps = MINIMAX_H3_TEXT_VIDEO_SAMPLE_FPS
    stride = fps / sample_fps
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(
            f"A reference video is read at {sample_fps:g} fps and its sampled frames are merged in groups of "
            f"{temporal_patch}, so it must run at least {minimum} frames at {fps:g} fps "
            f"({minimum / fps:.2g} seconds), got {frames.shape[0]}."
        )

    timestamps = [index / sample_fps for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    block_timestamps = [
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    ]
    return [frames[index] for index in indices], block_timestamps
