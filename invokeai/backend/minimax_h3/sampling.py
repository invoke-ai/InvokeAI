"""Denoise-state construction for MiniMax H3 (FL2VA / T2VA).

First-party port of the state-preparation blocks of the diffusers MiniMax-H3 integration
(``modular_pipelines/minimax_h3/before_denoise.py`` at the commit recorded in ``__init__``).
The packed-sequence geometry itself is the vendored :mod:`invokeai.backend.minimax_h3.packing`;
this module only sequences it: noise draws, conditioning-row noise augmentation, schedules and
the per-step row-timestep plan.

Reproducibility contract (mirrors upstream): a request draws every stream from ONE generator,
in a fixed order — the keyframe conditioning noise first (one draw per keyframe), then the
video noise as a 5D latent tensor that is patchified afterwards, then the audio noise directly
in row layout. A CPU-seeded generator keeps the draws identical across CUDA/ROCm/MPS.
"""

from dataclasses import dataclass

import torch
from diffusers.utils.torch_utils import randn_tensor

from invokeai.backend.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    MINIMAX_H3_SHORT_EDGE,
    MiniMaxH3PackedSequence,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
)
from invokeai.backend.minimax_h3.presets import MINIMAX_H3_MIN_VIDEO_FRAMES
from invokeai.backend.minimax_h3.scheduling_minimax_h3 import MiniMaxH3Scheduler

# The released FL2VA checkpoint's two sigma shifts (scheduler/ and audio_scheduler/ configs).
MINIMAX_H3_VIDEO_FLOW_SHIFT = 12.0
MINIMAX_H3_AUDIO_FLOW_SHIFT = 3.0

# Transformer geometry of the released checkpoints (transformer/config.json). The vendored
# transformer validates these against its own config at forward time; keeping them as module
# constants lets the state be built without the 33B model in memory.
MINIMAX_H3_PATCH_SIZE = (1, 2, 2)
MINIMAX_H3_VAE_LATENT_CHANNELS = 24
MINIMAX_H3_AUDIO_LATENT_CHANNELS = 32
MINIMAX_H3_SPATIAL_COMPRESSION = 16

# A single 17-frame-block clip (5 pixel frames, ~0.2 s) is the still-image path: the video VAE
# can decode it, but it sits far below the 5 s floor the model was trained for, so it is only
# offered for frame extraction, not as a video duration.
MINIMAX_H3_STILL_NUM_FRAMES = 5


def validate_num_frames(num_frames: int) -> None:
    """Reject frame counts the FL2VA checkpoint cannot generate.

    Valid values are ``17 * n + 5`` (the video VAE's chunk grid), from 90 frames (3.75 s — below
    the released 5 s training window, but accepted for fast test renders) up to the model's 15 s
    ceiling — except for the single-block minimum of 5 frames, which is allowed as the
    still-image (frame-extraction) path.
    """
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(
            f"num_frames must be of the form 17 * n + 5 for the MiniMax H3 video VAE "
            f"(5, 22, 39, ..., 124, ...); got {num_frames}."
        )
    if num_frames == MINIMAX_H3_STILL_NUM_FRAMES:
        return
    duration = num_frames / MINIMAX_H3_FPS
    if num_frames < MINIMAX_H3_MIN_VIDEO_FRAMES or duration > MINIMAX_H3_MAX_DURATION:
        raise ValueError(
            f"MiniMax H3 video requests must be between {MINIMAX_H3_MIN_VIDEO_FRAMES} and "
            f"{int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS)} frames on the 17n+5 grid "
            f"({MINIMAX_H3_MIN_VIDEO_FRAMES / MINIMAX_H3_FPS:g}-{MINIMAX_H3_MAX_DURATION:g} seconds at "
            f"{MINIMAX_H3_FPS} fps), or exactly {MINIMAX_H3_STILL_NUM_FRAMES} frames for a still image; "
            f"got {num_frames}."
        )


def validate_canvas(height: int, width: int) -> None:
    """Reject canvases the FL2VA checkpoint was not released for.

    H3 was released for a 768 px short edge under a soft ``768 * 1344`` area cap, with aspect
    ratios from 1:4 to 4:1 and both axes on the 32 px grid. Without this the packed sequence
    simply grows with the canvas — a 4096x4096 request builds a ~607k-row sequence against the
    native canvas's ~38k — and the run surfaces as an out-of-memory failure rather than a named
    error, while ``num_frames`` one off the grid is rejected precisely.

    The area test carries the rounding slack rather than comparing to the cap directly:
    :func:`~invokeai.backend.minimax_h3.packing.resolve_canvas_size` rounds both axes to the 32
    grid *after* applying the area cap, so its own outputs reach ~1.04x the cap (e.g. 576x1856).
    Shrinking each axis by the maximum rounding-up slack asks the real question — could this
    canvas have come from rounding a within-budget canvas up? — and accepts every canvas
    ``resolve_canvas_size`` can emit.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Canvas dimensions must be positive, got {width}x{height}.")
    if height % MINIMAX_H3_CANVAS_MULTIPLE or width % MINIMAX_H3_CANVAS_MULTIPLE:
        raise ValueError(
            f"MiniMax H3 requires both canvas axes to be multiples of "
            f"{MINIMAX_H3_CANVAS_MULTIPLE}, got {width}x{height}."
        )

    ratio = width / height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(f"MiniMax H3 supports aspect ratios from 1:4 to 4:1, got {width}x{height} ({ratio:g}).")

    slack = MINIMAX_H3_CANVAS_MULTIPLE // 2
    if (height - slack) * (width - slack) > MINIMAX_H3_MAX_PIXELS:
        raise ValueError(
            f"MiniMax H3 was released for a {MINIMAX_H3_SHORT_EDGE}px short edge with a soft area cap of "
            f"{MINIMAX_H3_MAX_PIXELS} pixels ({MINIMAX_H3_SHORT_EDGE}x{MINIMAX_H3_MAX_PIXELS // MINIMAX_H3_SHORT_EDGE}); "
            f"{width}x{height} is {width * height / MINIMAX_H3_MAX_PIXELS:.1f}x that budget. Larger canvases grow the "
            "packed sequence past what the model can attend over and will exhaust device memory."
        )


def build_schedulers(num_inference_steps: int, device: torch.device) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler]:
    """The two schedules of a request: video (shift 12.0) and audio (shift 3.0)."""
    scheduler = MiniMaxH3Scheduler(shift=MINIMAX_H3_VIDEO_FLOW_SHIFT)
    audio_scheduler = MiniMaxH3Scheduler(shift=MINIMAX_H3_AUDIO_FLOW_SHIFT)
    scheduler.set_timesteps(num_inference_steps, device=device)
    audio_scheduler.set_timesteps(num_inference_steps, device=device)
    return scheduler, audio_scheduler


@dataclass
class MiniMaxH3DenoiseState:
    """Everything the denoise loop consumes, on the execution device."""

    layout: MiniMaxH3PackedSequence
    video_rows: torch.Tensor
    """Video rows of the packed sequence, conditioning rows first. Shape (C + V, 96), float32."""
    audio_rows: torch.Tensor
    """Channel-major audio rows. Shape (A, 32), float32."""
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    timesteps: torch.Tensor
    audio_timesteps: torch.Tensor
    row_timestep_plan: list[tuple[torch.Tensor, torch.Tensor]]
    scheduler: MiniMaxH3Scheduler
    audio_scheduler: MiniMaxH3Scheduler


def build_denoise_state(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    num_inference_steps: int,
    seed: int,
    device: torch.device,
    keyframe_anchors: tuple[str, ...] = (),
    clean_condition_rows: torch.Tensor | None = None,
) -> MiniMaxH3DenoiseState:
    """Build the packed layout, the noise, and the schedules of one FL2VA / T2VA request.

    ``clean_condition_rows`` are the un-noised keyframe conditioning rows (as produced by
    :func:`invokeai.backend.minimax_h3.keyframe_conditioning.encode_keyframes`); they are
    noise-augmented to ``t = 0.999`` here so that the request's single generator makes its
    three draws in upstream's order (keyframe noise, video noise, audio noise).
    """
    if (clean_condition_rows is None) != (len(keyframe_anchors) == 0):
        raise ValueError("clean_condition_rows and keyframe_anchors must be provided together.")

    layout = build_packed_sequence(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        MINIMAX_H3_PATCH_SIZE,
        keyframe_anchors,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Draw 1 (optional): keyframe conditioning noise, one draw per keyframe, in packed order.
    condition_rows: torch.Tensor | None = None
    if clean_condition_rows is not None:
        expected_rows = layout.num_condition_video_rows
        if clean_condition_rows.shape[0] != expected_rows:
            raise ValueError(
                f"Keyframe conditioning carries {clean_condition_rows.shape[0]} rows but the layout "
                f"expects {expected_rows}. The frame-conditioning node must be run with the same "
                "width/height as the denoise node."
            )
        noise = keyframe_condition_noise(
            ((1, latent_height, latent_width),) * len(keyframe_anchors),
            MINIMAX_H3_PATCH_SIZE,
            MINIMAX_H3_VAE_LATENT_CHANNELS,
            generator=generator,
            device=device,
        )
        scale_noise_scheduler = MiniMaxH3Scheduler(shift=MINIMAX_H3_VIDEO_FLOW_SHIFT)
        condition_rows = scale_noise_scheduler.scale_noise(
            clean_condition_rows.to(device=device, dtype=torch.float32), MINIMAX_H3_KEYFRAME_NOISE_AUG, noise
        )

    # Draw 2: video noise, as a 5D latent tensor, patchified afterwards (upstream order).
    video_noise = randn_tensor(
        (1, MINIMAX_H3_VAE_LATENT_CHANNELS, num_latent_frames, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    video_rows = patchify_video_latents(video_noise, MINIMAX_H3_PATCH_SIZE).to(device)

    # Draw 3: audio noise, directly in row layout.
    audio_rows = randn_tensor(
        (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_LATENT_CHANNELS),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(device)

    if condition_rows is not None:
        video_rows = torch.cat([condition_rows, video_rows])

    scheduler, audio_scheduler = build_schedulers(num_inference_steps, device)

    row_timestep_plan = [
        tuple(
            tensor.to(device)
            for tensor in build_row_timesteps(
                layout,
                float(timestep),
                float(audio_timestep),
                max(float(timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                1.0,
            )
        )
        for timestep, audio_timestep in zip(scheduler.timesteps, audio_scheduler.timesteps, strict=True)
    ]

    return MiniMaxH3DenoiseState(
        layout=layout,
        video_rows=video_rows,
        audio_rows=audio_rows,
        position_ids=layout.position_ids.to(device),
        token_tags=layout.token_tags.to(device),
        video_indices=layout.video_indices.to(device),
        audio_indices=layout.audio_indices.to(device),
        text_indices=layout.text_indices.to(device),
        timesteps=scheduler.timesteps,
        audio_timesteps=audio_scheduler.timesteps,
        row_timestep_plan=row_timestep_plan,
        scheduler=scheduler,
        audio_scheduler=audio_scheduler,
    )
