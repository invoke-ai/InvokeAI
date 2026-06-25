"""Ideogram 4 denoising loop.

Ports ``Ideogram4Pipeline.__call__``'s sampling loop, decoupled from model loading
and text encoding. Runs the Euler flow-matching loop with dual-branch asymmetric
CFG: the conditional transformer over the packed ``[text][image]`` sequence and the
unconditional transformer over image-only tokens with zeroed conditioning.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Transformer
from invokeai.backend.ideogram4.sampling_utils import (
    LATENT_DIM,
    build_denoise_inputs,
    pack_latents_to_grid,
)
from invokeai.backend.ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals

# Called after each completed step with (step_index, total_steps, latents).
StepCallback = Callable[[int, int, torch.Tensor], None]


@torch.no_grad()
def run_ideogram4_denoise(
    *,
    conditional_transformer: Ideogram4Transformer,
    unconditional_transformer: Ideogram4Transformer,
    llm_features: torch.Tensor,
    height: int,
    width: int,
    num_steps: int,
    mu: float,
    std: float,
    guidance_schedule: Optional[Sequence[float]] = None,
    guidance_scale: float = 7.0,
    seed: Optional[int] = None,
    device: torch.device,
    step_callback: Optional[StepCallback] = None,
) -> torch.Tensor:
    """Sample latents for a single image.

    Args:
        conditional_transformer / unconditional_transformer: the two DiT branches.
        llm_features: ``(num_text_tokens, 53248)`` text conditioning on ``device``.
        guidance_schedule: per-step guidance weights in loop-INDEX order (index 0 is
            the last/polish step), length ``num_steps``. Falls back to a constant
            ``guidance_scale`` when ``None``.

    Returns:
        Packed latents ``(1, LATENT_DIM, grid_h, grid_w)``.
    """
    num_text_tokens = int(llm_features.shape[0])
    llm_dim = int(llm_features.shape[-1])

    inputs = build_denoise_inputs(num_text_tokens, height, width, device)
    num_image_tokens = inputs["num_image_tokens"]
    grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]

    schedule = get_schedule_for_resolution((height, width), known_mean=mu, std=std)
    step_intervals = make_step_intervals(num_steps).to(device)

    if guidance_schedule is not None:
        gw_per_step = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)
        if gw_per_step.shape != (num_steps,):
            raise ValueError(
                f"guidance_schedule must have length {num_steps}, got {tuple(gw_per_step.shape)}"
            )
    else:
        gw_per_step = torch.full((num_steps,), float(guidance_scale), dtype=torch.float32, device=device)

    # Conditional branch: text features followed by zeros for the image tokens.
    llm_features_full = torch.zeros(
        1, num_text_tokens + num_image_tokens, llm_dim, dtype=llm_features.dtype, device=device
    )
    llm_features_full[0, :num_text_tokens] = llm_features.to(device)

    # Unconditional (negative) branch is image-only with zeroed conditioning.
    neg_position_ids = inputs["position_ids"][:, num_text_tokens:]
    neg_segment_ids = inputs["segment_ids"][:, num_text_tokens:]
    neg_indicator = inputs["indicator"][:, num_text_tokens:]
    neg_llm_features = torch.zeros(1, num_image_tokens, llm_dim, dtype=llm_features.dtype, device=device)

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    z = torch.randn(
        1, num_image_tokens, LATENT_DIM, dtype=torch.float32, device=device, generator=generator
    )
    text_z_padding = torch.zeros(1, num_text_tokens, LATENT_DIM, dtype=torch.float32, device=device)

    for i in range(num_steps - 1, -1, -1):
        t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
        s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
        t = torch.full((1,), t_val, dtype=torch.float32, device=device)

        pos_z = torch.cat([text_z_padding, z], dim=1)
        pos_out = conditional_transformer(
            llm_features=llm_features_full,
            x=pos_z,
            t=t,
            position_ids=inputs["position_ids"],
            segment_ids=inputs["segment_ids"],
            indicator=inputs["indicator"],
        )
        pos_v = pos_out[:, num_text_tokens:]

        neg_v = unconditional_transformer(
            llm_features=neg_llm_features,
            x=z,
            t=t,
            position_ids=neg_position_ids,
            segment_ids=neg_segment_ids,
            indicator=neg_indicator,
        )

        gw_i = gw_per_step[i]
        v = gw_i * pos_v + (1.0 - gw_i) * neg_v
        z = z + v * (s_val - t_val)

        if step_callback is not None:
            step_callback(num_steps - i, num_steps, z)

    return pack_latents_to_grid(z, grid_h, grid_w)
