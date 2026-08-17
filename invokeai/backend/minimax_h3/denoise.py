"""The MiniMax H3 denoising loop (FL2VA / T2VA).

First-party port of ``modular_pipelines/minimax_h3/denoise.py`` (commit recorded in
``__init__``): one transformer forward per step over the packed sequence — every row at its own
noise level — followed by one scheduler step per modality on the *generated* rows only. The
conditioning rows are never written, so the anchors survive the loop by construction. The
checkpoint is guidance-distilled: no unconditional pass, no CFG.
"""

from typing import Callable

import torch

from invokeai.backend.minimax_h3.sampling import MiniMaxH3DenoiseState
from invokeai.backend.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel


def denoise(
    transformer: MiniMaxH3Transformer3DModel,
    state: MiniMaxH3DenoiseState,
    prompt_embeds: torch.Tensor,
    step_callback: Callable[[int, int, torch.Tensor], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full denoising schedule over the packed sequence.

    Args:
        transformer: The FL2VA transformer.
        state: The prepared denoise state (rows, layout, schedules).
        prompt_embeds: The layer-50 Qwen3-VL hidden states, shape ``(1, num_text_tokens, text_dim)``.
        step_callback: Called after every step with ``(step_index, total_steps, pred_x0_video_rows)``
            — the step's *predicted-clean* (x-hat-0) estimate of the GENERATED video rows
            (conditioning rows excluded), float32, for previews. Unlike the noisy running
            latents, the prediction is decodable at every step.
        is_canceled: Polled once per step; a True return raises ``KeyboardInterrupt``-free
            cancellation by letting the caller's exception type propagate from the callback.

    Returns:
        The denoised ``(video_rows, audio_rows)`` (conditioning/reference rows still included).
    """
    from invokeai.app.services.session_processor.session_processor_common import CanceledException

    num_condition_video_rows = state.layout.num_condition_video_rows
    num_condition_audio_rows = state.layout.num_condition_audio_rows

    latents = state.video_rows
    audio_latents = state.audio_rows
    prompt_embeds = prompt_embeds.to(latents.device)

    total_steps = len(state.timesteps)
    for i, t in enumerate(state.timesteps):
        if is_canceled is not None and is_canceled():
            raise CanceledException

        unique_timesteps, timestep_indices = state.row_timestep_plan[i]
        noise_pred, audio_noise_pred = transformer(
            hidden_states=latents[None],
            audio_hidden_states=audio_latents[None],
            encoder_hidden_states=prompt_embeds,
            timestep=unique_timesteps,
            timestep_indices=timestep_indices,
            token_tags=state.token_tags,
            position_ids=state.position_ids,
            video_indices=state.video_indices,
            audio_indices=state.audio_indices,
            text_indices=state.text_indices,
            attention_kwargs=None,
            return_dict=False,
        )

        pred_x0_video_rows: torch.Tensor | None = None
        if step_callback is not None:
            # The scheduler's own denoised estimate (`x0 = x_t + sigma * v`, data-ward velocity),
            # taken BEFORE the in-place Euler update below overwrites x_t.
            sigma_video = 1.0 - t.to(torch.float32)
            pred_x0_video_rows = latents[num_condition_video_rows:].to(torch.float32) + sigma_video * noise_pred[
                0, num_condition_video_rows:
            ].to(torch.float32)

        latents[num_condition_video_rows:] = state.scheduler.step(
            noise_pred[0, num_condition_video_rows:].float(),
            t,
            latents[num_condition_video_rows:],
            return_dict=False,
        )[0]
        audio_latents[num_condition_audio_rows:] = state.audio_scheduler.step(
            audio_noise_pred[0, num_condition_audio_rows:].float(),
            state.audio_timesteps[i],
            audio_latents[num_condition_audio_rows:],
            return_dict=False,
        )[0]

        if step_callback is not None:
            assert pred_x0_video_rows is not None
            step_callback(i + 1, total_steps, pred_x0_video_rows)

    return latents, audio_latents
