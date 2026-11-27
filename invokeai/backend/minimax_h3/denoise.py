"""The MiniMax H3 denoising loop (FL2VA / T2VA).

First-party port of ``modular_pipelines/minimax_h3/denoise.py`` (commit recorded in
``__init__``): one transformer forward per step over the packed sequence — every row at its own
noise level — followed by one scheduler step per modality on the *generated* rows only. The
conditioning rows are never written, so the anchors survive the loop by construction. The
checkpoint is guidance-distilled: no unconditional pass, no CFG.

Diagnostics: set ``INVOKEAI_PROFILE_H3_DENOISE`` to profile exactly one denoise step with
``torch.profiler`` — the second step when there is one (the first step pays allocator warm-up
and, under partial loading, first-touch weight streaming). The kernel-level summary table is
logged at INFO and a chrome trace (viewable at ``chrome://tracing`` or https://ui.perfetto.dev)
is written to the directory the variable names (or the working directory when set to ``1``).
Unset, the loop behaves exactly as before.
"""

import os
import time
from pathlib import Path
from typing import Callable

import torch

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.minimax_h3.sampling import MiniMaxH3DenoiseState
from invokeai.backend.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from invokeai.backend.util.cancel_hooks import cancel_before_forward
from invokeai.backend.util.logging import InvokeAILogger

PROFILE_ENV_VAR = "INVOKEAI_PROFILE_H3_DENOISE"


def _kineto_gpu_events_visible(device: torch.device) -> bool:
    """Probe whether torch.profiler (kineto) can actually see GPU kernels here.

    Accepting ``ProfilerActivity.CUDA`` is no guarantee: CUPTI can fail to initialize with
    ``CUPTI_ERROR_INVALID_DEVICE`` (observed on consumer Blackwell cards), and the first
    profiler session starting on a session-worker thread trips kineto's "External init
    callback must run in same thread as registerClient" — in both cases the profile comes
    back with zero GPU events and only CPU-side times. The probe pays a few milliseconds
    once; when it sees no GPU events the step is profiled with the legacy cudaEvent
    fallback instead, which times each op without CUPTI. ROCm (roctracer) passes the probe.
    """
    from torch.profiler import ProfilerActivity, profile

    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            (torch.ones(8, device=device) + 1).sum().item()
        return any(
            (getattr(evt, "self_device_time_total", 0) or getattr(evt, "self_cuda_time_total", 0)) > 0
            for evt in prof.key_averages()
        )
    except Exception:
        return False


def _profile_one_step(trace_target: str, device: torch.device, run_step: Callable[[], None]) -> None:
    """Run one denoise step under torch.profiler; log the kernel table and write a chrome trace.

    The device is synchronized before and after the profiled region so the window contains this
    step's kernels only (not the tail of the previous step's asynchronous queue).
    """
    from torch.profiler import ProfilerActivity, profile

    logger = InvokeAILogger.get_logger(__name__)
    is_gpu = device.type == "cuda"

    if is_gpu and _kineto_gpu_events_visible(device):
        profiler_cm = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    elif is_gpu:
        logger.warning(
            "MiniMax H3 denoise: torch.profiler cannot see GPU kernels on this device (CUPTI unavailable or "
            "mis-initialized); profiling with the cudaEvent fallback instead. GPU times below are per-op event "
            "timings rather than true kernel activities."
        )
        try:
            profiler_cm = torch.autograd.profiler.profile(use_device="cuda")
        except TypeError:
            # Older torch spells it use_cuda.
            profiler_cm = torch.autograd.profiler.profile(use_cuda=True)
    else:
        profiler_cm = profile(activities=[ProfilerActivity.CPU])

    if is_gpu:
        torch.cuda.synchronize(device)
    wall_start = time.time()
    with profiler_cm as prof:
        run_step()
        if is_gpu:
            torch.cuda.synchronize(device)
    wall_elapsed = time.time() - wall_start

    # The sort key was renamed cuda -> device across torch releases; try newest first. The CPU
    # key always exists, so the loop cannot fall through with `table` still empty.
    table = ""
    for sort_key in ("self_device_time_total", "self_cuda_time_total", "self_cpu_time_total"):
        try:
            table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
            break
        except Exception:
            continue
    logger.info(f"MiniMax H3 denoise: profiled one step in {wall_elapsed:.2f}s wall time. Kernel summary:\n{table}")

    trace_dir = Path(trace_target) if trace_target.lower() not in ("1", "true", "yes") else Path.cwd()
    trace_path = trace_dir / "h3_denoise_step_trace.json"
    table_path = trace_dir / "h3_denoise_step_profile.txt"
    try:
        trace_dir.mkdir(parents=True, exist_ok=True)
        # Persist the table too: the log line above can scroll away, and the cudaEvent fallback's
        # chrome trace carries no GPU lanes — on that path this file is the only durable record of
        # the GPU times.
        table_path.write_text(
            f"MiniMax H3 denoise step profile ({time.strftime('%Y-%m-%d %H:%M:%S')})\n"
            f"device: {torch.cuda.get_device_name(device) if is_gpu else device}\n"
            f"wall time for the profiled step: {wall_elapsed:.2f}s (includes profiler overhead)\n\n"
            f"{table}\n",
            encoding="utf-8",
        )
        prof.export_chrome_trace(str(trace_path))
        logger.info(f"MiniMax H3 denoise: wrote {table_path} and chrome trace {trace_path}")
    except Exception:
        # The table above is the primary deliverable; a failed file export must not kill the
        # generation mid-denoise.
        logger.warning(f"MiniMax H3 denoise: failed to write profile files to {trace_dir}", exc_info=True)


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
        is_canceled: Polled before every step and before every transformer block within a step,
            after the previous block's kernels have finished; a True return raises
            ``CanceledException`` from inside the forward, so a cancel idles the GPU within one
            block rather than at the end of a 40-100 s step.

    Returns:
        The denoised ``(video_rows, audio_rows)`` (conditioning/reference rows still included).
    """
    num_condition_video_rows = state.layout.num_condition_video_rows
    num_condition_audio_rows = state.layout.num_condition_audio_rows

    latents = state.video_rows
    audio_latents = state.audio_rows
    prompt_embeds = prompt_embeds.to(latents.device)

    total_steps = len(state.timesteps)

    profile_target = os.environ.get(PROFILE_ENV_VAR)
    profile_step_index = (1 if total_steps > 1 else 0) if profile_target else None

    def run_step(i: int, t: torch.Tensor) -> None:
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

    # One step is a forward through the full block stack (40-100 s for a video on the reference
    # dual-GPU rig), so the step loop's own poll is not enough: poll before every block too.
    with cancel_before_forward(transformer.transformer_blocks, is_canceled, latents.device):
        for i, t in enumerate(state.timesteps):
            if is_canceled is not None and is_canceled():
                raise CanceledException

            if i == profile_step_index:
                assert profile_target is not None
                _profile_one_step(profile_target, latents.device, lambda i=i, t=t: run_step(i, t))
            else:
                run_step(i, t)

    return latents, audio_latents
