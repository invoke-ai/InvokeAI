"""Flux2 Klein Denoising Function.

This module provides the denoising function for FLUX.2 Klein models,
which use Qwen3 as the text encoder instead of CLIP+T5.
"""

import math
from typing import Any, Callable

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm

from invokeai.backend.flux.schedulers import (
    is_flow_match_lcm_scheduler,
    set_heun_timesteps_from_sigmas,
    set_timesteps_from_sigmas,
)
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.util.devices import TorchDevice


def denoise(
    model: torch.nn.Module,
    # model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    # sampling parameters
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    guidance: float,
    cfg_scale: list[float],
    # Negative conditioning for CFG
    neg_txt: torch.Tensor | None = None,
    neg_txt_ids: torch.Tensor | None = None,
    # Diffusers Euler, Heun or LCM scheduler that runs `timesteps`. Manual Euler when omitted.
    scheduler: SchedulerMixin | None = None,
    # Inpainting extension for merging latents during denoising
    inpaint_extension: RectifiedFlowInpaintExtension | None = None,
    # Reference image conditioning (multi-reference image editing)
    img_cond_seq: torch.Tensor | None = None,
    img_cond_seq_ids: torch.Tensor | None = None,
    # Optional joint_attention_kwargs (e.g. {"attention_mask": ...}) applied to positive forward
    # passes only. Negative forwards always run unmasked.
    pos_joint_attention_kwargs: dict[str, Any] | None = None,
    # RNG for stochastic schedulers. FlowMatchLCMScheduler.step re-noises the sample every step.
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Denoise latents using a FLUX.2 Klein transformer model.

    This is a simplified denoise function for FLUX.2 Klein models that uses
    the diffusers Flux2Transformer2DModel interface.

    All current FLUX.2 Klein variants (4B, 4B Base, 9B, 9B Base) have guidance_embeds=False
    in their HF transformer config (or absent/zeroed projection weights), so the guidance
    value is passed but effectively ignored by the model. The argument is retained for
    node-graph compatibility and future variants that may ship trained guidance projections.
    CFG is applied externally using negative conditioning when cfg_scale != 1.0.

    Args:
        model: The Flux2Transformer2DModel from diffusers.
        img: Packed latent image tensor of shape (B, seq_len, channels).
        img_ids: Image position IDs tensor.
        txt: Text encoder hidden states (Qwen3 embeddings).
        txt_ids: Text position IDs tensor.
        timesteps: The shifted, clipped sigma schedule: the sigmas the model is evaluated at, followed
            by the terminal sigma the last step ends at.
        step_callback: Callback function for progress updates.
        guidance: Guidance strength. Inert for all current FLUX.2 Klein variants
            (their guidance_embeds projection weights are absent/zero).
        cfg_scale: List of CFG scale values per step.
        neg_txt: Negative text embeddings for CFG (optional).
        neg_txt_ids: Negative text position IDs (optional).
        scheduler: Optional diffusers scheduler (Euler, Heun, LCM) without a shift of its own; it runs
            `timesteps` as given. If None, uses manual Euler.

    Returns:
        Denoised latent tensor.
    """
    total_steps = len(timesteps) - 1

    # Sequence length of the generated latents, i.e. everything that is *not* reference conditioning.
    original_seq_len = img.shape[1]

    # Reference image conditioning (multi-reference image editing) is constant context: it is
    # concatenated onto the latents for every forward pass, but it must NOT be advanced by the
    # sampler. Concatenating it onto `img` itself would make every sampler step integrate the
    # reference tokens along the model's velocity field, so the reference drifts away from the
    # encoded image over the schedule (and drags the generated image with it). Instead, build the
    # model input per step and slice the prediction back to the generated tokens - same as the
    # FLUX.1 Kontext path and diffusers' Flux2KleinPipeline.
    # The position IDs for the concatenated sequence are constant, so precompute them once.
    if img_cond_seq is not None:
        assert img_cond_seq_ids is not None, "You need to provide either both or neither of the sequence conditioning"
        model_img_ids = torch.cat([img_ids, img_cond_seq_ids], dim=1)
    else:
        model_img_ids = img_ids

    # The transformer forward() requires a guidance tensor even when guidance_embeds=False,
    # because the Flux2TimestepGuidanceEmbeddings forward signature takes it unconditionally.
    # All current Klein variants have guidance_embeds=False, so the value is ignored internally.
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # Use scheduler if provided
    use_scheduler = scheduler is not None
    if use_scheduler:
        # Every scheduler runs InvokeAI's schedule: the model-evaluation sigmas plus the requested
        # terminal sigma, which is not 0 when denoising_end < 1. Unlike FLUX.1, LCM gets it too: the
        # FLUX.2 schedule is shifted from mu and these schedulers do not shift, so a schedule LCM built
        # itself would be the unshifted linear one.
        is_heun = isinstance(scheduler, FlowMatchHeunDiscreteScheduler)
        if is_heun:
            set_heun_timesteps_from_sigmas(scheduler, timesteps[:-1], timesteps[-1], img.device)
        elif isinstance(scheduler, FlowMatchEulerDiscreteScheduler) or is_flow_match_lcm_scheduler(scheduler):
            set_timesteps_from_sigmas(scheduler, timesteps[:-1], timesteps[-1], img.device)
        else:
            raise TypeError(f"Unsupported FLUX.2 scheduler: {type(scheduler).__name__}")
        num_scheduler_steps = len(scheduler.timesteps)
        user_step = 0

        pbar = tqdm(total=total_steps, desc=f"Denoising{TorchDevice.get_session_device_label()}")
        for step_index in range(num_scheduler_steps):
            timestep = scheduler.timesteps[step_index]
            # Convert scheduler timestep (0-1000) to normalized (0-1) for the model
            t_curr = timestep.item() / scheduler.config.num_train_timesteps
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            # Track if we're in first or second order step (for Heun)
            in_first_order = scheduler.state_in_first_order if is_heun else True

            # Append the (unchanged) reference conditioning for this forward pass only.
            img_input = torch.cat([img, img_cond_seq], dim=1) if img_cond_seq is not None else img

            # Run the transformer model (matching diffusers: guidance=guidance, return_dict=False)
            output = model(
                hidden_states=img_input,
                encoder_hidden_states=txt,
                timestep=t_vec,
                img_ids=model_img_ids,
                txt_ids=txt_ids,
                guidance=guidance_vec,
                joint_attention_kwargs=pos_joint_attention_kwargs,
                return_dict=False,
            )

            # Extract the sample from the output (return_dict=False returns tuple)
            pred = output[0] if isinstance(output, tuple) else output

            # Drop the prediction for the reference tokens - they are context, not sampled state.
            if img_cond_seq is not None:
                pred = pred[:, :original_seq_len]

            step_cfg_scale = cfg_scale[min(user_step, len(cfg_scale) - 1)]

            # Apply CFG if scale is not 1.0
            if not math.isclose(step_cfg_scale, 1.0):
                if neg_txt is None:
                    raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

                neg_output = model(
                    hidden_states=img_input,
                    encoder_hidden_states=neg_txt,
                    timestep=t_vec,
                    img_ids=model_img_ids,
                    txt_ids=neg_txt_ids if neg_txt_ids is not None else txt_ids,
                    guidance=guidance_vec,
                    return_dict=False,
                )

                neg_pred = neg_output[0] if isinstance(neg_output, tuple) else neg_output
                if img_cond_seq is not None:
                    neg_pred = neg_pred[:, :original_seq_len]
                pred = neg_pred + step_cfg_scale * (pred - neg_pred)

            # Use the sample and sigma of this model evaluation; after the step, `img` is already at
            # the next sigma. For Heun the interval's preview comes from its second-order evaluation, or
            # from the final first-order step, which has none.
            should_emit_preview = (
                not is_heun or not in_first_order or step_index == num_scheduler_steps - 1
            ) and user_step < total_steps
            preview_img = img - t_curr * pred if should_emit_preview else None

            # Use scheduler.step() for the update. `generator` matters for LCM, which re-noises every
            # step. Euler and Heun don't use step noise here: with `s_churn` at 0 it is never applied.
            step_output = scheduler.step(model_output=pred, timestep=timestep, sample=img, generator=generator)
            img = step_output.prev_sample

            # Get t_prev for inpainting (next sigma value)
            if step_index + 1 < len(scheduler.sigmas):
                t_prev = scheduler.sigmas[step_index + 1].item()
            else:
                t_prev = 0.0

            # Apply inpainting merge at each step
            if inpaint_extension is not None:
                img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)

            # For Heun, a user step completes with its second-order evaluation. The final interval has
            # none: Diffusers ends it with a first-order step to the terminal sigma.
            if is_heun:
                if not in_first_order or step_index == num_scheduler_steps - 1:
                    user_step += 1
                    if user_step <= total_steps:
                        pbar.update(1)
                        assert preview_img is not None
                        if inpaint_extension is not None:
                            preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(
                                preview_img, 0.0
                            )
                        step_callback(
                            PipelineIntermediateState(
                                step=user_step,
                                # Progress over completed user steps, not the solver's order.
                                order=1,
                                total_steps=total_steps,
                                timestep=int(t_curr * 1000),
                                latents=preview_img,
                            ),
                        )
            else:
                user_step += 1
                if user_step <= total_steps:
                    pbar.update(1)
                    assert preview_img is not None
                    if inpaint_extension is not None:
                        preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(preview_img, 0.0)
                    step_callback(
                        PipelineIntermediateState(
                            step=user_step,
                            order=1,
                            total_steps=total_steps,
                            timestep=int(t_curr * 1000),
                            latents=preview_img,
                        ),
                    )

        pbar.close()
    else:
        # Manual Euler stepping (original behavior)
        for step_index, (t_curr, t_prev) in tqdm(
            list(enumerate(zip(timesteps[:-1], timesteps[1:], strict=True))),
            desc=f"Denoising{TorchDevice.get_session_device_label()}",
        ):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            # Append the (unchanged) reference conditioning for this forward pass only.
            img_input = torch.cat([img, img_cond_seq], dim=1) if img_cond_seq is not None else img

            # Run the transformer model (matching diffusers: guidance=guidance, return_dict=False)
            output = model(
                hidden_states=img_input,
                encoder_hidden_states=txt,
                timestep=t_vec,
                img_ids=model_img_ids,
                txt_ids=txt_ids,
                guidance=guidance_vec,
                joint_attention_kwargs=pos_joint_attention_kwargs,
                return_dict=False,
            )

            # Extract the sample from the output (return_dict=False returns tuple)
            pred = output[0] if isinstance(output, tuple) else output

            # Drop the prediction for the reference tokens - they are context, not sampled state.
            if img_cond_seq is not None:
                pred = pred[:, :original_seq_len]

            step_cfg_scale = cfg_scale[step_index]

            # Apply CFG if scale is not 1.0
            if not math.isclose(step_cfg_scale, 1.0):
                if neg_txt is None:
                    raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

                neg_output = model(
                    hidden_states=img_input,
                    encoder_hidden_states=neg_txt,
                    timestep=t_vec,
                    img_ids=model_img_ids,
                    txt_ids=neg_txt_ids if neg_txt_ids is not None else txt_ids,
                    guidance=guidance_vec,
                    return_dict=False,
                )

                neg_pred = neg_output[0] if isinstance(neg_output, tuple) else neg_output
                if img_cond_seq is not None:
                    neg_pred = neg_pred[:, :original_seq_len]
                pred = neg_pred + step_cfg_scale * (pred - neg_pred)

            # Euler step
            preview_img = img - t_curr * pred
            img = img + (t_prev - t_curr) * pred

            # Apply inpainting merge at each step
            if inpaint_extension is not None:
                img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)
                preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(preview_img, 0.0)

            step_callback(
                PipelineIntermediateState(
                    step=step_index + 1,
                    order=1,
                    total_steps=total_steps,
                    timestep=int(t_curr),
                    latents=preview_img,
                ),
            )

    return img
