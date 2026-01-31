"""Flux2 Klein Denoising Function.

This module provides the denoising function for FLUX.2 Klein models,
which use Qwen3 as the text encoder instead of CLIP+T5.
"""

import inspect
import math
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


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
    cfg_scale: list[float],
    # Negative conditioning for CFG
    neg_txt: torch.Tensor | None = None,
    neg_txt_ids: torch.Tensor | None = None,
    # Scheduler for stepping (e.g., FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler)
    scheduler: Any = None,
    # Dynamic shifting parameter for FLUX.2 Klein (computed from image resolution)
    mu: float | None = None,
    # Inpainting extension for merging latents during denoising
    inpaint_extension: RectifiedFlowInpaintExtension | None = None,
    # Reference image conditioning (multi-reference image editing)
    img_cond_seq: torch.Tensor | None = None,
    img_cond_seq_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Denoise latents using a FLUX.2 Klein transformer model.

    This is a simplified denoise function for FLUX.2 Klein models that uses
    the diffusers Flux2Transformer2DModel interface.

    Note: FLUX.2 Klein has guidance_embeds=False, so no guidance parameter is used.
    CFG is applied externally using negative conditioning when cfg_scale != 1.0.

    Args:
        model: The Flux2Transformer2DModel from diffusers.
        img: Packed latent image tensor of shape (B, seq_len, channels).
        img_ids: Image position IDs tensor.
        txt: Text encoder hidden states (Qwen3 embeddings).
        txt_ids: Text position IDs tensor.
        timesteps: List of timesteps for denoising schedule (linear sigmas from 1.0 to 1/n).
        step_callback: Callback function for progress updates.
        cfg_scale: List of CFG scale values per step.
        neg_txt: Negative text embeddings for CFG (optional).
        neg_txt_ids: Negative text position IDs (optional).
        scheduler: Optional diffusers scheduler (Euler, Heun, LCM). If None, uses manual Euler.
        mu: Dynamic shifting parameter computed from image resolution. Required when scheduler
            has use_dynamic_shifting=True.

    Returns:
        Denoised latent tensor.
    """
    total_steps = len(timesteps) - 1

    # Store original sequence length for extracting output later (before concatenating reference images)
    original_seq_len = img.shape[1]

    # Concatenate reference image conditioning if provided (multi-reference image editing)
    if img_cond_seq is not None and img_cond_seq_ids is not None:
        img = torch.cat([img, img_cond_seq], dim=1)
        img_ids = torch.cat([img_ids, img_cond_seq_ids], dim=1)

    # Klein has guidance_embeds=False, but the transformer forward() still requires a guidance tensor
    # We pass a dummy value (1.0) since it won't affect the output when guidance_embeds=False
    guidance = torch.full((img.shape[0],), 1.0, device=img.device, dtype=img.dtype)

    # Use scheduler if provided
    use_scheduler = scheduler is not None
    if use_scheduler:
        # Set up scheduler with sigmas and mu for dynamic shifting
        # Convert timesteps (0-1 range) to sigmas for the scheduler
        # The scheduler will apply dynamic shifting internally using mu (if enabled in scheduler config)
        sigmas = np.array(timesteps[:-1], dtype=np.float32)  # Exclude final 0.0

        # Check if scheduler supports sigmas parameter using inspect.signature
        # FlowMatchHeunDiscreteScheduler and FlowMatchLCMScheduler don't support sigmas
        set_timesteps_sig = inspect.signature(scheduler.set_timesteps)
        supports_sigmas = "sigmas" in set_timesteps_sig.parameters
        if supports_sigmas and mu is not None:
            # Pass mu if provided - it will only be used if scheduler has use_dynamic_shifting=True
            scheduler.set_timesteps(sigmas=sigmas.tolist(), mu=mu, device=img.device)
        elif supports_sigmas:
            scheduler.set_timesteps(sigmas=sigmas.tolist(), device=img.device)
        else:
            # Scheduler doesn't support sigmas (e.g., Heun, LCM) - use num_inference_steps
            scheduler.set_timesteps(num_inference_steps=len(sigmas), device=img.device)
        num_scheduler_steps = len(scheduler.timesteps)
        is_heun = hasattr(scheduler, "state_in_first_order")
        user_step = 0

        pbar = tqdm(total=total_steps, desc="Denoising")
        for step_index in range(num_scheduler_steps):
            timestep = scheduler.timesteps[step_index]
            # Convert scheduler timestep (0-1000) to normalized (0-1) for the model
            t_curr = timestep.item() / scheduler.config.num_train_timesteps
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            # Track if we're in first or second order step (for Heun)
            in_first_order = scheduler.state_in_first_order if is_heun else True

            # Run the transformer model (matching diffusers: guidance=guidance, return_dict=False)
            output = model(
                hidden_states=img,
                encoder_hidden_states=txt,
                timestep=t_vec,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )

            # Extract the sample from the output (return_dict=False returns tuple)
            pred = output[0] if isinstance(output, tuple) else output

            step_cfg_scale = cfg_scale[min(user_step, len(cfg_scale) - 1)]

            # Apply CFG if scale is not 1.0
            if not math.isclose(step_cfg_scale, 1.0):
                if neg_txt is None:
                    raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

                neg_output = model(
                    hidden_states=img,
                    encoder_hidden_states=neg_txt,
                    timestep=t_vec,
                    img_ids=img_ids,
                    txt_ids=neg_txt_ids if neg_txt_ids is not None else txt_ids,
                    guidance=guidance,
                    return_dict=False,
                )

                neg_pred = neg_output[0] if isinstance(neg_output, tuple) else neg_output
                pred = neg_pred + step_cfg_scale * (pred - neg_pred)

            # Use scheduler.step() for the update
            step_output = scheduler.step(model_output=pred, timestep=timestep, sample=img)
            img = step_output.prev_sample

            # Get t_prev for inpainting (next sigma value)
            if step_index + 1 < len(scheduler.sigmas):
                t_prev = scheduler.sigmas[step_index + 1].item()
            else:
                t_prev = 0.0

            # Apply inpainting merge at each step
            if inpaint_extension is not None:
                # Separate the generated latents from the reference conditioning
                gen_img = img[:, :original_seq_len, :]
                ref_img = img[:, original_seq_len:, :]

                # Merge only the generated part
                gen_img = inpaint_extension.merge_intermediate_latents_with_init_latents(gen_img, t_prev)

                # Concatenate back together
                img = torch.cat([gen_img, ref_img], dim=1)

            # For Heun, only increment user step after second-order step completes
            if is_heun:
                if not in_first_order:
                    user_step += 1
                    if user_step <= total_steps:
                        pbar.update(1)
                        preview_img = img - t_curr * pred
                        if inpaint_extension is not None:
                            preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(
                                preview_img, 0.0
                            )
                        step_callback(
                            PipelineIntermediateState(
                                step=user_step,
                                order=2,
                                total_steps=total_steps,
                                timestep=int(t_curr * 1000),
                                latents=preview_img,
                            ),
                        )
            else:
                user_step += 1
                if user_step <= total_steps:
                    pbar.update(1)
                    preview_img = img - t_curr * pred
                    if inpaint_extension is not None:
                        preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(preview_img, 0.0)
                    # Extract only the generated image portion for preview (exclude reference images)
                    callback_latents = preview_img[:, :original_seq_len, :] if img_cond_seq is not None else preview_img
                    step_callback(
                        PipelineIntermediateState(
                            step=user_step,
                            order=1,
                            total_steps=total_steps,
                            timestep=int(t_curr * 1000),
                            latents=callback_latents,
                        ),
                    )

        pbar.close()
    else:
        # Manual Euler stepping (original behavior)
        for step_index, (t_curr, t_prev) in tqdm(list(enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)))):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            # Run the transformer model (matching diffusers: guidance=guidance, return_dict=False)
            output = model(
                hidden_states=img,
                encoder_hidden_states=txt,
                timestep=t_vec,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )

            # Extract the sample from the output (return_dict=False returns tuple)
            pred = output[0] if isinstance(output, tuple) else output

            step_cfg_scale = cfg_scale[step_index]

            # Apply CFG if scale is not 1.0
            if not math.isclose(step_cfg_scale, 1.0):
                if neg_txt is None:
                    raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

                neg_output = model(
                    hidden_states=img,
                    encoder_hidden_states=neg_txt,
                    timestep=t_vec,
                    img_ids=img_ids,
                    txt_ids=neg_txt_ids if neg_txt_ids is not None else txt_ids,
                    guidance=guidance,
                    return_dict=False,
                )

                neg_pred = neg_output[0] if isinstance(neg_output, tuple) else neg_output
                pred = neg_pred + step_cfg_scale * (pred - neg_pred)

            # Euler step
            preview_img = img - t_curr * pred
            img = img + (t_prev - t_curr) * pred

            # Apply inpainting merge at each step
            if inpaint_extension is not None:
                # Separate the generated latents from the reference conditioning
                gen_img = img[:, :original_seq_len, :]
                ref_img = img[:, original_seq_len:, :]

                # Merge only the generated part
                gen_img = inpaint_extension.merge_intermediate_latents_with_init_latents(gen_img, t_prev)

                # Concatenate back together
                img = torch.cat([gen_img, ref_img], dim=1)

                # Handling preview images
                preview_gen = preview_img[:, :original_seq_len, :]
                preview_gen = inpaint_extension.merge_intermediate_latents_with_init_latents(preview_gen, 0.0)

            # Extract only the generated image portion for preview (exclude reference images)
            callback_latents = preview_img[:, :original_seq_len, :] if img_cond_seq is not None else preview_img
            step_callback(
                PipelineIntermediateState(
                    step=step_index + 1,
                    order=1,
                    total_steps=total_steps,
                    timestep=int(t_curr),
                    latents=callback_latents,
                ),
            )

    # Extract only the generated image portion (exclude concatenated reference images)
    if img_cond_seq is not None:
        img = img[:, :original_seq_len, :]

    return img
