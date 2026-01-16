"""Flux2 Klein Denoising Function.

This module provides the denoising function for FLUX.2 Klein models,
which use Qwen3 as the text encoder instead of CLIP+T5.
"""

import math
from typing import Any, Callable, Union

import torch
from diffusers.models.transformers.transformer_flux import Transformer2DModelOutput
from tqdm import tqdm

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
        timesteps: List of timesteps for denoising schedule.
        step_callback: Callback function for progress updates.
        cfg_scale: List of CFG scale values per step.
        neg_txt: Negative text embeddings for CFG (optional).
        neg_txt_ids: Negative text position IDs (optional).
        scheduler: Optional diffusers scheduler (Euler, Heun, LCM). If None, uses manual Euler.

    Returns:
        Denoised latent tensor.
    """
    total_steps = len(timesteps) - 1

    # Klein has guidance_embeds=False, but diffusers forward() still expects the parameter.
    # We pass a dummy value (1.0) since it won't be used for embeddings.
    guidance_vec = torch.full((img.shape[0],), 1.0, device=img.device, dtype=img.dtype)

    # Use scheduler if provided
    use_scheduler = scheduler is not None
    if use_scheduler:
        # Set up scheduler timesteps (convert 0-1 range to 0-1000)
        scheduler_timesteps = [int(t * 1000) for t in timesteps[:-1]]
        scheduler.set_timesteps(timesteps=scheduler_timesteps, device=img.device)
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

            # Run the transformer model
            output: Union[torch.Tensor, Transformer2DModelOutput] = model(
                hidden_states=img,
                encoder_hidden_states=txt,
                timestep=t_vec,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance_vec,
                return_dict=True,
            )

            # Extract the sample from the output
            if isinstance(output, Transformer2DModelOutput):
                pred = output.sample
            else:
                pred = output

            step_cfg_scale = cfg_scale[min(user_step, len(cfg_scale) - 1)]

            # Apply CFG if scale is not 1.0
            if not math.isclose(step_cfg_scale, 1.0):
                if neg_txt is None:
                    raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

                neg_output: Union[torch.Tensor, Transformer2DModelOutput] = model(
                    hidden_states=img,
                    encoder_hidden_states=neg_txt,
                    timestep=t_vec,
                    img_ids=img_ids,
                    txt_ids=neg_txt_ids if neg_txt_ids is not None else txt_ids,
                    guidance=guidance_vec,
                    return_dict=True,
                )

                if isinstance(neg_output, Transformer2DModelOutput):
                    neg_pred = neg_output.sample
                else:
                    neg_pred = neg_output

                pred = neg_pred + step_cfg_scale * (pred - neg_pred)

            # Use scheduler.step() for the update
            step_output = scheduler.step(model_output=pred, timestep=timestep, sample=img)
            img = step_output.prev_sample

            # For Heun, only increment user step after second-order step completes
            if is_heun:
                if not in_first_order:
                    user_step += 1
                    if user_step <= total_steps:
                        pbar.update(1)
                        preview_img = img - t_curr * pred
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
        for step_index, (t_curr, t_prev) in tqdm(list(enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)))):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            # Run the transformer model (Klein ignores guidance due to guidance_embeds=False)
            output: Union[torch.Tensor, Transformer2DModelOutput] = model(
                hidden_states=img,
                encoder_hidden_states=txt,
                timestep=t_vec,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance_vec,
                return_dict=True,
            )

            # Extract the sample from the output
            if isinstance(output, Transformer2DModelOutput):
                pred = output.sample
            else:
                pred = output

            step_cfg_scale = cfg_scale[step_index]

            # Apply CFG if scale is not 1.0
            if not math.isclose(step_cfg_scale, 1.0):
                if neg_txt is None:
                    raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

                neg_output: Union[torch.Tensor, Transformer2DModelOutput] = model(
                    hidden_states=img,
                    encoder_hidden_states=neg_txt,
                    timestep=t_vec,
                    img_ids=img_ids,
                    txt_ids=neg_txt_ids if neg_txt_ids is not None else txt_ids,
                    guidance=guidance_vec,
                    return_dict=True,
                )

                if isinstance(neg_output, Transformer2DModelOutput):
                    neg_pred = neg_output.sample
                else:
                    neg_pred = neg_output

                pred = neg_pred + step_cfg_scale * (pred - neg_pred)

            # Euler step
            preview_img = img - t_curr * pred
            img = img + (t_prev - t_curr) * pred

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
