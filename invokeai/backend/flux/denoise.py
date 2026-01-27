import inspect
import math
from typing import Callable

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm

from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput, sum_controlnet_flux_outputs
from invokeai.backend.flux.extensions.dype_extension import DyPEExtension
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.extensions.xlabs_controlnet_extension import XLabsControlNetExtension
from invokeai.backend.flux.extensions.xlabs_ip_adapter_extension import XLabsIPAdapterExtension
from invokeai.backend.flux.model import Flux
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


def denoise(
    model: Flux,
    # model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    pos_regional_prompting_extension: RegionalPromptingExtension,
    neg_regional_prompting_extension: RegionalPromptingExtension | None,
    # sampling parameters
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    guidance: float,
    cfg_scale: list[float],
    inpaint_extension: RectifiedFlowInpaintExtension | None,
    controlnet_extensions: list[XLabsControlNetExtension | InstantXControlNetExtension],
    pos_ip_adapter_extensions: list[XLabsIPAdapterExtension],
    neg_ip_adapter_extensions: list[XLabsIPAdapterExtension],
    # extra img tokens (channel-wise)
    img_cond: torch.Tensor | None,
    # extra img tokens (sequence-wise) - for Kontext conditioning
    img_cond_seq: torch.Tensor | None = None,
    img_cond_seq_ids: torch.Tensor | None = None,
    # DyPE extension for high-resolution generation
    dype_extension: DyPEExtension | None = None,
    # Optional scheduler for alternative sampling methods
    scheduler: SchedulerMixin | None = None,
):
    # Determine if we're using a diffusers scheduler or the built-in Euler method
    use_scheduler = scheduler is not None

    if use_scheduler:
        # Initialize scheduler with timesteps
        # The timesteps list contains values in [0, 1] range (sigmas)
        # LCM should use num_inference_steps (it has its own sigma schedule),
        # while other schedulers can use custom sigmas if supported
        is_lcm = scheduler.__class__.__name__ == "FlowMatchLCMScheduler"
        set_timesteps_sig = inspect.signature(scheduler.set_timesteps)
        if not is_lcm and "sigmas" in set_timesteps_sig.parameters:
            # Scheduler supports custom sigmas - use InvokeAI's time-shifted schedule
            scheduler.set_timesteps(sigmas=timesteps, device=img.device)
        else:
            # LCM or scheduler doesn't support custom sigmas - use num_inference_steps
            # The schedule will be computed by the scheduler itself
            num_inference_steps = len(timesteps) - 1
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=img.device)

        # For schedulers like Heun, the number of actual steps may differ
        # (Heun doubles timesteps internally)
        num_scheduler_steps = len(scheduler.timesteps)
        # For user-facing step count, use the original number of denoising steps
        total_steps = len(timesteps) - 1
    else:
        total_steps = len(timesteps) - 1
        num_scheduler_steps = total_steps

    # guidance_vec is ignored for schnell.
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # Store original sequence length for slicing predictions
    original_seq_len = img.shape[1]

    # DyPE: Patch model with DyPE-aware position embedder
    dype_embedder = None
    original_pe_embedder = None
    if dype_extension is not None:
        dype_embedder, original_pe_embedder = dype_extension.patch_model(model)

    try:
        # Track the actual step for user-facing progress (accounts for Heun's double steps)
        user_step = 0

        if use_scheduler:
            # Use diffusers scheduler for stepping
            # Use tqdm with total_steps (user-facing steps) not num_scheduler_steps (internal steps)
            # This ensures progress bar shows 1/8, 2/8, etc. even when scheduler uses more internal steps
            pbar = tqdm(total=total_steps, desc="Denoising")
            for step_index in range(num_scheduler_steps):
                timestep = scheduler.timesteps[step_index]
                # Convert scheduler timestep (0-1000) to normalized (0-1) for the model
                t_curr = timestep.item() / scheduler.config.num_train_timesteps
                t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

                # DyPE: Update step state for timestep-dependent scaling
                if dype_extension is not None and dype_embedder is not None:
                    dype_extension.update_step_state(
                        embedder=dype_embedder,
                        timestep=t_curr,
                        timestep_index=user_step,
                        total_steps=total_steps,
                    )

                # For Heun scheduler, track if we're in first or second order step
                is_heun = hasattr(scheduler, "state_in_first_order")
                in_first_order = scheduler.state_in_first_order if is_heun else True

                # Run ControlNet models
                controlnet_residuals: list[ControlNetFluxOutput] = []
                for controlnet_extension in controlnet_extensions:
                    controlnet_residuals.append(
                        controlnet_extension.run_controlnet(
                            timestep_index=user_step,
                            total_num_timesteps=total_steps,
                            img=img,
                            img_ids=img_ids,
                            txt=pos_regional_prompting_extension.regional_text_conditioning.t5_embeddings,
                            txt_ids=pos_regional_prompting_extension.regional_text_conditioning.t5_txt_ids,
                            y=pos_regional_prompting_extension.regional_text_conditioning.clip_embeddings,
                            timesteps=t_vec,
                            guidance=guidance_vec,
                        )
                    )

                merged_controlnet_residuals = sum_controlnet_flux_outputs(controlnet_residuals)

                # Prepare input for model
                img_input = img
                img_input_ids = img_ids

                if img_cond is not None:
                    img_input = torch.cat((img_input, img_cond), dim=-1)

                if img_cond_seq is not None:
                    assert img_cond_seq_ids is not None
                    img_input = torch.cat((img_input, img_cond_seq), dim=1)
                    img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

                pred = model(
                    img=img_input,
                    img_ids=img_input_ids,
                    txt=pos_regional_prompting_extension.regional_text_conditioning.t5_embeddings,
                    txt_ids=pos_regional_prompting_extension.regional_text_conditioning.t5_txt_ids,
                    y=pos_regional_prompting_extension.regional_text_conditioning.clip_embeddings,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    timestep_index=user_step,
                    total_num_timesteps=total_steps,
                    controlnet_double_block_residuals=merged_controlnet_residuals.double_block_residuals,
                    controlnet_single_block_residuals=merged_controlnet_residuals.single_block_residuals,
                    ip_adapter_extensions=pos_ip_adapter_extensions,
                    regional_prompting_extension=pos_regional_prompting_extension,
                )

                if img_cond_seq is not None:
                    pred = pred[:, :original_seq_len]

                # Get CFG scale for current user step
                step_cfg_scale = cfg_scale[min(user_step, len(cfg_scale) - 1)]

                if not math.isclose(step_cfg_scale, 1.0):
                    if neg_regional_prompting_extension is None:
                        raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

                    neg_img_input = img
                    neg_img_input_ids = img_ids

                    if img_cond is not None:
                        neg_img_input = torch.cat((neg_img_input, img_cond), dim=-1)

                    if img_cond_seq is not None:
                        neg_img_input = torch.cat((neg_img_input, img_cond_seq), dim=1)
                        neg_img_input_ids = torch.cat((neg_img_input_ids, img_cond_seq_ids), dim=1)

                    neg_pred = model(
                        img=neg_img_input,
                        img_ids=neg_img_input_ids,
                        txt=neg_regional_prompting_extension.regional_text_conditioning.t5_embeddings,
                        txt_ids=neg_regional_prompting_extension.regional_text_conditioning.t5_txt_ids,
                        y=neg_regional_prompting_extension.regional_text_conditioning.clip_embeddings,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                        timestep_index=user_step,
                        total_num_timesteps=total_steps,
                        controlnet_double_block_residuals=None,
                        controlnet_single_block_residuals=None,
                        ip_adapter_extensions=neg_ip_adapter_extensions,
                        regional_prompting_extension=neg_regional_prompting_extension,
                    )

                    if img_cond_seq is not None:
                        neg_pred = neg_pred[:, :original_seq_len]
                    pred = neg_pred + step_cfg_scale * (pred - neg_pred)

                # Use scheduler.step() for the update
                step_output = scheduler.step(model_output=pred, timestep=timestep, sample=img)
                img = step_output.prev_sample

                # Get t_prev for inpainting (next sigma value)
                if step_index + 1 < len(scheduler.sigmas):
                    t_prev = scheduler.sigmas[step_index + 1].item()
                else:
                    t_prev = 0.0

                if inpaint_extension is not None:
                    img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)

                # For Heun, only increment user step after second-order step completes
                if is_heun:
                    if not in_first_order:
                        # Second order step completed
                        user_step += 1
                        # Only call step_callback if we haven't exceeded total_steps
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
                    # For LCM and other first-order schedulers
                    user_step += 1
                    # Only call step_callback if we haven't exceeded total_steps
                    # (LCM scheduler may have more internal steps than user-facing steps)
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
                                order=1,
                                total_steps=total_steps,
                                timestep=int(t_curr * 1000),
                                latents=preview_img,
                            ),
                        )

            pbar.close()
            return img

        # Original Euler implementation (when scheduler is None)
        for step_index, (t_curr, t_prev) in tqdm(list(enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)))):
            # DyPE: Update step state for timestep-dependent scaling
            if dype_extension is not None and dype_embedder is not None:
                dype_extension.update_step_state(
                    embedder=dype_embedder,
                    timestep=t_curr,
                    timestep_index=step_index,
                    total_steps=total_steps,
                )

            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            # Run ControlNet models.
            controlnet_residuals: list[ControlNetFluxOutput] = []
            for controlnet_extension in controlnet_extensions:
                controlnet_residuals.append(
                    controlnet_extension.run_controlnet(
                        timestep_index=step_index,
                        total_num_timesteps=total_steps,
                        img=img,
                        img_ids=img_ids,
                        txt=pos_regional_prompting_extension.regional_text_conditioning.t5_embeddings,
                        txt_ids=pos_regional_prompting_extension.regional_text_conditioning.t5_txt_ids,
                        y=pos_regional_prompting_extension.regional_text_conditioning.clip_embeddings,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
                )

            # Merge the ControlNet residuals from multiple ControlNets.
            # TODO(ryand): We may want to calculate the sum just-in-time to keep peak memory low. Keep in mind, that the
            # controlnet_residuals datastructure is efficient in that it likely contains multiple references to the same
            # tensors. Calculating the sum materializes each tensor into its own instance.
            merged_controlnet_residuals = sum_controlnet_flux_outputs(controlnet_residuals)

            # Prepare input for model - concatenate fresh each step
            img_input = img
            img_input_ids = img_ids

            # Add channel-wise conditioning (for ControlNet, FLUX Fill, etc.)
            if img_cond is not None:
                img_input = torch.cat((img_input, img_cond), dim=-1)

            # Add sequence-wise conditioning (for Kontext)
            if img_cond_seq is not None:
                assert img_cond_seq_ids is not None, (
                    "You need to provide either both or neither of the sequence conditioning"
                )
                img_input = torch.cat((img_input, img_cond_seq), dim=1)
                img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

            pred = model(
                img=img_input,
                img_ids=img_input_ids,
                txt=pos_regional_prompting_extension.regional_text_conditioning.t5_embeddings,
                txt_ids=pos_regional_prompting_extension.regional_text_conditioning.t5_txt_ids,
                y=pos_regional_prompting_extension.regional_text_conditioning.clip_embeddings,
                timesteps=t_vec,
                guidance=guidance_vec,
                timestep_index=step_index,
                total_num_timesteps=total_steps,
                controlnet_double_block_residuals=merged_controlnet_residuals.double_block_residuals,
                controlnet_single_block_residuals=merged_controlnet_residuals.single_block_residuals,
                ip_adapter_extensions=pos_ip_adapter_extensions,
                regional_prompting_extension=pos_regional_prompting_extension,
            )

            # Slice prediction to only include the main image tokens
            if img_cond_seq is not None:
                pred = pred[:, :original_seq_len]

            step_cfg_scale = cfg_scale[step_index]

            # If step_cfg_scale, is 1.0, then we don't need to run the negative prediction.
            if not math.isclose(step_cfg_scale, 1.0):
                # TODO(ryand): Add option to run positive and negative predictions in a single batch for better performance
                # on systems with sufficient VRAM.

                if neg_regional_prompting_extension is None:
                    raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

                # For negative prediction with Kontext, we need to include the reference images
                # to maintain consistency between positive and negative passes. Without this,
                # CFG would create artifacts as the attention mechanism would see different
                # spatial structures in each pass
                neg_img_input = img
                neg_img_input_ids = img_ids

                # Add channel-wise conditioning for negative pass if present
                if img_cond is not None:
                    neg_img_input = torch.cat((neg_img_input, img_cond), dim=-1)

                # Add sequence-wise conditioning (Kontext) for negative pass
                # This ensures reference images are processed consistently
                if img_cond_seq is not None:
                    neg_img_input = torch.cat((neg_img_input, img_cond_seq), dim=1)
                    neg_img_input_ids = torch.cat((neg_img_input_ids, img_cond_seq_ids), dim=1)

                neg_pred = model(
                    img=neg_img_input,
                    img_ids=neg_img_input_ids,
                    txt=neg_regional_prompting_extension.regional_text_conditioning.t5_embeddings,
                    txt_ids=neg_regional_prompting_extension.regional_text_conditioning.t5_txt_ids,
                    y=neg_regional_prompting_extension.regional_text_conditioning.clip_embeddings,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    timestep_index=step_index,
                    total_num_timesteps=total_steps,
                    controlnet_double_block_residuals=None,
                    controlnet_single_block_residuals=None,
                    ip_adapter_extensions=neg_ip_adapter_extensions,
                    regional_prompting_extension=neg_regional_prompting_extension,
                )

                # Slice negative prediction to match main image tokens
                if img_cond_seq is not None:
                    neg_pred = neg_pred[:, :original_seq_len]
                pred = neg_pred + step_cfg_scale * (pred - neg_pred)

            preview_img = img - t_curr * pred
            img = img + (t_prev - t_curr) * pred

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

    finally:
        # DyPE: Restore original position embedder
        if original_pe_embedder is not None:
            DyPEExtension.restore_model(model, original_pe_embedder)
