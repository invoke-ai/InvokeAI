import inspect
import math
from typing import Callable

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm

from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput, sum_controlnet_flux_outputs
from invokeai.backend.flux.extensions.dype_extension import DyPEExtension
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.extensions.xlabs_controlnet_extension import XLabsControlNetExtension
from invokeai.backend.flux.extensions.xlabs_ip_adapter_extension import XLabsIPAdapterExtension
from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.schedulers import is_flow_match_lcm_scheduler, set_heun_timesteps_from_sigmas
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.util.devices import TorchDevice


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
    # Defaults to Diffusers Euler when omitted.
    scheduler: SchedulerMixin | None = None,
):
    if len(timesteps) <= 1:
        return img
    if scheduler is None:
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    # Initialize scheduler with timesteps
    # The timesteps list contains values in [0, 1] range (sigmas)
    # LCM should use num_inference_steps (it has its own sigma schedule),
    # while other schedulers can use custom sigmas if supported
    is_lcm = is_flow_match_lcm_scheduler(scheduler)
    set_timesteps_sig = inspect.signature(scheduler.set_timesteps)
    if isinstance(scheduler, FlowMatchHeunDiscreteScheduler):
        set_heun_timesteps_from_sigmas(scheduler, timesteps[:-1], timesteps[-1], img.device)
    elif not is_lcm and "sigmas" in set_timesteps_sig.parameters:
        # InvokeAI supplies model-evaluation sigmas plus its requested terminal endpoint.
        scheduler.set_timesteps(sigmas=timesteps[:-1], device=img.device)
        if hasattr(scheduler, "sigmas"):
            scheduler.sigmas[-1] = torch.as_tensor(
                timesteps[-1], dtype=scheduler.sigmas.dtype, device=scheduler.sigmas.device
            )
    else:
        # LCM or scheduler doesn't support custom sigmas - use num_inference_steps
        # The schedule will be computed by the scheduler itself. For LCM,
        # clipping controls the requested step count but not exact start/end
        # sigmas; keep that scheduler-defined behavior intact.
        #
        # Important for img2img callers: if the initial latent/noise blend was
        # computed from a separate pre-scheduler schedule, that preblend may not
        # match this scheduler's true first step exactly.
        num_inference_steps = len(timesteps) - 1
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=img.device)

    # For schedulers like Heun, the number of actual steps may differ
    # (Heun doubles timesteps internally)
    num_scheduler_steps = len(scheduler.timesteps)
    # For user-facing step count, use the original number of denoising steps
    total_steps = len(timesteps) - 1

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

        # Use diffusers scheduler for stepping
        # Use tqdm with total_steps (user-facing steps) not num_scheduler_steps (internal steps)
        # This ensures progress bar shows 1/8, 2/8, etc. even when scheduler uses more internal steps
        pbar = tqdm(total=total_steps, desc=f"Denoising{TorchDevice.get_session_device_label()}")
        for step_index in range(num_scheduler_steps):
            timestep = scheduler.timesteps[step_index]
            # Convert scheduler timestep (0-1000) to normalized (0-1) for the model
            t_curr = timestep.item() / scheduler.config.num_train_timesteps
            dype_sigma = DyPEExtension.resolve_step_sigma(
                fallback_sigma=t_curr,
                step_index=step_index,
                scheduler_sigmas=getattr(scheduler, "sigmas", None),
            )
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            # DyPE: Update step state for timestep-dependent scaling
            if dype_extension is not None and dype_embedder is not None:
                dype_extension.update_step_state(
                    embedder=dype_embedder,
                    sigma=dype_sigma,
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

            # Use the sample and sigma from this model evaluation. This is also
            # the clean prediction used by Heun and FlowMatch LCM internally.
            should_emit_preview = (
                not is_heun or not in_first_order or step_index == num_scheduler_steps - 1
            ) and user_step < total_steps
            preview_img = img - t_curr * pred if should_emit_preview else None

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
                if not in_first_order or step_index == num_scheduler_steps - 1:
                    # Second order step completed
                    user_step += 1
                    # Only call step_callback if we haven't exceeded total_steps
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
                                order=2 if not in_first_order else 1,
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
        return img

    finally:
        # DyPE: Restore original position embedder
        if original_pe_embedder is not None:
            DyPEExtension.restore_model(model, original_pe_embedder)
