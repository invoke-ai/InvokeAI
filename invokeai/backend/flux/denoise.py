import math
from typing import Callable

import torch
from tqdm import tqdm

from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput, sum_controlnet_flux_outputs
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
):
    # step 0 is the initial state
    total_steps = len(timesteps) - 1
    step_callback(
        PipelineIntermediateState(
            step=0,
            order=1,
            total_steps=total_steps,
            timestep=int(timesteps[0]),
            latents=img,
        ),
    )
    # guidance_vec is ignored for schnell.
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # Store original sequence length for slicing predictions
    original_seq_len = img.shape[1]

    for step_index, (t_curr, t_prev) in tqdm(list(enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)))):
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
