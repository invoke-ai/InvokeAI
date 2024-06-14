from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Callable, List, Optional

import torch

from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import IPAdapterData, TextConditioningData
from invokeai.backend.stable_diffusion.diffusion.unet_attention_patcher import UNetAttentionPatcher, UNetIPAdapterData


class MultiDiffusionPipeline(StableDiffusionGeneratorPipeline):
    """A Stable Diffusion pipeline that uses Multi-Diffusion (https://arxiv.org/pdf/2302.08113) for denoising."""

    def latents_from_embeddings(
        self,
        latents: torch.Tensor,
        scheduler_step_kwargs: dict[str, Any],
        conditioning_data: TextConditioningData,
        noise: Optional[torch.Tensor],
        seed: int,
        timesteps: torch.Tensor,
        init_timestep: torch.Tensor,
        callback: Callable[[PipelineIntermediateState], None],
        control_data: list[ControlNetData] | None = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
        mask: Optional[torch.Tensor] = None,
        masked_latents: Optional[torch.Tensor] = None,
        is_gradient_mask: bool = False,
    ) -> torch.Tensor:
        # TODO(ryand): Figure out why this condition is necessary, and document it. My guess is that it's to handle
        # cases where densoisings_start and denoising_end are set such that there are no timesteps.
        if init_timestep.shape[0] == 0 or timesteps.shape[0] == 0:
            return latents

        orig_latents = latents.clone()

        batch_size = latents.shape[0]
        batched_init_timestep = init_timestep.expand(batch_size)

        # noise can be None if the latents have already been noised (e.g. when running the SDXL refiner).
        if noise is not None:
            # TODO(ryand): I'm pretty sure we should be applying init_noise_sigma in cases where we are starting with
            # full noise. Investigate the history of why this got commented out.
            # latents = noise * self.scheduler.init_noise_sigma # it's like in t2l according to diffusers
            latents = self.scheduler.add_noise(latents, noise, batched_init_timestep)

        self._adjust_memory_efficient_attention(latents)

        # Handle mask guidance (a.k.a. inpainting).
        mask_guidance: AddsMaskGuidance | None = None
        if mask is not None and not is_inpainting_model(self.unet):
            # We are doing inpainting, since a mask is provided, but we are not using an inpainting model, so we will
            # apply mask guidance to the latents.

            # 'noise' might be None if the latents have already been noised (e.g. when running the SDXL refiner).
            # We still need noise for inpainting, so we generate it from the seed here.
            if noise is None:
                noise = torch.randn(
                    orig_latents.shape,
                    dtype=torch.float32,
                    device="cpu",
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                ).to(device=orig_latents.device, dtype=orig_latents.dtype)

            mask_guidance = AddsMaskGuidance(
                mask=mask,
                mask_latents=orig_latents,
                scheduler=self.scheduler,
                noise=noise,
                is_gradient_mask=is_gradient_mask,
            )

        use_ip_adapter = ip_adapter_data is not None
        use_regional_prompting = (
            conditioning_data.cond_regions is not None or conditioning_data.uncond_regions is not None
        )
        unet_attention_patcher = None
        attn_ctx = nullcontext()

        if use_ip_adapter or use_regional_prompting:
            ip_adapters: Optional[List[UNetIPAdapterData]] = (
                [{"ip_adapter": ipa.ip_adapter_model, "target_blocks": ipa.target_blocks} for ipa in ip_adapter_data]
                if use_ip_adapter
                else None
            )
            unet_attention_patcher = UNetAttentionPatcher(ip_adapters)
            attn_ctx = unet_attention_patcher.apply_ip_adapter_attention(self.invokeai_diffuser.model)

        with attn_ctx:
            callback(
                PipelineIntermediateState(
                    step=-1,
                    order=self.scheduler.order,
                    total_steps=len(timesteps),
                    timestep=self.scheduler.config.num_train_timesteps,
                    latents=latents,
                )
            )

            for i, t in enumerate(self.progress_bar(timesteps)):
                batched_t = t.expand(batch_size)
                step_output = self.step(
                    t=batched_t,
                    latents=latents,
                    conditioning_data=conditioning_data,
                    step_index=i,
                    total_step_count=len(timesteps),
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    mask_guidance=mask_guidance,
                    mask=mask,
                    masked_latents=masked_latents,
                    control_data=control_data,
                    ip_adapter_data=ip_adapter_data,
                    t2i_adapter_data=t2i_adapter_data,
                )
                latents = step_output.prev_sample
                predicted_original = getattr(step_output, "pred_original_sample", None)

                callback(
                    PipelineIntermediateState(
                        step=i,
                        order=self.scheduler.order,
                        total_steps=len(timesteps),
                        timestep=int(t),
                        latents=latents,
                        predicted_original=predicted_original,
                    )
                )

        # restore unmasked part after the last step is completed
        # in-process masking happens before each step
        if mask is not None:
            if is_gradient_mask:
                latents = torch.where(mask > 0, latents, orig_latents)
            else:
                latents = torch.lerp(
                    orig_latents, latents.to(dtype=orig_latents.dtype), mask.to(dtype=orig_latents.dtype)
                )

        return latents

    @torch.inference_mode()
    def step(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        conditioning_data: TextConditioningData,
        step_index: int,
        total_step_count: int,
        scheduler_step_kwargs: dict[str, Any],
        mask_guidance: AddsMaskGuidance | None,
        mask: torch.Tensor | None,
        masked_latents: torch.Tensor | None,
        control_data: list[ControlNetData] | None = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
    ):
        # invokeai_diffuser has batched timesteps, but diffusers schedulers expect a single value
        timestep = t[0]

        # Handle masked image-to-image (a.k.a inpainting).
        if mask_guidance is not None:
            # NOTE: This is intentionally done *before* self.scheduler.scale_model_input(...).
            latents = mask_guidance(latents, timestep)

        # TODO: should this scaling happen here or inside self._unet_forward?
        #     i.e. before or after passing it to InvokeAIDiffuserComponent
        latent_model_input = self.scheduler.scale_model_input(latents, timestep)

        # Handle ControlNet(s)
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        if control_data is not None:
            down_block_additional_residuals, mid_block_additional_residual = self.invokeai_diffuser.do_controlnet_step(
                control_data=control_data,
                sample=latent_model_input,
                timestep=timestep,
                step_index=step_index,
                total_step_count=total_step_count,
                conditioning_data=conditioning_data,
            )

        # Handle T2I-Adapter(s)
        down_intrablock_additional_residuals = None
        if t2i_adapter_data is not None:
            accum_adapter_state = None
            for single_t2i_adapter_data in t2i_adapter_data:
                # Determine the T2I-Adapter weights for the current denoising step.
                first_t2i_adapter_step = math.floor(single_t2i_adapter_data.begin_step_percent * total_step_count)
                last_t2i_adapter_step = math.ceil(single_t2i_adapter_data.end_step_percent * total_step_count)
                t2i_adapter_weight = (
                    single_t2i_adapter_data.weight[step_index]
                    if isinstance(single_t2i_adapter_data.weight, list)
                    else single_t2i_adapter_data.weight
                )
                if step_index < first_t2i_adapter_step or step_index > last_t2i_adapter_step:
                    # If the current step is outside of the T2I-Adapter's begin/end step range, then set its weight to 0
                    # so it has no effect.
                    t2i_adapter_weight = 0.0

                # Apply the t2i_adapter_weight, and accumulate.
                if accum_adapter_state is None:
                    # Handle the first T2I-Adapter.
                    accum_adapter_state = [val * t2i_adapter_weight for val in single_t2i_adapter_data.adapter_state]
                else:
                    # Add to the previous adapter states.
                    for idx, value in enumerate(single_t2i_adapter_data.adapter_state):
                        accum_adapter_state[idx] += value * t2i_adapter_weight

            down_intrablock_additional_residuals = accum_adapter_state

        # Handle inpainting models.
        if is_inpainting_model(self.unet):
            # NOTE: These calls to add_inpainting_channels_to_latents(...) are intentionally done *after*
            # self.scheduler.scale_model_input(...) so that the scaling is not applied to the mask or reference image
            # latents.
            if mask is not None:
                if masked_latents is None:
                    raise ValueError("Source image required for inpaint mask when inpaint model used!")
                latent_model_input = self.add_inpainting_channels_to_latents(
                    latents=latent_model_input, masked_ref_image_latents=masked_latents, inpainting_mask=mask
                )
            else:
                # We are using an inpainting model, but no mask was provided, so we are not really "inpainting".
                # We generate a global mask and empty original image so that we can still generate in this
                # configuration.
                # TODO(ryand): Should we just raise an exception here instead? I can't think of a use case for wanting
                # to do this.
                # TODO(ryand): If we decide that there is a good reason to keep this, then we should generate the 'fake'
                # mask and original image once rather than on every denoising step.
                latent_model_input = self.add_inpainting_channels_to_latents(
                    latents=latent_model_input,
                    masked_ref_image_latents=torch.zeros_like(latent_model_input[:1]),
                    inpainting_mask=torch.ones_like(latent_model_input[:1, :1]),
                )

        uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
            sample=latent_model_input,
            timestep=t,  # TODO: debug how handled batched and non batched timesteps
            step_index=step_index,
            total_step_count=total_step_count,
            conditioning_data=conditioning_data,
            ip_adapter_data=ip_adapter_data,
            down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
            mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
        )

        guidance_scale = conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[step_index]

        noise_pred = self.invokeai_diffuser._combine(uc_noise_pred, c_noise_pred, guidance_scale)
        guidance_rescale_multiplier = conditioning_data.guidance_rescale_multiplier
        if guidance_rescale_multiplier > 0:
            noise_pred = self._rescale_cfg(
                noise_pred,
                c_noise_pred,
                guidance_rescale_multiplier,
            )

        # compute the previous noisy sample x_t -> x_t-1
        step_output = self.scheduler.step(noise_pred, timestep, latents, **scheduler_step_kwargs)

        # TODO: discuss injection point options. For now this is a patch to get progress images working with inpainting
        # again.
        if mask_guidance is not None:
            # Apply the mask to any "denoised" or "pred_original_sample" fields.
            if hasattr(step_output, "denoised"):
                step_output.pred_original_sample = mask_guidance(step_output.denoised, self.scheduler.timesteps[-1])
            elif hasattr(step_output, "pred_original_sample"):
                step_output.pred_original_sample = mask_guidance(
                    step_output.pred_original_sample, self.scheduler.timesteps[-1]
                )
            else:
                step_output.pred_original_sample = mask_guidance(latents, self.scheduler.timesteps[-1])

        return step_output
