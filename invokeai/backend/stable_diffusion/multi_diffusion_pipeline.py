from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, List, Optional

import torch

from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    AddsMaskGuidance,
    ControlNetData,
    PipelineIntermediateState,
    StableDiffusionGeneratorPipeline,
    T2IAdapterData,
    is_inpainting_model,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import IPAdapterData, TextConditioningData
from invokeai.backend.stable_diffusion.diffusion.unet_attention_patcher import UNetAttentionPatcher, UNetIPAdapterData


class MultiDiffusionPipeline(StableDiffusionGeneratorPipeline):
    """A Stable Diffusion pipeline that uses Multi-Diffusion (https://arxiv.org/pdf/2302.08113) for denoising."""

    # Plan:
    # - latents_from_embeddings(...) will accept all of the same global params, but the "local" params will be bundled
    #     together with tile locations.
    # - What is "local"?:
    #   - conditioning_data could be local, but for upscaling will be global
    #   - control_data makes more sense as global, then we split it up as we split up the latents
    #   - ip_adapter_data sort of has 3 modes to consider:
    #     - global style: applied in the same way to all tiles
    #     - local style: apply different IP-Adapters to each tile
    #     - global structure: we want to crop the input image and run the IP-Adapter on each separately
    #   - t2i_adapter_data won't be supported at first - it's not popular enough
    #   - All the inpainting params are global and need to be cropped accordingly
    # - Local:
    #  - latents
    #  - conditioning_data
    #  - noise
    #  - control_data
    #  - ip_adapter_data (skip for now)
    #  - t2i_adapter_data (skip for now)
    #  - mask
    #  - masked_latents
    #  - is_gradient_mask ???
    # - Can we support inpainting models in this node?
    #   - TBD, need to think about this more
    # - step(...) remains mostly unmodified, is not overriden in this sub-class.
    # - May need a cleaner AddsMaskGuidance implementation to handle this plan... we'll see.
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
