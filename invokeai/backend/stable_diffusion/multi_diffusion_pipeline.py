from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    PipelineIntermediateState,
    StableDiffusionGeneratorPipeline,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import TextConditioningData
from invokeai.backend.tiles.utils import TBLR


@dataclass
class MultiDiffusionRegionConditioning:
    # Region coords in latent space.
    region: TBLR
    text_conditioning_data: TextConditioningData
    control_data: list[ControlNetData]


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
    def multi_diffusion_denoise(
        self,
        multi_diffusion_conditioning: list[MultiDiffusionRegionConditioning],
        latents: torch.Tensor,
        scheduler_step_kwargs: dict[str, Any],
        noise: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        init_timestep: torch.Tensor,
        callback: Callable[[PipelineIntermediateState], None],
    ) -> torch.Tensor:
        # TODO(ryand): Figure out why this condition is necessary, and document it. My guess is that it's to handle
        # cases where densoisings_start and denoising_end are set such that there are no timesteps.
        if init_timestep.shape[0] == 0 or timesteps.shape[0] == 0:
            return latents

        batch_size, _, latent_height, latent_width = latents.shape
        batched_init_timestep = init_timestep.expand(batch_size)

        # noise can be None if the latents have already been noised (e.g. when running the SDXL refiner).
        if noise is not None:
            # TODO(ryand): I'm pretty sure we should be applying init_noise_sigma in cases where we are starting with
            # full noise. Investigate the history of why this got commented out.
            # latents = noise * self.scheduler.init_noise_sigma # it's like in t2l according to diffusers
            latents = self.scheduler.add_noise(latents, noise, batched_init_timestep)

        # TODO(ryand): Look into the implications of passing in latents here that are larger than they will be after
        # cropping into regions.
        self._adjust_memory_efficient_attention(latents)

        # Populate a weighted mask that will be used to combine the results from each region after every step.
        # For now, we assume that each regions has the same weight (1.0).
        region_weight_mask = torch.zeros(
            (1, 1, latent_height, latent_width), device=latents.device, dtype=latents.dtype
        )
        for region_conditioning in multi_diffusion_conditioning:
            region = region_conditioning.region
            region_weight_mask[:, :, region.top : region.bottom, region.left : region.right] += 1.0

        # Many of the diffusers schedulers are stateful (i.e. they update internal state in each call to step()). Since
        # we are calling step() multiple times at the same timestep (once for each region batch), we must maintain a
        # separate scheduler state for each region batch.
        region_batch_schedulers: list[SchedulerMixin] = [
            copy.copy(self.scheduler) for _ in multi_diffusion_conditioning
        ]

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

            merged_latents = torch.zeros_like(latents)
            merged_pred_original: torch.Tensor | None = None
            for region_idx, region_conditioning in enumerate(multi_diffusion_conditioning):
                # Switch to the scheduler for the region batch.
                self.scheduler = region_batch_schedulers[region_idx]

                # Run a denoising step on the region.
                step_output = self._region_step(
                    region_conditioning=region_conditioning,
                    t=batched_t,
                    latents=latents,
                    step_index=i,
                    total_step_count=len(timesteps),
                    scheduler_step_kwargs=scheduler_step_kwargs,
                )

                # Store the results from the region.
                region = region_conditioning.region
                merged_latents[:, :, region.top : region.bottom, region.left : region.right] += step_output.prev_sample
                pred_orig_sample = getattr(step_output, "pred_original_sample", None)
                if pred_orig_sample is not None:
                    # If one region has pred_original_sample, then we can assume that all regions will have it, because
                    # they all use the same scheduler.
                    if merged_pred_original is None:
                        merged_pred_original = torch.zeros_like(latents)
                    merged_pred_original[:, :, region.top : region.bottom, region.left : region.right] += (
                        pred_orig_sample
                    )

            # Normalize the merged results.
            latents = torch.where(region_weight_mask > 0, merged_latents / region_weight_mask, merged_latents)
            predicted_original = None
            if merged_pred_original is not None:
                predicted_original = torch.where(
                    region_weight_mask > 0, merged_pred_original / region_weight_mask, merged_pred_original
                )

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

        return latents

    @torch.inference_mode()
    def _region_step(
        self,
        region_conditioning: MultiDiffusionRegionConditioning,
        t: torch.Tensor,
        latents: torch.Tensor,
        step_index: int,
        total_step_count: int,
        scheduler_step_kwargs: dict[str, Any],
    ):
        use_regional_prompting = (
            region_conditioning.text_conditioning_data.cond_regions is not None
            or region_conditioning.text_conditioning_data.uncond_regions is not None
        )
        if use_regional_prompting:
            raise NotImplementedError("Regional prompting is not yet supported in Multi-Diffusion.")

        # Crop the inputs to the region.
        region_latents = latents[
            :,
            :,
            region_conditioning.region.top : region_conditioning.region.bottom,
            region_conditioning.region.left : region_conditioning.region.right,
        ]

        # Run the denoising step on the region.
        return self.step(
            t=t,
            latents=region_latents,
            conditioning_data=region_conditioning.text_conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            scheduler_step_kwargs=scheduler_step_kwargs,
            mask_guidance=None,
            mask=None,
            masked_latents=None,
            control_data=region_conditioning.control_data,
        )
