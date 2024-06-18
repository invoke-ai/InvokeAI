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

# The maximum number of regions with compatible sizes that will be batched together.
# Larger batch sizes improve speed, but require more device memory.
MAX_REGION_BATCH_SIZE = 4


@dataclass
class MultiDiffusionRegionConditioning:
    # Region coords in latent space.
    region: TBLR
    text_conditioning_data: TextConditioningData
    control_data: list[ControlNetData]


class MultiDiffusionPipeline(StableDiffusionGeneratorPipeline):
    """A Stable Diffusion pipeline that uses Multi-Diffusion (https://arxiv.org/pdf/2302.08113) for denoising."""

    def _split_into_region_batches(
        self, multi_diffusion_conditioning: list[MultiDiffusionRegionConditioning]
    ) -> list[list[MultiDiffusionRegionConditioning]]:
        # Group the regions by shape. Only regions with the same shape can be batched together.
        conditioning_by_shape: dict[tuple[int, int], list[MultiDiffusionRegionConditioning]] = {}
        for region_conditioning in multi_diffusion_conditioning:
            shape_hw = (
                region_conditioning.region.bottom - region_conditioning.region.top,
                region_conditioning.region.right - region_conditioning.region.left,
            )
            # In python, a tuple of hashable objects is hashable, so can be used as a key in a dict.
            if shape_hw not in conditioning_by_shape:
                conditioning_by_shape[shape_hw] = []
            conditioning_by_shape[shape_hw].append(region_conditioning)

        # Split the regions into batches, respecting the MAX_REGION_BATCH_SIZE constraint.
        region_conditioning_batches = []
        for region_conditioning_batch in conditioning_by_shape.values():
            for i in range(0, len(region_conditioning_batch), MAX_REGION_BATCH_SIZE):
                region_conditioning_batches.append(region_conditioning_batch[i : i + MAX_REGION_BATCH_SIZE])

        return region_conditioning_batches

    def _check_regional_prompting(self, multi_diffusion_conditioning: list[MultiDiffusionRegionConditioning]):
        """Check the input conditioning and confirm that regional prompting is not used."""
        for region_conditioning in multi_diffusion_conditioning:
            if (
                region_conditioning.text_conditioning_data.cond_regions is not None
                or region_conditioning.text_conditioning_data.uncond_regions is not None
            ):
                raise NotImplementedError("Regional prompting is not yet supported in Multi-Diffusion.")

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
        self._check_regional_prompting(multi_diffusion_conditioning)

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
        # For now, we assume that each region has the same weight (1.0).
        region_weight_mask = torch.zeros(
            (1, 1, latent_height, latent_width), device=latents.device, dtype=latents.dtype
        )
        for region_conditioning in multi_diffusion_conditioning:
            region = region_conditioning.region
            region_weight_mask[:, :, region.top : region.bottom, region.left : region.right] += 1.0

        # Group the region conditioning into batches for faster processing.
        # region_conditioning_batches[b][r] is the r'th region in the b'th batch.
        region_conditioning_batches = self._split_into_region_batches(multi_diffusion_conditioning)

        # Many of the diffusers schedulers are stateful (i.e. they update internal state in each call to step()). Since
        # we are calling step() multiple times at the same timestep (once for each region batch), we must maintain a
        # separate scheduler state for each region batch.
        region_batch_schedulers: list[SchedulerMixin] = [
            copy.deepcopy(self.scheduler) for _ in region_conditioning_batches
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
            for region_batch_idx, region_conditioning_batch in enumerate(region_conditioning_batches):
                # Switch to the scheduler for the region batch.
                self.scheduler = region_batch_schedulers[region_batch_idx]

                # TODO(ryand): This logic has not yet been tested with input latents with a batch_size > 1.

                # Prepare the latents for the region batch.
                batch_latents = torch.cat(
                    [
                        latents[
                            :,
                            :,
                            region_conditioning.region.top : region_conditioning.region.bottom,
                            region_conditioning.region.left : region_conditioning.region.right,
                        ]
                        for region_conditioning in region_conditioning_batch
                    ],
                )

                # TODO(ryand): Do we have to repeat the text_conditioning_data to match the batch size? Or does step()
                # handle broadcasting properly?

                # TODO(ryand): Resume here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # Run the denoising step on the region.
                step_output = self.step(
                    t=batched_t,
                    latents=batch_latents,
                    conditioning_data=region_conditioning.text_conditioning_data,
                    step_index=i,
                    total_step_count=total_step_count,
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    mask_guidance=None,
                    mask=None,
                    masked_latents=None,
                    control_data=region_conditioning.control_data,
                )
                # Run a denoising step on the region.
                # step_output = self._region_step(
                #     region_conditioning=region_conditioning,
                #     t=batched_t,
                #     latents=latents,
                #     step_index=i,
                #     total_step_count=len(timesteps),
                #     scheduler_step_kwargs=scheduler_step_kwargs,
                # )

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
    def _region_batch_step(
        self,
        region_conditioning: MultiDiffusionRegionConditioning,
        t: torch.Tensor,
        latents: torch.Tensor,
        step_index: int,
        total_step_count: int,
        scheduler_step_kwargs: dict[str, Any],
    ):
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
