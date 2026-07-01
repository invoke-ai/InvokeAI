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
from invokeai.backend.tiles.utils import Tile


@dataclass
class MultiDiffusionRegionConditioning:
    # Region coords in latent space.
    region: Tile
    text_conditioning_data: TextConditioningData
    control_data: list[ControlNetData]


class MultiDiffusionPipeline(StableDiffusionGeneratorPipeline):
    """A Stable Diffusion pipeline that uses Multi-Diffusion (https://arxiv.org/pdf/2302.08113) for denoising."""

    def _check_regional_prompting(self, multi_diffusion_conditioning: list[MultiDiffusionRegionConditioning]):
        """Validate that regional conditioning is not used."""
        for region_conditioning in multi_diffusion_conditioning:
            if (
                region_conditioning.text_conditioning_data.cond_regions is not None
                or region_conditioning.text_conditioning_data.uncond_regions is not None
            ):
                raise NotImplementedError("Regional prompting is not yet supported in Multi-Diffusion.")

    def multi_diffusion_denoise(
        self,
        multi_diffusion_conditioning: list[MultiDiffusionRegionConditioning],
        target_overlap: int,
        latents: torch.Tensor,
        scheduler_step_kwargs: dict[str, Any],
        noise: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        init_timestep: torch.Tensor,
        callback: Callable[[PipelineIntermediateState], None],
    ) -> torch.Tensor:
        self._check_regional_prompting(multi_diffusion_conditioning)

        if init_timestep.shape[0] == 0:
            return latents

        batch_size, _, latent_height, latent_width = latents.shape
        batched_init_timestep = init_timestep.expand(batch_size)

        # noise can be None if the latents have already been noised (e.g. when running the SDXL refiner).
        if noise is not None:
            # TODO(ryand): I'm pretty sure we should be applying init_noise_sigma in cases where we are starting with
            # full noise. Investigate the history of why this got commented out.
            # latents = noise * self.scheduler.init_noise_sigma # it's like in t2l according to diffusers
            latents = self.scheduler.add_noise(latents, noise, batched_init_timestep)
            assert isinstance(latents, torch.Tensor)  # For static type checking.

        # TODO(ryand): Look into the implications of passing in latents here that are larger than they will be after
        # cropping into regions.
        self._adjust_memory_efficient_attention(latents)

        # Many of the diffusers schedulers are stateful (i.e. they update internal state in each call to step()). Since
        # we are calling step() multiple times at the same timestep (once for each region batch), we must maintain a
        # separate scheduler state for each region batch.
        # TODO(ryand): This solution allows all schedulers to **run**, but does not fully solve the issue of scheduler
        # statefulness. Some schedulers store previous model outputs in their state, but these values become incorrect
        # as Multi-Diffusion blending is applied (e.g. the PNDMScheduler). This can result in a blurring effect when
        # multiple MultiDiffusion regions overlap. Solving this properly would require a case-by-case review of each
        # scheduler to determine how it's state needs to be updated for compatibilty with Multi-Diffusion.
        region_batch_schedulers: list[SchedulerMixin] = [
            copy.deepcopy(self.scheduler) for _ in multi_diffusion_conditioning
        ]

        callback(
            PipelineIntermediateState(
                step=0,
                order=self.scheduler.order,
                total_steps=len(timesteps),
                timestep=self.scheduler.config.num_train_timesteps,
                latents=latents,
            )
        )

        for i, t in enumerate(self.progress_bar(timesteps)):
            batched_t = t.expand(batch_size)

            merged_latents = torch.zeros_like(latents)
            merged_latents_weights = torch.zeros(
                (1, 1, latent_height, latent_width), device=latents.device, dtype=latents.dtype
            )
            merged_pred_original: torch.Tensor | None = None
            for region_idx, region_conditioning in enumerate(multi_diffusion_conditioning):
                # Switch to the scheduler for the region batch.
                self.scheduler = region_batch_schedulers[region_idx]

                # Crop the inputs to the region.
                region_latents = latents[
                    :,
                    :,
                    region_conditioning.region.coords.top : region_conditioning.region.coords.bottom,
                    region_conditioning.region.coords.left : region_conditioning.region.coords.right,
                ]

                # Run the denoising step on the region.
                step_output = self.step(
                    t=batched_t,
                    latents=region_latents,
                    conditioning_data=region_conditioning.text_conditioning_data,
                    step_index=i,
                    total_step_count=len(timesteps),
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    mask_guidance=None,
                    mask=None,
                    masked_latents=None,
                    control_data=region_conditioning.control_data,
                )

                # Build a region_weight matrix that applies gradient blending to the edges of the region.
                region = region_conditioning.region
                _, _, region_height, region_width = step_output.prev_sample.shape
                region_weight = torch.ones(
                    (1, 1, region_height, region_width),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                if region.overlap.left > 0:
                    left_grad = torch.linspace(
                        0, 1, region.overlap.left, device=latents.device, dtype=latents.dtype
                    ).view((1, 1, 1, -1))
                    region_weight[:, :, :, : region.overlap.left] *= left_grad
                if region.overlap.top > 0:
                    top_grad = torch.linspace(
                        0, 1, region.overlap.top, device=latents.device, dtype=latents.dtype
                    ).view((1, 1, -1, 1))
                    region_weight[:, :, : region.overlap.top, :] *= top_grad
                if region.overlap.right > 0:
                    right_grad = torch.linspace(
                        1, 0, region.overlap.right, device=latents.device, dtype=latents.dtype
                    ).view((1, 1, 1, -1))
                    region_weight[:, :, :, -region.overlap.right :] *= right_grad
                if region.overlap.bottom > 0:
                    bottom_grad = torch.linspace(
                        1, 0, region.overlap.bottom, device=latents.device, dtype=latents.dtype
                    ).view((1, 1, -1, 1))
                    region_weight[:, :, -region.overlap.bottom :, :] *= bottom_grad

                # Update the merged results with the region results.
                merged_latents[
                    :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
                ] += step_output.prev_sample * region_weight
                merged_latents_weights[
                    :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
                ] += region_weight

                pred_orig_sample = getattr(step_output, "pred_original_sample", None)
                if pred_orig_sample is not None:
                    # If one region has pred_original_sample, then we can assume that all regions will have it, because
                    # they all use the same scheduler.
                    if merged_pred_original is None:
                        merged_pred_original = torch.zeros_like(latents)
                    merged_pred_original[
                        :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
                    ] += pred_orig_sample

            # Normalize the merged results.
            latents = torch.where(merged_latents_weights > 0, merged_latents / merged_latents_weights, merged_latents)
            # For debugging, uncomment this line to visualize the region seams:
            # latents = torch.where(merged_latents_weights > 1, 0.0, latents)
            predicted_original = None
            if merged_pred_original is not None:
                predicted_original = torch.where(
                    merged_latents_weights > 0, merged_pred_original / merged_latents_weights, merged_pred_original
                )

            callback(
                PipelineIntermediateState(
                    step=i + 1,
                    order=self.scheduler.order,
                    total_steps=len(timesteps),
                    timestep=int(t),
                    latents=latents,
                    predicted_original=predicted_original,
                )
            )

        return latents
