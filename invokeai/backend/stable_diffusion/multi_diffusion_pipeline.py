from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch

from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    PipelineIntermediateState,
    StableDiffusionGeneratorPipeline,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import TextConditioningData
from invokeai.backend.stable_diffusion.diffusion.unet_attention_patcher import UNetAttentionPatcher
from invokeai.backend.tiles.utils import Tile


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
        regions: list[Tile],
        latents: torch.Tensor,
        scheduler_step_kwargs: dict[str, Any],
        conditioning_data: TextConditioningData,
        noise: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        init_timestep: torch.Tensor,
        callback: Callable[[PipelineIntermediateState], None],
        control_data: list[ControlNetData] | None = None,
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

        use_regional_prompting = (
            conditioning_data.cond_regions is not None or conditioning_data.uncond_regions is not None
        )
        unet_attention_patcher = None
        attn_ctx = nullcontext()

        if use_regional_prompting:
            unet_attention_patcher = UNetAttentionPatcher(ip_adapter_data=None)
            attn_ctx = unet_attention_patcher.apply_ip_adapter_attention(self.invokeai_diffuser.model)

        # Populate a weighted mask that will be used to combine the results from each region after every step.
        # For now, we assume that each regions has the same weight (1.0).
        region_weight_mask = torch.zeros(
            (1, 1, latent_height, latent_width), device=latents.device, dtype=latents.dtype
        )
        for region in regions:
            region_weight_mask[
                :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
            ] += 1.0

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

                prev_samples_by_region: list[torch.Tensor] = []
                pred_original_by_region: list[torch.Tensor | None] = []
                for region in regions:
                    # Run a denoising step on the region.
                    step_output = self._region_step(
                        region=region,
                        t=batched_t,
                        latents=latents,
                        conditioning_data=conditioning_data,
                        step_index=i,
                        total_step_count=len(timesteps),
                        scheduler_step_kwargs=scheduler_step_kwargs,
                        control_data=control_data,
                    )
                    prev_samples_by_region.append(step_output.prev_sample)
                    pred_original_by_region.append(getattr(step_output, "pred_original_sample", None))

                # Merge the prev_sample results from each region.
                merged_latents = torch.zeros_like(latents)
                for region_idx, region in enumerate(regions):
                    merged_latents[
                        :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
                    ] += prev_samples_by_region[region_idx]
                latents = merged_latents / region_weight_mask

                # Merge the predicted_original results from each region.
                predicted_original = None
                if all(pred_original_by_region):
                    merged_pred_original = torch.zeros_like(latents)
                    for region_idx, region in enumerate(regions):
                        merged_pred_original[
                            :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
                        ] += pred_original_by_region[region_idx]
                    predicted_original = merged_pred_original / region_weight_mask

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
        region: Tile,
        t: torch.Tensor,
        latents: torch.Tensor,
        conditioning_data: TextConditioningData,
        step_index: int,
        total_step_count: int,
        scheduler_step_kwargs: dict[str, Any],
        control_data: list[ControlNetData] | None = None,
    ):
        # Crop the inputs to the region.
        region_latents = latents[
            :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
        ]

        region_control_data: list[ControlNetData] | None = None
        if control_data is not None:
            region_control_data = [self._crop_controlnet_data(c, region) for c in control_data]

        # Run the denoising step on the region.
        return self.step(
            t=t,
            latents=region_latents,
            conditioning_data=conditioning_data,
            step_index=step_index,
            total_step_count=total_step_count,
            scheduler_step_kwargs=scheduler_step_kwargs,
            mask_guidance=None,
            mask=None,
            masked_latents=None,
            control_data=region_control_data,
        )

    def _crop_controlnet_data(self, control_data: ControlNetData, region: Tile) -> ControlNetData:
        """Crop a ControlNetData object to a region."""
        # Create a shallow copy of the control_data object.
        control_data_copy = copy.copy(control_data)
        # The ControlNet reference image is the only attribute that needs to be cropped.
        control_data_copy.image_tensor = control_data.image_tensor[
            :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
        ]
        return control_data_copy
