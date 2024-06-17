from __future__ import annotations

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

        batch_size = latents.shape[0]
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
                    mask_guidance=None,
                    mask=None,
                    masked_latents=None,
                    control_data=control_data,
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

        return latents
