from __future__ import annotations

import copy
import torch
from typing import TYPE_CHECKING, List, Union, Callable, Optional
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, override, modifier
from invokeai.backend.tiles.tiles import calc_tiles_min_overlap
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager


class TiledDenoiseExt(ExtensionBase):
    def __init__(
        self,
        tile_width: int,
        tile_height: int,
        tile_overlap: int,
        priority: int,
    ):
        super().__init__(priority=priority)
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.tile_overlap = tile_overlap

    @dataclass
    class FakeSchedulerOutput(SchedulerOutput): #BaseOutput
        #prev_sample: torch.Tensor
        pred_original_sample: Optional[torch.Tensor] = None

    @modifier("pre_denoise_loop")
    def init_tiles(self, ctx: DenoiseContext):
        _, _, latent_height, latent_width = ctx.latents.shape
        latent_tile_height = self.tile_height // LATENT_SCALE_FACTOR
        latent_tile_width = self.tile_width // LATENT_SCALE_FACTOR
        latent_tile_overlap = self.tile_overlap // LATENT_SCALE_FACTOR

        from invokeai.backend.tiles.tiles import calc_tiles_min_overlap
        self.tiles = calc_tiles_min_overlap(
            image_height=latent_height,
            image_width=latent_width,
            tile_height=latent_tile_height,
            tile_width=latent_tile_width,
            min_overlap=latent_tile_overlap,
        )

    @override("step")
    def tiled_step(self, orig_step: Callable, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        batch_size, _, latent_height, latent_width = ctx.latents.shape
        region_batch_schedulers: list[SchedulerMixin] = [
            copy.deepcopy(ctx.scheduler) for _ in self.tiles
        ]

        merged_latents = torch.zeros_like(ctx.latents)
        merged_latents_weights = torch.zeros(
            (1, 1, latent_height, latent_width), device=ctx.latents.device, dtype=ctx.latents.dtype
        )
        merged_pred_original: torch.Tensor | None = None
        for region_idx, tile_region in enumerate(self.tiles):
            # Crop the inputs to the region.
            region_latents = ctx.latents[
                :,
                :,
                tile_region.coords.top : tile_region.coords.bottom,
                tile_region.coords.left : tile_region.coords.right,
            ]

            region_ctx = DenoiseContext(**vars(ctx))
            region_ctx.latents = region_latents
            region_ctx.scheduler = region_batch_schedulers[region_idx]
            #region_ctx.conditioning_data = region_conditioning.text_conditioning_data
            region_ctx.extra["tile_coords"] = tile_region.coords

            # Run the denoising step on the region.
            step_output = orig_step(region_ctx, ext_manager)

            # Store the results from the region.
            # If two tiles overlap by more than the target overlap amount, crop the left and top edges of the
            # affected tiles to achieve the target overlap.
            target_overlap = self.tile_overlap // LATENT_SCALE_FACTOR
            top_adjustment = max(0, tile_region.overlap.top - target_overlap)
            left_adjustment = max(0, tile_region.overlap.left - target_overlap)
            region_height_slice = slice(tile_region.coords.top + top_adjustment, tile_region.coords.bottom)
            region_width_slice = slice(tile_region.coords.left + left_adjustment, tile_region.coords.right)
            merged_latents[:, :, region_height_slice, region_width_slice] += step_output.prev_sample[
                :, :, top_adjustment:, left_adjustment:
            ]
            # For now, we treat every region as having the same weight.
            merged_latents_weights[:, :, region_height_slice, region_width_slice] += 1.0

            # TODO: denoised
            pred_orig_sample = getattr(step_output, "pred_original_sample", None)
            if pred_orig_sample is not None:
                # If one region has pred_original_sample, then we can assume that all regions will have it, because
                # they all use the same scheduler.
                if merged_pred_original is None:
                    merged_pred_original = torch.zeros_like(ctx.latents)
                merged_pred_original[:, :, region_height_slice, region_width_slice] += pred_orig_sample[
                    :, :, top_adjustment:, left_adjustment:
                ]

        # Normalize the merged results.
        latents = torch.where(merged_latents_weights > 0, merged_latents / merged_latents_weights, merged_latents)
        # For debugging, uncomment this line to visualize the region seams:
        # latents = torch.where(merged_latents_weights > 1, 0.0, latents)
        predicted_original = None
        if merged_pred_original is not None:
            predicted_original = torch.where(
                merged_latents_weights > 0, merged_pred_original / merged_latents_weights, merged_pred_original
            )

        return self.FakeSchedulerOutput(
            prev_sample=latents,
            pred_original_sample=predicted_original,
        )
