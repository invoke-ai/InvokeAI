from dataclasses import dataclass
import torch
from typing import Callable, Optional
from .base import ExtensionBase, modifier
from ..denoise_context import DenoiseContext


@dataclass
class PipelineIntermediateState:
    step: int
    order: int
    total_steps: int
    timestep: int
    latents: torch.Tensor
    predicted_original: Optional[torch.Tensor] = None


class PreviewExt(ExtensionBase):
    def __init__(self, callback: Callable[[PipelineIntermediateState], None], priority: int):
        super().__init__(priority=priority)
        self.callback = callback

    # do last so that all other changes shown
    @modifier("pre_denoise_loop", order="last")
    def initial_preview(self, ctx: DenoiseContext):
        self.callback(
            PipelineIntermediateState(
                step=-1,
                order=ctx.scheduler.order,
                total_steps=len(ctx.timesteps),
                timestep=int(ctx.scheduler.config.num_train_timesteps), # TODO: is there any code which uses it?
                latents=ctx.latents,
            )
        )

    # do last so that all other changes shown
    @modifier("post_step", order="last")
    def step_preview(self, ctx: DenoiseContext):
        if hasattr(ctx.step_output, "denoised"):
            predicted_original = ctx.step_output.denoised
        elif hasattr(ctx.step_output, "pred_original_sample"):
            predicted_original = ctx.step_output.pred_original_sample
        else:
            predicted_original = ctx.step_output.prev_sample


        self.callback(
            PipelineIntermediateState(
                step=ctx.step_index,
                order=ctx.scheduler.order,
                total_steps=len(ctx.timesteps),
                timestep=int(ctx.timestep), # TODO: is there any code which uses it?
                latents=ctx.step_output.prev_sample,
                predicted_original=predicted_original, # TODO: is there any reason for additional field?
            )
        )
