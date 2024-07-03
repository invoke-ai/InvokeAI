import torch
import math
from typing import List, Union
from .base import ExtensionBase, modifier
from ..denoise_context import DenoiseContext


class T2IAdapterExt(ExtensionBase):
    adapter_state: List[torch.Tensor]
    weight: Union[float, List[float]]
    begin_step_percent: float
    end_step_percent: float

    def __init__(
        self,
        adapter_state: List[torch.Tensor],
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
        priority: int,
    ):
        super().__init__(priority=priority)
        self.adapter_state = adapter_state
        self.weight = weight
        self.begin_step_percent = begin_step_percent
        self.end_step_percent = end_step_percent

    @modifier("pre_unet_forward")
    def pre_unet_step(self, ctx: DenoiseContext):
        # skip if model not active in current step
        total_steps = len(ctx.timesteps)
        first_step = math.floor(self.begin_step_percent * total_steps)
        last_step  = math.ceil(self.end_step_percent * total_steps)
        if ctx.step_index < first_step or ctx.step_index > last_step:
            return

        weight = self.weight
        if isinstance(weight, list):
            weight = weight[ctx.step_index]

        # TODO: conditioning_mode?
        if ctx.unet_kwargs.down_intrablock_additional_residuals is None:
            ctx.unet_kwargs.down_intrablock_additional_residuals = [v * weight for v in self.adapter_state]
        else:
            for i, value in enumerate(self.adapter_state):
                ctx.unet_kwargs.down_intrablock_additional_residuals[i] += value * weight
