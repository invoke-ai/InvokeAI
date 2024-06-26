from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch
from pydantic import Field

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import TextConditioningData
from .base import AddonBase


@dataclass
class T2IAdapterAddon(AddonBase):
    adapter_state: List[torch.Tensor] = Field() # TODO:  why here was dict before
    weight: Union[float, List[float]] = Field(default=1.0)
    begin_step_percent: float = Field(default=0.0)
    end_step_percent: float = Field(default=1.0)

    def pre_unet_step(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        step_index: int,
        total_steps: int,
        conditioning_data: TextConditioningData,

        unet_kwargs: Dict[str, Any],
        conditioning_mode: str,
    ):
        # skip if model not active in current step
        first_step = math.floor(self.begin_step_percent * total_steps)
        last_step  = math.ceil(self.end_step_percent * total_steps)
        if step_index < first_step or step_index > last_step:
            return

        weight = self.weight
        if isinstance(weight, list):
            weight = weight[step_index]

        # TODO: conditioning_mode?
        down_intrablock_additional_residuals = unet_kwargs.get("down_intrablock_additional_residuals", None)
        if down_intrablock_additional_residuals is None:
            down_intrablock_additional_residuals = [v * weight for v in self.adapter_state]
        else:
            for i, value in enumerate(self.adapter_state):
                down_intrablock_additional_residuals[i] += value * weight

        unet_kwargs.update(dict(
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
        ))
