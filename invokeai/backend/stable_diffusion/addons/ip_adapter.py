from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch
from pydantic import Field

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import TextConditioningData
from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
from .base import AddonBase


@dataclass
class IPAdapterAddon(AddonBase):
    model: IPAdapter
    conditioning: IPAdapterConditioningInfo
    mask: torch.Tensor
    target_blocks: List[str]
    weight: Union[float, List[float]] = 1.0
    begin_step_percent: float = 0.0
    end_step_percent: float = 1.0

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
        last_step = math.ceil(self.end_step_percent * total_steps)
        if step_index < first_step or step_index > last_step:
            return

        weight = self.weight
        if isinstance(weight, List):
            weight = weight[step_index]

        if conditioning_mode == "both":
            embeds = torch.stack([self.conditioning.uncond_image_prompt_embeds, self.conditioning.cond_image_prompt_embeds])
        elif conditioning_mode == "negative":
            embeds = torch.stack([self.conditioning.uncond_image_prompt_embeds])
        else: # elif conditioning_mode == "positive":
            embeds = torch.stack([self.conditioning.cond_image_prompt_embeds])


        cross_attention_kwargs = unet_kwargs.get("cross_attention_kwargs", None)
        if cross_attention_kwargs is None:
            cross_attention_kwargs = dict()
            unet_kwargs.update(dict(cross_attention_kwargs=cross_attention_kwargs))


        regional_ip_data = cross_attention_kwargs.get("regional_ip_data", None)
        if regional_ip_data is None:
            regional_ip_data = RegionalIPData(
                image_prompt_embeds=[],
                scales=[],
                masks=[],
                dtype=sample.dtype,
                device=sample.device,
            )
            cross_attention_kwargs.update(dict(
                regional_ip_data=regional_ip_data,
            ))


        regional_ip_data.add(
            embeds=embeds,
            scale=weight,
            mask=self.mask,
        )
