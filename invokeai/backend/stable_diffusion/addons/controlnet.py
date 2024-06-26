from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch
from pydantic import Field

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import TextConditioningData
from invokeai.backend.util.hotfixes import ControlNetModel
from .base import AddonBase


@dataclass
class ControlNetAddon(AddonBase):
    model: ControlNetModel = Field(default=None)
    image_tensor: torch.Tensor = Field(default=None)
    weight: Union[float, List[float]] = Field(default=1.0)
    begin_step_percent: float = Field(default=0.0)
    end_step_percent: float = Field(default=1.0)
    control_mode: str = Field(default="balanced")
    resize_mode: str = Field(default="just_resize")

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

        # convert mode to internal flags
        soft_injection = self.control_mode in ["more_prompt", "more_control"]
        cfg_injection  = self.control_mode in ["more_control", "unbalanced"]

        # skip, as negative not runned in cfg_injection mode
        if cfg_injection and conditioning_mode == "negative":
            return

        cn_unet_kwargs = dict(
            cross_attention_kwargs=dict(
                percent_through=step_index / total_steps,
            )
        )

        if conditioning_mode == "both":
            if cfg_injection:
                conditioning_data.to_unet_kwargs(cn_unet_kwargs, conditioning_mode="positive")

                down_samples, mid_sample = self._run(
                    sample=sample,
                    timestep=timestep,
                    step_index=step_index,
                    guess_mode=soft_injection,
                    unet_kwargs=cn_unet_kwargs,
                )
                # add zeros as samples for negative conditioning
                down_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_samples]
                mid_sample = torch.cat([torch.zeros_like(mid_sample), mid_sample])

            else:
                conditioning_data.to_unet_kwargs(cn_unet_kwargs, conditioning_mode="both")
                down_samples, mid_sample = self._run(
                    sample=torch.cat([sample] * 2),
                    timestep=timestep,
                    step_index=step_index,
                    guess_mode=soft_injection,
                    unet_kwargs=cn_unet_kwargs,
                )

        else: # elif in ["negative", "positive"]:
            conditioning_data.to_unet_kwargs(cn_unet_kwargs, conditioning_mode=conditioning_mode)

            down_samples, mid_sample = self._run(
                sample=sample,
                timestep=timestep,
                step_index=step_index,
                guess_mode=soft_injection,
                unet_kwargs=cn_unet_kwargs,
            )


        down_block_additional_residuals = unet_kwargs.get("down_block_additional_residuals", None)
        mid_block_additional_residual = unet_kwargs.get("mid_block_additional_residual", None)

        if down_block_additional_residuals is None and mid_block_additional_residual is None:
            down_block_additional_residuals, mid_block_additional_residual = down_samples, mid_sample
        else:
            # add controlnet outputs together if have multiple controlnets
            down_block_additional_residuals = [
                samples_prev + samples_curr
                for samples_prev, samples_curr in zip(down_block_additional_residuals, down_samples, strict=True)
            ]
            mid_block_additional_residual += mid_sample

        unet_kwargs.update(dict(
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        ))


    def _run(
        self,
        sample,
        timestep,
        step_index,
        guess_mode,
        unet_kwargs,
    ):
        # get static weight, or weight corresponding to current step
        weight = self.weight
        if isinstance(weight, list):
            weight = weight[step_index]

        # controlnet(s) inference
        down_samples, mid_sample = self.model(
            sample=sample,
            timestep=timestep,
            controlnet_cond=self.image_tensor,
            conditioning_scale=weight,  # controlnet specific, NOT the guidance scale
            guess_mode=guess_mode,  # this is still called guess_mode in diffusers ControlNetModel
            return_dict=False,


            **unet_kwargs,
            #added_cond_kwargs=added_cond_kwargs,
            #encoder_hidden_states=encoder_hidden_states,
            #encoder_attention_mask=encoder_attention_mask,
        )

        return down_samples, mid_sample
