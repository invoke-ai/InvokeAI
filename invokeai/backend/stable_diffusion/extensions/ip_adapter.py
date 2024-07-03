from __future__ import annotations

import math
import torch
from typing import Any, List, Dict, Union
from diffusers import UNet2DConditionModel

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import IPAdapterConditioningInfo
from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
from .base import ExtensionBase, modifier
from ..denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import (
    CustomAttnProcessor2_0,
    IPAdapterAttentionWeights,
)


class IPAdapterExt(ExtensionBase):
    def __init__(
        self,
        model: IPAdapter,
        conditioning: IPAdapterConditioningInfo,
        mask: torch.Tensor,
        target_blocks: List[str],
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
        priority: int,
    ):
        super().__init__(priority=priority)
        self.model = model
        self.conditioning = conditioning
        self.mask = mask
        self.target_blocks = target_blocks
        self.weight = weight
        self.begin_step_percent = begin_step_percent
        self.end_step_percent = end_step_percent

    def patch_unet(self, unet: UNet2DConditionModel):
        for idx, name in enumerate(unet.attn_processors.keys()):
            if name.endswith("attn1.processor"):
                continue

            ip_adapter_weights = self.model.attn_weights.get_attention_processor_weights(idx)
            skip = True
            for block in self.target_blocks:
                if block in name:
                    skip = False
                    break

            assert isinstance(unet.attn_processors[name], CustomAttnProcessor2_0)
            unet.attn_processors[name].add_ip_adapter(
                IPAdapterAttentionWeights(
                    ip_adapter_weights=ip_adapter_weights,
                    skip=skip,
                )
            )

    def unpatch_unet(self, unet: UNet2DConditionModel):
        # nop, as it unpatched with attention processor
        pass

    @modifier("pre_unet_forward")
    def pre_unet_step(self, ctx: DenoiseContext):
        # skip if model not active in current step
        total_steps = len(ctx.timesteps)
        first_step = math.floor(self.begin_step_percent * total_steps)
        last_step = math.ceil(self.end_step_percent * total_steps)
        if ctx.step_index < first_step or ctx.step_index > last_step:
            return

        weight = self.weight
        if isinstance(weight, List):
            weight = weight[ctx.step_index]

        if ctx.conditioning_mode == "both":
            embeds = torch.stack([self.conditioning.uncond_image_prompt_embeds, self.conditioning.cond_image_prompt_embeds])
        elif ctx.conditioning_mode == "negative":
            embeds = torch.stack([self.conditioning.uncond_image_prompt_embeds])
        else: # elif ctx.conditioning_mode == "positive":
            embeds = torch.stack([self.conditioning.cond_image_prompt_embeds])

        if ctx.unet_kwargs.cross_attention_kwargs is None:
            ctx.unet_kwargs.cross_attention_kwargs = dict()

        regional_ip_data = ctx.unet_kwargs.cross_attention_kwargs.get("regional_ip_data", None)
        if regional_ip_data is None:
            regional_ip_data = RegionalIPData(
                image_prompt_embeds=[],
                scales=[],
                masks=[],
                dtype=ctx.latent_model_input.dtype,
                device=ctx.latent_model_input.device,
            )
            ctx.unet_kwargs.cross_attention_kwargs.update(dict(
                regional_ip_data=regional_ip_data,
            ))


        regional_ip_data.add(
            embeds=embeds,
            scale=weight,
            mask=self.mask,
        )
