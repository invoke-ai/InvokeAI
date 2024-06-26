from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from pydantic import Field

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import TextConditioningData
from .base import AddonBase


@dataclass
class InpaintModelAddon(AddonBase):
    mask: Optional[torch.Tensor] = None
    masked_latents: Optional[torch.Tensor] = None

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
        batch_size = sample.shape[0]
        if conditioning_mode == "both":
            batch_size *= 2

        if self.mask is None:
            self.mask = torch.ones_like(sample[:1, :1])

        if self.masked_latents is None:
            self.masked_latents = torch.zeros_like(sample[:1])

        b_mask = torch.cat([self.mask] * batch_size)
        b_masked_latents = torch.cat([self.masked_latents] * batch_size)

        extra_channels = torch.cat([b_mask, b_masked_latents], dim=1).to(device=sample.device, dtype=sample.dtype)

        unet_kwargs.update(dict(
            extra_channels=extra_channels,
        ))
