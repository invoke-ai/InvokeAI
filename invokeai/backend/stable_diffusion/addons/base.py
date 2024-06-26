from __future__ import annotations

import torch
from typing import Any, Dict
from abc import ABC, abstractmethod
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import TextConditioningData


class AddonBase(ABC):

    @abstractmethod
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
        pass
