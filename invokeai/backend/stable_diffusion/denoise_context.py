from __future__ import annotations

import torch
import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Union, Dict, Tuple
from diffusers import UNet2DConditionModel
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.diffusion.conditioning_data import TextConditioningData



@dataclass
class UNetKwargs:
    sample: torch.Tensor
    timestep: Union[torch.Tensor, float, int]
    encoder_hidden_states: torch.Tensor

    class_labels: Optional[torch.Tensor] = None
    timestep_cond: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None
    mid_block_additional_residual: Optional[torch.Tensor] = None
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    #return_dict: bool = True


@dataclass
class DenoiseContext:
    latents: torch.Tensor
    scheduler_step_kwargs: dict[str, Any]
    conditioning_data: TextConditioningData
    noise: Optional[torch.Tensor]
    seed: int
    timesteps: torch.Tensor
    init_timestep: torch.Tensor
    
    unet: UNet2DConditionModel
    scheduler: SchedulerMixin

    orig_latents: Optional[torch.Tensor] = None
    step_index: Optional[int] = None
    timestep: Optional[torch.Tensor] = None
    unet_kwargs: Optional[UNetKwargs] = None
    step_output: Optional[SchedulerOutput] = None

    latent_model_input: Optional[torch.Tensor] = None
    conditioning_mode: Optional[str] = None
    negative_noise_pred: Optional[torch.Tensor] = None
    positive_noise_pred: Optional[torch.Tensor] = None
    noise_pred: Optional[torch.Tensor] = None

    def __delattr__(self, name: str):
        setattr(self, name, None)

    @classmethod
    def from_kwargs(cls, **kwargs) -> DenoiseContext:
        names = set([f.name for f in dataclasses.fields(cls)])
        return cls(**{
            k: v for k, v in kwargs.items() 
            if k in names
        })