import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch

from .cross_attention_control import Arguments


@dataclass
class ExtraConditioningInfo:
    tokens_count_including_eos_bos: int
    cross_attention_control_args: Optional[Arguments] = None

    @property
    def wants_cross_attention_control(self):
        return self.cross_attention_control_args is not None


@dataclass
class BasicConditioningInfo:
    embeds: torch.Tensor
    extra_conditioning: Optional[ExtraConditioningInfo]

    def to(self, device, dtype=None):
        self.embeds = self.embeds.to(device=device, dtype=dtype)
        return self


@dataclass
class ConditioningFieldData:
    conditionings: List[BasicConditioningInfo]


@dataclass
class SDXLConditioningInfo(BasicConditioningInfo):
    pooled_embeds: torch.Tensor
    add_time_ids: torch.Tensor

    def to(self, device, dtype=None):
        self.pooled_embeds = self.pooled_embeds.to(device=device, dtype=dtype)
        self.add_time_ids = self.add_time_ids.to(device=device, dtype=dtype)
        return super().to(device=device, dtype=dtype)


@dataclass(frozen=True)
class PostprocessingSettings:
    threshold: float
    warmup: float
    h_symmetry_time_pct: Optional[float]
    v_symmetry_time_pct: Optional[float]


@dataclass
class IPAdapterConditioningInfo:
    cond_image_prompt_embeds: torch.Tensor
    """IP-Adapter image encoder conditioning embeddings.
    Shape: (num_images, num_tokens, encoding_dim).
    """
    uncond_image_prompt_embeds: torch.Tensor
    """IP-Adapter image encoding embeddings to use for unconditional generation.
    Shape: (num_images, num_tokens, encoding_dim).
    """


@dataclass
class ConditioningData:
    unconditioned_embeddings: BasicConditioningInfo
    text_embeddings: BasicConditioningInfo
    """
    Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
    `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
    Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate
    images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
    """
    guidance_scale: Union[float, List[float]]
    """ for models trained using zero-terminal SNR ("ztsnr"), it's suggested to use guidance_rescale_multiplier of 0.7 .
     ref [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf)
    """
    guidance_rescale_multiplier: float = 0
    scheduler_args: dict[str, Any] = field(default_factory=dict)
    """
    Additional arguments to pass to invokeai_diffuser.do_latent_postprocessing().
    """
    postprocessing_settings: Optional[PostprocessingSettings] = None

    ip_adapter_conditioning: Optional[list[IPAdapterConditioningInfo]] = None

    @property
    def dtype(self):
        return self.text_embeddings.dtype

    def add_scheduler_args_if_applicable(self, scheduler, **kwargs):
        scheduler_args = dict(self.scheduler_args)
        step_method = inspect.signature(scheduler.step)
        for name, value in kwargs.items():
            try:
                step_method.bind_partial(**{name: value})
            except TypeError:
                # FIXME: don't silently discard arguments
                pass  # debug("%s does not accept argument named %r", scheduler, name)
            else:
                scheduler_args[name] = value
        return dataclasses.replace(self, scheduler_args=scheduler_args)
