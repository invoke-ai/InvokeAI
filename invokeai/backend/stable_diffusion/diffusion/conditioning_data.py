from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from .cross_attention_control import Arguments


@dataclass
class ExtraConditioningInfo:
    """Extra conditioning information produced by Compel.

    This is used for prompt-to-prompt cross-attention control (a.k.a. `.swap()` in Compel).
    """

    tokens_count_including_eos_bos: int
    cross_attention_control_args: Optional[Arguments] = None

    @property
    def wants_cross_attention_control(self):
        return self.cross_attention_control_args is not None


@dataclass
class BasicConditioningInfo:
    """SD 1/2 text conditioning information produced by Compel."""

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
    """SDXL text conditioning information produced by Compel."""

    pooled_embeds: torch.Tensor
    add_time_ids: torch.Tensor

    def to(self, device, dtype=None):
        self.pooled_embeds = self.pooled_embeds.to(device=device, dtype=dtype)
        self.add_time_ids = self.add_time_ids.to(device=device, dtype=dtype)
        return super().to(device=device, dtype=dtype)


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
class Range:
    start: int
    end: int


class TextConditioningRegions:
    def __init__(
        self,
        masks: torch.Tensor,
        ranges: list[Range],
        mask_weights: list[float],
    ):
        # A binary mask indicating the regions of the image that the prompt should be applied to.
        # Shape: (1, num_prompts, height, width)
        # Dtype: torch.bool
        self.masks = masks

        # A list of ranges indicating the start and end indices of the embeddings that corresponding mask applies to.
        # ranges[i] contains the embedding range for the i'th prompt / mask.
        self.ranges = ranges

        self.mask_weights = mask_weights

        assert self.masks.shape[1] == len(self.ranges) == len(self.mask_weights)


class TextConditioningData:
    def __init__(
        self,
        uncond_text: Union[BasicConditioningInfo, SDXLConditioningInfo],
        cond_text: Union[BasicConditioningInfo, SDXLConditioningInfo],
        uncond_regions: Optional[TextConditioningRegions],
        cond_regions: Optional[TextConditioningRegions],
        guidance_scale: Union[float, List[float]],
        guidance_rescale_multiplier: float = 0,
    ):
        self.uncond_text = uncond_text
        self.cond_text = cond_text
        self.uncond_regions = uncond_regions
        self.cond_regions = cond_regions
        # Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
        # `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
        # Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate
        # images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
        self.guidance_scale = guidance_scale
        # For models trained using zero-terminal SNR ("ztsnr"), it's suggested to use guidance_rescale_multiplier of 0.7.
        # See [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
        self.guidance_rescale_multiplier = guidance_rescale_multiplier

    def is_sdxl(self):
        assert isinstance(self.uncond_text, SDXLConditioningInfo) == isinstance(self.cond_text, SDXLConditioningInfo)
        return isinstance(self.cond_text, SDXLConditioningInfo)
