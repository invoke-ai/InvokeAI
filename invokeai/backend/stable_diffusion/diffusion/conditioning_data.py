import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter


@dataclass
class BasicConditioningInfo:
    """SD 1/2 text conditioning information produced by Compel."""

    embeds: torch.Tensor

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
class IPAdapterData:
    ip_adapter_model: IPAdapter
    ip_adapter_conditioning: IPAdapterConditioningInfo
    mask: torch.Tensor
    target_blocks: List[str]

    # Either a single weight applied to all steps, or a list of weights for each step.
    weight: Union[float, List[float]] = 1.0
    begin_step_percent: float = 0.0
    end_step_percent: float = 1.0

    def scale_for_step(self, step_index: int, total_steps: int) -> float:
        first_adapter_step = math.floor(self.begin_step_percent * total_steps)
        last_adapter_step = math.ceil(self.end_step_percent * total_steps)
        weight = self.weight[step_index] if isinstance(self.weight, List) else self.weight
        if step_index >= first_adapter_step and step_index <= last_adapter_step:
            # Only apply this IP-Adapter if the current step is within the IP-Adapter's begin/end step range.
            return weight
        # Otherwise, set the IP-Adapter's scale to 0, so it has no effect.
        return 0.0


@dataclass
class Range:
    start: int
    end: int


class TextConditioningRegions:
    def __init__(
        self,
        masks: torch.Tensor,
        ranges: list[Range],
    ):
        # A binary mask indicating the regions of the image that the prompt should be applied to.
        # Shape: (1, num_prompts, height, width)
        # Dtype: torch.bool
        self.masks = masks

        # A list of ranges indicating the start and end indices of the embeddings that corresponding mask applies to.
        # ranges[i] contains the embedding range for the i'th prompt / mask.
        self.ranges = ranges

        assert self.masks.shape[1] == len(self.ranges)


class SDRegionalTextConditioning:
    def __init__(self, text_embeds: list[torch.Tensor], masks: Optional[list[torch.Tensor]]):
        if masks is not None:
            assert len(text_embeds) == len(masks)

        # A list of text embeddings. text_embeds[i] contains the text embeddings for the i'th prompt.
        self.text_embeds = text_embeds
        # A list of masks indicating the regions of the image that the prompts should be applied to. masks[i] contains
        # the mask for the i'th prompt. Each mask has shape (1, height, width).
        self.masks = masks

    def uses_regional_prompts(self):
        # If there is more than one prompt, we treat this as regional prompting, even if there are no masks, because
        # the regional prompting logic is used to combine the information from multiple prompts.
        return len(self.text_embeds) > 1 or self.masks is not None


class SDXLRegionalTextConditioning(SDRegionalTextConditioning):
    def __init__(
        self,
        pooled_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        text_embeds: list[torch.Tensor],
        masks: Optional[list[torch.Tensor]],
    ):
        super().__init__(text_embeds, masks)

        # Pooled embeddings for the global prompt.
        self.pooled_embeds = pooled_embeds
        # Additional global conditioning inputs for SDXL. The name "time_ids" comes from diffusers, and is a bit of a
        # misnomer. This Tensor contains original_size, crop_coords, and target_size conditioning.
        self.add_time_ids = add_time_ids


class TextConditioningData:
    def __init__(
        self,
        uncond_text: Union[SDRegionalTextConditioning, SDXLRegionalTextConditioning],
        cond_text: Union[SDRegionalTextConditioning, SDXLRegionalTextConditioning],
        guidance_scale: Union[float, List[float]],
        guidance_rescale_multiplier: float = 0,
    ):
        self.uncond_text = uncond_text
        self.cond_text = cond_text
        # Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
        # `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf).
        # Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate
        # images that are closely linked to the text `prompt`, usually at the expense of lower image quality.
        self.guidance_scale = guidance_scale
        # For models trained using zero-terminal SNR ("ztsnr"), it's suggested to use guidance_rescale_multiplier of 0.7.
        # See [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
        self.guidance_rescale_multiplier = guidance_rescale_multiplier

    def is_sdxl(self):
        assert isinstance(self.uncond_text, SDXLRegionalTextConditioning) == isinstance(
            self.cond_text, SDXLRegionalTextConditioning
        )
        return isinstance(self.cond_text, SDXLRegionalTextConditioning)
