import math
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.stable_diffusion.diffusion.regional_prompt_data import RegionalPromptData


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

    def to_unet_kwargs(self, unet_kwargs, conditioning_mode):
        if conditioning_mode == "both":
            encoder_hidden_states, encoder_attention_mask = self._concat_conditionings_for_batch(
                self.uncond_text.embeds, self.cond_text.embeds
            )
        elif conditioning_mode == "positive":
            encoder_hidden_states = self.cond_text.embeds
            encoder_attention_mask = None
        else: # elif conditioning_mode == "negative":
            encoder_hidden_states = self.uncond_text.embeds
            encoder_attention_mask = None

        unet_kwargs.encoder_hidden_states=encoder_hidden_states
        unet_kwargs.encoder_attention_mask=encoder_attention_mask

        if self.is_sdxl():
            if conditioning_mode == "negative":
                added_cond_kwargs = dict(
                    text_embeds=self.cond_text.pooled_embeds,
                    time_ids=self.cond_text.add_time_ids,
                )
            elif conditioning_mode == "positive":
                added_cond_kwargs = dict(
                    text_embeds=self.uncond_text.pooled_embeds,
                    time_ids=self.uncond_text.add_time_ids,
                )
            else: # elif conditioning_mode == "both":
                added_cond_kwargs = dict(
                    text_embeds=torch.cat(
                        [
                            # TODO: how to pad? just by zeros? or even truncate?
                            self.uncond_text.pooled_embeds,
                            self.cond_text.pooled_embeds,
                        ],
                    ),
                    time_ids=torch.cat(
                        [
                            self.uncond_text.add_time_ids,
                            self.cond_text.add_time_ids,
                        ],
                    ),
                )

            unet_kwargs.added_cond_kwargs=added_cond_kwargs

        if self.cond_regions is not None or self.uncond_regions is not None:
            # TODO(ryand): We currently initialize RegionalPromptData for every denoising step. The text conditionings
            # and masks are not changing from step-to-step, so this really only needs to be done once. While this seems
            # painfully inefficient, the time spent is typically negligible compared to the forward inference pass of
            # the UNet. The main reason that this hasn't been moved up to eliminate redundancy is that it is slightly
            # awkward to handle both standard conditioning and sequential conditioning further up the stack.

            _tmp_regions = self.cond_regions if self.cond_regions is not None else self.uncond_regions
            _, _, h, w = _tmp_regions.masks.shape
            dtype = self.cond_text.embeds.dtype
            device = self.cond_text.embeds.device

            regions = []
            for c, r in [
                (self.uncond_text, self.uncond_regions),
                (self.cond_text, self.cond_regions),
            ]:
                if r is None:
                    # Create a dummy mask and range for text conditioning that doesn't have region masks.
                    r = TextConditioningRegions(
                        masks=torch.ones((1, 1, h, w), dtype=dtype),
                        ranges=[Range(start=0, end=c.embeds.shape[1])],
                    )
                regions.append(r)

            if unet_kwargs.cross_attention_kwargs is None:
                unet_kwargs.cross_attention_kwargs = dict()

            unet_kwargs.cross_attention_kwargs.update(dict(
                regional_prompt_data=RegionalPromptData(
                    regions=regions, device=device, dtype=dtype
                ),
            ))

    def _concat_conditionings_for_batch(self, unconditioning, conditioning):
        def _pad_conditioning(cond, target_len, encoder_attention_mask):
            conditioning_attention_mask = torch.ones(
                (cond.shape[0], cond.shape[1]), device=cond.device, dtype=cond.dtype
            )

            if cond.shape[1] < max_len:
                conditioning_attention_mask = torch.cat(
                    [
                        conditioning_attention_mask,
                        torch.zeros((cond.shape[0], max_len - cond.shape[1]), device=cond.device, dtype=cond.dtype),
                    ],
                    dim=1,
                )

                cond = torch.cat(
                    [
                        cond,
                        torch.zeros(
                            (cond.shape[0], max_len - cond.shape[1], cond.shape[2]),
                            device=cond.device,
                            dtype=cond.dtype,
                        ),
                    ],
                    dim=1,
                )

            if encoder_attention_mask is None:
                encoder_attention_mask = conditioning_attention_mask
            else:
                encoder_attention_mask = torch.cat(
                    [
                        encoder_attention_mask,
                        conditioning_attention_mask,
                    ]
                )

            return cond, encoder_attention_mask

        encoder_attention_mask = None
        if unconditioning.shape[1] != conditioning.shape[1]:
            max_len = max(unconditioning.shape[1], conditioning.shape[1])
            unconditioning, encoder_attention_mask = _pad_conditioning(unconditioning, max_len, encoder_attention_mask)
            conditioning, encoder_attention_mask = _pad_conditioning(conditioning, max_len, encoder_attention_mask)

        return torch.cat([unconditioning, conditioning]), encoder_attention_mask
