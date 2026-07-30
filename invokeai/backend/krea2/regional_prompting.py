from dataclasses import dataclass

import torch
import torchvision

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range
from invokeai.backend.util.mask import to_standard_float_mask


@dataclass
class Krea2TextConditioning:
    prompt_embeds: torch.Tensor
    mask: torch.Tensor | None


@dataclass
class Krea2RegionalTextConditioning:
    prompt_embeds: torch.Tensor
    image_masks: list[torch.Tensor | None]
    embedding_ranges: list[Range]


class Krea2RegionalPromptingExtension:
    """Concatenates Krea-2 text conditionings and lazily builds Flux-style regional attention masks."""

    def __init__(self, regional_text_conditioning: Krea2RegionalTextConditioning, image_seq_len: int) -> None:
        self.regional_text_conditioning = regional_text_conditioning
        self.image_seq_len = image_seq_len
        self._attention_mask: torch.Tensor | None = None

    @property
    def has_regional_masks(self) -> bool:
        return any(mask is not None for mask in self.regional_text_conditioning.image_masks)

    @property
    def attention_mask_numel(self) -> int:
        if not self.has_regional_masks:
            return 0
        total_seq_len = self.regional_text_conditioning.prompt_embeds.shape[1] + self.image_seq_len
        return total_seq_len**2

    @property
    def attention_mask_build_scratch_numel(self) -> int:
        """Peak boolean scratch allocation used while constructing the image-to-image attention block."""
        if not self.has_regional_masks:
            return 0
        return self.image_seq_len**2

    @classmethod
    def from_text_conditionings(
        cls, text_conditionings: list[Krea2TextConditioning], image_seq_len: int
    ) -> "Krea2RegionalPromptingExtension":
        if not text_conditionings:
            raise ValueError("At least one Krea-2 text conditioning is required.")

        prompt_embeds: list[torch.Tensor] = []
        image_masks: list[torch.Tensor | None] = []
        embedding_ranges: list[Range] = []
        current_start = 0
        for conditioning in text_conditionings:
            sequence_length = conditioning.prompt_embeds.shape[1]
            if conditioning.mask is not None and conditioning.mask.numel() != image_seq_len:
                raise ValueError(
                    f"Krea-2 regional mask has {conditioning.mask.numel()} values, expected {image_seq_len}."
                )
            prompt_embeds.append(conditioning.prompt_embeds)
            image_masks.append(conditioning.mask)
            embedding_ranges.append(Range(start=current_start, end=current_start + sequence_length))
            current_start += sequence_length

        regional_text_conditioning = Krea2RegionalTextConditioning(
            prompt_embeds=torch.cat(prompt_embeds, dim=1),
            image_masks=image_masks,
            embedding_ranges=embedding_ranges,
        )
        return cls(regional_text_conditioning=regional_text_conditioning, image_seq_len=image_seq_len)

    def get_attention_mask(self) -> torch.Tensor | None:
        if not self.has_regional_masks:
            return None
        if self._attention_mask is None:
            self._attention_mask = self._build_attention_mask()
        return self._attention_mask

    def _build_attention_mask(self) -> torch.Tensor:
        conditioning = self.regional_text_conditioning
        text_seq_len = conditioning.prompt_embeds.shape[1]
        total_seq_len = text_seq_len + self.image_seq_len
        device = conditioning.prompt_embeds.device
        attention_mask = torch.zeros((total_seq_len, total_seq_len), device=device, dtype=torch.bool)

        background_mask = torch.ones(self.image_seq_len, device=device, dtype=torch.bool)
        for image_mask in conditioning.image_masks:
            if image_mask is not None:
                background_mask &= ~(image_mask.reshape(-1) > 0.5)

        image_attention_mask = attention_mask[text_seq_len:, text_seq_len:]
        for image_mask, embedding_range in zip(conditioning.image_masks, conditioning.embedding_ranges, strict=True):
            text_slice = slice(embedding_range.start, embedding_range.end)
            attention_mask[text_slice, text_slice] = True

            if image_mask is None:
                attention_mask[text_slice, text_seq_len:] = background_mask
                attention_mask[text_seq_len:, text_slice] = background_mask[:, None]
                continue

            region_mask = image_mask.reshape(-1) > 0.5
            attention_mask[text_slice, text_seq_len:] = region_mask
            attention_mask[text_seq_len:, text_slice] = region_mask[:, None]
            image_attention_mask |= region_mask[:, None] & region_mask[None, :]

        image_attention_mask |= background_mask[:, None] | background_mask[None, :]
        return attention_mask

    @staticmethod
    def preprocess_regional_prompt_mask(
        mask: torch.Tensor,
        grid_height: int,
        grid_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        mask = to_standard_float_mask(mask, out_dtype=dtype)
        resize = torchvision.transforms.Resize(
            (grid_height, grid_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )
        return resize(mask.unsqueeze(0)).flatten(start_dim=2).to(device=device)
