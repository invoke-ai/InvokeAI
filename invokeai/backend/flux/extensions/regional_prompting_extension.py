from typing import Optional

import torch
import torchvision

from invokeai.backend.flux.text_conditioning import FluxRegionalTextConditioning, FluxTextConditioning
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.mask import to_standard_float_mask


class RegionalPromptingExtension:
    """A class for managing regional prompting with FLUX."""

    def __init__(self, regional_text_conditioning: FluxRegionalTextConditioning):
        self.regional_text_conditioning = regional_text_conditioning

    @classmethod
    def from_text_conditioning(cls, text_conditioning: list[FluxTextConditioning]):
        return cls(regional_text_conditioning=cls._concat_regional_text_conditioning(text_conditioning))

    @classmethod
    def _concat_regional_text_conditioning(
        cls,
        text_conditionings: list[FluxTextConditioning],
    ) -> FluxRegionalTextConditioning:
        """Concatenate regional text conditioning data into a single conditioning tensor (with associated masks)."""
        concat_t5_embeddings: list[torch.Tensor] = []
        concat_clip_embeddings: list[torch.Tensor] = []
        concat_image_masks: list[torch.Tensor] = []
        concat_t5_embedding_ranges: list[Range] = []
        concat_clip_embedding_ranges: list[Range] = []

        cur_t5_embedding_len = 0
        cur_clip_embedding_len = 0
        for text_conditioning in text_conditionings:
            concat_t5_embeddings.append(text_conditioning.t5_embeddings)
            concat_clip_embeddings.append(text_conditioning.clip_embeddings)

            concat_t5_embedding_ranges.append(
                Range(start=cur_t5_embedding_len, end=cur_t5_embedding_len + text_conditioning.t5_embeddings.shape[1])
            )
            concat_clip_embedding_ranges.append(
                Range(
                    start=cur_clip_embedding_len,
                    end=cur_clip_embedding_len + text_conditioning.clip_embeddings.shape[1],
                )
            )

            concat_image_masks.append(text_conditioning.mask)

            cur_t5_embedding_len += text_conditioning.t5_embeddings.shape[1]
            cur_clip_embedding_len += text_conditioning.clip_embeddings.shape[1]

        t5_embeddings = torch.cat(concat_t5_embeddings, dim=1)

        # Initialize the txt_ids tensor.
        pos_bs, pos_t5_seq_len, _ = t5_embeddings.shape
        t5_txt_ids = torch.zeros(
            pos_bs, pos_t5_seq_len, 3, dtype=t5_embeddings.dtype, device=TorchDevice.choose_torch_device()
        )

        return FluxRegionalTextConditioning(
            t5_embeddings=t5_embeddings,
            clip_embeddings=torch.cat(concat_clip_embeddings, dim=1),
            t5_txt_ids=t5_txt_ids,
            image_masks=torch.cat(concat_image_masks, dim=1),
            t5_embedding_ranges=concat_t5_embedding_ranges,
            clip_embedding_ranges=concat_clip_embedding_ranges,
        )

    @staticmethod
    def preprocess_regional_prompt_mask(
        mask: Optional[torch.Tensor], target_height: int, target_width: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Preprocess a regional prompt mask to match the target height and width.
        If mask is None, returns a mask of all ones with the target height and width.
        If mask is not None, resizes the mask to the target height and width using 'nearest' interpolation.

        Returns:
            torch.Tensor: The processed mask. shape: (1, 1, target_height, target_width).
        """

        if mask is None:
            return torch.ones((1, 1, target_height, target_width), dtype=dtype)

        mask = to_standard_float_mask(mask, out_dtype=dtype)

        tf = torchvision.transforms.Resize(
            (target_height, target_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # Add a batch dimension to the mask, because torchvision expects shape (batch, channels, h, w).
        mask = mask.unsqueeze(0)  # Shape: (1, h, w) -> (1, 1, h, w)
        resized_mask = tf(mask)
        return resized_mask
