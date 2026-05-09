"""Regional prompting extension for Anima.

Anima's architecture uses separate cross-attention in each DiT block: image tokens
(in 5D spatial layout) cross-attend to context tokens (LLM Adapter output). This is
different from Z-Image's unified [img, txt] sequence with self-attention.

For regional prompting, we:
1. Run the LLM Adapter separately for each regional prompt
2. Concatenate the resulting context vectors
3. Build a cross-attention mask that restricts each image region to attend only to
   its corresponding context tokens
4. Patch the DiT's cross-attention to use this mask

The mask alternation strategy (masked on even blocks, full on odd blocks) helps
maintain global coherence across regions.
"""

from typing import Optional

import torch
import torchvision

from invokeai.backend.anima.conditioning_data import AnimaRegionalTextConditioning
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.mask import to_standard_float_mask


class AnimaRegionalPromptingExtension:
    """Manages regional prompting for Anima's cross-attention.

    Unlike Z-Image which uses a unified [img, txt] sequence, Anima has separate
    cross-attention where image tokens (query) attend to context tokens (key/value).
    The cross-attention mask shape is (img_seq_len, context_seq_len).
    """

    def __init__(
        self,
        regional_text_conditioning: AnimaRegionalTextConditioning,
        cross_attn_mask: torch.Tensor | None = None,
    ):
        self.regional_text_conditioning = regional_text_conditioning
        self.cross_attn_mask = cross_attn_mask

    def get_cross_attn_mask(self, block_index: int) -> torch.Tensor | None:
        """Get the cross-attention mask for a given block index.

        Uses alternating pattern: apply mask on even blocks, no mask on odd blocks.
        This helps balance regional control with global coherence.
        """
        if block_index % 2 == 0:
            return self.cross_attn_mask
        return None

    @classmethod
    def from_regional_conditioning(
        cls,
        regional_text_conditioning: AnimaRegionalTextConditioning,
        img_seq_len: int,
    ) -> "AnimaRegionalPromptingExtension":
        """Create extension from pre-processed regional conditioning.

        Args:
            regional_text_conditioning: Regional conditioning with concatenated context and masks.
            img_seq_len: Number of image tokens (H_patches * W_patches).
        """
        cross_attn_mask = cls._prepare_cross_attn_mask(regional_text_conditioning, img_seq_len)
        return cls(
            regional_text_conditioning=regional_text_conditioning,
            cross_attn_mask=cross_attn_mask,
        )

    @classmethod
    def _prepare_cross_attn_mask(
        cls,
        regional_text_conditioning: AnimaRegionalTextConditioning,
        img_seq_len: int,
    ) -> torch.Tensor | None:
        """Prepare a cross-attention mask for regional prompting.

        The mask shape is (img_seq_len, context_seq_len) where:
        - Each image token can attend to context tokens from its assigned region
        - Global prompts (mask=None) attend to background regions

        Args:
            regional_text_conditioning: The regional text conditioning data.
            img_seq_len: Number of image tokens.

        Returns:
            Cross-attention mask of shape (img_seq_len, context_seq_len), or None
            if no regional masks are present.
        """
        has_regional_masks = any(mask is not None for mask in regional_text_conditioning.image_masks)
        if not has_regional_masks:
            return None

        # Identify background region (area not covered by any mask)
        background_region_mask: torch.Tensor | None = None
        for image_mask in regional_text_conditioning.image_masks:
            if image_mask is not None:
                mask_flat = image_mask.view(-1)
                if background_region_mask is None:
                    background_region_mask = torch.ones_like(mask_flat)
                background_region_mask = background_region_mask * (1 - mask_flat)

        device = TorchDevice.choose_torch_device()
        context_seq_len = regional_text_conditioning.context_embeds.shape[0]

        # Cross-attention mask: (img_seq_len, context_seq_len)
        # img tokens are queries, context tokens are keys/values
        cross_attn_mask = torch.zeros((img_seq_len, context_seq_len), device=device, dtype=torch.float16)

        for image_mask, context_range in zip(
            regional_text_conditioning.image_masks,
            regional_text_conditioning.context_ranges,
            strict=True,
        ):
            ctx_start = context_range.start
            ctx_end = context_range.end

            if image_mask is not None:
                # Regional prompt: only masked image tokens attend to this region's context
                mask_flat = image_mask.view(img_seq_len)
                cross_attn_mask[:, ctx_start:ctx_end] = mask_flat.view(img_seq_len, 1)
            else:
                # Global prompt: background image tokens attend to this context
                if background_region_mask is not None:
                    cross_attn_mask[:, ctx_start:ctx_end] = background_region_mask.view(img_seq_len, 1)
                else:
                    cross_attn_mask[:, ctx_start:ctx_end] = 1.0

        # Convert to boolean
        cross_attn_mask = cross_attn_mask > 0.5
        return cross_attn_mask

    @staticmethod
    def preprocess_regional_prompt_mask(
        mask: Optional[torch.Tensor],
        target_height: int,
        target_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Preprocess a regional prompt mask to match the target image token grid.

        Args:
            mask: Input mask tensor. If None, returns a mask of all ones.
            target_height: Height of the image token grid (H // patch_size).
            target_width: Width of the image token grid (W // patch_size).
            dtype: Target dtype for the mask.
            device: Target device for the mask.

        Returns:
            Processed mask of shape (1, 1, target_height * target_width).
        """
        img_seq_len = target_height * target_width

        if mask is None:
            return torch.ones((1, 1, img_seq_len), dtype=dtype, device=device)

        mask = to_standard_float_mask(mask, out_dtype=dtype)

        tf = torchvision.transforms.Resize(
            (target_height, target_width),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        resized_mask = tf(mask)
        return resized_mask.flatten(start_dim=2).to(device=device)
