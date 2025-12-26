from typing import Optional

import torch
import torchvision

from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.mask import to_standard_float_mask
from invokeai.backend.z_image.text_conditioning import ZImageRegionalTextConditioning, ZImageTextConditioning


class ZImageRegionalPromptingExtension:
    """A class for managing regional prompting with Z-Image.

    This implementation is inspired by the FLUX regional prompting extension and
    the paper https://arxiv.org/pdf/2411.02395.

    Key difference from FLUX: Z-Image uses sequence order [img_tokens, txt_tokens],
    while FLUX uses [txt_tokens, img_tokens]. The attention mask construction
    accounts for this difference.
    """

    def __init__(
        self,
        regional_text_conditioning: ZImageRegionalTextConditioning,
        regional_attn_mask: torch.Tensor | None = None,
    ):
        self.regional_text_conditioning = regional_text_conditioning
        self.regional_attn_mask = regional_attn_mask

    def get_attn_mask(self, block_index: int) -> torch.Tensor | None:
        """Get the attention mask for a given block index.

        Uses alternating pattern: apply mask on even blocks, no mask on odd blocks.
        This helps balance regional control with global coherence.
        """
        order = [self.regional_attn_mask, None]
        return order[block_index % len(order)]

    @classmethod
    def from_text_conditionings(
        cls,
        text_conditionings: list[ZImageTextConditioning],
        img_seq_len: int,
    ) -> "ZImageRegionalPromptingExtension":
        """Create a ZImageRegionalPromptingExtension from a list of text conditionings.

        Args:
            text_conditionings: List of text conditionings with optional masks.
            img_seq_len: The image sequence length (i.e. (H // patch_size) * (W // patch_size)).

        Returns:
            A configured ZImageRegionalPromptingExtension.
        """
        regional_text_conditioning = ZImageRegionalTextConditioning.from_text_conditionings(text_conditionings)
        attn_mask = cls._prepare_regional_attn_mask(regional_text_conditioning, img_seq_len)
        return cls(
            regional_text_conditioning=regional_text_conditioning,
            regional_attn_mask=attn_mask,
        )

    @classmethod
    def _prepare_regional_attn_mask(
        cls,
        regional_text_conditioning: ZImageRegionalTextConditioning,
        img_seq_len: int,
    ) -> torch.Tensor | None:
        """Prepare a regional attention mask for Z-Image.

        The mask controls which tokens can attend to each other:
        - Image tokens within a region attend only to each other
        - Image tokens attend only to their corresponding regional text
        - Text tokens attend only to their corresponding regional image
        - Text tokens attend to themselves

        Z-Image sequence order: [img_tokens, txt_tokens]

        Args:
            regional_text_conditioning: The regional text conditioning data.
            img_seq_len: Number of image tokens.

        Returns:
            Attention mask of shape (img_seq_len + txt_seq_len, img_seq_len + txt_seq_len).
            Returns None if no regional masks are present.
        """
        # Check if any regional masks exist
        has_regional_masks = any(mask is not None for mask in regional_text_conditioning.image_masks)
        if not has_regional_masks:
            # No regional masks, return None to use default attention
            return None

        # Identify background region (area not covered by any mask)
        background_region_mask: torch.Tensor | None = None
        for image_mask in regional_text_conditioning.image_masks:
            if image_mask is not None:
                # image_mask shape: (1, 1, img_seq_len) -> flatten to (img_seq_len,)
                mask_flat = image_mask.view(-1)
                if background_region_mask is None:
                    background_region_mask = torch.ones_like(mask_flat)
                background_region_mask = background_region_mask * (1 - mask_flat)

        device = TorchDevice.choose_torch_device()
        txt_seq_len = regional_text_conditioning.prompt_embeds.shape[0]
        total_seq_len = img_seq_len + txt_seq_len

        # Initialize empty attention mask
        # Z-Image sequence: [img_tokens (0:img_seq_len), txt_tokens (img_seq_len:total_seq_len)]
        regional_attention_mask = torch.zeros((total_seq_len, total_seq_len), device=device, dtype=torch.float16)

        for image_mask, embedding_range in zip(
            regional_text_conditioning.image_masks,
            regional_text_conditioning.embedding_ranges,
            strict=True,
        ):
            # Calculate text token positions in the unified sequence
            txt_start = img_seq_len + embedding_range.start
            txt_end = img_seq_len + embedding_range.end

            # 1. txt attends to itself
            regional_attention_mask[txt_start:txt_end, txt_start:txt_end] = 1.0

            if image_mask is not None:
                # Flatten mask: (1, 1, img_seq_len) -> (img_seq_len,)
                mask_flat = image_mask.view(img_seq_len)

                # 2. img attends to corresponding regional txt
                # Reshape mask to (img_seq_len, 1) for broadcasting
                regional_attention_mask[:img_seq_len, txt_start:txt_end] = mask_flat.view(img_seq_len, 1)

                # 3. txt attends to corresponding regional img
                # Reshape mask to (1, img_seq_len) for broadcasting
                regional_attention_mask[txt_start:txt_end, :img_seq_len] = mask_flat.view(1, img_seq_len)

                # 4. img self-attention within region
                # mask @ mask.T creates pairwise attention within the masked region
                regional_attention_mask[:img_seq_len, :img_seq_len] += mask_flat.view(img_seq_len, 1) @ mask_flat.view(
                    1, img_seq_len
                )
            else:
                # Global prompt: allow attention to/from background regions only
                if background_region_mask is not None:
                    # 2. background img attends to global txt
                    regional_attention_mask[:img_seq_len, txt_start:txt_end] = background_region_mask.view(
                        img_seq_len, 1
                    )

                    # 3. global txt attends to background img
                    regional_attention_mask[txt_start:txt_end, :img_seq_len] = background_region_mask.view(
                        1, img_seq_len
                    )
                else:
                    # No regional masks at all, allow full attention
                    regional_attention_mask[:img_seq_len, txt_start:txt_end] = 1.0
                    regional_attention_mask[txt_start:txt_end, :img_seq_len] = 1.0

        # Allow background regions to attend to themselves
        if background_region_mask is not None:
            bg_mask = background_region_mask.view(img_seq_len, 1)
            regional_attention_mask[:img_seq_len, :img_seq_len] += bg_mask @ bg_mask.T

        # Convert to boolean mask
        regional_attention_mask = regional_attention_mask > 0.5

        return regional_attention_mask

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

        # Resize mask to target dimensions
        tf = torchvision.transforms.Resize(
            (target_height, target_width),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )

        # Add batch dimension if needed: (h, w) -> (1, h, w) -> (1, 1, h, w)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        resized_mask = tf(mask)

        # Flatten to (1, 1, img_seq_len)
        return resized_mask.flatten(start_dim=2).to(device=device)
