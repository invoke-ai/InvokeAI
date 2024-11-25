from typing import Optional

import torch
import torchvision

from invokeai.backend.flux.text_conditioning import FluxRegionalTextConditioning, FluxTextConditioning
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.mask import to_standard_float_mask


class RegionalPromptingExtension:
    """A class for managing regional prompting with FLUX.

    Implementation inspired by: https://arxiv.org/pdf/2411.02395
    """

    def __init__(
        self,
        regional_text_conditioning: FluxRegionalTextConditioning,
        attn_mask_with_restricted_img_self_attn: torch.Tensor | None = None,
        attn_mask_with_unrestricted_img_self_attn: torch.Tensor | None = None,
    ):
        self.regional_text_conditioning = regional_text_conditioning
        self.attn_mask_with_restricted_img_self_attn = attn_mask_with_restricted_img_self_attn
        self.attn_mask_with_unrestricted_img_self_attn = attn_mask_with_unrestricted_img_self_attn

    def get_double_stream_attn_mask(self, block_index: int) -> torch.Tensor | None:
        return self.attn_mask_with_unrestricted_img_self_attn

    def get_single_stream_attn_mask(self) -> torch.Tensor | None:
        return self.attn_mask_with_unrestricted_img_self_attn

    @classmethod
    def from_text_conditioning(cls, text_conditioning: list[FluxTextConditioning], img_seq_len: int):
        """Create a RegionalPromptingExtension from a list of text conditionings.

        Args:
            text_conditioning (list[FluxTextConditioning]): The text conditionings to use for regional prompting.
            img_seq_len (int): The image sequence length (i.e. packed_height * packed_width).
        """
        regional_text_conditioning = cls._concat_regional_text_conditioning(text_conditioning)
        attn_mask_with_restricted_img_self_attn = cls._prepare_attn_mask(
            regional_text_conditioning, img_seq_len, restrict_img_self_attn=True
        )
        attn_mask_with_unrestricted_img_self_attn = cls._prepare_attn_mask(
            regional_text_conditioning, img_seq_len, restrict_img_self_attn=False
        )
        return cls(
            regional_text_conditioning=regional_text_conditioning,
            attn_mask_with_restricted_img_self_attn=attn_mask_with_restricted_img_self_attn,
            attn_mask_with_unrestricted_img_self_attn=attn_mask_with_unrestricted_img_self_attn,
        )

    @classmethod
    def _prepare_attn_mask(
        cls,
        regional_text_conditioning: FluxRegionalTextConditioning,
        img_seq_len: int,
        restrict_img_self_attn: bool,
    ) -> torch.Tensor:
        device = TorchDevice.choose_torch_device()

        # Infer txt_seq_len from the t5_embeddings tensor.
        txt_seq_len = regional_text_conditioning.t5_embeddings.shape[1]

        # Decide whether to compute the img self-attention region mask.
        # When compute_img_self_attn_region_mask is True, img self attention is only allowed within regions.
        # When compute_img_self_attn_region_mask is False, img self attention is not constrained.
        has_region_masks = any(mask is not None for mask in regional_text_conditioning.image_masks)
        compute_img_self_attn_region_mask = restrict_img_self_attn and has_region_masks

        # In the double stream attention blocks, the txt seq and img seq are concatenated and then attention is applied.
        # Concatenation happens in the following order: [txt_seq, img_seq].
        # There are 4 portions of the attention mask to consider as we prepare it:
        # 1. txt attends to itself
        # 2. txt attends to corresponding regional img
        # 3. regional img attends to corresponding txt
        # 4. regional img attends to itself

        # Initialize empty attention mask.
        regional_attention_mask = torch.zeros(
            (txt_seq_len + img_seq_len, txt_seq_len + img_seq_len), device=device, dtype=torch.float16
        )

        for image_mask, t5_embedding_range in zip(
            regional_text_conditioning.image_masks, regional_text_conditioning.t5_embedding_ranges, strict=True
        ):
            # 1. txt attends to itself
            regional_attention_mask[
                t5_embedding_range.start : t5_embedding_range.end, t5_embedding_range.start : t5_embedding_range.end
            ] = 1.0

            # 2. txt attends to corresponding regional img
            # Note that we reshape to (1, img_seq_len) to ensure broadcasting works as desired.
            fill_value = image_mask.view(1, img_seq_len) if image_mask is not None else 1.0
            regional_attention_mask[t5_embedding_range.start : t5_embedding_range.end, txt_seq_len:] = fill_value

            # 3. regional img attends to corresponding txt
            # Note that we reshape to (img_seq_len, 1) to ensure broadcasting works as desired.
            fill_value = image_mask.view(img_seq_len, 1) if image_mask is not None else 1.0
            regional_attention_mask[txt_seq_len:, t5_embedding_range.start : t5_embedding_range.end] = fill_value

            # 4. regional img attends to itself
            if compute_img_self_attn_region_mask and image_mask is not None:
                image_mask = image_mask.view(img_seq_len, 1)
                regional_attention_mask[txt_seq_len:, txt_seq_len:] += image_mask @ image_mask.T

        if not compute_img_self_attn_region_mask:
            # Allow unrestricted img self attention.
            regional_attention_mask[txt_seq_len:, txt_seq_len:] = 1.0

        # Convert attention mask to boolean.
        regional_attention_mask = regional_attention_mask > 0.5

        return regional_attention_mask

    @classmethod
    def _concat_regional_text_conditioning(
        cls,
        text_conditionings: list[FluxTextConditioning],
    ) -> FluxRegionalTextConditioning:
        """Concatenate regional text conditioning data into a single conditioning tensor (with associated masks)."""
        concat_t5_embeddings: list[torch.Tensor] = []
        concat_t5_embedding_ranges: list[Range] = []
        image_masks: list[torch.Tensor | None] = []

        # Choose global CLIP embedding.
        # Use the first global prompt's CLIP embedding as the global CLIP embedding. If there is no global prompt, use
        # the first prompt's CLIP embedding.
        global_clip_embedding: torch.Tensor = text_conditionings[0].clip_embeddings
        for text_conditioning in text_conditionings:
            if text_conditioning.mask is None:
                global_clip_embedding = text_conditioning.clip_embeddings
                break

        cur_t5_embedding_len = 0
        for text_conditioning in text_conditionings:
            concat_t5_embeddings.append(text_conditioning.t5_embeddings)

            concat_t5_embedding_ranges.append(
                Range(start=cur_t5_embedding_len, end=cur_t5_embedding_len + text_conditioning.t5_embeddings.shape[1])
            )

            image_masks.append(text_conditioning.mask)

            cur_t5_embedding_len += text_conditioning.t5_embeddings.shape[1]

        t5_embeddings = torch.cat(concat_t5_embeddings, dim=1)

        # Initialize the txt_ids tensor.
        pos_bs, pos_t5_seq_len, _ = t5_embeddings.shape
        t5_txt_ids = torch.zeros(
            pos_bs, pos_t5_seq_len, 3, dtype=t5_embeddings.dtype, device=TorchDevice.choose_torch_device()
        )

        return FluxRegionalTextConditioning(
            t5_embeddings=t5_embeddings,
            clip_embeddings=global_clip_embedding,
            t5_txt_ids=t5_txt_ids,
            image_masks=image_masks,
            t5_embedding_ranges=concat_t5_embedding_ranges,
        )

    @staticmethod
    def preprocess_regional_prompt_mask(
        mask: Optional[torch.Tensor], packed_height: int, packed_width: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Preprocess a regional prompt mask to match the target height and width.
        If mask is None, returns a mask of all ones with the target height and width.
        If mask is not None, resizes the mask to the target height and width using 'nearest' interpolation.

        packed_height and packed_width are the target height and width of the mask in the 'packed' latent space.

        Returns:
            torch.Tensor: The processed mask. shape: (1, 1, packed_height * packed_width).
        """

        if mask is None:
            return torch.ones((1, 1, packed_height * packed_width), dtype=dtype, device=device)

        mask = to_standard_float_mask(mask, out_dtype=dtype)

        tf = torchvision.transforms.Resize(
            (packed_height, packed_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # Add a batch dimension to the mask, because torchvision expects shape (batch, channels, h, w).
        mask = mask.unsqueeze(0)  # Shape: (1, h, w) -> (1, 1, h, w)
        resized_mask = tf(mask)

        # Flatten the height and width dimensions into a single image_seq_len dimension.
        return resized_mask.flatten(start_dim=2)
