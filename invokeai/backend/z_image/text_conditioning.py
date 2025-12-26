from dataclasses import dataclass

import torch

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range


@dataclass
class ZImageTextConditioning:
    """Z-Image text conditioning with optional regional mask.

    Attributes:
        prompt_embeds: Text embeddings from Qwen3 encoder. Shape: (seq_len, hidden_size).
        mask: Optional binary mask for regional prompting. If None, the prompt is global.
              Shape: (1, 1, img_seq_len) where img_seq_len = (H // patch_size) * (W // patch_size).
    """

    prompt_embeds: torch.Tensor
    mask: torch.Tensor | None = None


@dataclass
class ZImageRegionalTextConditioning:
    """Container for multiple regional text conditionings concatenated together.

    In Z-Image, the unified sequence is [img_tokens, txt_tokens], which is different
    from FLUX where it's [txt_tokens, img_tokens]. The attention mask must account for this.

    Attributes:
        prompt_embeds: Concatenated text embeddings from all regional prompts.
                       Shape: (total_seq_len, hidden_size).
        image_masks: List of binary masks for each regional prompt.
                     image_masks[i] corresponds to embedding_ranges[i].
                     If None, the prompt is global (applies to entire image).
                     Shape: (1, 1, img_seq_len).
        embedding_ranges: List of ranges indicating which portion of prompt_embeds
                         corresponds to each regional prompt.
    """

    prompt_embeds: torch.Tensor
    image_masks: list[torch.Tensor | None]
    embedding_ranges: list[Range]

    @classmethod
    def from_text_conditionings(
        cls,
        text_conditionings: list[ZImageTextConditioning],
    ) -> "ZImageRegionalTextConditioning":
        """Create a ZImageRegionalTextConditioning from a list of ZImageTextConditioning objects.

        Args:
            text_conditionings: List of text conditionings, each with optional mask.

        Returns:
            A single ZImageRegionalTextConditioning with concatenated embeddings.
        """
        concat_embeds: list[torch.Tensor] = []
        concat_ranges: list[Range] = []
        image_masks: list[torch.Tensor | None] = []

        cur_embed_len = 0
        for tc in text_conditionings:
            concat_embeds.append(tc.prompt_embeds)
            concat_ranges.append(Range(start=cur_embed_len, end=cur_embed_len + tc.prompt_embeds.shape[0]))
            image_masks.append(tc.mask)
            cur_embed_len += tc.prompt_embeds.shape[0]

        prompt_embeds = torch.cat(concat_embeds, dim=0)

        return cls(
            prompt_embeds=prompt_embeds,
            image_masks=image_masks,
            embedding_ranges=concat_ranges,
        )
