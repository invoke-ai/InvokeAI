from dataclasses import dataclass

import torch

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range


@dataclass
class FluxTextConditioning:
    t5_embeddings: torch.Tensor
    clip_embeddings: torch.Tensor
    mask: torch.Tensor


@dataclass
class FluxRegionalTextConditioning:
    # Concatenated text embeddings.
    # Shape: (1, concatenated_txt_seq_len, 4096)
    t5_embeddings: torch.Tensor
    # Shape: (1, concatenated_txt_seq_len, 3)
    t5_txt_ids: torch.Tensor

    # Global CLIP embeddings.
    # Shape: (1, 768)
    clip_embeddings: torch.Tensor

    # A binary mask indicating the regions of the image that the prompt should be applied to.
    # Shape: (1, num_prompts, image_seq_len)
    # Dtype: torch.bool
    image_masks: torch.Tensor

    # List of ranges that represent the embedding ranges for each mask.
    # t5_embedding_ranges[i] contains the range of the t5 embeddings that correspond to image_masks[i].
    t5_embedding_ranges: list[Range]
