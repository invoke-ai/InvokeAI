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
    t5_embeddings: torch.Tensor
    clip_embeddings: torch.Tensor

    t5_txt_ids: torch.Tensor

    # A binary mask indicating the regions of the image that the prompt should be applied to.
    # Shape: (1, num_prompts, height, width)
    # Dtype: torch.bool
    image_masks: torch.Tensor

    # List of ranges that represent the embedding ranges for each mask.
    # t5_embedding_ranges[i] contains the range of the t5 embeddings that correspond to image_masks[i].
    # clip_embedding_ranges[i] contains the range of the clip embeddings that correspond to image_masks[i].
    t5_embedding_ranges: list[Range]
    clip_embedding_ranges: list[Range]
