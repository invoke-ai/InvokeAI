from dataclasses import dataclass

import torch

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range


@dataclass
class FluxTextConditioning:
    t5_embeddings: torch.Tensor
    clip_embeddings: torch.Tensor
    # If mask is None, the prompt is a global prompt.
    mask: torch.Tensor | None


@dataclass
class FluxReduxConditioning:
    redux_embeddings: torch.Tensor
    # If mask is None, the prompt is a global prompt.
    mask: torch.Tensor | None


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

    # A binary mask indicating the regions of the image that the prompt should be applied to. If None, the prompt is a
    # global prompt.
    # image_masks[i] is the mask for the ith prompt.
    # image_masks[i] has shape (1, image_seq_len) and dtype torch.bool.
    image_masks: list[torch.Tensor | None]

    # List of ranges that represent the embedding ranges for each mask.
    # t5_embedding_ranges[i] contains the range of the t5 embeddings that correspond to image_masks[i].
    t5_embedding_ranges: list[Range]
