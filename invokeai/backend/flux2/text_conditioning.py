from dataclasses import dataclass

import torch

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range


@dataclass
class Flux2TextConditioning:
    """Single FLUX.2 Klein text conditioning entry (Qwen3 embeddings) with optional regional mask."""

    txt_embeddings: torch.Tensor
    mask: torch.Tensor | None


@dataclass
class Flux2RegionalTextConditioning:
    """Concatenated FLUX.2 Klein regional text conditioning data.

    FLUX.2 Klein uses Qwen3 embeddings only (no CLIP pooled). The denoise loop already
    treats `clip_embeds` as unused, so we omit it here.
    """

    txt_embeddings: torch.Tensor
    txt_ids: torch.Tensor
    image_masks: list[torch.Tensor | None]
    embedding_ranges: list[Range]
