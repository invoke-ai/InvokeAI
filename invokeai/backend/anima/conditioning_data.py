"""Anima text conditioning data structures.

Anima uses a dual-conditioning scheme:
- Qwen3 0.6B hidden states (continuous embeddings)
- T5-XXL token IDs (discrete IDs, embedded by the LLM Adapter inside the transformer)

Both are produced by the text encoder invocation and stored together.

For regional prompting, multiple conditionings (each with an optional spatial mask)
are concatenated and processed together. The LLM Adapter runs on each region's
conditioning separately, producing per-region context vectors that are concatenated
for the DiT's cross-attention layers. An attention mask restricts which image tokens
attend to which regional context tokens.
"""

from dataclasses import dataclass

import torch

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range


@dataclass
class AnimaTextConditioning:
    """Anima text conditioning with Qwen3 hidden states, T5-XXL token IDs, and optional mask.

    Attributes:
        qwen3_embeds: Text embeddings from Qwen3 0.6B encoder.
            Shape: (seq_len, hidden_size) where hidden_size=1024.
        t5xxl_ids: T5-XXL token IDs for the same prompt.
            Shape: (seq_len,).
        t5xxl_weights: Per-token weights for prompt weighting.
            Shape: (seq_len,). Defaults to all ones if not provided.
        mask: Optional binary mask for regional prompting. If None, the prompt is global.
              Shape: (1, 1, img_seq_len) where img_seq_len = (H // patch_size) * (W // patch_size).
    """

    qwen3_embeds: torch.Tensor
    t5xxl_ids: torch.Tensor
    t5xxl_weights: torch.Tensor | None = None
    mask: torch.Tensor | None = None


@dataclass
class AnimaRegionalTextConditioning:
    """Container for multiple regional text conditionings processed by the LLM Adapter.

    After the LLM Adapter processes each region's conditioning, the outputs are concatenated.
    The DiT cross-attention then uses an attention mask to restrict which image tokens
    attend to which region's context tokens.

    Attributes:
        context_embeds: Concatenated LLM Adapter outputs from all regional prompts.
                        Shape: (total_context_len, 1024).
        image_masks: List of binary masks for each regional prompt.
                     If None, the prompt is global (applies to entire image).
                     Shape: (1, 1, img_seq_len).
        context_ranges: List of ranges indicating which portion of context_embeds
                       corresponds to each regional prompt.
    """

    context_embeds: torch.Tensor
    image_masks: list[torch.Tensor | None]
    context_ranges: list[Range]
