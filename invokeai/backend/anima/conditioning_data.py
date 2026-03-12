"""Anima text conditioning data structures.

Anima uses a dual-conditioning scheme:
- Qwen3 0.6B hidden states (continuous embeddings)
- T5-XXL token IDs (discrete IDs, embedded by the LLM Adapter inside the transformer)

Both are produced by the text encoder invocation and stored together.
"""

from dataclasses import dataclass

import torch


@dataclass
class AnimaTextConditioning:
    """Anima text conditioning with Qwen3 hidden states and T5-XXL token IDs.

    Attributes:
        qwen3_embeds: Text embeddings from Qwen3 0.6B encoder.
            Shape: (seq_len, hidden_size) where hidden_size=1024.
        t5xxl_ids: T5-XXL token IDs for the same prompt.
            Shape: (seq_len,).
        t5xxl_weights: Per-token weights for prompt weighting.
            Shape: (seq_len,). Defaults to all ones if not provided.
    """

    qwen3_embeds: torch.Tensor
    t5xxl_ids: torch.Tensor
    t5xxl_weights: torch.Tensor | None = None
