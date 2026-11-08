"""Utilities for patching the AnimaTransformer to support regional cross-attention masks."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange


@dataclass
class AnimaRegionalPromptingState:
    """The active CFG branch's mask, scoped to one denoise invocation."""

    cross_attn_mask: torch.Tensor | None = None


def _patched_cross_attn_forward(
    original_forward,
    state: AnimaRegionalPromptingState,
):
    """Create a patched forward for CosmosAttention that injects a cross-attention mask.

    Args:
        original_forward: The original CosmosAttention.forward method (bound to self).
        state: Selects the current pass's cross-attention mask.
    """

    def forward(x, context=None, rope_emb=None):
        attn_mask = state.cross_attn_mask
        if attn_mask is None:
            return original_forward(x, context, rope_emb=rope_emb)

        self = original_forward.__self__

        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = (rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim) for t in (q, k, v))

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        if self.is_selfattn and rope_emb is not None:
            from invokeai.backend.anima.anima_transformer import apply_rotary_pos_emb_cosmos

            q = apply_rotary_pos_emb_cosmos(q, rope_emb)
            k = apply_rotary_pos_emb_cosmos(k, rope_emb)

        in_q_shape = q.shape
        in_k_shape = k.shape
        q = rearrange(q, "b ... h d -> b h ... d").reshape(in_q_shape[0], in_q_shape[-2], -1, in_q_shape[-1])
        k = rearrange(k, "b ... h d -> b h ... d").reshape(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])
        v = rearrange(v, "b ... h d -> b h ... d").reshape(in_k_shape[0], in_k_shape[-2], -1, in_k_shape[-1])

        # SDPA accepts the shared boolean mask directly: True allows attention.
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.unsqueeze(0).unsqueeze(0))
        result = rearrange(result, "b h s d -> b s (h d)")
        return self.output_dropout(self.output_proj(result))

    return forward


@contextmanager
def patch_anima_for_regional_prompting(
    transformer,
    state: AnimaRegionalPromptingState,
):
    """Context manager to temporarily patch the Anima transformer for regional prompting.

    Patches the cross-attention in each DiT block to use a regional attention mask.
    Uses alternating pattern: masked on even blocks, unmasked on odd blocks for
    global coherence.

    Args:
        transformer: The AnimaTransformer instance.
        state: The caller selects a branch's mask before each transformer pass.

    Yields:
        The (possibly patched) transformer.
    """
    original_forwards = []
    try:
        for block in transformer.blocks[::2]:
            original_forwards.append((block.cross_attn, block.cross_attn.forward))
            block.cross_attn.forward = _patched_cross_attn_forward(block.cross_attn.forward, state)
        yield transformer
    finally:
        state.cross_attn_mask = None
        for attention, original_forward in reversed(original_forwards):
            attention.forward = original_forward
