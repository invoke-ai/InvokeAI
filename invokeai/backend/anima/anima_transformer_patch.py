"""Utilities for patching the AnimaTransformer to support regional cross-attention masks."""

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from invokeai.backend.anima.regional_prompting import AnimaRegionalPromptingExtension


def _patched_cross_attn_forward(
    original_forward,
    attn_mask: torch.Tensor,
):
    """Create a patched forward for CosmosAttention that injects a cross-attention mask.

    Args:
        original_forward: The original CosmosAttention.forward method (bound to self).
        attn_mask: Cross-attention mask of shape (img_seq_len, context_seq_len).
    """

    def forward(x, context=None, rope_emb=None):
        # If the context sequence length doesn't match the mask (e.g. negative conditioning
        # has a different number of tokens than positive regional conditioning), skip masking
        # and use the original unmasked forward.
        actual_context = x if context is None else context
        if actual_context.shape[-2] != attn_mask.shape[1]:
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

        # Convert boolean mask to float additive mask for SDPA
        # True (attend) -> 0.0, False (block) -> -inf
        # Shape: (img_seq_len, context_seq_len) -> (1, 1, img_seq_len, context_seq_len)
        float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        float_mask[~attn_mask] = float("-inf")
        expanded_mask = float_mask.unsqueeze(0).unsqueeze(0)

        result = F.scaled_dot_product_attention(q, k, v, attn_mask=expanded_mask)
        result = rearrange(result, "b h s d -> b s (h d)")
        return self.output_dropout(self.output_proj(result))

    return forward


@contextmanager
def patch_anima_for_regional_prompting(
    transformer,
    regional_extension: Optional[AnimaRegionalPromptingExtension],
):
    """Context manager to temporarily patch the Anima transformer for regional prompting.

    Patches the cross-attention in each DiT block to use a regional attention mask.
    Uses alternating pattern: masked on even blocks, unmasked on odd blocks for
    global coherence.

    Args:
        transformer: The AnimaTransformer instance.
        regional_extension: The regional prompting extension. If None or no mask, no patching.

    Yields:
        The (possibly patched) transformer.
    """
    if regional_extension is None or regional_extension.cross_attn_mask is None:
        yield transformer
        return

    # Store original forwards
    original_forwards = []
    for block_idx, block in enumerate(transformer.blocks):
        original_forwards.append(block.cross_attn.forward)

        mask = regional_extension.get_cross_attn_mask(block_idx)
        if mask is not None:
            block.cross_attn.forward = _patched_cross_attn_forward(block.cross_attn.forward, mask)

    try:
        yield transformer
    finally:
        # Restore original forwards
        for block_idx, block in enumerate(transformer.blocks):
            block.cross_attn.forward = original_forwards[block_idx]
