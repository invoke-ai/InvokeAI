"""Memory-efficient attention processor for the Krea-2 transformer.

The stock ``Krea2AttnProcessor`` calls ``scaled_dot_product_attention`` with ``enable_gqa=True`` (Krea-2 uses
grouped-query attention: 48 query heads, 12 key/value heads). PyTorch's fused flash / memory-efficient SDPA
kernels do **not** support ``enable_gqa``, so this forces the *math* backend, which materializes the full
``[heads, seq, seq]`` score matrix. At 1280x720 (3600 image tokens) that is ~5.7 GB **per attention**, and it
grows O(seq^2) — ~40 GB at 2560x1440 — so generation OOMs or the cache offloads the transformer to RAM.

This processor instead expands the K/V heads to match the query heads (``repeat_interleave``) so ``enable_gqa``
is not needed, and runs under the memory-efficient SDPA backend (which supports the additive padding mask and
is O(seq) in memory). Measured: the same 3600-token attention drops from ~5.7 GB to ~0.19 GB.

The math is otherwise identical to ``Krea2AttnProcessor`` (q/k RMSNorm, rotary embeddings, sigmoid output gate).
"""

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from torch.nn.attention import SDPBackend, sdpa_kernel

# Prefer the memory-efficient kernel; fall back to flash (if the build has it) then math so we never hard-fail.
_KREA2_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]


class Krea2MemoryEfficientAttnProcessor:
    """Drop-in replacement for ``Krea2AttnProcessor`` that avoids the ``enable_gqa`` math fallback."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # [B, S, H, D] -> [B, H, S, D] for scaled_dot_product_attention.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Expand K/V heads to the query head count so we can drop enable_gqa (which forces the math backend).
        if attn.num_heads != attn.num_kv_heads:
            repeats = attn.num_heads // attn.num_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)

        with sdpa_kernel(_KREA2_SDPA_BACKENDS):
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        # [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D], matching Krea2AttnProcessor's output layout.
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return attn.to_out[0](hidden_states)
