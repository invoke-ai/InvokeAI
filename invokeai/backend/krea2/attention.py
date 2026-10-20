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

import re
from dataclasses import dataclass
from typing import Iterable, Protocol

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from torch.nn.attention import SDPBackend, sdpa_kernel

from invokeai.backend.krea2.style_reference import (
    Krea2StyleReferenceMode,
    Krea2StyleReferenceState,
    apply_style_reference,
    capture_style_reference,
)

# Prefer the memory-efficient kernel; fall back to flash (if the build has it) then math so we never hard-fail.
_KREA2_SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]


@dataclass
class Krea2RegionalPromptingState:
    """Mutable per-forward regional attention state shared by Krea-2 transformer-block processors."""

    attention_mask: torch.Tensor | None = None

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self.attention_mask = attention_mask


class Krea2MemoryEfficientAttnProcessor:
    """Drop-in replacement for ``Krea2AttnProcessor`` that avoids the ``enable_gqa`` math fallback."""

    def __init__(
        self,
        regional_prompting_state: Krea2RegionalPromptingState | None = None,
        style_reference_state: Krea2StyleReferenceState | None = None,
        block_index: int | None = None,
    ) -> None:
        self.regional_prompting_state = regional_prompting_state
        # Only set on blocks selected for style reference; None everywhere else, so an inactive block costs
        # nothing beyond the identity check below.
        self.style_reference_state = style_reference_state
        self.block_index = block_index

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.regional_prompting_state is not None and self.regional_prompting_state.attention_mask is not None:
            regional_attention_mask = self.regional_prompting_state.attention_mask
            if regional_attention_mask.shape != (hidden_states.shape[1], hidden_states.shape[1]):
                raise ValueError(
                    f"Krea-2 regional attention mask shape {tuple(regional_attention_mask.shape)} does not match "
                    f"the transformer sequence length {hidden_states.shape[1]}."
                )
            attention_mask = (
                regional_attention_mask if attention_mask is None else attention_mask & regional_attention_mask
            )

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

        # Style reference hooks in here: after RoPE (so the captured keys carry their rotation) but before
        # the GQA head expansion, which keeps the retained cache 4x smaller. See style_reference.py.
        style_state = self.style_reference_state
        style_mode = Krea2StyleReferenceMode.OFF if style_state is None else style_state.mode
        injection = None
        if style_state is not None and self.block_index is not None:
            if style_mode is Krea2StyleReferenceMode.CAPTURE:
                capture_style_reference(style_state, self.block_index, query, key, value)
            elif style_mode is Krea2StyleReferenceMode.INJECT:
                injection = apply_style_reference(style_state, self.block_index, query, key, value)

        def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            # Expand K/V heads to the query head count so we can drop enable_gqa (which forces the math backend).
            if attn.num_heads != attn.num_kv_heads:
                repeats = attn.num_heads // attn.num_kv_heads
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)
            with sdpa_kernel(_KREA2_SDPA_BACKENDS):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        if injection is None:
            hidden_states = attend(query, key, value, attention_mask)
        else:
            # The reference keys/values are appended along the token axis, so a regional mask has to grow
            # with them.
            hidden_states = attend(
                injection.query,
                injection.key,
                injection.value,
                style_state.pad_attention_mask(attention_mask),
            )
            if injection.attention_mix < 1.0:
                # Blend against the same (AdaIN'd) query/key attending to the target's own tokens only.
                # At the default style_strength of 1.0 this branch is dead and the second attention is skipped.
                sequence_length = injection.query.shape[2]
                native = attend(
                    injection.query,
                    injection.key[:, :, :sequence_length, :],
                    injection.value[:, :, :sequence_length, :],
                    attention_mask,
                )
                hidden_states = native * (1.0 - injection.attention_mix) + hidden_states * injection.attention_mix

        # [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D], matching Krea2AttnProcessor's output layout.
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return attn.to_out[0](hidden_states)


class _Krea2AttentionProcessorContainer(Protocol):
    @property
    def attn_processors(self) -> dict[str, object]: ...


def build_krea2_attention_processors(
    transformer: _Krea2AttentionProcessorContainer,
    regional_prompting_state: Krea2RegionalPromptingState,
    style_reference_state: Krea2StyleReferenceState | None = None,
    style_reference_blocks: Iterable[int] | None = None,
) -> dict[str, Krea2MemoryEfficientAttnProcessor]:
    """Build processors that apply regional masks to alternating main transformer blocks only.

    Style reference runs over its own, independent band of blocks (upstream's default is 7-27, i.e. both
    parities), so it gets a second state object rather than sharing the regional one. Leaving both style
    arguments unset reproduces the pre-style behaviour exactly.

    The text-fusion blocks do not match the main-block pattern, so they never receive either state. That is
    correct: they only ever see text tokens.
    """

    style_blocks = frozenset(style_reference_blocks or ())
    processors: dict[str, Krea2MemoryEfficientAttnProcessor] = {}
    for name in transformer.attn_processors:
        match = re.fullmatch(r"transformer_blocks\.(\d+)\.attn\.processor", name)
        block_index = int(match.group(1)) if match is not None else None
        regional = regional_prompting_state if block_index is not None and block_index % 2 == 0 else None
        style = (
            style_reference_state
            if style_reference_state is not None and block_index is not None and block_index in style_blocks
            else None
        )
        processors[name] = Krea2MemoryEfficientAttnProcessor(
            regional_prompting_state=regional,
            style_reference_state=style,
            block_index=block_index,
        )
    return processors
