"""Hand the MiniMax H3 attention kernel head-major-contiguous Q/K/V.

``MiniMaxH3AttnProcessor`` computes Q/K/V as ``[B, seq, heads, dim]`` (sequence-major) tensors
and diffusers' ``dispatch_attention_fn`` native backend permutes them to ``[B, heads, seq, dim]``
*views* for ``scaled_dot_product_attention`` — so the flash kernel receives transposed,
non-contiguous inputs. At video sequence lengths that layout costs real throughput, measured
with the H3 attention shape (56 heads, head_dim 128, bf16) on the two dev machines:

- AMD W7900 (torch 2.10+rocm7.1, AOTriton flash), N=59500: 8.8 TFLOPS effective transposed vs
  17.9 contiguous — 2.0x. This matched a profiled real denoise step (9.33 s/call, 90% of a
  540 s step) and was the dominant term in H3's ROCm step-time deficit.
- NVIDIA RTX 5060 Ti (cu128), N=59500: 34.3 transposed vs 45.3 contiguous — 1.32x. Neutral at
  N=25000 on NVIDIA; still +20% on AMD there.

The patched ``__call__`` below is upstream's byte-for-byte (see the vendoring note in
``__init__``: the vendored module itself must stay untouched) plus one step before dispatch:
re-materialize each tensor so its memory is head-major ("BHSD") contiguous while keeping the
sequence-major shape dispatch expects — the native backend's permute then yields fully
contiguous kernel inputs. Tensors are converted one at a time and rebound immediately, so at
most one extra ``[seq, heads*dim]`` bf16 tensor (~14 KB per row) is alive beyond the unpatched
peak — inside the denoise working-memory estimate's per-row headroom.

Applied at transformer load time (both the diffusers-folder and single-file loaders),
class-level and idempotent, on every platform: the copies cost ~milliseconds against
attention-kernel savings that reach seconds per call.
"""

import torch

from invokeai.backend.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3AttnProcessor,
    _apply_rotary_emb,
    dispatch_attention_fn,
)

_SENTINEL = "_invokeai_contiguous_qkv"


def _head_major_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Return ``x`` (shape ``[B, seq, heads, dim]``) with head-major-contiguous memory.

    The result compares equal element-wise and keeps the same shape; only the underlying
    layout changes, so the downstream ``permute(0, 2, 1, 3)`` produces a contiguous tensor.
    """
    head_major = x.transpose(1, 2)
    if head_major.is_contiguous():
        return x
    return head_major.contiguous().transpose(1, 2)


def _contiguous_qkv_call(
    self,
    attn: "torch.nn.Module",
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # --- begin upstream MiniMaxH3AttnProcessor.__call__ body (unchanged) ---
    if attn.fused_projections:
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if rotary_emb is not None:
        query = _apply_rotary_emb(query, *rotary_emb)
        key = _apply_rotary_emb(key, *rotary_emb)
    # --- end upstream body ---

    # InvokeAI addition: one tensor at a time so the replaced original is freed before the
    # next copy (bounds the transient to one extra tensor).
    query = _head_major_contiguous(query)
    key = _head_major_contiguous(key)
    value = _head_major_contiguous(value)

    # --- begin upstream body (unchanged) ---
    # Without padding rows the packed sequence is a single attention document and no mask is needed (passing an
    # all-zero float mask here would hard-fail the flash / sage backends). When padding rows are present, the
    # caller supplies a boolean mask that keeps them in their own attention document, mirroring the reference's
    # `cu_seqlens = [0, used, S]` split; masked backends (SDPA & co.) are required in that case.
    hidden_states = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=self._parallel_config,
    )
    hidden_states = hidden_states.flatten(2, 3).type_as(query)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states
    # --- end upstream body ---


def patch_minimax_h3_attention_contiguous_qkv() -> None:
    """Swap ``MiniMaxH3AttnProcessor.__call__`` for the contiguous-Q/K/V variant.

    Class-level and idempotent, mirroring ``patch_minimax_h3_causal_conv3d_for_rocm``.
    Unconditional across platforms: measured neutral-to-2x on every device tested.
    """
    if getattr(MiniMaxH3AttnProcessor, _SENTINEL, False):
        return
    MiniMaxH3AttnProcessor.__call__ = _contiguous_qkv_call
    setattr(MiniMaxH3AttnProcessor, _SENTINEL, True)
