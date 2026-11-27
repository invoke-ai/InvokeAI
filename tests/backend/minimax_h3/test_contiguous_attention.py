"""Tests for the contiguous-Q/K/V attention layout patch.

The patch must be a pure layout change: same math (within kernel-selection float tolerance),
same shapes, applied idempotently at the class level, covering the pruned subclass too.
"""

import pytest
import torch

from invokeai.backend.minimax_h3.contiguous_attention import (
    _contiguous_qkv_call,
    _head_major_contiguous,
    patch_minimax_h3_attention_contiguous_qkv,
)
from invokeai.backend.minimax_h3.transformer_minimax_h3 import MiniMaxH3Attention, MiniMaxH3AttnProcessor


@pytest.fixture
def restore_processor():
    """Run a test with the pristine upstream __call__, restoring it afterward."""
    original_call = MiniMaxH3AttnProcessor.__dict__.get("__call__")
    sentinel = getattr(MiniMaxH3AttnProcessor, "_invokeai_contiguous_qkv", None)
    yield
    if original_call is not None:
        MiniMaxH3AttnProcessor.__call__ = original_call
    if sentinel is None and hasattr(MiniMaxH3AttnProcessor, "_invokeai_contiguous_qkv"):
        delattr(MiniMaxH3AttnProcessor, "_invokeai_contiguous_qkv")


def test_head_major_contiguous_is_elementwise_identity():
    x = torch.randn(1, 64, 4, 8)  # [B, seq, heads, dim], seq-major contiguous
    y = _head_major_contiguous(x)
    assert torch.equal(x, y)
    assert y.shape == x.shape
    # The memory layout changed: the head-major permute of the result is contiguous.
    assert y.transpose(1, 2).is_contiguous()
    # Already-head-major input is returned untouched (no copy).
    z = _head_major_contiguous(y)
    assert z is y


def test_patched_call_matches_upstream(restore_processor):
    torch.manual_seed(0)
    attn = MiniMaxH3Attention(hidden_size=32, heads=4, dim_head=8).eval()
    hidden_states = torch.randn(1, 64, 32)
    # cos/sin are (seq_len, rotary_dim) with an even rotary_dim <= head_dim.
    angles = torch.outer(torch.arange(64, dtype=torch.float32), torch.tensor([1.0, 0.5, 0.25, 0.125]))
    cos, sin = torch.cos(angles), torch.sin(angles)

    with torch.no_grad():
        reference = MiniMaxH3AttnProcessor()(attn, hidden_states, rotary_emb=(cos, sin))
        processor = MiniMaxH3AttnProcessor()
        patched = _contiguous_qkv_call(processor, attn, hidden_states, rotary_emb=(cos, sin))

    assert patched.shape == reference.shape
    torch.testing.assert_close(patched, reference)


def test_patch_is_class_level_and_idempotent(restore_processor):
    patch_minimax_h3_attention_contiguous_qkv()
    first = MiniMaxH3AttnProcessor.__call__
    patch_minimax_h3_attention_contiguous_qkv()
    assert MiniMaxH3AttnProcessor.__call__ is first
    assert first is _contiguous_qkv_call


def test_patch_covers_pruned_transformer_blocks(restore_processor):
    """The pruned subclass reuses the vendored attention classes, so the class-level patch
    must reach a block instantiated from either model class."""
    from invokeai.backend.minimax_h3.transformer_minimax_h3 import MiniMaxH3TransformerBlock

    patch_minimax_h3_attention_contiguous_qkv()
    block = MiniMaxH3TransformerBlock(
        hidden_size=32,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        time_embed_dim=16,
        norm_eps=1e-6,
        qk_norm_eps=1e-5,
    )
    assert type(block.attn.processor).__call__ is _contiguous_qkv_call
