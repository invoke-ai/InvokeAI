import torch
from diffusers.models.transformers.transformer_krea2 import Krea2Attention, Krea2AttnProcessor

from invokeai.backend.krea2.attention import Krea2MemoryEfficientAttnProcessor


def _build_gqa_attention() -> Krea2Attention:
    # Krea-2's main blocks use grouped-query attention: more query heads than key/value heads.
    torch.manual_seed(0)
    attn = Krea2Attention(hidden_size=256, num_heads=8, num_kv_heads=2, eps=1e-5).eval()
    assert attn.num_heads != attn.num_kv_heads
    return attn


def test_memory_efficient_processor_matches_stock_processor() -> None:
    # The memory-efficient processor expands the KV heads and uses the O(seq) SDPA kernel instead of the
    # enable_gqa math path, but must be numerically equivalent to the stock Krea2AttnProcessor.
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    mask = torch.ones(1, 1, 1, 24, dtype=torch.bool)

    with torch.no_grad():
        attn.set_processor(Krea2AttnProcessor())
        out_stock = attn(hidden_states, attention_mask=mask, image_rotary_emb=None)
        attn.set_processor(Krea2MemoryEfficientAttnProcessor())
        out_efficient = attn(hidden_states, attention_mask=mask, image_rotary_emb=None)

    assert out_stock.shape == out_efficient.shape
    assert torch.allclose(out_stock, out_efficient, atol=1e-4, rtol=1e-4)


def test_memory_efficient_processor_handles_equal_head_counts() -> None:
    # The text-fusion attention has num_heads == num_kv_heads (no GQA); the processor must skip the KV expansion
    # and still produce the right result.
    torch.manual_seed(0)
    attn = Krea2Attention(hidden_size=256, num_heads=8, num_kv_heads=8, eps=1e-5).eval()
    hidden_states = torch.randn(1, 24, attn.hidden_size)

    with torch.no_grad():
        attn.set_processor(Krea2AttnProcessor())
        out_stock = attn(hidden_states, attention_mask=None, image_rotary_emb=None)
        attn.set_processor(Krea2MemoryEfficientAttnProcessor())
        out_efficient = attn(hidden_states, attention_mask=None, image_rotary_emb=None)

    assert torch.allclose(out_stock, out_efficient, atol=1e-4, rtol=1e-4)
