import pytest
import torch
from diffusers.models.transformers.transformer_krea2 import Krea2Attention, Krea2AttnProcessor
from torch.nn.attention import SDPBackend

import invokeai.backend.krea2.attention as krea2_attention
from invokeai.backend.krea2.attention import Krea2MemoryEfficientAttnProcessor, Krea2RegionalPromptingState


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


def test_regional_state_matches_stock_processor_with_a_dense_attention_mask() -> None:
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    mask = torch.tril(torch.ones(24, 24, dtype=torch.bool))
    state = Krea2RegionalPromptingState(attention_mask=mask)

    with torch.no_grad():
        attn.set_processor(Krea2AttnProcessor())
        out_stock = attn(hidden_states, attention_mask=mask, image_rotary_emb=None)
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))
        out_regional = attn(hidden_states, attention_mask=None, image_rotary_emb=None)

    assert torch.allclose(out_stock, out_regional, atol=1e-4, rtol=1e-4)


def test_regional_state_rejects_a_mask_sized_for_a_different_conditioning() -> None:
    # The positive and negative conditionings can tokenize to different lengths, so each denoise pass must
    # install its own mask. If the wrong mask is left on the shared state the processor must fail loudly
    # rather than broadcast a mismatched mask into attention.
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    state = Krea2RegionalPromptingState(attention_mask=torch.ones(20, 20, dtype=torch.bool))
    attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))

    with pytest.raises(ValueError, match=r"\(20, 20\) does not match the transformer sequence length 24"):
        attn(hidden_states, attention_mask=None, image_rotary_emb=None)


def test_processor_without_regional_state_ignores_the_shared_mask() -> None:
    # Odd-numbered main blocks are built with regional_prompting_state=None so they stay unrestricted. Setting
    # a mask on the shared state must not leak into them.
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    state = Krea2RegionalPromptingState(attention_mask=torch.block_diag(*[torch.ones(12, 12, dtype=torch.bool)] * 2))

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=None))
        out_unrestricted = attn(hidden_states, attention_mask=None, image_rotary_emb=None)
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))
        out_restricted = attn(hidden_states, attention_mask=None, image_rotary_emb=None)

    # Sanity check that the mask is strong enough to change the result at all, then that the unrestricted
    # processor is unaffected by it.
    assert not torch.allclose(out_unrestricted, out_restricted, atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor())
        out_no_state = attn(hidden_states, attention_mask=None, image_rotary_emb=None)
    assert torch.equal(out_unrestricted, out_no_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise fused SDPA")
def test_cuda_memory_efficient_sdpa_accepts_dense_regional_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    attn = _build_gqa_attention().to(device="cuda", dtype=torch.float16)
    hidden_states = torch.randn(1, 24, attn.hidden_size, device="cuda", dtype=torch.float16)
    mask = torch.block_diag(
        torch.ones(12, 12, device="cuda", dtype=torch.bool),
        torch.ones(12, 12, device="cuda", dtype=torch.bool),
    )
    state = Krea2RegionalPromptingState(attention_mask=mask)
    monkeypatch.setattr(krea2_attention, "_KREA2_SDPA_BACKENDS", [SDPBackend.EFFICIENT_ATTENTION])

    head_dim = attn.hidden_size // attn.num_heads
    sdpa_tensor = torch.empty(1, attn.num_heads, 24, head_dim, device="cuda", dtype=torch.float16)
    sdpa_params = torch.backends.cuda.SDPAParams(sdpa_tensor, sdpa_tensor, sdpa_tensor, mask, 0.0, False, False)
    if not torch.backends.cuda.can_use_efficient_attention(sdpa_params):
        pytest.skip("This CUDA device/build does not support dense masks with memory-efficient SDPA")

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(regional_prompting_state=state))
        output = attn(hidden_states, attention_mask=None, image_rotary_emb=None)

    assert output.is_cuda
    assert torch.isfinite(output).all()
