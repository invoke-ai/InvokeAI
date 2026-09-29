import pytest
import torch
from diffusers.models.transformers.transformer_krea2 import Krea2Attention, Krea2AttnProcessor
from torch.nn.attention import SDPBackend

import invokeai.backend.krea2.attention as krea2_attention
from invokeai.backend.krea2.attention import (
    Krea2MemoryEfficientAttnProcessor,
    Krea2RegionalPromptingState,
    build_krea2_attention_processors,
)
from invokeai.backend.krea2.style_reference import (
    Krea2StyleReferenceSettings,
    Krea2StyleReferenceState,
    resolve_effective_settings,
)


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


# --- style reference -----------------------------------------------------------------------------


class _StubTransformer:
    def __init__(self, num_blocks: int) -> None:
        self.attn_processors = {f"transformer_blocks.{i}.attn.processor": object() for i in range(num_blocks)}
        self.attn_processors["text_fusion.layerwise_blocks.0.attn.processor"] = object()


def _style_state(image_seq_len: int, **overrides) -> Krea2StyleReferenceState:
    # head_dim is 32 for the test attention (hidden 256 / 8 heads), so the axes must sum to 32.
    return Krea2StyleReferenceState(
        settings=resolve_effective_settings(Krea2StyleReferenceSettings(**overrides)),
        image_seq_len=image_seq_len,
        axes_dims_rope=(8, 12, 12),
    )


def test_builder_gives_the_style_state_only_to_the_configured_blocks() -> None:
    regional = Krea2RegionalPromptingState()
    style = _style_state(4)

    processors = build_krea2_attention_processors(
        _StubTransformer(12), regional, style_reference_state=style, style_reference_blocks={7, 8}
    )

    styled = {name for name, p in processors.items() if p.style_reference_state is not None}
    assert styled == {"transformer_blocks.7.attn.processor", "transformer_blocks.8.attn.processor"}


def test_builder_keeps_the_regional_mask_on_even_blocks_only_when_style_is_active() -> None:
    # Style runs over both parities (7-27); that must not widen the regional mask's even-only band.
    regional = Krea2RegionalPromptingState()
    processors = build_krea2_attention_processors(
        _StubTransformer(12), regional, style_reference_state=_style_state(4), style_reference_blocks=set(range(7, 12))
    )

    for index in range(12):
        processor = processors[f"transformer_blocks.{index}.attn.processor"]
        assert (processor.regional_prompting_state is not None) == (index % 2 == 0)
    # Block 8 is even and inside the style band, so it carries both states at once.
    both = processors["transformer_blocks.8.attn.processor"]
    assert both.regional_prompting_state is not None and both.style_reference_state is not None


def test_builder_without_style_arguments_reproduces_the_previous_behaviour() -> None:
    processors = build_krea2_attention_processors(_StubTransformer(4), Krea2RegionalPromptingState())
    assert all(processor.style_reference_state is None for processor in processors.values())


def test_builder_never_styles_the_text_fusion_blocks() -> None:
    # They only ever see text tokens, so there is no image-token range to capture.
    processors = build_krea2_attention_processors(
        _StubTransformer(4), Krea2RegionalPromptingState(), _style_state(4), style_reference_blocks=set(range(4))
    )
    assert processors["text_fusion.layerwise_blocks.0.attn.processor"].style_reference_state is None


def test_capture_pass_leaves_the_attention_output_unchanged() -> None:
    # The reference pass must be a plain forward; it only observes.
    attn = _build_gqa_attention()
    hidden_states = torch.randn(1, 24, attn.hidden_size)
    state = _style_state(16)
    state.begin_capture()

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor())
        out_plain = attn(hidden_states, attention_mask=None, image_rotary_emb=None)
        attn.set_processor(Krea2MemoryEfficientAttnProcessor(style_reference_state=state, block_index=0))
        out_capture = attn(hidden_states, attention_mask=None, image_rotary_emb=None)

    assert torch.equal(out_plain, out_capture)
    assert state.get(0).reference_key.shape == (1, attn.num_kv_heads, 16, attn.head_dim)


def test_inject_pass_changes_the_output_and_preserves_its_shape() -> None:
    attn = _build_gqa_attention()
    reference = torch.randn(1, 24, attn.hidden_size)
    target = torch.randn(1, 24, attn.hidden_size)
    state = _style_state(16)

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor())
        out_plain = attn(target, attention_mask=None, image_rotary_emb=None)

        processor = Krea2MemoryEfficientAttnProcessor(style_reference_state=state, block_index=0)
        attn.set_processor(processor)
        state.begin_capture()
        attn(reference, attention_mask=None, image_rotary_emb=None)
        state.begin_inject(0.0)
        out_styled = attn(target, attention_mask=None, image_rotary_emb=None)

    assert out_styled.shape == out_plain.shape
    assert not torch.allclose(out_styled, out_plain, atol=1e-4)


def test_style_strength_of_zero_reproduces_the_unstyled_output() -> None:
    attn = _build_gqa_attention()
    reference = torch.randn(1, 24, attn.hidden_size)
    target = torch.randn(1, 24, attn.hidden_size)
    state = _style_state(16, style_strength=0.0)

    with torch.no_grad():
        attn.set_processor(Krea2MemoryEfficientAttnProcessor())
        out_plain = attn(target, attention_mask=None, image_rotary_emb=None)

        attn.set_processor(Krea2MemoryEfficientAttnProcessor(style_reference_state=state, block_index=0))
        state.begin_capture()
        attn(reference, attention_mask=None, image_rotary_emb=None)
        state.begin_inject(0.5)
        out_styled = attn(target, attention_mask=None, image_rotary_emb=None)

    assert torch.allclose(out_plain, out_styled, atol=1e-6)


def test_regional_mask_is_key_padded_when_the_reference_is_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    # The reference keys are appended along the token axis, so a square regional mask no longer fits.
    attn = _build_gqa_attention()
    reference = torch.randn(1, 24, attn.hidden_size)
    target = torch.randn(1, 24, attn.hidden_size)
    style = _style_state(16)
    regional = Krea2RegionalPromptingState(attention_mask=torch.tril(torch.ones(24, 24, dtype=torch.bool)))

    seen: list[torch.Tensor | None] = []
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def record(query, key, value, attn_mask=None, **kwargs):
        seen.append(attn_mask)
        return original_sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(krea2_attention.F, "scaled_dot_product_attention", record)

    with torch.no_grad():
        processor = Krea2MemoryEfficientAttnProcessor(
            regional_prompting_state=regional, style_reference_state=style, block_index=0
        )
        attn.set_processor(processor)
        style.begin_capture()
        regional.set_attention_mask(None)
        attn(reference, attention_mask=None, image_rotary_emb=None)
        style.begin_inject(0.0)
        regional.set_attention_mask(torch.tril(torch.ones(24, 24, dtype=torch.bool)))
        attn(target, attention_mask=None, image_rotary_emb=None)

    styled_mask = seen[-1]
    assert styled_mask is not None
    assert styled_mask.shape == (24, 24 + 16)
    # Every target query may see the reference, in every region.
    assert bool(styled_mask[:, 24:].all())
