"""Regional masks select a CFG branch explicitly, never by its context length."""

from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.anima.anima_transformer import CosmosAttention
from invokeai.backend.anima.anima_transformer_patch import (
    AnimaRegionalPromptingState,
    patch_anima_for_regional_prompting,
)
from invokeai.backend.anima.conditioning_data import AnimaRegionalTextConditioning
from invokeai.backend.anima.regional_prompting import AnimaRegionalPromptingExtension
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range


def _attention() -> CosmosAttention:
    attention = CosmosAttention(query_dim=2, context_dim=2, n_heads=1, head_dim=2)
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.k_proj.weight.zero_()
        attention.v_proj.weight.copy_(torch.eye(2))
        attention.output_proj.weight.copy_(torch.eye(2))
    return attention


@pytest.mark.parametrize("fail", [False, True])
def test_switches_equal_length_masks_without_affecting_unmasked_passes_and_restores(fail: bool) -> None:
    transformer = SimpleNamespace(blocks=[SimpleNamespace(cross_attn=_attention()) for _ in range(2)])
    original_forwards = [block.cross_attn.forward for block in transformer.blocks]
    state = AnimaRegionalPromptingState()
    query = torch.zeros(1, 2, 2)
    context = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])

    class Stop(Exception):
        pass

    try:
        with patch_anima_for_regional_prompting(transformer, state):
            state.cross_attn_mask = torch.eye(2, dtype=torch.bool)
            torch.testing.assert_close(transformer.blocks[0].cross_attn(query, context), context)
            # Odd blocks retain unrestricted attention to keep global coherence.
            torch.testing.assert_close(
                transformer.blocks[1].cross_attn(query, context), torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
            )
            state.cross_attn_mask = ~state.cross_attn_mask
            torch.testing.assert_close(transformer.blocks[0].cross_attn(query, context), context.flip(1))
            state.cross_attn_mask = None
            torch.testing.assert_close(
                transformer.blocks[0].cross_attn(query, context), torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
            )
            if fail:
                state.cross_attn_mask = torch.eye(2, dtype=torch.bool)
                raise Stop
    except Stop:
        pass
    assert state.cross_attn_mask is None
    assert [block.cross_attn.forward for block in transformer.blocks] == original_forwards


def test_restores_already_patched_blocks_when_installation_fails() -> None:
    attention = _attention()
    transformer = SimpleNamespace(blocks=[SimpleNamespace(cross_attn=attention), None, SimpleNamespace()])
    original_forward = attention.forward
    state = AnimaRegionalPromptingState(torch.ones(2, 2, dtype=torch.bool))
    with pytest.raises(AttributeError):
        with patch_anima_for_regional_prompting(transformer, state):
            pytest.fail("The invalid block should fail patch installation")
    assert attention.forward == original_forward
    assert state.cross_attn_mask is None


def test_masked_only_conditioning_leaves_uncovered_queries_finite(monkeypatch) -> None:
    monkeypatch.setattr(
        "invokeai.backend.anima.regional_prompting.TorchDevice.choose_torch_device", lambda: torch.device("cpu")
    )
    conditioning = AnimaRegionalTextConditioning(
        context_embeds=torch.tensor([[2.0, 4.0]]),
        image_masks=[torch.tensor([[[1.0, 0.0]]])],
        context_ranges=[Range(start=0, end=1)],
    )
    extension = AnimaRegionalPromptingExtension.from_regional_conditioning(conditioning, img_seq_len=2)
    state = AnimaRegionalPromptingState(extension.cross_attn_mask)
    attention = _attention()
    with patch_anima_for_regional_prompting(SimpleNamespace(blocks=[SimpleNamespace(cross_attn=attention)]), state):
        output = attention(torch.zeros(1, 2, 2), conditioning.context_embeds.unsqueeze(0))
    torch.testing.assert_close(output, torch.tensor([[[2.0, 4.0], [0.0, 0.0]]]))


def test_boolean_mask_matches_additive_bias_attention(monkeypatch) -> None:
    """Keep the previous masked attention's numerics while avoiding a Python-side bias per block."""
    import torch.nn.functional as functional

    generator = torch.Generator().manual_seed(47)
    transformer = SimpleNamespace(blocks=[SimpleNamespace(cross_attn=CosmosAttention(8, 8, n_heads=2, head_dim=4))])
    query = torch.randn(1, 3, 8, generator=generator)
    context = torch.randn(1, 4, 8, generator=generator)
    mask = torch.tensor([[True, False, True, True], [False, True, False, False], [False, False, False, False]])
    state = AnimaRegionalPromptingState(mask)
    original_sdpa = functional.scaled_dot_product_attention

    def additive_sdpa(q, k, v, *, attn_mask):
        bias = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(~attn_mask, -torch.inf)
        return original_sdpa(q, k, v, attn_mask=bias)

    with patch_anima_for_regional_prompting(transformer, state):
        boolean_output = transformer.blocks[0].cross_attn(query, context)
        monkeypatch.setattr(functional, "scaled_dot_product_attention", additive_sdpa)
        additive_output = transformer.blocks[0].cross_attn(query, context)
    torch.testing.assert_close(boolean_output, additive_output)
