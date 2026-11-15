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


@pytest.fixture
def _masks_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        "invokeai.backend.anima.regional_prompting.TorchDevice.choose_torch_device", lambda: torch.device("cpu")
    )


@pytest.mark.usefixtures("_masks_on_cpu")
def test_unmasked_conditioning_covers_the_whole_image_only_in_negative_mode() -> None:
    # Context: [global token, region token]; image: [region token, background token].
    conditioning = AnimaRegionalTextConditioning(
        context_embeds=torch.zeros(2, 2),
        image_masks=[None, torch.tensor([[[1.0, 0.0]]])],
        context_ranges=[Range(start=0, end=1), Range(start=1, end=2)],
    )
    positive = AnimaRegionalPromptingExtension.from_regional_conditioning(conditioning, img_seq_len=2)
    negative = AnimaRegionalPromptingExtension.from_regional_conditioning(
        conditioning, img_seq_len=2, unmasked_covers_whole_image=True
    )
    assert torch.equal(positive.cross_attn_mask, torch.tensor([[False, True], [True, False]]))
    assert torch.equal(negative.cross_attn_mask, torch.tensor([[True, True], [True, False]]))

    global_only = AnimaRegionalTextConditioning(
        context_embeds=torch.zeros(1, 2), image_masks=[None], context_ranges=[Range(start=0, end=1)]
    )
    unrestricted = AnimaRegionalPromptingExtension.from_regional_conditioning(
        global_only, img_seq_len=2, unmasked_covers_whole_image=True
    )
    assert unrestricted.cross_attn_mask is None


@pytest.mark.usefixtures("_masks_on_cpu")
def test_image_tokens_outside_every_region_of_a_masked_only_list_attend_all_context() -> None:
    conditioning = AnimaRegionalTextConditioning(
        context_embeds=torch.tensor([[2.0, 0.0], [0.0, 4.0]]),
        image_masks=[torch.tensor([[[1.0, 0.0, 0.0]]]), torch.tensor([[[0.0, 1.0, 0.0]]])],
        context_ranges=[Range(start=0, end=1), Range(start=1, end=2)],
    )
    extension = AnimaRegionalPromptingExtension.from_regional_conditioning(conditioning, img_seq_len=3)
    assert torch.equal(extension.cross_attn_mask, torch.tensor([[True, False], [False, True], [True, True]]))

    state = AnimaRegionalPromptingState(extension.cross_attn_mask)
    attention = _attention()
    with patch_anima_for_regional_prompting(SimpleNamespace(blocks=[SimpleNamespace(cross_attn=attention)]), state):
        output = attention(torch.zeros(1, 3, 2), conditioning.context_embeds.unsqueeze(0))
    assert not output.isnan().any()
    torch.testing.assert_close(output, torch.tensor([[[2.0, 0.0], [0.0, 4.0], [1.0, 2.0]]]))


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
