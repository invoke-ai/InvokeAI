import pytest
import torch
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel

from invokeai.backend.krea2.attention import (
    Krea2RegionalPromptingState,
    build_krea2_attention_processors,
)
from invokeai.backend.krea2.regional_prompting import (
    Krea2RegionalPromptingExtension,
    Krea2TextConditioning,
)
from invokeai.backend.krea2.sampling_utils import prepare_position_ids


def _conditioning(length: int, value: float, mask: torch.Tensor | None = None) -> Krea2TextConditioning:
    return Krea2TextConditioning(prompt_embeds=torch.full((1, length, 12, 8), value), mask=mask)


def test_no_regional_masks_concatenates_conditionings_without_allocating_an_attention_mask() -> None:
    extension = Krea2RegionalPromptingExtension.from_text_conditionings(
        [_conditioning(2, 1.0), _conditioning(3, 2.0)],
        image_seq_len=4,
    )

    regional = extension.regional_text_conditioning
    assert regional.prompt_embeds.shape == (1, 5, 12, 8)
    assert [(item.start, item.end) for item in regional.embedding_ranges] == [(0, 2), (2, 5)]
    assert extension.attention_mask_numel == 0
    assert extension.attention_mask_build_scratch_numel == 0
    assert extension.get_attention_mask() is None


def test_restricted_attention_mask_matches_flux_style_region_semantics() -> None:
    # Sequence: [global text, regional text, region image token, background image token].
    region_mask = torch.tensor([[[1.0, 0.0]]])
    extension = Krea2RegionalPromptingExtension.from_text_conditionings(
        [_conditioning(1, 1.0), _conditioning(1, 2.0, region_mask)],
        image_seq_len=2,
    )

    assert extension.attention_mask_numel == 16
    assert extension.attention_mask_build_scratch_numel == 4
    mask = extension.get_attention_mask()
    assert mask is not None
    assert mask.dtype == torch.bool
    assert torch.equal(
        mask,
        torch.tensor(
            [
                [True, False, False, True],
                [False, True, True, False],
                [False, True, True, True],
                [True, False, True, True],
            ]
        ),
    )
    assert bool(mask.any(dim=1).all())


def test_fully_covered_disjoint_regions_cannot_attend_across_regions() -> None:
    left_mask = torch.tensor([[[1.0, 0.0]]])
    right_mask = torch.tensor([[[0.0, 1.0]]])
    extension = Krea2RegionalPromptingExtension.from_text_conditionings(
        [_conditioning(1, 1.0, left_mask), _conditioning(1, 2.0, right_mask)],
        image_seq_len=2,
    )

    mask = extension.get_attention_mask()
    assert mask is not None
    # Regional text/image cross-attention remains local.
    assert mask[0, 2] and mask[2, 0]
    assert mask[1, 3] and mask[3, 1]
    assert not mask[0, 3] and not mask[3, 0]
    assert not mask[1, 2] and not mask[2, 1]
    # With no background, image self-attention is restricted to each region.
    assert mask[2, 2] and mask[3, 3]
    assert not mask[2, 3] and not mask[3, 2]


def test_global_conditioning_applies_image_wide_when_regions_cover_the_full_image() -> None:
    left_mask = torch.tensor([[[1.0, 0.0]]])
    right_mask = torch.tensor([[[0.0, 1.0]]])
    extension = Krea2RegionalPromptingExtension.from_text_conditionings(
        [
            _conditioning(1, 1.0),
            _conditioning(1, 2.0, left_mask),
            _conditioning(1, 3.0, right_mask),
        ],
        image_seq_len=2,
    )

    mask = extension.get_attention_mask()
    assert mask is not None
    # With no uncovered background, the unmasked/global conditioning falls back to the full image.
    assert bool(mask[0, 3:].all())
    assert bool(mask[3:, 0].all())
    # Regional text and image self-attention remain isolated.
    assert mask[1, 3] and not mask[1, 4]
    assert mask[2, 4] and not mask[2, 3]
    assert not mask[3, 4] and not mask[4, 3]


def test_preprocess_mask_resizes_thresholds_and_flattens() -> None:
    raw_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )

    mask = Krea2RegionalPromptingExtension.preprocess_regional_prompt_mask(
        mask=raw_mask,
        grid_height=2,
        grid_width=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert mask.shape == (1, 1, 4)
    assert torch.equal(mask.view(2, 2), torch.tensor([[1.0, 0.0], [1.0, 0.0]]))


def test_preprocess_mask_rejects_an_unsupported_mask_rank() -> None:
    # Workflow users can wire any tensor into the encoder's mask input; an unusable rank must be rejected
    # with a clear error instead of silently reshaping into a wrong region.
    with pytest.raises(ValueError, match="Unsupported mask shape"):
        Krea2RegionalPromptingExtension.preprocess_regional_prompt_mask(
            mask=torch.ones(1, 3, 4, 4),
            grid_height=2,
            grid_width=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def _tiny_transformer() -> Krea2Transformer2DModel:
    return Krea2Transformer2DModel(
        in_channels=4,
        num_layers=3,
        attention_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        timestep_embed_dim=8,
        text_hidden_dim=16,
        num_text_layers=2,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=16,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 2, 4),
    )


def test_attention_processor_map_applies_regional_state_to_even_main_blocks_only() -> None:
    transformer = _tiny_transformer()
    state = Krea2RegionalPromptingState()

    processors = build_krea2_attention_processors(transformer, state)

    assert processors.keys() == transformer.attn_processors.keys()
    assert processors["text_fusion.layerwise_blocks.0.attn.processor"].regional_prompting_state is None
    assert processors["text_fusion.refiner_blocks.0.attn.processor"].regional_prompting_state is None
    assert processors["transformer_blocks.0.attn.processor"].regional_prompting_state is state
    assert processors["transformer_blocks.1.attn.processor"].regional_prompting_state is None
    assert processors["transformer_blocks.2.attn.processor"].regional_prompting_state is state


def test_only_even_main_blocks_apply_the_regional_mask_during_a_forward() -> None:
    # The processor map is only a wiring detail; this asserts the actual effect during a real forward pass.
    # Every block's attention input is captured while the mask is installed, then each block's attention is
    # re-run on that exact input with the mask cleared. Even blocks must change, odd blocks must not.
    torch.manual_seed(0)
    transformer = _tiny_transformer().eval()
    state = Krea2RegionalPromptingState()
    transformer.set_attn_processor(build_krea2_attention_processors(transformer, state))

    grid_height = grid_width = 2
    image_seq_len = grid_height * grid_width
    region_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])

    # _tiny_transformer uses num_text_layers=2 / text_hidden_dim=16, so it needs its own embedding shape.
    def _tiny_conditioning(mask: torch.Tensor | None) -> Krea2TextConditioning:
        return Krea2TextConditioning(prompt_embeds=torch.randn(1, 2, 2, 16), mask=mask)

    extension = Krea2RegionalPromptingExtension.from_text_conditionings(
        [_tiny_conditioning(None), _tiny_conditioning(region_mask)], image_seq_len=image_seq_len
    )
    prompt_embeds = extension.regional_text_conditioning.prompt_embeds
    state.set_attention_mask(extension.get_attention_mask())

    captured: dict[int, tuple[torch.Tensor, dict, torch.Tensor]] = {}

    def _make_hook(index: int):
        def _hook(_module, args, kwargs, output):
            captured[index] = (args[0].detach().clone(), kwargs, output.detach().clone())

        return _hook

    handles = [
        block.attn.register_forward_hook(_make_hook(index), with_kwargs=True)
        for index, block in enumerate(transformer.transformer_blocks)
    ]
    try:
        with torch.no_grad():
            transformer(
                hidden_states=torch.randn(1, image_seq_len, 4),
                encoder_hidden_states=prompt_embeds,
                timestep=torch.tensor([0.5]),
                position_ids=prepare_position_ids(prompt_embeds.shape[1], grid_height, grid_width, torch.device("cpu")),
                return_dict=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert set(captured) == {0, 1, 2}
    state.set_attention_mask(None)
    for index, (attn_input, attn_kwargs, masked_output) in captured.items():
        with torch.no_grad():
            unmasked_output = transformer.transformer_blocks[index].attn(attn_input, **attn_kwargs)
        if index % 2 == 0:
            assert not torch.allclose(masked_output, unmasked_output), f"block {index} was not restricted"
        else:
            assert torch.equal(masked_output, unmasked_output), f"block {index} should be unrestricted"


def test_regional_state_can_switch_between_positive_and_negative_masks() -> None:
    state = Krea2RegionalPromptingState()
    positive = torch.eye(4, dtype=torch.bool)
    negative = ~positive

    state.set_attention_mask(positive)
    assert state.attention_mask is positive
    state.set_attention_mask(negative)
    assert state.attention_mask is negative
    state.set_attention_mask(None)
    assert state.attention_mask is None
