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
