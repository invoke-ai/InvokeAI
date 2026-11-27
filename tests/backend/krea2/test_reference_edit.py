"""Krea-2 edit references: clean reference tokens ride along in the sequence and are modulated at t=0.

`forward_with_references` restates the diffusers forward to give the reference span its own modulation,
so these tests pin it against the stock modules (drift guard) and against an independent formulation of
the reference method (ComfyUI `timestep_zero_index`, ostris `_forward_with_refs`). The K/V-cache mode is
pinned against the same joint pass with ai-toolkit's isolating mask written out.
"""

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_krea2 import (
    Krea2AttnProcessor,
    Krea2RotaryPosEmbed,
    Krea2Transformer2DModel,
    Krea2TransformerBlock,
)

from invokeai.backend.krea2.attention import Krea2RegionalPromptingState, build_krea2_attention_processors
from invokeai.backend.krea2.reference_edit import (
    Krea2ReferenceKVCache,
    block_forward_split,
    forward_with_references,
    precompute_reference_kv,
)
from invokeai.backend.krea2.sampling_utils import (
    prepare_position_ids,
    prepare_position_ids_with_references,
    prepare_reference_position_ids,
)

TEXT_LEN = 3
TARGET_GRID = (2, 3)
# Two references with grids that differ from the target and from each other.
REFERENCE_GRIDS = [(2, 2), (1, 3)]


def _randomize(module: torch.nn.Module, seed: int = 0) -> None:
    # diffusers zero-initializes the modulation tables and the zero-centered RMSNorm scales, under which
    # every block shares one modulation and a per-span bug can hide. Random weights everywhere.
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)


def _tiny_transformer() -> Krea2Transformer2DModel:
    model = Krea2Transformer2DModel(
        in_channels=64,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        timestep_embed_dim=8,
        text_hidden_dim=8,
        num_text_layers=3,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=16,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    _randomize(model)
    return model


def _inputs(seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target_len = TARGET_GRID[0] * TARGET_GRID[1]
    reference_len = sum(h * w for h, w in REFERENCE_GRIDS)
    return {
        "hidden_states": torch.randn(1, target_len, 64, generator=generator),
        "encoder_hidden_states": torch.randn(1, TEXT_LEN, 3, 8, generator=generator),
        "references": torch.randn(1, reference_len, 64, generator=generator),
    }


def _reference_position_ids() -> torch.Tensor:
    return prepare_position_ids_with_references(TEXT_LEN, *TARGET_GRID, REFERENCE_GRIDS, torch.device("cpu"))


def _per_token_reference_forward(
    model: Krea2Transformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    position_ids: torch.Tensor,
    references: torch.Tensor,
    isolate_references: bool = False,
) -> torch.Tensor:
    """The reference method written per token: every token picks its modulation by a mask, no slicing.

    ``isolate_references`` adds ai-toolkit's ``isolate_refs`` mask: reference queries see only reference keys,
    every other query sees everything.
    """
    text = model.txt_in(model.text_fusion(encoder_hidden_states))
    sequence = torch.cat([text, model.img_in(hidden_states), model.img_in(references)], dim=1)
    target_end = text.shape[1] + hidden_states.shape[1]
    is_reference = (torch.arange(sequence.shape[1]) >= target_end)[None, :, None, None]

    temb = model.time_embed(timestep, dtype=sequence.dtype)
    real_vec = model.time_mod_proj(F.gelu(temb, approximate="tanh"))
    zero_vec = model.time_mod_proj(
        F.gelu(model.time_embed(torch.zeros_like(timestep), dtype=sequence.dtype), approximate="tanh")
    )
    rotary = model.rotary_emb(position_ids)
    mask = None
    if isolate_references:
        query_is_reference = is_reference[0, :, 0, 0][:, None]
        key_is_reference = is_reference[0, :, 0, 0][None, :]
        mask = (~query_is_reference | key_is_reference)[None, None]

    for block in model.transformer_blocks:
        real = real_vec.unflatten(-1, (6, -1)) + block.scale_shift_table
        zero = zero_vec.unflatten(-1, (6, -1)) + block.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = torch.where(is_reference, zero, real).unbind(-2)
        attn = block.attn(
            (1 + prescale) * block.norm1(sequence) + preshift, attention_mask=mask, image_rotary_emb=rotary
        )
        sequence = sequence + pregate * attn
        sequence = sequence + postgate * block.ff((1 + postscale) * block.norm2(sequence) + postshift)

    return model.final_layer(sequence, temb)[:, text.shape[1] : target_end]


@torch.no_grad()
def test_split_block_with_one_modulation_is_the_stock_block() -> None:
    # Pins block_forward_split to the diffusers block it restates: a diffusers change to the block
    # (a new modulation term, a reordered gate) must fail here rather than silently diverge.
    block = Krea2TransformerBlock(hidden_size=16, intermediate_size=32, num_heads=2, num_kv_heads=1, norm_eps=1e-5)
    _randomize(block)
    generator = torch.Generator().manual_seed(2)
    hidden_states = torch.randn(1, 10, 16, generator=generator)
    temb_mod = torch.randn(1, 1, 6 * 16, generator=generator)
    position_ids = prepare_position_ids(4, 2, 3, torch.device("cpu"))
    rotary = Krea2RotaryPosEmbed(theta=1000, axes_dim=[2, 2, 4])(position_ids)
    attention_mask = torch.ones(1, 1, 1, 10, dtype=torch.bool)
    attention_mask[..., 1] = False

    expected = block(hidden_states, temb_mod, rotary, attention_mask)
    actual = block_forward_split(block, hidden_states, temb_mod, temb_mod, 6, rotary, attention_mask)

    assert torch.equal(actual, expected)


@torch.no_grad()
def test_without_references_the_forward_is_the_stock_forward() -> None:
    model = _tiny_transformer()
    inputs = _inputs()
    timestep = torch.tensor([0.7])
    position_ids = prepare_position_ids(TEXT_LEN, *TARGET_GRID, torch.device("cpu"))
    encoder_attention_mask = torch.tensor([[True, True, False]])

    expected = model(
        hidden_states=inputs["hidden_states"],
        encoder_hidden_states=inputs["encoder_hidden_states"],
        timestep=timestep,
        position_ids=position_ids,
        encoder_attention_mask=encoder_attention_mask,
        return_dict=False,
    )[0]
    actual = forward_with_references(
        model,
        inputs["hidden_states"],
        inputs["encoder_hidden_states"],
        timestep,
        position_ids,
        None,
        encoder_attention_mask=encoder_attention_mask,
    )

    assert torch.equal(actual, expected)


@torch.no_grad()
def test_with_references_the_forward_matches_the_per_token_formulation() -> None:
    model = _tiny_transformer()
    inputs = _inputs()
    timestep = torch.tensor([0.7])
    position_ids = _reference_position_ids()

    expected = _per_token_reference_forward(
        model, inputs["hidden_states"], inputs["encoder_hidden_states"], timestep, position_ids, inputs["references"]
    )
    # The production path runs under InvokeAI's own attention processors; the expectation under diffusers'.
    model.set_attn_processor(build_krea2_attention_processors(model, Krea2RegionalPromptingState()))
    actual = forward_with_references(
        model, inputs["hidden_states"], inputs["encoder_hidden_states"], timestep, position_ids, inputs["references"]
    )
    model.set_attn_processor(Krea2AttnProcessor())

    assert actual.shape == inputs["hidden_states"].shape
    torch.testing.assert_close(actual, expected)


@torch.no_grad()
def test_at_t0_references_are_ordinary_appended_image_tokens() -> None:
    """At t=0 the two modulations coincide, so the reference method collapses to the stock forward over
    [target | references] with the output cut back to the target. Anywhere else they must differ: the
    references are always presented as clean, whatever the step."""
    model = _tiny_transformer()
    inputs = _inputs()
    position_ids = _reference_position_ids()
    target_len = inputs["hidden_states"].shape[1]

    def stock_over_the_whole_sequence(timestep: torch.Tensor) -> torch.Tensor:
        return model(
            hidden_states=torch.cat([inputs["hidden_states"], inputs["references"]], dim=1),
            encoder_hidden_states=inputs["encoder_hidden_states"],
            timestep=timestep,
            position_ids=position_ids,
            return_dict=False,
        )[0][:, :target_len]

    def with_references(timestep: torch.Tensor) -> torch.Tensor:
        return forward_with_references(
            model,
            inputs["hidden_states"],
            inputs["encoder_hidden_states"],
            timestep,
            position_ids,
            inputs["references"],
        )

    torch.testing.assert_close(with_references(torch.tensor([0.0])), stock_over_the_whole_sequence(torch.tensor([0.0])))
    assert not torch.allclose(
        with_references(torch.tensor([0.7])), stock_over_the_whole_sequence(torch.tensor([0.7])), atol=1e-3
    )


def test_a_position_id_count_that_omits_the_references_is_refused() -> None:
    model = _tiny_transformer()
    inputs = _inputs()

    with pytest.raises(ValueError, match="text, target and reference tokens"):
        forward_with_references(
            model,
            inputs["hidden_states"],
            inputs["encoder_hidden_states"],
            torch.tensor([0.5]),
            prepare_position_ids(TEXT_LEN, *TARGET_GRID, torch.device("cpu")),
            inputs["references"],
        )


def test_reference_position_ids_number_each_reference_on_its_own_grid() -> None:
    ids = prepare_position_ids_with_references(2, 1, 2, [(1, 2), (2, 1)], torch.device("cpu"))

    expected = torch.tensor(
        [
            # text
            [0, 0, 0],
            [0, 0, 0],
            # target, 1x2
            [0, 0, 0],
            [0, 0, 1],
            # reference 1, 1x2
            [1, 0, 0],
            [1, 0, 1],
            # reference 2, 2x1
            [2, 0, 0],
            [2, 1, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(ids, expected)
    assert torch.equal(
        prepare_position_ids_with_references(2, 1, 2, [], torch.device("cpu")),
        prepare_position_ids(2, 1, 2, torch.device("cpu")),
    )


@pytest.mark.parametrize("reference_grids", [REFERENCE_GRIDS[:1], REFERENCE_GRIDS], ids=["one", "two"])
@torch.no_grad()
def test_the_kv_cache_is_the_joint_pass_with_isolated_references(reference_grids) -> None:
    """Precomputed reference K/V appended to the stock [text | target] pass reproduce the joint pass in which
    reference queries see only reference keys -- what ai-toolkit's kv_cache LoRAs are trained with."""
    model = _tiny_transformer()
    inputs = _inputs()
    references = inputs["references"][:, : sum(h * w for h, w in reference_grids)]
    timestep = torch.tensor([0.7])
    cpu = torch.device("cpu")

    expected = _per_token_reference_forward(
        model,
        inputs["hidden_states"],
        inputs["encoder_hidden_states"],
        timestep,
        prepare_position_ids_with_references(TEXT_LEN, *TARGET_GRID, reference_grids, cpu),
        references,
        isolate_references=True,
    )
    cache = Krea2ReferenceKVCache()
    model.set_attn_processor(
        build_krea2_attention_processors(model, Krea2RegionalPromptingState(), reference_kv_cache=cache)
    )
    precompute_reference_kv(model, references, prepare_reference_position_ids(reference_grids, cpu), cache)
    actual = model(
        hidden_states=inputs["hidden_states"],
        encoder_hidden_states=inputs["encoder_hidden_states"],
        timestep=timestep,
        position_ids=prepare_position_ids(TEXT_LEN, *TARGET_GRID, cpu),
        return_dict=False,
    )[0]

    assert len(cache) == len(model.transformer_blocks)
    torch.testing.assert_close(actual, expected)


@torch.no_grad()
def test_isolated_references_differ_from_joint_references() -> None:
    # Guards the comparison above: if isolation changed nothing on this model, it would prove nothing.
    model = _tiny_transformer()
    inputs = _inputs()
    args = (
        model,
        inputs["hidden_states"],
        inputs["encoder_hidden_states"],
        torch.tensor([0.7]),
        _reference_position_ids(),
        inputs["references"],
    )

    assert not torch.allclose(
        _per_token_reference_forward(*args), _per_token_reference_forward(*args, isolate_references=True), atol=1e-3
    )
