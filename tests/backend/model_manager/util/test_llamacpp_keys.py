"""Tests for the shared llama.cpp decoder GGUF key converter.

Used by three encoder GGUF loaders: text-only Qwen3 (Z-Image / FLUX.2 Klein), the Qwen3-VL language
tower (Krea-2 / Ideogram 4) and Mistral / Ministral 3 (FLUX.2 [dev]). A key that maps to *nothing*
fails loudly — the loaders sweep for leftover meta tensors afterwards and name the offending ones.
What this table has to protect against is the silent case: a key mapped to a different but
shape-compatible module. `attn_q`/`attn_k` and `ffn_gate`/`ffn_up` are same-shaped pairs, so
transposing either pair would load cleanly and produce wrong conditioning with nothing logged.
"""

import pytest

from invokeai.backend.model_manager.util.llamacpp_keys import (
    convert_llamacpp_decoder_keys,
    is_llamacpp_decoder_state_dict,
)


def test_block_components_map_to_the_transformers_decoder_layout() -> None:
    """Pinned individually: a swap within the same-shaped q/k or gate/up pair is undetectable later."""
    sd = {
        "blk.0.attn_q.weight": "q",
        "blk.0.attn_k.weight": "k",
        "blk.0.attn_v.weight": "v",
        "blk.0.attn_output.weight": "o",
        "blk.0.attn_q_norm.weight": "qn",
        "blk.0.attn_k_norm.weight": "kn",
        "blk.0.ffn_gate.weight": "gate",
        "blk.0.ffn_up.weight": "up",
        "blk.0.ffn_down.weight": "down",
        "blk.0.attn_norm.weight": "in_ln",
        "blk.0.ffn_norm.weight": "post_ln",
    }

    assert convert_llamacpp_decoder_keys(sd) == {
        "model.layers.0.self_attn.q_proj.weight": "q",
        "model.layers.0.self_attn.k_proj.weight": "k",
        "model.layers.0.self_attn.v_proj.weight": "v",
        "model.layers.0.self_attn.o_proj.weight": "o",
        "model.layers.0.self_attn.q_norm.weight": "qn",
        "model.layers.0.self_attn.k_norm.weight": "kn",
        "model.layers.0.mlp.gate_proj.weight": "gate",
        "model.layers.0.mlp.up_proj.weight": "up",
        "model.layers.0.mlp.down_proj.weight": "down",
        "model.layers.0.input_layernorm.weight": "in_ln",
        "model.layers.0.post_attention_layernorm.weight": "post_ln",
    }


def test_multi_digit_layer_indices_survive() -> None:
    """Qwen3-VL 4B/8B have 36 layers, so two-digit indices are the common case, not an edge one."""
    assert convert_llamacpp_decoder_keys({"blk.35.attn_q.weight": "w"}) == {
        "model.layers.35.self_attn.q_proj.weight": "w"
    }


def test_top_level_tensors_map_to_their_transformers_names() -> None:
    assert convert_llamacpp_decoder_keys(
        {
            "token_embd.weight": "embed",
            "output_norm.weight": "norm",
            "output.weight": "head",
        }
    ) == {
        "model.embed_tokens.weight": "embed",
        "model.norm.weight": "norm",
        "lm_head.weight": "head",
    }


def test_unrecognized_keys_pass_through_instead_of_being_dropped() -> None:
    """An unknown tensor must stay visible so the load complains about it by its real name."""
    assert convert_llamacpp_decoder_keys(
        {
            "blk.2.some_future_thing.weight": "a",
            "rope_freqs.weight": "b",
            "blk.3.bare_component": "c",
        }
    ) == {
        "model.layers.2.some_future_thing.weight": "a",
        "rope_freqs.weight": "b",
        "model.layers.3.bare_component": "c",
    }


def test_non_string_keys_are_preserved() -> None:
    """State dicts are typed `str | int` across the config layer; a non-string key must survive."""
    assert convert_llamacpp_decoder_keys({0: "x"}) == {0: "x"}


@pytest.mark.parametrize(
    "sd, expected",
    [
        ({"blk.0.attn_q.weight": None, "token_embd.weight": None}, True),
        ({"model.layers.0.self_attn.q_proj.weight": None}, False),
        ({0: None}, False),
        ({}, False),
    ],
)
def test_llamacpp_layout_detection(sd: dict, expected: bool) -> None:
    """ComfyUI-converted GGUFs already use the transformers naming and must not be re-mapped."""
    assert is_llamacpp_decoder_state_dict(sd) is expected


def test_a_component_is_matched_by_position_not_by_containment() -> None:
    """The replaced Mistral converter rewrote with unanchored `str.replace`, so a known component
    name was rewritten wherever it appeared in the key -- not only as the component.

    These two inputs are the ones that actually separate the implementations; I checked the deleted
    one to be sure. `blk.0.cross_attn_q.weight` became `...cross_self_attn.q_proj.weight` under it,
    and a nested `attn_v` was rewritten mid-path. Both land under a wrong but plausibly
    shape-compatible module, which is the failure that loads cleanly and conditions wrongly.
    (A name that merely *starts* with a known component, like `ffn_gate_exps`, was already safe
    there -- every replacement carried a trailing dot -- so it would not discriminate.)
    """
    assert convert_llamacpp_decoder_keys(
        {
            "blk.0.cross_attn_q.weight": "cross_q",
            "blk.0.some.attn_v.weight": "nested_v",
        }
    ) == {
        "model.layers.0.cross_attn_q.weight": "cross_q",
        "model.layers.0.some.attn_v.weight": "nested_v",
    }
