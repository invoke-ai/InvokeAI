"""FLUX.3 prompt encoding: the token window and the hidden states it hands on.

The tokenizer is the real Qwen3-VL one InvokeAI bundles; its chat template, token ids and pad token are the
ones the FLUX.3 repository's processor produces. The encoder is a stand-in that records what it was given and
returns hidden states that name their own layer, so the stacking order is visible in the result.
"""

from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.flux3.text_encoding import (
    FLUX3_TEXT_ENCODER_LAYERS,
    encode_flux3_prompt,
    flux3_prompt_text,
    tokenize_flux3_prompt,
)
from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return load_bundled_qwen3_vl_tokenizer()


def test_the_prompt_is_one_user_turn_with_the_generation_prompt(tokenizer) -> None:
    assert flux3_prompt_text(tokenizer, "a fox") == "<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"


@pytest.mark.parametrize("words", [1, 40, 120])
def test_the_window_is_right_padded_to_a_multiple_of_80(tokenizer, words: int) -> None:
    input_ids, mask, position_ids = tokenize_flux3_prompt(tokenizer, " ".join(["fox"] * words))

    real = int(mask.sum())
    length = input_ids.shape[1]
    assert length % 80 == 0 and length - 80 < real <= length
    # Right padding: the real tokens come first, then only padding.
    assert mask[0, :real].all() and not mask[0, real:].any()
    assert (input_ids[0, real:] == tokenizer.pad_token_id).all()
    # Real tokens count up from 0; the padding sits at position 1, as in the training environment.
    assert position_ids[0, :real].tolist() == list(range(real))
    assert (position_ids[0, real:] == 1).all()


def test_the_window_is_capped(tokenizer) -> None:
    input_ids, _, _ = tokenize_flux3_prompt(tokenizer, " ".join(["fox"] * 200), max_tokens=120)
    assert input_ids.shape[1] == 120


def test_the_shared_tokenizer_is_not_reconfigured(tokenizer) -> None:
    before = tokenizer.padding_side
    tokenize_flux3_prompt(tokenizer, "a fox")
    assert tokenizer.padding_side == before


def test_hidden_states_4_to_32_are_stacked_per_token_with_the_mask_passed(tokenizer) -> None:
    hidden = 3
    seen = {}

    def text_encoder(**kwargs):
        seen.update(kwargs)
        batch, length = kwargs["input_ids"].shape
        # hidden_states[k] is filled with k, so the stack order can be read off the output.
        states = tuple(torch.full((batch, length, hidden), float(k)) for k in range(37))
        return SimpleNamespace(hidden_states=states)

    embeds = encode_flux3_prompt("a fox", tokenizer, text_encoder, torch.device("cpu"))

    assert seen["attention_mask"] is not None and seen["position_ids"] is not None
    assert seen["output_hidden_states"] is True
    assert embeds.shape == (1, 80, len(FLUX3_TEXT_ENCODER_LAYERS) * hidden)
    per_token = embeds[0, 0].reshape(len(FLUX3_TEXT_ENCODER_LAYERS), hidden)[:, 0].tolist()
    assert per_token == [float(k) for k in FLUX3_TEXT_ENCODER_LAYERS]
