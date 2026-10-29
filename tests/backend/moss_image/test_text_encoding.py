"""MOSS-Image encodes a prompt as the user turn of a thinking chat, read after Qwen3.5's final norm.

Run with the vendored Qwen3.5 tokenizer -- the one a single-file 9B encoder loads -- and a two-layer
encoder wide enough for its vocabulary, so the template's special tokens embed.
"""

import logging

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from invokeai.backend.moss_image import text_encoding
from invokeai.backend.moss_image.text_encoding import encode_moss_prompt
from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder, load_bundled_qwen3_5_tokenizer

_CPU = torch.device("cpu")
_HIDDEN = 16


def _formatted(prompt: str) -> str:
    """The reference pipeline's rendering of one user message, with the generation prompt and thinking on."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"


@pytest.fixture(scope="module")
def encoder() -> Qwen35Encoder:
    config = Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=_HIDDEN,
        intermediate_size=16,
        num_hidden_layers=2,
        full_attention_interval=2,  # one linear-attention layer, one full-attention layer
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    encoder = Qwen35Encoder(config, final_norm=True).eval()
    for parameter in encoder.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    return encoder


def _expected(encoder: Qwen35Encoder, ids: list[int]) -> torch.Tensor:
    return encoder.last_hidden_state(torch.tensor([ids]))[0]


def test_the_prompt_is_encoded_as_the_user_turn_of_a_thinking_chat(encoder: Qwen35Encoder) -> None:
    tokenizer = load_bundled_qwen3_5_tokenizer()
    prompt = "A red fox in fresh snow, 雪中的红狐 🦊"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assert rendered == _formatted(prompt)  # what the vendored template renders, so the ids below are the reference's

    states = encode_moss_prompt(prompt, tokenizer, encoder, device=_CPU)

    ids = tokenizer(_formatted(prompt)).input_ids
    assert states.shape == (len(ids), _HIDDEN)
    torch.testing.assert_close(states, _expected(encoder, ids))


def test_an_empty_prompt_encodes_the_template_alone(encoder: Qwen35Encoder) -> None:
    tokenizer = load_bundled_qwen3_5_tokenizer()

    states = encode_moss_prompt("", tokenizer, encoder, device=_CPU)

    ids = tokenizer(_formatted("")).input_ids
    assert len(ids) > 0
    torch.testing.assert_close(states, _expected(encoder, ids))


def test_a_prompt_longer_than_max_length_keeps_its_head_and_says_so(
    encoder: Qwen35Encoder, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer = load_bundled_qwen3_5_tokenizer()
    prompt = "a lighthouse on a cliff at dusk"
    ids = tokenizer(_formatted(prompt)).input_ids

    with caplog.at_level(logging.WARNING, logger=text_encoding.__name__):
        exact = encode_moss_prompt(prompt, tokenizer, encoder, max_length=len(ids), device=_CPU)
    assert caplog.text == ""  # filling the budget exactly loses nothing

    with caplog.at_level(logging.WARNING, logger=text_encoding.__name__):
        truncated = encode_moss_prompt(prompt, tokenizer, encoder, max_length=len(ids) - 3, device=_CPU)
    assert "truncated" in caplog.text

    torch.testing.assert_close(exact, _expected(encoder, ids))
    torch.testing.assert_close(truncated, _expected(encoder, ids[:-3]))
