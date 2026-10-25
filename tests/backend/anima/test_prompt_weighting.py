from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.anima.prompt_weighting import parse_prompt_attention, tokenize_t5_with_weights
from invokeai.backend.t5.t5_tokenizer import load_bundled_t5_tokenizer


def _spans(prompt: str, ranges: list[tuple[int, int, float]]) -> list[tuple[str, float]]:
    return [(prompt[start:end], weight) for start, end, weight in ranges]


def test_compel_style_numeric_group_and_word_suffixes() -> None:
    prompt, ranges = parse_prompt_attention("a portrait, (long hair)2, blue eyes++")

    assert prompt == "a portrait, long hair, blue eyes"
    assert _spans(prompt, ranges) == [
        ("a portrait, ", 1.0),
        ("long hair", 2.0),
        (", blue ", 1.0),
        ("eyes", pytest.approx(1.21)),
    ]


def test_group_suffixes_and_nested_weights_multiply() -> None:
    prompt, ranges = parse_prompt_attention("((long hair)2)0.5, (blue eyes)--")

    assert prompt == "long hair, blue eyes"
    assert _spans(prompt, ranges) == [("long hair, ", 1.0), ("blue eyes", pytest.approx(0.81))]


def test_bare_groups_are_neutral_and_escaped_parentheses_are_literal() -> None:
    prompt, ranges = parse_prompt_attention(r"hatsune miku (cosplay), \(literal\)")

    assert prompt == "hatsune miku cosplay, (literal)"
    assert ranges == [(0, len(prompt), 1.0)]


def test_comfyui_colon_syntax_is_not_weighting_in_compel_style() -> None:
    prompt, ranges = parse_prompt_attention("(long hair:2)")

    assert prompt == "long hair:2"
    assert ranges == [(0, len(prompt), 1.0)]


def test_lone_plus_between_words_and_hyphen_inside_word_are_literal() -> None:
    prompt = "a+b, blue-green"
    assert parse_prompt_attention(prompt) == (prompt, [(0, len(prompt), 1.0)])


def test_weight_suffix_before_adjacent_word_keeps_the_words_separate() -> None:
    prompt, ranges = parse_prompt_attention("(long hair)2next, blue++eyes")

    assert prompt == "long hair next, blue eyes"
    assert _spans(prompt, ranges) == [
        ("long hair", 2.0),
        (" next, ", 1.0),
        ("blue", pytest.approx(1.21)),
        (" eyes", 1.0),
    ]


def test_adjacent_groups_and_words_do_not_merge() -> None:
    prompt, ranges = parse_prompt_attention("(red)(blue) sky, (dark)forest")

    assert prompt == "red blue sky, dark forest"
    assert ranges == [(0, len(prompt), 1.0)]


@pytest.mark.parametrize("weight", [0.0, 6.0])
def test_zero_and_above_five_weights_reach_t5_tokens(weight: float) -> None:
    tokenizer = load_bundled_t5_tokenizer()
    prompt, ranges = parse_prompt_attention(f"a portrait, (long hair){weight}, blue eyes")

    assert prompt == "a portrait, long hair, blue eyes"
    ids, token_weights = tokenize_t5_with_weights(tokenizer, prompt, ranges, max_length=512)
    assert token_weights is not None
    assert ids.shape == token_weights.shape

    offsets = tokenizer(prompt, return_offsets_mapping=True).offset_mapping
    start = prompt.index("long hair")
    end = start + len("long hair")
    expected = torch.tensor([weight if right > start and left < end else 1.0 for left, right in offsets])
    torch.testing.assert_close(token_weights, expected)


def test_nonrepresentable_decimal_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite and within float16 magnitude"):
        parse_prompt_attention("(long hair)" + "9" * 100)


def test_weight_must_survive_float16_cast() -> None:
    prompt, ranges = parse_prompt_attention("(long hair)65504")
    assert prompt == "long hair"
    assert ranges == [(0, len(prompt), 65504.0)]

    with pytest.raises(ValueError, match="float16 magnitude"):
        parse_prompt_attention("(long hair)65505")
    with pytest.raises(ValueError, match="float16 magnitude"):
        parse_prompt_attention("((long hair)300)300")


def test_deep_and_unbalanced_groups_are_iterative() -> None:
    unbalanced = "(" * 20_000 + "long hair"
    assert parse_prompt_attention(unbalanced)[0] == unbalanced

    nested = "(" * 1_500 + "long hair" + ")" * 1_500 + "2"
    prompt, ranges = parse_prompt_attention(nested)
    assert prompt == "long hair"
    assert ranges == [(0, len(prompt), 2.0)]


def test_many_literal_signs_preserve_the_prompt() -> None:
    prompt = "a+b-c" * 10_000
    assert parse_prompt_attention(prompt) == (prompt, [(0, len(prompt), 1.0)])


def test_weighted_t5_tokenization_keeps_clean_ids_and_unweighted_none() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    plain_prompt, plain_ranges = parse_prompt_attention("a portrait, long hair, blue eyes")
    weighted_prompt, weighted_ranges = parse_prompt_attention("a portrait, (long hair)2, blue eyes")

    plain_ids, plain_weights = tokenize_t5_with_weights(tokenizer, plain_prompt, plain_ranges, max_length=512)
    weighted_ids, weighted_weights = tokenize_t5_with_weights(
        tokenizer, weighted_prompt, weighted_ranges, max_length=512
    )

    assert plain_prompt == weighted_prompt
    torch.testing.assert_close(plain_ids, weighted_ids)
    assert plain_weights is None
    assert weighted_weights is not None


def test_partial_token_overlap_averages_character_weights() -> None:
    class CombinedTokenTokenizer:
        def __call__(self, prompt, **kwargs):
            assert prompt == "unhappy"
            assert kwargs["return_offsets_mapping"]
            return SimpleNamespace(
                input_ids=torch.tensor([[1, 2]]),
                offset_mapping=torch.tensor([[[0, 7], [0, 0]]]),
                special_tokens_mask=torch.tensor([[0, 1]]),
            )

    # A tokenizer offset may cross a weight boundary even when prompt syntax is separate.
    prompt = "unhappy"
    ranges = [(0, 2, 2.0), (2, len(prompt), 1.0)]
    ids, weights = tokenize_t5_with_weights(CombinedTokenTokenizer(), prompt, ranges, max_length=512)

    assert prompt == "unhappy"
    torch.testing.assert_close(ids, torch.tensor([1, 2]))
    assert weights is not None
    torch.testing.assert_close(weights, torch.tensor([9 / 7, 1.0]))


def test_t5_truncation_keeps_ids_and_weights_aligned() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    prompt, ranges = parse_prompt_attention(" ".join(["(long hair)2"] * 600))

    ids, weights = tokenize_t5_with_weights(tokenizer, prompt, ranges, max_length=512)

    assert weights is not None
    assert ids.shape == weights.shape == (512,)
