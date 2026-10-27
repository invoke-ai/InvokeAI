import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.anima.prompt_weighting import parse_prompt_attention, tokenize_t5_with_weights
from invokeai.backend.t5.t5_tokenizer import load_bundled_t5_tokenizer


def _spans(prompt: str, ranges: list[tuple[int, int, float]]) -> list[tuple[str, float]]:
    return [(prompt[start:end], weight) for start, end, weight in ranges]


def test_compel_style_numeric_group_and_word_suffixes() -> None:
    prompt, ranges = parse_prompt_attention("a portrait, (long hair)2, blue eyes++")

    assert prompt == "a portrait, long hair , blue eyes"
    assert _spans(prompt, ranges) == [
        ("a portrait, ", 1.0),
        ("long hair", 2.0),
        (" , blue ", 1.0),
        ("eyes", pytest.approx(1.21)),
    ]


def test_group_suffixes_and_nested_weights_multiply() -> None:
    prompt, ranges = parse_prompt_attention("((long hair)2)0.5, (blue eyes)--")

    assert prompt == "long hair , blue eyes"
    assert _spans(prompt, ranges) == [("long hair , ", 1.0), ("blue eyes", pytest.approx(0.81))]


def test_bare_groups_are_neutral_and_escaped_parentheses_are_literal() -> None:
    prompt, ranges = parse_prompt_attention(r"hatsune miku (cosplay), \(literal\)")

    assert prompt == "hatsune miku cosplay , (literal)"
    assert ranges == [(0, len(prompt), 1.0)]


def test_comfyui_colon_syntax_is_not_weighting_in_compel_style() -> None:
    prompt, ranges = parse_prompt_attention("(long hair:2)")

    assert prompt == "long hair:2"
    assert ranges == [(0, len(prompt), 1.0)]


def test_lone_plus_between_words_and_hyphen_inside_word_are_literal() -> None:
    prompt = "a+b, blue-green"
    assert parse_prompt_attention(prompt) == (prompt, [(0, len(prompt), 1.0)])


@pytest.mark.parametrize(
    ("prompt", "cleaned", "spans"),
    [
        ("blue-green+", "blue-green", [("blue-green", 1.1)]),
        ("hair!+", "hair!", [("hair!", 1.1)]),
        ("a+b+", "a+b", [("a+b", 1.1)]),
        ("foo/bar+", "foo/bar", [("foo/bar", 1.1)]),
        ("foo,bar+", "foo, bar", [("foo, ", 1.0), ("bar", 1.1)]),
        ("foo.bar+", "foo.bar+", [("foo.bar+", 1.0)]),
        ("foo=bar+", "foo=bar+", [("foo=bar+", 1.0)]),
        ("++x+", "++x+", [("++x+", 1.0)]),
        ("+-", "+-", [("+-", 1.0)]),
        (r"hair\)+", "hair)", [("hair)", 1.1)]),
    ],
)
def test_bare_word_suffix_uses_compel_word_boundaries(
    prompt: str, cleaned: str, spans: list[tuple[str, float]]
) -> None:
    actual_prompt, ranges = parse_prompt_attention(prompt)

    assert actual_prompt == cleaned
    assert _spans(actual_prompt, ranges) == spans


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

    assert prompt == "a portrait, long hair , blue eyes"
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


@pytest.mark.parametrize("sign", ["+", "-"])
def test_large_literal_sign_runs_use_bounded_parser_reads(sign: str) -> None:
    class BudgetedPrompt(str):
        reads = 0

        def __getitem__(self, key):
            self.reads += 1
            if self.reads > len(self) * 4:
                raise AssertionError("A literal sign run was rescanned instead of consumed once")
            return super().__getitem__(key)

    prompt = BudgetedPrompt(" " + sign * 20_000)
    assert parse_prompt_attention(prompt) == (prompt, [(0, len(prompt), 1.0)])


def test_weighted_t5_tokenization_keeps_clean_ids_and_unweighted_none() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    plain_prompt, plain_ranges = parse_prompt_attention("a portrait, long hair , blue eyes")
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


_CORPUS = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "invokeai/frontend/webv2/src/features/generation/core/__fixtures__/animaPromptAttention.json"
    ).read_text()
)["cases"]


@pytest.mark.parametrize("case", _CORPUS, ids=lambda case: case["input"])
def test_shared_compel_attention_corpus(case: dict) -> None:
    cleaned, ranges = parse_prompt_attention(case["input"])
    assert re.sub(r"[ \t\r\n\v\f]+", " ", cleaned).strip(" ") == re.sub(r"[ \t\r\n\v\f]+", " ", case["text"]).strip(" ")
    actual = [
        (char, weight) for start, end, weight in ranges for char in cleaned[start:end] if char not in " \t\r\n\v\f"
    ]
    expected = [(char, weight) for text, weight in case["fragments"] for char in text if char not in " \t\r\n\v\f"]
    assert [char for char, _ in actual] == [char for char, _ in expected]
    assert [weight for _, weight in actual] == pytest.approx([weight for _, weight in expected])


@pytest.mark.parametrize(
    ("source", "cleaned"),
    [
        ("red+(blue)2", "red blue"),
        ("(red)2_blue", "red _blue"),
        ("(foo.bar+)2", "foo . bar"),
        ("red,blue+", "red, blue"),
        ("(red)2,blue", "red , blue"),
        ("(red)2.blue", "red blue"),
        ("(cat)2\n\nnext", "cat\n\nnext"),
        ("( red  blue )2", " red  blue "),
        ("red  blue", "red  blue"),
        ("foo.bar+", "foo.bar+"),
        ("a(b)2", "a(b)2"),
        ("cat)2", "cat)2"),
        ("((red)2", "((red)2"),
    ],
)
def test_fragment_separation_and_plain_source_preservation(source: str, cleaned: str) -> None:
    assert parse_prompt_attention(source)[0] == cleaned


def test_attention_truncated_away_has_no_weight_tensor() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    cleaned, ranges = parse_prompt_attention("cat dog (blue)2")
    ids, weights = tokenize_t5_with_weights(tokenizer, cleaned, ranges, max_length=2)
    assert len(ids) == 2
    assert weights is None


@pytest.mark.parametrize("source", ["((cat)65505)0", "((cat)0)65505"])
def test_local_weight_bound_is_enforced_even_when_composition_is_zero(source: str) -> None:
    with pytest.raises(ValueError, match="float16 magnitude"):
        parse_prompt_attention(source)


@pytest.mark.parametrize(
    ("authored", "expected_text", "weighted_word", "expected_weight"),
    [
        ("red+(blue)2", "red blue", "blue", 2.0),
        ("(foo.bar+)2", "foo . bar", "bar", 2.2),
        ('"red (blue)2"+', "red blue", "blue", 2.2),
        ("target .attend (2)", "target", "target", 2.0),
        ("red\u0085blue+", "red\u0085blue", "red", 1.1),
        (".and(dog+)", ".and(dog)", "dog", 1.1),
    ],
)
def test_fragment_boundaries_and_weights_reach_real_t5_tokens(
    authored: str, expected_text: str, weighted_word: str, expected_weight: float
) -> None:
    tokenizer = load_bundled_t5_tokenizer()
    cleaned, ranges = parse_prompt_attention(authored)
    assert cleaned == expected_text
    ids, weights = tokenize_t5_with_weights(tokenizer, cleaned, ranges, max_length=512)
    assert weights is not None
    assert ids.tolist() == tokenizer(expected_text).input_ids
    start = expected_text.index(weighted_word)
    offsets = tokenizer(expected_text, return_offsets_mapping=True).offset_mapping
    matching = [
        index for index, (left, right) in enumerate(offsets) if right > start and left < start + len(weighted_word)
    ]
    assert matching
    torch.testing.assert_close(weights[matching], torch.full((len(matching),), expected_weight))


@pytest.mark.parametrize("depth", [1000, 2000, 4000])
@pytest.mark.parametrize(
    ("context", "prefix", "inner", "suffix", "copy_budget"),
    [
        ("parentheses", "", "=", "", 2),
        ("outer_quote", '"', "=", '"', 2),
        ("inner_quote", "", '"="', "", 2),
        ("discarded_attention", "", "(red)2 =", "2", 2),
        ("method_boundary", ".and(", "(red)2 =", ", dog+)", 2),
        ("malformed_quote", "", '"cat("', "65505", 2),
        ("decoded_before_malformed_quote", "", '"cat\\ndog" "cat("', "65505", 3),
    ],
)
def test_invalid_enclosing_group_rollback_copies_source_linearly(
    depth: int, context: str, prefix: str, inner: str, suffix: str, copy_budget: int
) -> None:
    class CountedPrompt(str):
        copied_chars = 0

        def __getitem__(self, key):
            value = super().__getitem__(key)
            if isinstance(key, slice):
                self.copied_chars += len(value)
            return value

    block = "(" * depth + inner + ")" * depth
    prompt = CountedPrompt(prefix + block + suffix)
    cleaned, ranges = parse_prompt_attention(prompt)

    # Count requested source characters, independently of timing and substring sharing.
    assert prompt.copied_chars <= copy_budget * len(prompt)
    if context == "method_boundary":
        assert cleaned == f".and({block} , dog)"
        assert _spans(cleaned, ranges) == [
            (f".and({block} , ", 1.0),
            ("dog", pytest.approx(1.1)),
            (")", 1.0),
        ]
    else:
        assert cleaned == prompt
        assert ranges == [(0, len(prompt), 1.0)]


@pytest.mark.parametrize("suffix", ["2", "65505", "9" * 310, "+" * 10000])
@pytest.mark.parametrize("child", ['"cat("', '"cat)"', '"cat\\n("'])
def test_malformed_quoted_children_roll_back_enclosing_weight_without_validation(child: str, suffix: str) -> None:
    prompt = f"(({child}){suffix})2"
    assert parse_prompt_attention(prompt) == (prompt, [(0, len(prompt), 1.0)])


@pytest.mark.parametrize(("escaped", "decoded"), [("n", "\n"), ("t", "\t"), ("r", "\r")])
def test_quoted_whitespace_decodes_before_word_attention_and_real_t5_tokenization(escaped: str, decoded: str) -> None:
    authored = '"cat\\' + escaped + 'dog+"2'
    cleaned, ranges = parse_prompt_attention(authored)
    assert cleaned == f"cat{decoded}dog"
    assert _spans(cleaned, ranges) == [(f"cat{decoded}", 2.0), ("dog", pytest.approx(2.2))]
    tokenizer = load_bundled_t5_tokenizer()
    ids, weights = tokenize_t5_with_weights(tokenizer, cleaned, ranges, max_length=512)
    assert ids.tolist() == tokenizer("cat dog").input_ids
    assert weights is not None
    offsets = tokenizer(cleaned, return_offsets_mapping=True).offset_mapping
    for index, (start, end) in enumerate(offsets):
        if end > start:
            assert weights[index].item() == pytest.approx(2.0 if start < 4 else 2.2)


def test_whitespace_only_quoted_form_feed_has_no_weighted_t5_tokens() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    cleaned, ranges = parse_prompt_attention(r'"\f"2')
    assert cleaned == "\f"
    ids, weights = tokenize_t5_with_weights(tokenizer, cleaned, ranges, max_length=512)
    assert ids.tolist() == tokenizer("").input_ids
    assert weights is None


@pytest.mark.parametrize("suffix", ["65505", "9" * 310, "+" * 10000])
@pytest.mark.parametrize("target", [r'"cat\fdog+"', '"cat("', '"cat)"', '"cat=dog"'])
def test_rejected_closed_quote_discards_weight_and_preserves_following_t5_attention(target: str, suffix: str) -> None:
    literal = target + suffix
    cleaned, ranges = parse_prompt_attention(literal + " bird+")
    assert cleaned == literal + " bird"
    assert _spans(cleaned, ranges) == [(literal + " ", 1.0), ("bird", pytest.approx(1.1))]
    tokenizer = load_bundled_t5_tokenizer()
    ids, weights = tokenize_t5_with_weights(tokenizer, cleaned, ranges, max_length=len(literal) + 32)
    assert ids.tolist() == tokenizer(cleaned).input_ids
    assert weights is not None
    offsets = tokenizer(cleaned, return_offsets_mapping=True).offset_mapping
    bird = cleaned.index("bird")
    matching = [index for index, (start, end) in enumerate(offsets) if end > bird and start < bird + 4]
    assert matching
    torch.testing.assert_close(weights[matching], torch.full((len(matching),), 1.1))


@pytest.mark.parametrize("depth", [500, 1000, 2000])
@pytest.mark.parametrize("separator", ["", " "])
def test_many_closed_rejected_quotes_recover_with_linear_source_copies(depth: int, separator: str) -> None:
    class CountedPrompt(str):
        copied_chars = 0

        def __getitem__(self, key):
            value = super().__getitem__(key)
            if isinstance(key, slice):
                self.copied_chars += len(value)
            return value

    literal = separator.join([r'"cat\fdog+"65505'] * depth)
    source = CountedPrompt(literal + " bird+")
    cleaned, ranges = parse_prompt_attention(source)
    assert source.copied_chars <= 2 * len(source)
    assert cleaned == literal + " bird"
    assert _spans(cleaned, ranges) == [(literal + " ", 1.0), ("bird", pytest.approx(1.1))]


def test_valid_attention_after_rejected_quote_still_enforces_its_weight_bound() -> None:
    with pytest.raises(ValueError, match="float16 magnitude"):
        parse_prompt_attention(r'"cat\fdog+"65505 bird.attend(65505)')
