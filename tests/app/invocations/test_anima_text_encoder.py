from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.anima_text_encoder import AnimaTextEncoderInvocation
from invokeai.backend.anima.prompt_weighting import parse_prompt_attention, tokenize_t5_with_weights
from invokeai.backend.t5.t5_tokenizer import load_bundled_t5_tokenizer


class FakeQwen3Encoder(torch.nn.Module):
    """Mimics the Qwen3 0.6B encoder.

    Its `.device` property reports CPU (as HF `PreTrainedModel.device` would when partial loading has offloaded
    every weight to RAM), while the intended compute device is carried separately by the LoadedModel. The forward
    records the device of its inputs so the test can assert where the encode actually ran.
    """

    def __init__(self):
        super().__init__()
        self.register_parameter("cpu_param", torch.nn.Parameter(torch.ones(1)))
        self.forward_input_device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, input_ids: torch.Tensor, attention_mask=None, output_hidden_states: bool = False):
        assert output_hidden_states
        self.forward_input_device = input_ids.device
        hidden = input_ids.unsqueeze(-1).float()
        return SimpleNamespace(hidden_states=[hidden, hidden + 1])


class FakeQwen3Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.prompt: str | None = None

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        del kwargs
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        )


class FakeT5Tokenizer:
    def __call__(self, prompt, **kwargs):
        del prompt, kwargs
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            offset_mapping=torch.tensor([[[0, 1], [1, 2], [0, 0]]], dtype=torch.long),
            special_tokens_mask=torch.tensor([[0, 0, 1]], dtype=torch.long),
        )


class FakeLoadedModel:
    def __init__(self, model, compute_device=torch.device("cpu")):
        self._model = model
        self._compute_device = compute_device

    @property
    def compute_device(self) -> torch.device:
        return self._compute_device

    @contextmanager
    def model_on_device(self):
        yield (None, self._model)


def _run_encode(
    monkeypatch, compute_device: torch.device, prompt: str = "test prompt"
) -> tuple[FakeQwen3Encoder, FakeQwen3Tokenizer]:
    module_path = "invokeai.app.invocations.anima_text_encoder"
    text_encoder = FakeQwen3Encoder()
    tokenizer = FakeQwen3Tokenizer()

    mock_context = MagicMock()
    mock_context.models.load.side_effect = [
        FakeLoadedModel(text_encoder, compute_device=compute_device),
        FakeLoadedModel(tokenizer),
    ]
    mock_context.util.signal_progress = MagicMock()

    # isinstance() guards in the invocation must accept the fakes.
    monkeypatch.setattr(f"{module_path}.PreTrainedModel", FakeQwen3Encoder)
    monkeypatch.setattr(f"{module_path}.PreTrainedTokenizerBase", FakeQwen3Tokenizer)
    monkeypatch.setattr(f"{module_path}.LayerPatcher.apply_smart_model_patches", lambda **kwargs: nullcontext())
    # Step 2 tokenizes with the bundled T5-XXL tokenizer; avoid touching the real bundled files.
    monkeypatch.setattr(f"{module_path}.load_bundled_t5_tokenizer", lambda: FakeT5Tokenizer())

    invocation = AnimaTextEncoderInvocation.model_construct(
        prompt=prompt,
        qwen3_encoder=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        mask=None,
    )

    invocation._encode_prompt(mock_context)
    return text_encoder, tokenizer


def test_anima_qwen3_encode_uses_compute_device(monkeypatch):
    # Regression test for #9373: the encoder's weights are offloaded to CPU (`.device` == CPU), but its intended
    # compute device is the accelerator. The encode must run on the intended compute device, not the current
    # residency, or the whole encode silently runs on the CPU.
    compute_device = torch.device("meta")
    text_encoder, _ = _run_encode(monkeypatch, compute_device)
    assert text_encoder.forward_input_device == compute_device


def test_anima_qwen3_encode_uses_cpu_for_cpu_only_model(monkeypatch):
    # A cpu_only encoder has compute_device == CPU; the encode must run on the CPU.
    text_encoder, _ = _run_encode(monkeypatch, torch.device("cpu"))
    assert text_encoder.forward_input_device == torch.device("cpu")


def test_parse_anima_prompt_attention() -> None:
    prompt, ranges = parse_prompt_attention("a (red) fox, (long hair:2), ((soft:0.5):2)")

    assert prompt == "a (red) fox, long hair, soft"
    assert [(prompt[start:end], weight) for start, end, weight in ranges] == [
        ("a (red) fox, ", 1.0),
        ("long hair", 2.0),
        (", ", 1.0),
        ("soft", 0.5),
    ]


def test_anima_bare_parentheses_are_preserved_without_explicit_weights() -> None:
    original = "hatsune miku (cosplay), long hair++"
    prompt, ranges = parse_prompt_attention(original)

    assert prompt == original
    assert ranges == [(0, len(original), 1.0)]


def test_anima_escaped_parentheses_remain_literal_without_weights() -> None:
    prompt, ranges = parse_prompt_attention(r"\(long hair:2\)")

    assert prompt == "(long hair:2)"
    assert ranges == [(0, len(prompt), 1.0)]


def test_anima_escaped_parentheses_inside_explicit_weight() -> None:
    prompt, ranges = parse_prompt_attention(r"(long hair \(blue\):2)")

    assert prompt == "long hair (blue)"
    assert ranges == [(0, len(prompt), 2.0)]


@pytest.mark.parametrize("weight", ["nan", "inf", "-inf", "1e309", "1e100"])
def test_anima_nonfinite_or_nonrepresentable_weights_are_rejected(weight: str) -> None:
    with pytest.raises(ValueError, match="Anima prompt weight must be finite and within float32 magnitude"):
        parse_prompt_attention(f"(long hair:{weight})")


@pytest.mark.parametrize("weight", [0.0, -1.0, 6.0])
def test_anima_zero_negative_and_above_five_weights_reach_t5_tokens(weight: float) -> None:
    tokenizer = load_bundled_t5_tokenizer()
    prompt, ranges = parse_prompt_attention(f"a portrait, (long hair:{weight}), blue eyes")

    assert prompt == "a portrait, long hair, blue eyes"
    assert (prompt.index("long hair"), prompt.index("long hair") + len("long hair"), weight) in ranges

    ids, token_weights = tokenize_t5_with_weights(tokenizer, prompt, ranges, max_length=512)
    assert token_weights is not None
    assert ids.shape == token_weights.shape

    offsets = tokenizer(prompt, return_offsets_mapping=True).offset_mapping
    start = prompt.index("long hair")
    end = start + len("long hair")
    expected = torch.tensor([weight if right > start and left < end else 1.0 for left, right in offsets])
    torch.testing.assert_close(token_weights, expected)


def test_anima_large_finite_weight_is_not_artificially_capped() -> None:
    prompt, ranges = parse_prompt_attention("(long hair:1e30)")
    assert prompt == "long hair"
    assert ranges == [(0, len(prompt), 1e30)]


def test_anima_deep_and_unbalanced_parentheses_do_not_recurse_or_rescan() -> None:
    unbalanced = "(" * 20_000 + "long hair"
    assert parse_prompt_attention(unbalanced)[0] == unbalanced

    deeply_nested = "(" * 1_500 + "(long hair:2)" + ")" * 1_500
    prompt, ranges = parse_prompt_attention(deeply_nested)
    assert prompt == "(" * 1_500 + "long hair" + ")" * 1_500
    assert (1_500, 1_500 + len("long hair"), 2.0) in ranges


def test_anima_weighted_t5_tokenization_preserves_clean_prompt_ids() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    plain_prompt, plain_ranges = parse_prompt_attention("a portrait, long hair, blue eyes")
    weighted_prompt, weighted_ranges = parse_prompt_attention("a portrait, (long hair:2), blue eyes")

    plain_ids, plain_weights = tokenize_t5_with_weights(tokenizer, plain_prompt, plain_ranges, max_length=512)
    weighted_ids, weighted_weights = tokenize_t5_with_weights(
        tokenizer, weighted_prompt, weighted_ranges, max_length=512
    )

    assert plain_prompt == weighted_prompt
    torch.testing.assert_close(plain_ids, weighted_ids)
    assert plain_weights is None
    assert weighted_weights is not None

    offsets = tokenizer(weighted_prompt, return_offsets_mapping=True).offset_mapping
    long_hair_start = weighted_prompt.index("long hair")
    long_hair_end = long_hair_start + len("long hair")
    expected_weights = torch.tensor(
        [
            2.0 if token_end > long_hair_start and token_start < long_hair_end else 1.0
            for token_start, token_end in offsets
        ]
    )
    torch.testing.assert_close(weighted_weights, expected_weights)


def test_anima_explicit_weight_one_matches_unweighted_prompt() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    plain_prompt, plain_ranges = parse_prompt_attention("long hair")
    weighted_prompt, weighted_ranges = parse_prompt_attention("(long hair:1)")

    plain_ids, plain_weights = tokenize_t5_with_weights(tokenizer, plain_prompt, plain_ranges, max_length=512)
    weighted_ids, weighted_weights = tokenize_t5_with_weights(
        tokenizer, weighted_prompt, weighted_ranges, max_length=512
    )

    assert plain_prompt == weighted_prompt
    torch.testing.assert_close(plain_ids, weighted_ids)
    assert plain_weights is None
    assert weighted_weights is None


def test_anima_partial_token_overlap_averages_character_weights() -> None:
    class CombinedTokenTokenizer:
        def __call__(self, prompt, **kwargs):
            assert prompt == "unhappy"
            assert kwargs["return_offsets_mapping"]
            return SimpleNamespace(
                input_ids=torch.tensor([[1, 2]]),
                offset_mapping=torch.tensor([[[0, 7], [0, 0]]]),
                special_tokens_mask=torch.tensor([[0, 1]]),
            )

    prompt, ranges = parse_prompt_attention("(un:2)happy")
    assert prompt == "unhappy"

    ids, weights = tokenize_t5_with_weights(CombinedTokenTokenizer(), prompt, ranges, max_length=512)

    torch.testing.assert_close(ids, torch.tensor([1, 2]))
    assert weights is not None
    torch.testing.assert_close(weights, torch.tensor([9 / 7, 1.0]))


def test_anima_t5_truncation_keeps_ids_and_weights_aligned() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    prompt, ranges = parse_prompt_attention(" ".join(["(long hair:2)"] * 600))

    ids, weights = tokenize_t5_with_weights(tokenizer, prompt, ranges, max_length=512)

    assert weights is not None
    assert ids.shape == weights.shape == (512,)


def test_anima_qwen3_does_not_receive_weighting_markup(monkeypatch) -> None:
    _, tokenizer = _run_encode(monkeypatch, torch.device("cpu"), "a portrait, (long hair:2), blue eyes")

    assert tokenizer.prompt == "a portrait, long hair, blue eyes"
