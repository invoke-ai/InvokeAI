"""Tests for the bundled Qwen3 tokenizer used by single-file / GGUF Qwen3 encoders.

Single-file and GGUF Qwen3 encoder checkpoints (Z-Image, Anima) ship weights only.
The tokenizer is vendored in the package so the encoder works fully offline instead
of pulling ``Qwen/Qwen3-4B`` from HuggingFace on first use.
"""

from invokeai.backend.qwen3.qwen3_tokenizer import load_bundled_qwen3_tokenizer

# The Qwen3 BPE tokenizer is shared across the 0.6B / 4B / 8B variants.
QWEN3_VOCAB_SIZE = 151643


def test_bundled_tokenizer_is_fast() -> None:
    tokenizer = load_bundled_qwen3_tokenizer()
    assert tokenizer.is_fast


def test_bundled_tokenizer_known_ids() -> None:
    tokenizer = load_bundled_qwen3_tokenizer()
    ids = tokenizer("A cinematic photo of a cat").input_ids
    assert ids == [32, 64665, 6548, 315, 264, 8251]


def test_bundled_tokenizer_roundtrip() -> None:
    tokenizer = load_bundled_qwen3_tokenizer()
    prompt = "A cinematic photo of a cat"
    ids = tokenizer(prompt).input_ids
    assert tokenizer.decode(ids) == prompt


def test_bundled_tokenizer_ids_within_vocab() -> None:
    tokenizer = load_bundled_qwen3_tokenizer()
    ids = tokenizer(
        "a very long and unusual prompt with rare tokens: zxqwv 12345",
    ).input_ids
    assert all(0 <= i < QWEN3_VOCAB_SIZE for i in ids)


def test_bundled_tokenizer_is_cached() -> None:
    assert load_bundled_qwen3_tokenizer() is load_bundled_qwen3_tokenizer()
