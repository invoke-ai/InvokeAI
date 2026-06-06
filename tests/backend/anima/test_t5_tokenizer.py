"""Tests for the bundled T5-XXL tokenizer used by Anima.

Anima feeds T5-XXL token IDs into the LLM Adapter's learned embedding table
(nn.Embedding(32128, 1024)). The tokenizer is vendored in the package so users
do not need to install a 9GB T5-XXL encoder just to obtain a ~2MB tokenizer.
"""

from invokeai.backend.anima.t5_tokenizer import ANIMA_T5_VOCAB_SIZE, load_bundled_t5_tokenizer


def test_bundled_tokenizer_is_fast() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    assert tokenizer.is_fast


def test_bundled_tokenizer_known_ids() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    ids = tokenizer("a cat sitting on a mat", truncation=True, max_length=512).input_ids
    assert ids == [3, 9, 1712, 3823, 30, 3, 9, 6928, 1]


def test_bundled_tokenizer_appends_eos() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    assert tokenizer("", truncation=True, max_length=512).input_ids == [1]


def test_bundled_tokenizer_ids_within_adapter_embedding() -> None:
    tokenizer = load_bundled_t5_tokenizer()
    ids = tokenizer(
        "a very long and unusual prompt with rare tokens: zxqwv 12345",
        truncation=True,
        max_length=512,
    ).input_ids
    assert all(0 <= i < ANIMA_T5_VOCAB_SIZE for i in ids)


def test_bundled_tokenizer_is_cached() -> None:
    assert load_bundled_t5_tokenizer() is load_bundled_t5_tokenizer()
