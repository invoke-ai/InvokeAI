"""Tests for the shared loader behind every vendored tokenizer in the package."""

import gzip
import json
from pathlib import Path

import pytest

from invokeai.backend.util.bundled_tokenizer import load_gzipped_tokenizer_dir


def _vendor(directory: Path, *, gzipped: bool = True, with_vocabulary: bool = True) -> Path:
    """Stage a minimal but real fast-tokenizer directory, the way the package ships one."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 8}),
        encoding="utf-8",
    )
    if not with_vocabulary:
        return directory

    vocabulary = {"a": 0, "b": 1, "ab": 2}
    payload = json.dumps(
        {
            "version": "1.0",
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "model": {"type": "WordLevel", "vocab": vocabulary, "unk_token": "a"},
        }
    ).encode("utf-8")

    if gzipped:
        with gzip.open(directory / "tokenizer.json.gz", "wb") as fh:
            fh.write(payload)
    else:
        (directory / "tokenizer.json").write_bytes(payload)
    return directory


def test_a_gzipped_vocabulary_is_expanded_and_loaded(tmp_path: Path) -> None:
    tokenizer = load_gzipped_tokenizer_dir(_vendor(tmp_path / "gz"))

    assert tokenizer("ab").input_ids == [2]


def test_plain_and_gzipped_members_are_staged_together(tmp_path: Path) -> None:
    """tokenizer_config.json ships uncompressed beside a gzipped vocabulary; both must arrive."""
    tokenizer = load_gzipped_tokenizer_dir(_vendor(tmp_path / "mixed"))

    assert tokenizer.model_max_length == 8


def test_an_uncompressed_vocabulary_is_also_accepted(tmp_path: Path) -> None:
    """The t5 bundle ships its vocabulary uncompressed; the helper must not require gzip."""
    tokenizer = load_gzipped_tokenizer_dir(_vendor(tmp_path / "plain", gzipped=False))

    assert tokenizer("ab").input_ids == [2]


def test_kwargs_reach_from_pretrained(tmp_path: Path) -> None:
    """The Qwen3-VL bundle passes extra_special_tokens={} to keep parity with its old hub call."""
    tokenizer = load_gzipped_tokenizer_dir(_vendor(tmp_path / "kw"), model_max_length=3)

    assert tokenizer.model_max_length == 3


def test_a_vendored_directory_without_its_vocabulary_is_refused_by_name(tmp_path: Path) -> None:
    """The silent failure this guard exists for.

    A wheel whose package-data globs matched ``tokenizer/*.json`` but not ``*.json.gz`` ships
    exactly this directory. transformers does not treat it as an error: it hands back a tokenizer
    with a one-token vocabulary that encodes every prompt to an empty sequence, so generation runs
    on no conditioning with nothing in the log. The load has to stop here instead.
    """
    incomplete = _vendor(tmp_path / "incomplete", with_vocabulary=False)

    with pytest.raises(FileNotFoundError, match="missing its vocabulary"):
        load_gzipped_tokenizer_dir(incomplete)


def test_a_missing_directory_is_refused_rather_than_raising_from_iterdir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing its vocabulary"):
        load_gzipped_tokenizer_dir(tmp_path / "absent")


def test_the_vendored_directory_is_not_written_to(tmp_path: Path) -> None:
    """Staging, not expanding in place: the package directory is read-only in most installs."""
    vendored = _vendor(tmp_path / "readonly")
    before = sorted(p.name for p in vendored.iterdir())

    load_gzipped_tokenizer_dir(vendored)

    assert sorted(p.name for p in vendored.iterdir()) == before
