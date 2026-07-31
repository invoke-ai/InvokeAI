"""The local-directory rungs of the Mistral tokenizer ladder must try `AutoTokenizer`, not just
`AutoProcessor`.

With transformers 5.5.4, `AutoProcessor.from_pretrained` on a directory whose `config.json` says
`model_type: "mistral3"` — the BFL-style standalone-encoder layout these rungs exist for — resolves
to a *multimodal* processor and raises `OSError: Can't load image processor ...` for the missing
`preprocessor_config.json`, before it ever looks at the tokenizer files sitting right next to it.
`AutoTokenizer` loads the same directory fine. Without the second loader class the ladder falls
through to the HF fetch, which raises `RuntimeError` when offline.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from transformers import AutoProcessor, AutoTokenizer

from invokeai.backend.model_manager.load.model_loaders import mistral_encoder


def test_ladder_contract() -> None:
    """The rest of this module monkeypatches the loader tuple, so pin what production actually uses."""
    assert mistral_encoder._TOKENIZER_LOADER_CLASSES == (AutoProcessor, AutoTokenizer)
    assert KeyError in mistral_encoder._TOKENIZER_LOAD_ERRORS


class _FakeLoader:
    """Stand-in for AutoProcessor / AutoTokenizer with a scripted `from_pretrained`."""

    def __init__(self, name: str, result: Any = None, raises: BaseException | None = None) -> None:
        self.__name__ = name
        self._result = result
        self._raises = raises
        self.calls: list[Path] = []

    def from_pretrained(self, path: Path, **kwargs: Any) -> Any:
        self.calls.append(Path(path))
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A BFL-style standalone encoder folder: weights + tokenizer files at the root, no
    `preprocessor_config.json` and no sibling `tokenizer/`."""
    (tmp_path / "config.json").write_text('{"model_type": "mistral3"}', encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return tmp_path


def _no_hf_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(logger: Any) -> Any:
        raise AssertionError("ladder fell through to the HuggingFace fetch")

    monkeypatch.setattr(mistral_encoder, "_load_tokenizer_from_hf", _fail)


def test_root_directory_falls_back_to_autotokenizer(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    tokenizer = object()
    processor = _FakeLoader("AutoProcessor", raises=OSError("Can't load image processor"))
    auto_tokenizer = _FakeLoader("AutoTokenizer", result=tokenizer)
    monkeypatch.setattr(mistral_encoder, "_TOKENIZER_LOADER_CLASSES", (processor, auto_tokenizer))
    _no_hf_fallback(monkeypatch)

    assert mistral_encoder._load_tokenizer_for_model(model_dir, MagicMock()) is tokenizer
    # Order matters: AutoProcessor stays the preferred loader, AutoTokenizer is the fallback.
    assert processor.calls == [model_dir]
    assert auto_tokenizer.calls == [model_dir]


def test_sibling_tokenizer_dir_falls_back_to_autotokenizer(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = object()
    processor = _FakeLoader("AutoProcessor", raises=OSError("Can't load image processor"))
    auto_tokenizer = _FakeLoader("AutoTokenizer", result=tokenizer)
    monkeypatch.setattr(mistral_encoder, "_TOKENIZER_LOADER_CLASSES", (processor, auto_tokenizer))
    _no_hf_fallback(monkeypatch)

    assert mistral_encoder._load_tokenizer_for_model(model_dir, MagicMock()) is tokenizer
    assert auto_tokenizer.calls == [tokenizer_dir]


def test_tekken_only_dir_keyerror_does_not_escape(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    """`AutoTokenizer` raises `KeyError: 'special_tokens'` on a directory carrying only
    `tekken.json`. That must fall through to the HF fetch, not crash the load."""
    processor = _FakeLoader("AutoProcessor", raises=OSError("Can't load image processor"))
    auto_tokenizer = _FakeLoader("AutoTokenizer", raises=KeyError("special_tokens"))
    monkeypatch.setattr(mistral_encoder, "_TOKENIZER_LOADER_CLASSES", (processor, auto_tokenizer))

    from_hf = object()
    monkeypatch.setattr(mistral_encoder, "_load_tokenizer_from_hf", lambda logger: from_hf)

    assert mistral_encoder._load_tokenizer_for_model(model_dir, MagicMock()) is from_hf


def test_hf_fallback_reports_every_attempt_instead_of_raising_keyerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HF rung shares the loader/except tuples, so a `KeyError` there is recorded in the
    actionable `RuntimeError` rather than escaping raw."""
    processor = _FakeLoader("AutoProcessor", raises=OSError("offline"))
    auto_tokenizer = _FakeLoader("AutoTokenizer", raises=KeyError("special_tokens"))
    monkeypatch.setattr(mistral_encoder, "_TOKENIZER_LOADER_CLASSES", (processor, auto_tokenizer))

    with pytest.raises(RuntimeError, match="Could not load FLUX.2 Mistral tokenizer") as exc_info:
        mistral_encoder._load_tokenizer_from_hf(MagicMock())

    assert "AutoTokenizer(local_only=True): KeyError" in str(exc_info.value)
