"""The local-directory rungs of the Mistral tokenizer ladder must try `AutoTokenizer`, not just
`AutoProcessor`.

With transformers 5.5.4, `AutoProcessor.from_pretrained` on a directory whose `config.json` says
`model_type: "mistral3"` — the BFL-style standalone-encoder layout these rungs exist for — resolves
to a *multimodal* processor and raises `OSError: Can't load image processor ...` for the missing
`preprocessor_config.json`, before it ever looks at the tokenizer files sitting right next to it.
`AutoTokenizer` loads the same directory fine. Without the second loader class the ladder falls
through to the HF fetch, which raises `RuntimeError` when offline.
"""

import base64
import json
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


# ---------------------------------------------------------------------------------------------
# A probe rung must never kill the load while a working fallback remains.
# ---------------------------------------------------------------------------------------------

# What transformers actually raises when `tokenizer_config.json` names a `tokenizer_class` the
# installed version does not know: `tokenizer_class_from_name` returns None and is dereferenced
# without a guard. Not an `OSError`/`ValueError`/`KeyError`, so it used to escape the ladder.
_UNKNOWN_TOKENIZER_CLASS_ERROR = AttributeError("'NoneType' object has no attribute 'from_pretrained'")


def test_unknown_tokenizer_class_attributeerror_does_not_escape(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path
) -> None:
    """A directory whose `tokenizer_config.json` declares `MistralCommonTokenizer` — exactly what
    transformers 4.5x's `save_pretrained` writes — makes `AutoTokenizer` raise `AttributeError`.
    That must fall through to the HF rung, not crash the whole load."""
    processor = _FakeLoader("AutoProcessor", raises=OSError("Can't load image processor"))
    auto_tokenizer = _FakeLoader("AutoTokenizer", raises=_UNKNOWN_TOKENIZER_CLASS_ERROR)
    monkeypatch.setattr(mistral_encoder, "_TOKENIZER_LOADER_CLASSES", (processor, auto_tokenizer))

    from_hf = object()
    monkeypatch.setattr(mistral_encoder, "_load_tokenizer_from_hf", lambda logger: from_hf)

    assert mistral_encoder._load_tokenizer_for_model(model_dir, MagicMock()) is from_hf
    assert auto_tokenizer.calls == [model_dir]


def test_hf_fallback_records_attributeerror_instead_of_escaping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same exception on the HF rung: recorded in the actionable `RuntimeError`, not raised raw."""
    processor = _FakeLoader("AutoProcessor", raises=OSError("offline"))
    auto_tokenizer = _FakeLoader("AutoTokenizer", raises=_UNKNOWN_TOKENIZER_CLASS_ERROR)
    monkeypatch.setattr(mistral_encoder, "_TOKENIZER_LOADER_CLASSES", (processor, auto_tokenizer))

    with pytest.raises(RuntimeError, match="Could not load FLUX.2 Mistral tokenizer") as exc_info:
        mistral_encoder._load_tokenizer_from_hf(MagicMock())

    assert "AutoTokenizer(local_only=True): AttributeError" in str(exc_info.value)


# ---------------------------------------------------------------------------------------------
# A real `tekken.json` must never be represented by a mistral-common tokenizer.
# ---------------------------------------------------------------------------------------------

# The FLUX.2 template's structural markers, plus the surrounding standard Mistral control tokens
# they are positioned among. Their *ranks* are what the adapter has to splice.
_TEKKEN_SPECIAL_TOKENS = (
    "<unk>", "<s>", "</s>", "[INST]", "[/INST]", "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]",
    "[TOOL_RESULTS]", "[/TOOL_RESULTS]", "[TOOL_CALLS]", "[IMG]", "<pad>", "[IMG_BREAK]",
    "[IMG_END]", "[PREFIX]", "[MIDDLE]", "[SUFFIX]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]",
    "[TOOL_CONTENT]",
)  # fmt: skip

_TEKKEN_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _write_valid_tekken(path: Path) -> None:
    """Write a *structurally valid* Tekken vocab file that `mistral_common` really parses.

    This is the point of the fixture: a malformed `tekken.json` makes `AutoTokenizer` raise and the
    ladder falls through, which is the easy path to test. A well-formed one does not raise — it
    short-circuits into a mistral-common-backed tokenizer — so only a parseable file exercises the
    behaviour that matters. Byte-fallback vocab keeps it small while staying a real Tekkenizer.
    """
    vocab = [{"rank": i, "token_bytes": base64.b64encode(bytes([i])).decode(), "token_str": None} for i in range(256)]
    path.write_text(
        json.dumps(
            {
                "config": {
                    "pattern": _TEKKEN_SPLIT_PATTERN,
                    "num_vocab_tokens": 256 + len(_TEKKEN_SPECIAL_TOKENS),
                    "default_vocab_size": 256 + len(_TEKKEN_SPECIAL_TOKENS),
                    "default_num_special_tokens": len(_TEKKEN_SPECIAL_TOKENS),
                    "version": "v7",
                },
                "vocab": vocab,
                "special_tokens": [
                    {"rank": i, "token_str": token, "is_control": True}
                    for i, token in enumerate(_TEKKEN_SPECIAL_TOKENS)
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def tekken_model_dir(model_dir: Path) -> Path:
    """A standalone encoder folder carrying a parseable `tekken.json` next to `config.json` —
    the layout of an official mistralai download."""
    _write_valid_tekken(model_dir / "tekken.json")
    return model_dir


def test_valid_tekken_file_is_read_before_transformers_sees_the_dir(
    monkeypatch: pytest.MonkeyPatch, tekken_model_dir: Path
) -> None:
    """With `config.json` present, `AutoTokenizer` *succeeds* on this layout and returns a
    mistral-common-backed tokenizer that BPE-encodes the template markers as literal text. The
    vocab must therefore be read directly, ahead of the transformers probes."""
    probe_result = object()
    probe = _FakeLoader("AutoTokenizer", result=probe_result)
    monkeypatch.setattr(mistral_encoder, "_TOKENIZER_LOADER_CLASSES", (probe,))
    _no_hf_fallback(monkeypatch)

    tokenizer = mistral_encoder._load_tokenizer_for_model(tekken_model_dir, MagicMock())

    assert isinstance(tokenizer, mistral_encoder._TekkenRawTextAdapter)
    assert tokenizer is not probe_result
    assert probe.calls == [], "the directory must never be handed to transformers"


def test_tekken_markers_encode_as_single_special_ids(tekken_model_dir: Path) -> None:
    """The reason the rung exists: the markers must be spliced as single Tekken ids. A
    mistral-common tokenizer would emit the BPE of the literal characters instead, which "works"
    and silently degrades conditioning."""
    tokenizer = mistral_encoder._load_tokenizer_for_model(tekken_model_dir, MagicMock())
    special_ids = tokenizer._special_ids

    assert set(special_ids) == {"[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[INST]", "[/INST]"}

    ids = tokenizer("[SYSTEM_PROMPT]x[/SYSTEM_PROMPT][INST]y[/INST]", padding=False)["input_ids"][0].tolist()
    assert ids[0] == mistral_encoder._TekkenRawTextAdapter._BOS_ID
    assert ids[1] == special_ids["[SYSTEM_PROMPT]"]
    assert ids[-1] == special_ids["[/INST]"]
    for marker, marker_id in special_ids.items():
        assert ids.count(marker_id) == 1, f"{marker} was not spliced as a single id"


class _FakeMistralCommonTokenizer:
    """Stand-in for transformers' mistral-common tokenizer.

    Detection is by module name rather than by class, because the class was renamed
    (`MistralCommonTokenizer` on 4.5x, `MistralCommonBackend` on 5.x), so mimic the module.
    """

    __module__ = "transformers.tokenization_mistral_common"

    def __init__(self, inner: Any) -> None:
        self.tokenizer = inner


def test_mistral_common_backed_result_is_rewrapped(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    """When a probe rung does hand back a mistral-common tokenizer, its underlying vocab is good —
    only its `__call__` is wrong. Re-wrap it rather than discarding a usable tokenizer."""
    inner = object()
    probe = _FakeLoader("AutoTokenizer", result=_FakeMistralCommonTokenizer(inner))
    monkeypatch.setattr(mistral_encoder, "_TOKENIZER_LOADER_CLASSES", (probe,))
    _no_hf_fallback(monkeypatch)

    tokenizer = mistral_encoder._load_tokenizer_for_model(model_dir, MagicMock())

    assert isinstance(tokenizer, mistral_encoder._TekkenRawTextAdapter)
    assert tokenizer._tok is inner


def test_mistral_common_backed_result_without_inner_tokenizer_keeps_falling(
    monkeypatch: pytest.MonkeyPatch, model_dir: Path
) -> None:
    """If the underlying tokenizer cannot be reached, keep falling: an actionable error from a
    later rung beats silently mis-encoded conditioning."""
    probe = _FakeLoader("AutoTokenizer", result=_FakeMistralCommonTokenizer(None))
    monkeypatch.setattr(mistral_encoder, "_TOKENIZER_LOADER_CLASSES", (probe,))

    from_hf = object()
    monkeypatch.setattr(mistral_encoder, "_load_tokenizer_from_hf", lambda logger: from_hf)

    assert mistral_encoder._load_tokenizer_for_model(model_dir, MagicMock()) is from_hf


def test_ordinary_tokenizers_are_not_rewrapped(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    """The re-wrap is narrow: an ordinary HF tokenizer must be returned untouched."""
    tokenizer = object()
    probe = _FakeLoader("AutoTokenizer", result=tokenizer)
    monkeypatch.setattr(mistral_encoder, "_TOKENIZER_LOADER_CLASSES", (probe,))
    _no_hf_fallback(monkeypatch)

    assert mistral_encoder._load_tokenizer_for_model(model_dir, MagicMock()) is tokenizer
