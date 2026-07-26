"""Regression tests for Gemma2 encoder (PiD caption encoder) config probing.

PiD's caption projection is hard-wired to Gemma-2-2b's 2304-dim hidden state. The classifier used
to accept every ``Gemma2ForCausalLM`` directory, so larger variants (9B → 3584, 27B → 4608) were
offered as compatible PiD encoders and then failed with a matrix-shape error deep inside inference.
The config now rejects any hidden size other than 2304 up front.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs import gemma2_encoder as gemma2_encoder_module
from invokeai.backend.model_manager.configs.gemma2_encoder import (
    Gemma2Encoder_Gemma2Encoder_Config,
    Gemma2Encoder_GGUF_Config,
)
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test-model",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _make_gemma_dir(root: Path, *, hidden_size: int) -> MagicMock:
    root.joinpath("config.json").write_text(
        json.dumps({"architectures": ["Gemma2ForCausalLM"], "hidden_size": hidden_size})
    )
    root.joinpath("tokenizer.json").write_text("{}")
    mod = MagicMock()
    mod.path = root
    return mod


def test_gemma_2_2b_matches() -> None:
    """The reference Gemma-2-2b encoder (2304-dim) is a valid PiD caption encoder."""
    with TemporaryDirectory() as tmpdir:
        mod = _make_gemma_dir(Path(tmpdir), hidden_size=2304)
        config = Gemma2Encoder_Gemma2Encoder_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "gemma2_encoder"


@pytest.mark.parametrize("hidden_size", [3584, 4608])
def test_incompatible_gemma_sizes_are_rejected(hidden_size: int) -> None:
    """Gemma 2 9B (3584) and 27B (4608) share the architecture but are incompatible with PiD."""
    with TemporaryDirectory() as tmpdir:
        mod = _make_gemma_dir(Path(tmpdir), hidden_size=hidden_size)
        with pytest.raises(NotAMatchError, match="hidden_size"):
            Gemma2Encoder_Gemma2Encoder_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def _make_gguf_file(root: Path, name: str = "gemma-2-2b-it-q4_k_m.gguf") -> MagicMock:
    # Content is irrelevant: _read_gguf_arch_and_hidden_size is monkeypatched in these tests. Only the
    # .gguf suffix and file existence (raise_if_not_file) are read from the real path.
    path = root / name
    path.write_bytes(b"GGUF")
    mod = MagicMock()
    mod.path = path
    return mod


def test_gemma_2_2b_gguf_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-file gemma2 GGUF reporting 2304-dim is a valid PiD caption encoder."""
    monkeypatch.setattr(gemma2_encoder_module, "_read_gguf_arch_and_hidden_size", lambda _p: ("gemma2", 2304))
    with TemporaryDirectory() as tmpdir:
        mod = _make_gguf_file(Path(tmpdir))
        config = Gemma2Encoder_GGUF_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "gemma2_encoder"
        assert config.format.value == "gguf_quantized"


@pytest.mark.parametrize("hidden_size", [3584, 4608])
def test_incompatible_gemma_gguf_sizes_are_rejected(monkeypatch: pytest.MonkeyPatch, hidden_size: int) -> None:
    monkeypatch.setattr(
        gemma2_encoder_module, "_read_gguf_arch_and_hidden_size", lambda _p: ("gemma2", hidden_size)
    )
    with TemporaryDirectory() as tmpdir:
        mod = _make_gguf_file(Path(tmpdir))
        with pytest.raises(NotAMatchError, match="embedding_length"):
            Gemma2Encoder_GGUF_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_non_gemma2_gguf_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemma2_encoder_module, "_read_gguf_arch_and_hidden_size", lambda _p: ("llama", 4096))
    with TemporaryDirectory() as tmpdir:
        mod = _make_gguf_file(Path(tmpdir))
        with pytest.raises(NotAMatchError, match="not 'gemma2'"):
            Gemma2Encoder_GGUF_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_non_gguf_file_is_rejected() -> None:
    """A non-.gguf file is rejected before any metadata read."""
    with TemporaryDirectory() as tmpdir:
        mod = _make_gguf_file(Path(tmpdir), name="model.bin")
        with pytest.raises(NotAMatchError, match="not a .gguf file"):
            Gemma2Encoder_GGUF_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
