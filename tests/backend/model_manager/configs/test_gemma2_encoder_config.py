"""Regression tests for Gemma2 encoder (PiD caption encoder) config probing.

PiD's caption projection is hard-wired to Gemma-2-2b's 2304-dim hidden state. The classifier used
to accept every ``Gemma2ForCausalLM`` directory, so larger variants (9B → 3584, 27B → 4608) were
offered as compatible PiD encoders and then failed with a matrix-shape error deep inside inference.
The config now rejects any hidden size other than 2304 up front.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock
from weakref import ref

import gguf
import numpy as np
import pytest

from invokeai.backend.model_manager.configs import gemma2_encoder as gemma2_encoder_module
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.gemma2_encoder import (
    Gemma2Encoder_Gemma2Encoder_Config,
    Gemma2Encoder_GGUF_Config,
)
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk

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
        # The Gemma encoder has no `variant` field. Re-identifying a model previously mis-detected as a
        # Qwen3 GGUF (which has a variant) to this config must therefore drop the stale variant — the
        # serialized record carries no variant key, so replace_model overwrites it away.
        assert "variant" not in config.model_dump()


@pytest.mark.parametrize("hidden_size", [3584, 4608])
def test_incompatible_gemma_gguf_sizes_are_rejected(monkeypatch: pytest.MonkeyPatch, hidden_size: int) -> None:
    monkeypatch.setattr(gemma2_encoder_module, "_read_gguf_arch_and_hidden_size", lambda _p: ("gemma2", hidden_size))
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


def _write_gemma2_gguf(path: Path, width: int, *, duplicate_header: bool = False) -> None:
    writer = gguf.GGUFWriter(path, "gemma2")
    writer.add_embedding_length(width)
    writer.add_tensor("probe.weight", np.zeros((1, 4) if width == 2304 else (2, 2), dtype=np.float32))
    if duplicate_header:
        writer.add_uint64("GGUF.tensor_count", 5)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize(
    ("width", "duplicate_header", "reason"),
    [
        (2304, False, None),
        (3584, False, "embedding_length 3584 is incompatible"),
        (2304, True, "not a readable GGUF file: 'Duplicate GGUF.tensor_count "),
    ],
    ids=["match", "wrong-width", "strict-duplicate-header"],
)
def test_the_factory_preserves_gemma2_gguf_metadata_checks(
    tmp_path: Path, width: int, duplicate_header: bool, reason: str | None
) -> None:
    path = tmp_path / "gemma2.gguf"
    _write_gemma2_gguf(path, width, duplicate_header=duplicate_header)

    result = ModelConfigFactory.from_model_on_disk(path, dict(_OVERRIDE_FIELDS), allow_unknown=False)

    if reason is None:
        assert isinstance(result.config, Gemma2Encoder_GGUF_Config)
    else:
        assert result.config is None
        refusal = result.details[Gemma2Encoder_GGUF_Config.__name__]
        assert isinstance(refusal, NotAMatchError)
        assert reason in str(refusal)
    path.rename(tmp_path / "moved.gguf")


@pytest.mark.parametrize(
    "atomic_replace",
    [False, pytest.param(True, marks=pytest.mark.skipif(os.name == "nt", reason="Windows locks a mapped file"))],
    ids=["mtime-change", "atomic-replace"],
)
def test_factory_refuses_a_changed_gguf_and_can_retry_the_same_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, atomic_replace: bool
) -> None:
    path = tmp_path / "gemma2.gguf"
    replacement = tmp_path / "replacement.gguf"
    _write_gemma2_gguf(path, 2304)
    _write_gemma2_gguf(replacement, 3584)
    mod = ModelOnDisk(path)
    real_hash = mod.hash
    reader_init = gguf.GGUFReader.__init__
    maps = []

    def track_map(reader: gguf.GGUFReader, reader_path: Path) -> None:
        reader_init(reader, reader_path)
        maps.append(ref(reader.data))

    def mutate_during_hash() -> str:
        assert mod.load_state_dict()["probe.weight"].tensor_shape == (1, 4)
        if atomic_replace:
            replacement.replace(path)
        else:
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
        return real_hash()

    monkeypatch.setattr(gguf.GGUFReader, "__init__", track_map)
    monkeypatch.setattr(mod, "hash", mutate_during_hash)
    result = ModelConfigFactory.from_model_on_disk(mod)

    assert result.config is None
    assert [str(error) for error in result.invalid_matches] == [
        "gemma2.gguf changed during GGUF identification. Retry with a stable file."
    ]
    assert [map_ref() for map_ref in maps] == [None]
    path.rename(tmp_path / "moved.gguf")
    (tmp_path / "moved.gguf" if atomic_replace else replacement).rename(path)

    retry = ModelConfigFactory.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS), allow_unknown=False)

    assert retry.config is None
    assert "embedding_length 3584 is incompatible" in str(retry.details[Gemma2Encoder_GGUF_Config.__name__])
    assert mod.metadata()["gemma2.embedding_length"] == 3584
    assert mod.load_state_dict()["probe.weight"].tensor_shape == (2, 2)


def test_a_duck_typed_disk_model_can_identify_a_real_gemma2_gguf(tmp_path: Path) -> None:
    path = tmp_path / "gemma2.gguf"
    _write_gemma2_gguf(path, 2304)
    model = ModelOnDisk(path)

    class Adapter:
        path = model.path
        name = model.name
        metadata = staticmethod(model.metadata)
        metadata_read_error = staticmethod(model.metadata_read_error)
        load_state_dict = staticmethod(model.load_state_dict)

    result = ModelConfigFactory.from_model_on_disk(Adapter(), dict(_OVERRIDE_FIELDS), allow_unknown=False)

    assert isinstance(result.config, Gemma2Encoder_GGUF_Config)
