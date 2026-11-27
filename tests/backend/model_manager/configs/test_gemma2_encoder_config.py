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
from weakref import ref

import gguf
import numpy as np
import pytest

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


def test_non_gguf_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.bin"
    path.touch()
    with pytest.raises(NotAMatchError, match=r"not a \.gguf file"):
        Gemma2Encoder_GGUF_Config.from_model_on_disk(ModelOnDisk(path), dict(_OVERRIDE_FIELDS))


def _write_gguf(path: Path, architecture: str, width: int, *, duplicate_header: bool = False) -> None:
    writer = gguf.GGUFWriter(path, architecture)
    writer.add_embedding_length(width)
    writer.add_name("synthetic-model")
    writer.add_tensor("probe.weight", np.zeros((1, 4), dtype=np.float32))
    if duplicate_header:
        writer.add_uint64("GGUF.tensor_count", 5)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize(
    ("name", "architecture", "width", "duplicate_header", "reason"),
    [
        ("gemma2.gguf", "gemma2", 2304, False, None),
        ("gemma2.GGUF", "gemma2", 2304, False, None),
        ("gemma2.gguf", "gemma2", 3584, False, "embedding_length 3584 is incompatible"),
        ("gemma2.gguf", "gemma2", 4608, False, "embedding_length 4608 is incompatible"),
        ("gemma2.gguf", "gemma2", 2304, True, "not a readable GGUF file: 'Duplicate GGUF.tensor_count "),
        ("llama.gguf", "llama", 4096, False, "not 'gemma2'"),
    ],
    ids=["match", "upper-case-suffix", "9b", "27b", "strict-duplicate-header", "wrong-architecture"],
)
def test_the_factory_preserves_gemma2_gguf_metadata_checks(
    tmp_path: Path, name: str, architecture: str, width: int, duplicate_header: bool, reason: str | None
) -> None:
    path = tmp_path / name
    _write_gguf(path, architecture, width, duplicate_header=duplicate_header)

    result = ModelConfigFactory.from_model_on_disk(path, dict(_OVERRIDE_FIELDS), allow_unknown=False)

    if reason is None:
        assert isinstance(result.config, Gemma2Encoder_GGUF_Config)
        assert result.config.type.value == "gemma2_encoder"
        assert result.config.format.value == "gguf_quantized"
        # A re-identified Gemma encoder record drops the stale variant of a previously mis-detected model.
        assert "variant" not in result.config.model_dump()
    else:
        assert result.config is None
        refusal = result.details[Gemma2Encoder_GGUF_Config.__name__]
        assert isinstance(refusal, NotAMatchError)
        assert reason in str(refusal)


def test_gemma2_gguf_releases_the_only_reader_before_hashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "gemma2.gguf"
    _write_gguf(path, "gemma2", 2304)
    mod = ModelOnDisk(path)
    reader_init = gguf.GGUFReader.__init__
    real_hash = mod.hash
    opened_paths: list[Path] = []
    readers = []
    mappings = []

    def track_open(reader: gguf.GGUFReader, reader_path: Path) -> None:
        reader_init(reader, reader_path)
        opened_paths.append(reader_path)
        readers.append(ref(reader))
        mappings.append(ref(reader.data))

    def hash_after_parse() -> str:
        assert opened_paths == [path]
        assert [reader() for reader in readers] == [None]
        assert [mapping() for mapping in mappings] == [None]
        return real_hash()

    monkeypatch.setattr(gguf.GGUFReader, "__init__", track_open)
    monkeypatch.setattr(mod, "hash", hash_after_parse)
    result = ModelConfigFactory.from_model_on_disk(mod, allow_unknown=False)

    assert isinstance(result.config, Gemma2Encoder_GGUF_Config)
    assert opened_paths == [path]


@pytest.mark.parametrize("duplicate_header", [False, True], ids=["strict-success", "strict-refusal"])
def test_uppercase_gemma2_only_decodes_probe_fields(tmp_path: Path, duplicate_header: bool) -> None:
    path = tmp_path / "gemma2.GGUF"
    _write_gguf(path, "gemma2", 2304, duplicate_header=duplicate_header)
    path.write_bytes(path.read_bytes().replace(b"synthetic-model", b"\xffynthetic-model"))

    result = ModelConfigFactory.from_model_on_disk(path, dict(_OVERRIDE_FIELDS), allow_unknown=False)

    if duplicate_header:
        assert result.config is None
        refusal = result.details[Gemma2Encoder_GGUF_Config.__name__]
        assert isinstance(refusal, NotAMatchError)
        assert "Duplicate GGUF.tensor_count" in str(refusal)
    else:
        assert isinstance(result.config, Gemma2Encoder_GGUF_Config)
