"""GGUF files no loader reads are refused for every model family, made through ``ModelConfigFactory``.

Run at the default ``allow_unknown``, because that is what decides what the user sees: a refusal with its reason
rather than an Unknown model. The parts are written by gguf-py's own splitting, so they are what llama.cpp's
tooling produces.
"""

import os
from collections.abc import Callable
from pathlib import Path
from weakref import ReferenceType, ref

import gguf
import numpy as np
import pytest

from invokeai.backend.model_manager.configs.factory import ModelClassificationResult, ModelConfigFactory
from invokeai.backend.model_manager.configs.unknown import Unknown_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk


def _identify(path: Path) -> ModelClassificationResult:
    return ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)


def _assert_renamable_on_windows(path: Path) -> None:
    # Windows refuses to rename a file while its memory map is open.
    path.rename(path.with_name("moved.gguf"))


def _unexpected_hash(self: ModelOnDisk) -> str:
    pytest.fail("GGUF refusal should happen before hashing")


def _write_gguf(path: Path, *, split_max_tensors: int = 0, original_shape: list[int] | None = None) -> None:
    writer = gguf.GGUFWriter(str(path), "llama", split_max_tensors=split_max_tensors)
    writer.add_uint32("llama.block_count", 2)
    writer.add_name("synthetic-model")
    if original_shape is not None:
        writer.add_array("comfy.gguf.orig_shape.token_embd.weight", original_shape)
    for name in ("token_embd.weight", "blk.0.attn_q.weight", "blk.1.attn_q.weight"):
        writer.add_tensor(name, np.zeros((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize("part_index", [0, 1, 2])
def test_every_part_of_a_multi_part_gguf_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, part_index: int
) -> None:
    # Part 1 must report the split even if there is also an invalid logical shape.
    _write_gguf(tmp_path / "model.gguf", split_max_tensors=1, original_shape=[2, 2])
    monkeypatch.setattr(ModelOnDisk, "hash", _unexpected_hash)
    parts = sorted(tmp_path.glob("model-*-of-*.gguf"))
    assert len(parts) == 3

    result = _identify(parts[part_index])

    assert result.config is None
    assert len(result.invalid_matches) == 1
    reason = str(result.invalid_matches[0])
    assert f"part {part_index + 1} of 3" in reason
    assert "llama-gguf-split --merge" in reason
    _assert_renamable_on_windows(parts[part_index])


def _cut_in_half(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


@pytest.mark.parametrize(
    "damage",
    [_cut_in_half, lambda path: path.write_bytes(b"hello, not a gguf"), lambda path: path.write_bytes(b"")],
    ids=["cut-short", "not-gguf", "empty"],
)
def test_an_unreadable_gguf_is_refused_with_the_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: Callable[[Path], object]
) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path)
    damage(path)
    monkeypatch.setattr(ModelOnDisk, "hash", _unexpected_hash)

    result = _identify(path)

    assert result.config is None
    assert len(result.invalid_matches) == 1
    assert "could not be read as a GGUF file" in str(result.invalid_matches[0])
    _assert_renamable_on_windows(path)


def test_a_readable_single_file_gguf_that_no_family_claims_is_not_refused(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path)

    result = _identify(path)

    assert isinstance(result.config, Unknown_Config)
    assert result.invalid_matches == []


def test_a_bad_logical_tensor_shape_is_a_candidate_error_and_leaves_the_file_unmapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path, original_shape=[2, 2])  # The stored tensor has 16 elements.
    mod = ModelOnDisk(path)
    mappings: list[ReferenceType[np.memmap]] = []
    reader_init = gguf.GGUFReader.__init__

    def track_mapping(reader: gguf.GGUFReader, reader_path: Path) -> None:
        reader_init(reader, reader_path)
        mappings.append(ref(reader.data))

    monkeypatch.setattr(gguf.GGUFReader, "__init__", track_mapping)
    result = ModelConfigFactory.from_model_on_disk(mod, allow_unknown=False)

    assert result.config is None
    assert "gguf_file" not in result.details
    assert any(
        isinstance(detail, ValueError) and "different element count" in str(detail)
        for detail in result.details.values()
    )
    assert mod.metadata()["llama.block_count"] == 2
    assert mod.metadata_read_error() is None
    with pytest.raises(ValueError, match="different element count"):
        mod.load_state_dict()
    assert [mapping() for mapping in mappings] == [None]
    _assert_renamable_on_windows(path)


def test_loading_gguf_state_first_also_caches_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path, original_shape=[2, 8])
    mod = ModelOnDisk(tmp_path)
    opened_paths: list[Path] = []
    reader_init = gguf.GGUFReader.__init__

    def count_open(reader: gguf.GGUFReader, reader_path: Path) -> None:
        opened_paths.append(reader_path)
        reader_init(reader, reader_path)

    monkeypatch.setattr(gguf.GGUFReader, "__init__", count_open)
    state = mod.load_state_dict(path)
    assert state["token_embd.weight"].tensor_shape == (2, 8)
    assert all(tensor.quantized_data.is_meta for tensor in state.values())
    assert mod.metadata(path)["llama.block_count"] == 2
    assert mod.metadata_read_error(path) is None
    assert opened_paths == [path]


@pytest.mark.skipif(os.name != "nt", reason="Windows Path equality folds differently spelled suffixes")
@pytest.mark.parametrize("uppercase_first", [False, True], ids=["lowercase-first", "uppercase-first"])
def test_gguf_caches_keep_suffix_spellings_distinct(tmp_path: Path, uppercase_first: bool) -> None:
    path = tmp_path / "model.gguf"
    uppercase = tmp_path / "model.GGUF"
    _write_gguf(path)
    mod = ModelOnDisk(path)

    if uppercase_first:
        assert mod.metadata(uppercase) == {}

    assert mod.load_state_dict(path)["token_embd.weight"].tensor_shape == (4, 4)
    assert mod.metadata(path)["general.architecture"] == "llama"
    assert mod.metadata(uppercase) == {}
    with pytest.raises(ValueError, match="Unrecognized model extension"):
        mod.load_state_dict(uppercase)


def test_a_scalar_decode_error_refuses_install_but_keeps_valid_meta_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path, original_shape=[2, 8])
    path.write_bytes(path.read_bytes().replace(b"synthetic-model", b"\xffynthetic-model"))
    mod = ModelOnDisk(path)
    opened_paths: list[Path] = []
    reader_init = gguf.GGUFReader.__init__

    def count_open(reader: gguf.GGUFReader, reader_path: Path) -> None:
        opened_paths.append(reader_path)
        reader_init(reader, reader_path)

    monkeypatch.setattr(gguf.GGUFReader, "__init__", count_open)
    assert mod.metadata() == {}
    assert (mod.metadata_read_error() or "").startswith("UnicodeDecodeError:")
    state = mod.load_state_dict()
    assert state["token_embd.weight"].tensor_shape == (2, 8)
    assert all(tensor.quantized_data.is_meta for tensor in state.values())

    monkeypatch.setattr(ModelOnDisk, "hash", _unexpected_hash)
    result = ModelConfigFactory.from_model_on_disk(mod)
    assert result.config is None
    assert "could not be read as a GGUF file (UnicodeDecodeError:" in str(result.details["gguf_file"])
    assert opened_paths == [path]
