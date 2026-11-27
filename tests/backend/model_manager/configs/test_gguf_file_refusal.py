"""GGUF files no loader reads are refused for every model family, made through ``ModelConfigFactory``.

Run at the default ``allow_unknown``, because that is what decides what the user sees: a refusal with its reason
rather than an Unknown model. The parts are written by gguf-py's own splitting, so they are what llama.cpp's
tooling produces.
"""

import os
import pickle
from asyncio import CancelledError
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from pathlib import Path
from weakref import ReferenceType, ref

import gguf
import numpy as np
import pytest

from invokeai.backend.model_manager import model_on_disk
from invokeai.backend.model_manager.configs.factory import ModelClassificationResult, ModelConfigFactory
from invokeai.backend.model_manager.configs.unknown import Unknown_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk


def _identify(path: Path) -> ModelClassificationResult:
    return ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)


def _write_gguf(path: Path, *, split_max_tensors: int = 0, original_shape: list[int] | None = None) -> None:
    writer = gguf.GGUFWriter(str(path), "llama", split_max_tensors=split_max_tensors)
    writer.add_uint32("llama.block_count", 2)
    if original_shape is not None:
        writer.add_array("comfy.gguf.orig_shape.token_embd.weight", original_shape)
    for name in ("token_embd.weight", "blk.0.attn_q.weight", "blk.1.attn_q.weight"):
        writer.add_tensor(name, np.zeros((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize("part_index", [0, 1, 2])
def test_every_part_of_a_multi_part_gguf_is_refused(tmp_path: Path, part_index: int) -> None:
    _write_gguf(tmp_path / "model.gguf", split_max_tensors=1)
    parts = sorted(tmp_path.glob("model-*-of-*.gguf"))
    assert len(parts) == 3

    result = _identify(parts[part_index])

    assert result.config is None
    assert len(result.invalid_matches) == 1
    reason = str(result.invalid_matches[0])
    assert f"part {part_index + 1} of 3" in reason
    assert "llama-gguf-split --merge" in reason
    parts[part_index].rename(tmp_path / "moved.gguf")


def _cut_in_half(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


@pytest.mark.parametrize(
    "damage",
    [_cut_in_half, lambda path: path.write_bytes(b"hello, not a gguf"), lambda path: path.write_bytes(b"")],
    ids=["cut-short", "not-gguf", "empty"],
)
def test_an_unreadable_gguf_is_refused_with_the_read_error(tmp_path: Path, damage: Callable[[Path], object]) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path)
    damage(path)

    result = _identify(path)

    assert result.config is None
    assert len(result.invalid_matches) == 1
    assert "could not be read as a GGUF file" in str(result.invalid_matches[0])
    path.rename(tmp_path / "moved.gguf")


def test_a_readable_single_file_gguf_that_no_family_claims_is_not_refused(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path)

    result = _identify(path)

    assert isinstance(result.config, Unknown_Config)
    assert result.invalid_matches == []


def test_a_bad_logical_tensor_shape_is_a_candidate_error_and_leaves_the_file_unmapped(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path, original_shape=[2, 2])  # The stored tensor has 16 elements.
    mod = ModelOnDisk(path)

    result = ModelConfigFactory.from_model_on_disk(mod, allow_unknown=False)

    assert result.config is None
    assert "gguf_file" not in result.details
    assert any("different element count" in str(detail) for detail in result.details.values())
    assert ModelOnDisk(path).metadata()["llama.block_count"] == 2
    path.rename(tmp_path / "moved.gguf")


@pytest.mark.parametrize(
    ("error_type", "change_file"),
    [(ValueError, False), (KeyboardInterrupt, True), (CancelledError, True)],
    ids=["hash-error", "interrupted-after-change", "cancelled-after-change"],
)
def test_gguf_classification_propagates_a_hash_failure_and_releases_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException], change_file: bool
) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path)
    mod = ModelOnDisk(path)
    maps: list[ReferenceType[np.memmap]] = []
    reader_init = gguf.GGUFReader.__init__
    failure = error_type("hash failed")

    def track_map(reader: gguf.GGUFReader, reader_path: Path) -> None:
        reader_init(reader, reader_path)
        maps.append(ref(reader.data))

    def fail_hash() -> str:
        if change_file:
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
        raise failure

    monkeypatch.setattr(gguf.GGUFReader, "__init__", track_map)
    monkeypatch.setattr(mod, "hash", fail_hash)
    with pytest.raises(error_type, match="hash failed") as error:
        ModelConfigFactory.from_model_on_disk(mod)

    assert error.value is failure
    assert [map_ref() for map_ref in maps] == [None]
    path.rename(tmp_path / "moved.gguf")
    assert error.traceback


@pytest.mark.parametrize("in_worker", [False, True], ids=["same-thread", "other-thread"])
def test_nested_gguf_classification_keeps_explicit_weight_paths_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_worker: bool
) -> None:
    primary = tmp_path / "primary.gguf"
    other = tmp_path / "other.gguf"
    _write_gguf(primary)
    _write_gguf(other, original_shape=[2, 8])
    mod = ModelOnDisk(primary)

    def probe_again() -> None:
        assert mod.load_state_dict(other)["token_embd.weight"].tensor_shape == (2, 8)
        result = ModelConfigFactory.from_model_on_disk(mod, {"hash": "blake3:nested"})
        assert isinstance(result.config, Unknown_Config)
        assert mod.load_state_dict(primary)["token_embd.weight"].tensor_shape == (4, 4)

    def probe_during_hash() -> str:
        if in_worker:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(probe_again).result()
        else:
            probe_again()
        return "blake3:primary"

    monkeypatch.setattr(mod, "hash", probe_during_hash)
    result = ModelConfigFactory.from_model_on_disk(mod)

    assert isinstance(result.config, Unknown_Config)
    primary.rename(tmp_path / "moved-primary.gguf")
    other.rename(tmp_path / "moved-other.gguf")


@pytest.mark.skipif(os.name != "nt", reason="Exercises case-folded WindowsPath equality")
def test_equal_windows_paths_do_not_lend_the_primary_gguf_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = tmp_path / "model.gguf"
    explicit = tmp_path / "Model.gguf"
    alternate = tmp_path / "alternate.gguf"
    assert explicit == primary and os.fspath(explicit) != os.fspath(primary)
    _write_gguf(primary)
    _write_gguf(alternate, original_shape=[2, 8])
    mod = ModelOnDisk(primary)
    reader_init = gguf.GGUFReader.__init__
    file_identity = model_on_disk._gguf_file_identity

    def physical_path(path: Path) -> Path:
        return alternate if os.fspath(path) == os.fspath(explicit) else path

    def open_case_sensitive(reader: gguf.GGUFReader, path: Path) -> None:
        reader_init(reader, physical_path(path))

    def probe_during_hash() -> str:
        assert mod.load_state_dict(explicit)["token_embd.weight"].tensor_shape == (2, 8)
        return "blake3:primary"

    monkeypatch.setattr(gguf.GGUFReader, "__init__", open_case_sensitive)
    monkeypatch.setattr(model_on_disk, "_gguf_file_identity", lambda path: file_identity(physical_path(path)))
    monkeypatch.setattr(mod, "hash", probe_during_hash)
    result = ModelConfigFactory.from_model_on_disk(mod)

    assert isinstance(result.config, Unknown_Config)
    assert mod.load_state_dict(primary)["token_embd.weight"].tensor_shape == (4, 4)


class _SlottedModelOnDisk(ModelOnDisk):
    __slots__ = ("label",)


@pytest.mark.parametrize("model_type", [ModelOnDisk, _SlottedModelOnDisk], ids=["base", "slotted"])
def test_copy_and_pickle_exclude_the_active_gguf_reader(tmp_path: Path, model_type: type[ModelOnDisk]) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path)
    mod = model_type(path)
    if isinstance(mod, _SlottedModelOnDisk):
        mod.label = "stored in a slot"

    with mod.reuse_gguf_header():
        assert mod.metadata()["llama.block_count"] == 2
        copies = [copy(mod), deepcopy(mod), pickle.loads(pickle.dumps(mod))]
        assert all(cloned._gguf_scope(path) is None for cloned in copies)

    path.rename(tmp_path / "moved.gguf")
    _write_gguf(path, original_shape=[2, 8])
    for cloned in copies:
        assert type(cloned) is model_type
        if isinstance(cloned, _SlottedModelOnDisk):
            assert cloned.label == "stored in a slot"
        assert cloned.load_state_dict()["token_embd.weight"].tensor_shape == (2, 8)


@pytest.mark.parametrize("with_slots", [False, True], ids=["dict", "tuple"])
def test_legacy_serialized_model_on_disk_keeps_its_non_gguf_caches(tmp_path: Path, with_slots: bool) -> None:
    path = tmp_path / "model.safetensors"
    legacy_state = {
        "path": path,
        "name": path.stem,
        "hash_algo": "blake3_single",
        "_state_dict_cache": {path: {"weight": "cached"}},
        "_metadata_cache": {path: {"description": "cached"}},
        "_metadata_errors": {},
    }
    if with_slots:
        mod = _SlottedModelOnDisk.__new__(_SlottedModelOnDisk)
        mod.__setstate__((legacy_state, {"label": "legacy slot"}))
        assert mod.label == "legacy slot"
    else:
        mod = ModelOnDisk.__new__(ModelOnDisk)
        mod.__setstate__(legacy_state)

    assert mod.load_state_dict(path) == {"weight": "cached"}
    assert mod.metadata() == {"description": "cached"}
