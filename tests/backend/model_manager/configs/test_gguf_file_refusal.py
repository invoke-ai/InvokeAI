"""GGUF files no loader reads are refused for every model family, made through ``ModelConfigFactory``.

Run at the default ``allow_unknown``, because that is what decides what the user sees: a refusal with its reason
rather than an Unknown model. The parts are written by gguf-py's own splitting, so they are what llama.cpp's
tooling produces.
"""

from collections.abc import Callable
from pathlib import Path

import gguf
import numpy as np
import pytest

from invokeai.backend.model_manager.configs.factory import ModelClassificationResult, ModelConfigFactory
from invokeai.backend.model_manager.configs.unknown import Unknown_Config


def _identify(path: Path) -> ModelClassificationResult:
    return ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)


def _write_gguf(path: Path, *, split_max_tensors: int = 0) -> None:
    writer = gguf.GGUFWriter(str(path), "llama", split_max_tensors=split_max_tensors)
    writer.add_uint32("llama.block_count", 2)
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


def test_a_readable_single_file_gguf_that_no_family_claims_is_not_refused(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(path)

    result = _identify(path)

    assert isinstance(result.config, Unknown_Config)
    assert result.invalid_matches == []
