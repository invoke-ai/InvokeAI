"""Identification of a llama.cpp Qwen3-VL ``mmproj`` GGUF as a vision tower.

The language-tower GGUF and the tower are two files of one repository, so the decisive tests go through
``ModelConfigFactory`` and assert ``match_count``: each file must be claimed by exactly one config, whichever
order the (set-ordered) candidates are probed in. Fixtures are real GGUF files of a few hundred bytes.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
    Qwen3VLEncoder_GGUF_Config,
    Qwen3VLVisionTower_GGUF_Config,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import ModelType, Qwen3VLVariantType

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

# What llama.cpp writes for the two released towers: the language model's width and the tower's depth.
_8B = {"projection_dim": 4096, "block_count": 27}
_4B = {"projection_dim": 2560, "block_count": 24}


def _write_mmproj(
    path: Path,
    *,
    projection_dim: int | None = 4096,
    block_count: int = 27,
    projector_type: str | None = "qwen3vl_merger",
    file_type: str | None = "mmproj",
    last_block: int | None = None,
    patch_frames: int = 2,
) -> Path:
    """A structurally honest mmproj GGUF: the metadata llama.cpp writes, and the tensors identification checks.

    `last_block` defaults to the declared depth's last block; a smaller one is a truncated file.
    """
    import gguf

    writer = gguf.GGUFWriter(str(path), "clip")
    if file_type is not None:
        writer.add_string("general.type", file_type)
    if projector_type is not None:
        writer.add_string("clip.projector_type", projector_type)
    if projection_dim is not None:
        writer.add_uint32("clip.vision.projection_dim", projection_dim)
    writer.add_uint32("clip.vision.block_count", block_count)
    stub = np.zeros((4, 4), dtype=np.float16)
    writer.add_tensor(f"v.blk.{block_count - 1 if last_block is None else last_block}.attn_qkv.weight", stub)
    writer.add_tensor("v.patch_embd.weight", stub)
    if patch_frames == 2:
        writer.add_tensor("v.patch_embd.weight.1", stub)
    writer.add_tensor("v.position_embd.weight", stub)
    writer.add_tensor("mm.2.weight", stub)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _write_language_tower(path: Path) -> Path:
    import gguf

    writer = gguf.GGUFWriter(str(path), "qwen3vl")
    writer.add_uint32("qwen3vl.block_count", 36)
    writer.add_uint32("qwen3vl.embedding_length", 4096)
    writer.add_tensor("token_embd.weight", np.zeros((8, 4096), dtype=np.float32))
    writer.add_tensor("blk.0.attn_q.weight", np.zeros((4, 4), dtype=np.float32))
    writer.add_tensor("blk.35.attn_q.weight", np.zeros((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _identify(path: Path, *, allow_unknown: bool = False) -> Any:
    return ModelConfigFactory.from_model_on_disk(path, dict(_OVERRIDE_FIELDS), allow_unknown=allow_unknown)


def _config_from(path: Path) -> Qwen3VLVisionTower_GGUF_Config:
    return Qwen3VLVisionTower_GGUF_Config.from_model_on_disk(ModelOnDisk(path), dict(_OVERRIDE_FIELDS))


@pytest.mark.parametrize(
    "dims, expected",
    [(_8B, Qwen3VLVariantType.Qwen3VL_8B), (_4B, Qwen3VLVariantType.Qwen3VL_4B)],
    ids=["8b", "4b"],
)
def test_an_mmproj_identifies_unambiguously_as_a_vision_tower(
    tmp_path: Path, dims: dict[str, int], expected: Qwen3VLVariantType
) -> None:
    result = _identify(_write_mmproj(tmp_path / "mmproj-Qwen3VL-Instruct-F16.gguf", **dims))

    assert result.match_count == 1
    assert isinstance(result.config, Qwen3VLVisionTower_GGUF_Config)
    assert result.config.type is ModelType.Qwen3VLVisionTower
    assert result.config.variant is expected


def test_the_language_tower_is_not_a_vision_tower_and_the_other_way_round(tmp_path: Path) -> None:
    language = _identify(_write_language_tower(tmp_path / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"))
    assert language.match_count == 1
    assert isinstance(language.config, Qwen3VLEncoder_GGUF_Config)

    with pytest.raises(NotAMatchError, match="mmproj"):
        _config_from(tmp_path / "Qwen3VL-8B-Instruct-Q4_K_M.gguf")


def test_another_models_projector_is_not_claimed(tmp_path: Path) -> None:
    path = _write_mmproj(tmp_path / "mmproj-gemma.gguf", projector_type="gemma3")

    with pytest.raises(NotAMatchError, match="gemma3"):
        _config_from(path)
    assert _identify(path, allow_unknown=True).config.type is ModelType.Unknown


def test_a_clip_file_that_is_not_an_mmproj_is_not_claimed(tmp_path: Path) -> None:
    with pytest.raises(NotAMatchError, match="mmproj"):
        _config_from(_write_mmproj(tmp_path / "clip.gguf", file_type=None))


@pytest.mark.parametrize("projection_dim", [5120, None], ids=["32b", "absent"])
def test_an_unsupported_width_is_refused_with_its_reason(tmp_path: Path, projection_dim: int | None) -> None:
    """Recognised as Qwen3-VL's projector and unusable: the refusal must be final, or the file registers as an
    Unknown model and the reason stays in the server log."""
    path = _write_mmproj(tmp_path / "mmproj-Qwen3VL-32B-F16.gguf", projection_dim=projection_dim)

    result = _identify(path, allow_unknown=True)

    assert result.config is None
    assert any("hidden size" in str(reason) for reason in result.invalid_matches)


@pytest.mark.parametrize(
    "broken",
    [{"last_block": 3}, {"patch_frames": 1}, {"block_count": 24}],
    ids=["truncated", "one_patch_frame", "depth_of_the_other_variant"],
)
def test_an_incomplete_or_mismatched_tower_is_refused(tmp_path: Path, broken: dict[str, int]) -> None:
    path = _write_mmproj(tmp_path / "mmproj.gguf", **_8B | broken)

    with pytest.raises(InvalidMatchError, match="incomplete or of another build"):
        _config_from(path)


def test_a_variant_override_does_not_skip_the_completeness_check(tmp_path: Path) -> None:
    path = _write_mmproj(tmp_path / "mmproj.gguf", **_8B, last_block=3)

    with pytest.raises(InvalidMatchError, match="incomplete"):
        Qwen3VLVisionTower_GGUF_Config.from_model_on_disk(
            ModelOnDisk(path), {**_OVERRIDE_FIELDS, "variant": Qwen3VLVariantType.Qwen3VL_8B}
        )
