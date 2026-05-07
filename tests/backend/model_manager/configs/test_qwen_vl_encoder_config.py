"""Tests for Qwen VL encoder config identification.

The single-file checkpoint identifier reads only the safetensors key index
instead of loading the full tensor data — a 7GB fp8 encoder otherwise pins
~7GB of RAM during model scan.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.qwen_vl_encoder import (
    QwenVLEncoder_Checkpoint_Config,
    _has_qwen_vl_keys,
    _read_safetensors_keys,
)

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test-model.safetensors",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _write_safetensors(path: Path, keys: list[str]) -> None:
    """Write a safetensors file with tiny placeholder tensors for the given keys."""
    sd = {k: torch.zeros(1, dtype=torch.float32) for k in keys}
    save_file(sd, str(path))


def test_has_qwen_vl_keys_accepts_lm_plus_visual() -> None:
    assert _has_qwen_vl_keys(["model.embed_tokens.weight", "visual.patch_embed.proj.weight"])
    assert _has_qwen_vl_keys(["model.layers.0.self_attn.q_proj.weight", "visual.blocks.0.norm1.weight"])


def test_has_qwen_vl_keys_rejects_lm_only() -> None:
    """Text-only Qwen3/Qwen2 encoders have LM keys but no visual tower — must not match."""
    assert not _has_qwen_vl_keys(["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"])


def test_has_qwen_vl_keys_rejects_empty() -> None:
    assert not _has_qwen_vl_keys([])


def test_read_safetensors_keys_returns_keys_without_loading_tensors() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tiny.safetensors"
        _write_safetensors(path, ["model.embed_tokens.weight", "visual.patch_embed.proj.weight"])

        keys = _read_safetensors_keys(path)

        assert set(keys) == {"model.embed_tokens.weight", "visual.patch_embed.proj.weight"}


def test_checkpoint_config_matches_qwen_vl_safetensors() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "qwen_vl.safetensors"
        _write_safetensors(path, ["model.embed_tokens.weight", "visual.patch_embed.proj.weight"])

        mod = MagicMock()
        mod.path = path

        config = QwenVLEncoder_Checkpoint_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "qwen_vl_encoder"
        assert config.format.value == "checkpoint"


def test_checkpoint_config_rejects_lm_only_safetensors() -> None:
    """A text-only LM checkpoint must not be identified as a Qwen VL encoder."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "text_only.safetensors"
        _write_safetensors(path, ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"])

        mod = MagicMock()
        mod.path = path

        with pytest.raises(NotAMatchError, match="does not look like a Qwen2.5-VL"):
            QwenVLEncoder_Checkpoint_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_checkpoint_config_rejects_non_safetensors_extension() -> None:
    """Bin/ckpt/pt files should be rejected cheaply without attempting to read the header."""
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "weights.bin"
        path.write_bytes(b"not a safetensors file")

        mod = MagicMock()
        mod.path = path

        with pytest.raises(NotAMatchError, match="expected a .safetensors file"):
            QwenVLEncoder_Checkpoint_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
