"""Tests for the WanT5Encoder config probe (UMT5-XXL diffusers folder)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.wan_t5_encoder import WanT5Encoder_WanT5Encoder_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


def _build_overrides(model_path: Path, name: str) -> dict:
    return {
        "hash": "test-hash",
        "path": str(model_path),
        "file_size": 0,
        "name": name,
        "source": str(model_path),
        "source_type": "path",
    }


def _make_mod(model_path: Path) -> MagicMock:
    mod = MagicMock()
    mod.path = model_path
    return mod


def _write_encoder_config(target: Path, model_type: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        json.dump({"model_type": model_type, "architectures": ["UMT5EncoderModel"]}, f)


class TestWanT5EncoderProbe:
    def test_accepts_nested_text_encoder_layout(self):
        """Standard layout: <root>/text_encoder/config.json with model_type=umt5."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "wan-encoder-bundle"
            root.mkdir()
            _write_encoder_config(root / "text_encoder" / "config.json", "umt5")

            cfg = WanT5Encoder_WanT5Encoder_Config.from_model_on_disk(
                _make_mod(root), _build_overrides(root, "wan-encoder")
            )

            assert cfg.base == BaseModelType.Any
            assert cfg.type == ModelType.WanT5Encoder
            assert cfg.format == ModelFormat.WanT5Encoder

    def test_accepts_flat_encoder_layout(self):
        """Flat layout: <root>/config.json directly (just the encoder folder)."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "umt5-xxl"
            root.mkdir()
            _write_encoder_config(root / "config.json", "umt5")

            cfg = WanT5Encoder_WanT5Encoder_Config.from_model_on_disk(
                _make_mod(root), _build_overrides(root, "umt5-xxl")
            )
            assert cfg.format == ModelFormat.WanT5Encoder

    def test_rejects_t5(self):
        """A regular T5-XXL encoder must not match (different vocabulary)."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "t5-xxl"
            root.mkdir()
            _write_encoder_config(root / "config.json", "t5")

            with pytest.raises(NotAMatchError, match="not 'umt5'"):
                WanT5Encoder_WanT5Encoder_Config.from_model_on_disk(
                    _make_mod(root), _build_overrides(root, "t5-xxl")
                )

    def test_rejects_full_pipeline(self):
        """A folder with model_index.json or transformer/ is a full pipeline, not an encoder."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "full-pipeline"
            root.mkdir()
            _write_encoder_config(root / "text_encoder" / "config.json", "umt5")
            (root / "model_index.json").touch()

            with pytest.raises(NotAMatchError, match="full Wan pipeline"):
                WanT5Encoder_WanT5Encoder_Config.from_model_on_disk(
                    _make_mod(root), _build_overrides(root, "full-pipeline")
                )

    def test_rejects_missing_config(self):
        """Empty directory has no encoder config to read."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty"
            root.mkdir()

            with pytest.raises(NotAMatchError, match="no encoder config"):
                WanT5Encoder_WanT5Encoder_Config.from_model_on_disk(
                    _make_mod(root), _build_overrides(root, "empty")
                )
