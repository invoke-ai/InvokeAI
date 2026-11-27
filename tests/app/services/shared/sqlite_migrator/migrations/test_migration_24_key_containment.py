"""migration_24 moves each Invoke-managed model into `models_dir/<key>`, taking the key from the stored config blob.
Older versions did not check that key, so the move is guarded: a key that would leave `models_dir` on this platform
leaves the model where it is, and a key that is merely illegal on Windows is an ordinary directory name on posix."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_24 import Migration24Callback
from invokeai.app.util import path_safety


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    (models / "sd-1").mkdir(parents=True)
    (models / "sd-1" / "model.safetensors").write_bytes(b"weights")
    return models


def _callback(models_dir: Path) -> Migration24Callback:
    app_config = MagicMock()
    app_config.models_path = models_dir
    return Migration24Callback(app_config=app_config, logger=logging.getLogger("test_migration_24"))


def test_a_key_that_escapes_leaves_the_model_in_place(models_dir: Path):
    result = _callback(models_dir)._normalize_model_storage("../escaped", "sd-1/model.safetensors")

    assert result.new_relative_path is None
    assert (models_dir / "sd-1" / "model.safetensors").exists()
    assert not (models_dir.parent / "escaped").exists()


def test_a_windows_only_key_is_left_in_place_under_windows_rules(models_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(path_safety, "_ON_WINDOWS", True)

    result = _callback(models_dir)._normalize_model_storage("..:stream", "sd-1/model.safetensors")

    assert result.new_relative_path is None
    assert (models_dir / "sd-1" / "model.safetensors").exists()


@pytest.mark.skipif(os.name == "nt", reason="Legacy colon keys could only have been stored on POSIX filesystems")
def test_a_legacy_posix_key_is_normalized_like_any_other(models_dir: Path):
    result = _callback(models_dir)._normalize_model_storage("legacy:key", "sd-1/model.safetensors")

    assert result.new_relative_path == "legacy:key/model.safetensors"
    assert (models_dir / "legacy:key" / "model.safetensors").exists()
