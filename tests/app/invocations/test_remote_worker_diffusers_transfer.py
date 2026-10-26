"""Safety and integrity checks for directory-based remote-worker model transfer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from invokeai.app.invocations.remote_worker.diffusers_transfer import (
    DirectoryFile,
    _safe_path,
    _validate_transferable_model_files,
    _validated_manifest,
    inventory_directory,
)
from invokeai.app.invocations.remote_worker.model_transfer import ModelTransferError


def test_rejects_unsafe_relative_paths() -> None:
    for path in (
        "../other",
        "/absolute",
        "a//b",
        "a/./b",
        "a/../b",
        "C:/model",
        "a\\b",
        "a/CON:",
    ):
        with pytest.raises(ValueError):
            _safe_path(path)


def test_directory_inventory_preserves_subfolders_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "model"
    (root / "unet").mkdir(parents=True)
    (root / "model_index.json").write_bytes(b"{}")
    (root / "unet" / "weights.safetensors").write_bytes(b"weights")

    files = inventory_directory(root)
    assert {file.path for file in files} == {"model_index.json", "unet/weights.safetensors"}
    weights = next(file for file in files if file.path == "unet/weights.safetensors")
    assert weights.size == 7
    assert weights.sha256 == hashlib.sha256(b"weights").hexdigest()


def test_directory_inventory_can_cancel_during_large_file_hashing(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "weights.safetensors").write_bytes(b"x" * (2 * 1024 * 1024 + 1))

    calls = 0

    def should_cancel() -> bool:
        nonlocal calls
        calls += 1
        # One directory-walk check, one per-file check, then the first hash
        # chunk is allowed. Cancel before hashing the second chunk.
        return calls >= 4

    with pytest.raises(ModelTransferError, match="preparation cancelled"):
        inventory_directory(root, should_cancel)


def test_directory_inventory_refuses_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "model_index.json").write_bytes(b"{}")
    link = root / "outside"
    try:
        link.symlink_to(root / "model_index.json")
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation unavailable")
    with pytest.raises(ModelTransferError):
        inventory_directory(root)


def test_receiver_rejects_manifest_path_traversal() -> None:
    for path in (
        "../outside",
        "a/../../outside",
        "/absolute",
        "C:/evil",
        "a\\b",
        "a//b",
    ):
        manifest = {
            "base_url": "http://127.0.0.1:9999/token",
            "name": "Test",
            "model_hash": "model-hash",
            "files": [{"path": path, "size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}],
        }
        with pytest.raises(ValueError):
            _validated_manifest(manifest)


@pytest.mark.parametrize(
    "paths",
    [
        ["model_index.json", "unet/diffusion_pytorch_model.safetensors"],
        ["text_encoder_2/model.safetensors", "tokenizer_2/tokenizer.json"],
        ["text_encoder_2/config.json", "tokenizer_2/tokenizer.json"],
        ["encoder/weights.gguf"],
        ["subfolder/modular_model_index.json"],
    ],
)
def test_accepts_pipeline_or_standalone_component(paths: list[str]) -> None:
    _validate_transferable_model_files([DirectoryFile(path, 123, "ignored") for path in paths])


@pytest.mark.parametrize(
    "files",
    [
        [DirectoryFile("tokenizer_2/tokenizer.json", 100, "ignored")],
        [DirectoryFile("tokenizer_2/tokenizer_config.json", 100, "ignored")],
        [DirectoryFile("text_encoder_2/model.safetensors", 0, "ignored")],
        [DirectoryFile("text_encoder_2/config.json", 0, "ignored")],
    ],
)
def test_rejects_incomplete_local_model(files: list[DirectoryFile]) -> None:
    with pytest.raises(ModelTransferError, match="Repair or reinstall"):
        _validate_transferable_model_files(files)
