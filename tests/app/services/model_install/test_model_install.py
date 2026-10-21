"""
Test the model installer
"""

import errno
import gc
import json
import os
import platform
import shutil
import sqlite3
import stat
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional
from unittest.mock import MagicMock

import pytest
from pydantic_core import Url
from requests import Request, Response, Timeout
from requests_testadapter import TestAdapter

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadJob, DownloadJobStatus, MultiFileDownloadJob
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.events.events_common import (
    ModelInstallCompleteEvent,
    ModelInstallDownloadProgressEvent,
    ModelInstallDownloadsCompleteEvent,
    ModelInstallDownloadStartedEvent,
    ModelInstallErrorEvent,
    ModelInstallStartedEvent,
)
from invokeai.app.services.model_install import (
    HFModelSource,
    ModelInstallService,
    ModelInstallServiceBase,
    model_install_default,
)
from invokeai.app.services.model_install.install_staging_ledger import (
    INSTALL_ACTIVE_SENTINEL,
    INSTALL_MARKER_FILENAME,
    INSTALL_MARKER_VERSION,
    INSTALL_RECOVERY_SENTINEL,
    TMPDIR_PREFIX,
    HeldClaim,
    InstallStaging,
    InstallStagingLedger,
)
from invokeai.app.services.model_install.model_install_common import (
    InstallCancellationConflictError,
    InstallDownloadConflictError,
    InstallRecoveryRequiredError,
    InstallStatus,
    InvalidModelConfigException,
    LocalModelSource,
    ModelInstallJob,
    URLModelSource,
)
from invokeai.app.services.model_records import ModelRecordChanges, ModelRecordServiceSQL, UnknownModelException
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig
from invokeai.backend.model_manager.metadata import RemoteModelFile
from invokeai.backend.model_manager.metadata.metadata_base import HF_METADATA_REQUEST_TIMEOUT_SECONDS
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelSourceType,
    ModelType,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403
from tests.test_nodes import TestEventService

OS = platform.uname().system


def _claim_path(root: Path) -> Path:
    return root.parent / f".{root.name}{INSTALL_ACTIVE_SENTINEL}"


def _recovery_path(root: Path) -> Path:
    return root.parent / f".{root.name}{INSTALL_RECOVERY_SENTINEL}"


def _read_marker(tmpdir: Path) -> Optional[dict[str, Any]]:
    path = tmpdir / INSTALL_MARKER_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _stage(job: ModelInstallJob, tmpdir: Optional[Path], models_path: Optional[Path] = None) -> InstallStaging:
    """Give a job built by the test the staging handle the installer creates with every job."""
    root = models_path or (tmpdir.parent if tmpdir is not None else None)
    assert root is not None
    job._staging = InstallStaging(InstallStagingLedger(root, InvokeAILogger.get_logger()), tmpdir)
    return job._staging


def _write_v1_marker(job: ModelInstallJob, status: InstallStatus) -> None:
    """Write the job's staging marker in format version 1, as any release since that format writes it."""
    assert job._staging is not None and job._staging.tmpdir is not None
    marker: dict[str, Any] = {
        "version": 1,
        "source": str(job.source),
        "access_token": getattr(job.source, "access_token", None),
        "config_in": job.config_in.model_dump(),
        "status": status.value,
        "updated_at": "2026-01-01T00:00:00",
        "files": list((job._resume_metadata or {}).values()),
    }
    if job._recovery_required:
        marker["recovery_required"] = True
    (job._staging.tmpdir / ".invokeai_install.json").write_text(json.dumps(marker), encoding="utf-8")


def _startup_cleanup(models_path: Path) -> None:
    InstallStagingLedger(models_path, InvokeAILogger.get_logger()).remove_dangling_staging([])


def _fake_move(monkeypatch: pytest.MonkeyPatch, move: Callable[[Path, Path], object]) -> None:
    """Install a move that fails on purpose. Its failures are scripted, so the retry backoff, which gives a real
    Windows file handle time to close, would only add sleep."""
    monkeypatch.setattr(model_install_default, "move", move)
    monkeypatch.setattr(model_install_default, "_MOVE_RETRY_DELAY_SECONDS", 0.0)


def test_registration(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    matches = store.search_by_attr(model_name="test_embedding")
    assert len(matches) == 0
    key = mm2_installer.register_path(embedding_file)
    # Not raising here is sufficient - key should be UUIDv4
    uuid.UUID(key, version=4)


def test_registration_meta(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record is not None
    assert model_record.name == "test_embedding"
    assert model_record.type == ModelType.TextualInversion
    assert Path(model_record.path) == embedding_file
    assert Path(model_record.path).exists()
    assert model_record.base == BaseModelType("sd-1")
    assert model_record.description is None
    assert model_record.source is not None
    assert Path(model_record.source) == embedding_file


def test_registration_meta_override_fail(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    with pytest.raises(InvalidModelConfigException):
        mm2_installer.register_path(embedding_file, ModelRecordChanges(name="banana_sushi", type=ModelType("lora")))


def test_registration_meta_override_succeed(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(
        embedding_file, ModelRecordChanges(name="banana_sushi", source="fake/repo_id", key="xyzzy")
    )
    model_record = store.get_model(key)
    assert model_record.name == "banana_sushi"
    assert model_record.source == "fake/repo_id"
    assert model_record.key == "xyzzy"


@pytest.mark.parametrize(("setting", "value"), [("fp8_storage", True), ("steps", 37)])
def test_registration_keeps_the_default_settings_sent_with_the_install(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, setting: str, value: bool | int
) -> None:
    """Identification computes a model's default settings; the ones a user picked while installing must survive it."""
    import torch
    from safetensors.torch import save_file

    checkpoint = tmp_path / "qwen_image.safetensors"
    save_file(
        {"img_in.weight": torch.zeros(8, 4), "txt_in.weight": torch.zeros(8, 4), "txt_norm.weight": torch.ones(4)},
        str(checkpoint),
    )

    key = mm2_installer.register_path(checkpoint, ModelRecordChanges(default_settings={setting: value}))

    record = mm2_installer.record_store.get_model(key)
    assert record.default_settings is not None
    assert getattr(record.default_settings, setting) == value


def test_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record.path.endswith(f"{key}/test_embedding.safetensors")
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert model_record.source == embedding_file.as_posix()


def test_file_install_retries_copy_then_unlink_permission_error(
    mm2_installer: ModelInstallServiceBase,
    embedding_file: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows can finish shutil.move's copy before failing to unlink its source."""
    original_bytes = embedding_file.read_bytes()
    real_move = shutil.move
    calls = 0

    def copy_then_fail_unlink(src: Path, dst: Path):
        nonlocal calls
        calls += 1
        if calls == 1:
            shutil.copy2(src, dst)
            raise PermissionError("simulated Windows source unlink failure")
        return real_move(src, dst)

    _fake_move(monkeypatch, copy_then_fail_unlink)

    key = mm2_installer.install_path(embedding_file)

    installed = mm2_installer.record_store.get_model(key)
    installed_path = mm2_app_config.models_path / installed.path
    assert installed_path.read_bytes() == original_bytes
    assert not embedding_file.exists()
    assert calls == 1


def test_destination_is_hidden_from_orphan_cleanup_during_install(
    mm2_installer: ModelInstallServiceBase,
    embedding_file: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invokeai.app.services.orphaned_models import OrphanedModelsService

    orphan_service = OrphanedModelsService(config=mm2_app_config, store=mm2_installer.record_store)
    observations: list[tuple[str, bool, set[str]]] = []
    real_move = shutil.move

    def move_then_scan(src: Path, dst: Path):
        result = real_move(src, dst)
        destination_root = dst.parent
        assert _claim_path(destination_root).exists()
        orphan_keys = {orphan.path for orphan in orphan_service.find_orphaned_models()}
        observations.append((destination_root.name, _recovery_path(destination_root).exists(), orphan_keys))
        return result

    monkeypatch.setattr(model_install_default, "move", move_then_scan)

    mm2_installer.install_path(embedding_file)

    assert len(observations) == 1
    destination_key, has_sentinel, orphan_keys = observations[0]
    assert has_sentinel
    assert destination_key not in orphan_keys
    assert not _recovery_path(mm2_app_config.models_path / destination_key).exists()


def test_managed_local_source_is_hidden_from_orphan_cleanup_during_install(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invokeai.app.services.orphaned_models import OrphanedModelsService

    source_root = mm2_app_config.models_path / f"local-source-{uuid.uuid4().hex}"
    shutil.copytree(diffusers_dir, source_root)
    orphan_service = OrphanedModelsService(config=mm2_app_config, store=mm2_installer.record_store)
    observations: list[tuple[bool, set[str], str]] = []
    real_probe = mm2_installer._probe
    real_move_with_retries = mm2_installer._move_with_retries

    def observe_source_during_operation(path: Path, config: ModelRecordChanges):
        assert path == source_root
        protected = _claim_path(source_root).exists()
        orphan_keys = {orphan.path for orphan in orphan_service.find_orphaned_models()}
        delete_result = orphan_service.delete_orphaned_models([source_root.name])[source_root.name]
        observations.append((protected, orphan_keys, delete_result))
        return real_probe(path, config)

    def observe_source_during_transfer(src: Path, dst: Path) -> None:
        assert _claim_path(source_root).exists()
        real_move_with_retries(src, dst)

    monkeypatch.setattr(mm2_installer, "_probe", observe_source_during_operation)
    monkeypatch.setattr(mm2_installer, "_move_with_retries", observe_source_during_transfer)

    job = mm2_installer.import_model(LocalModelSource(path=source_root, inplace=False))
    assert job is not None
    mm2_installer.wait_for_installs()

    assert observations
    assert observations[0][0]
    assert source_root.name not in observations[0][1]
    assert observations[0][2] == "error: path is reserved by an active install or install recovery"
    assert job.complete
    assert job.config_out is not None
    assert mm2_installer.record_store.get_model(job.config_out.key)
    assert not _claim_path(source_root).exists()
    assert not _recovery_path(source_root).exists()


def test_orphan_delete_claim_prevents_a_local_install_from_starting(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invokeai.app.services.orphaned_models import OrphanedModelsService, orphaned_models_service

    source_root = mm2_app_config.models_path / f"local-source-{uuid.uuid4().hex}"
    shutil.copytree(diffusers_dir, source_root)
    orphan_service = OrphanedModelsService(config=mm2_app_config, store=mm2_installer.record_store)
    real_rmtree = orphaned_models_service.shutil.rmtree
    deletion_claimed = threading.Event()
    release_deletion = threading.Event()
    results: dict[str, str] = {}

    def blocked_rmtree(path: Path) -> None:
        deletion_claimed.set()
        assert release_deletion.wait(timeout=5)
        real_rmtree(path)

    monkeypatch.setattr(orphaned_models_service.shutil, "rmtree", blocked_rmtree)
    delete_thread = threading.Thread(
        target=lambda: results.update(orphan_service.delete_orphaned_models([source_root.name]))
    )
    delete_thread.start()
    try:
        assert deletion_claimed.wait(timeout=5)
        assert _claim_path(source_root).exists()
        with pytest.raises(InstallCancellationConflictError, match="another install or orphan cleanup"):
            mm2_installer.install_path(source_root)
        assert source_root.exists()
    finally:
        release_deletion.set()
        delete_thread.join(timeout=10)

    assert not delete_thread.is_alive()
    assert results[source_root.name] == "deleted"
    assert not source_root.exists()


def test_unregistered_destination_remains_protected_after_registration_failure(
    mm2_installer: ModelInstallServiceBase,
    embedding_file: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invokeai.app.services.orphaned_models import OrphanedModelsService

    orphan_service = OrphanedModelsService(config=mm2_app_config, store=mm2_installer.record_store)

    def fail_recording(_config: Any) -> None:
        raise RuntimeError("simulated model record failure")

    monkeypatch.setattr(mm2_installer.record_store, "add_model", fail_recording)

    with pytest.raises(RuntimeError, match="simulated model record failure"):
        mm2_installer.install_path(embedding_file)

    protected_roots = [
        path for path in mm2_app_config.models_path.iterdir() if path.is_dir() and _recovery_path(path).exists()
    ]
    assert len(protected_roots) == 1
    recovery_root = protected_roots[0]
    assert list(recovery_root.iterdir())
    assert recovery_root.name not in {orphan.path for orphan in orphan_service.find_orphaned_models()}


@pytest.mark.parametrize("is_directory", [False, True])
def test_rollback_restores_source_across_filesystems_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_directory: bool
) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "models"
    source_root.mkdir()
    destination_root.mkdir()
    original = source_root / "model"
    if is_directory:
        original.mkdir()
        (original / "weights.safetensors").write_bytes(b"model weights")
    else:
        original.write_bytes(b"model weights")
    moved = destination_root / "model"
    shutil.move(original, moved)
    real_rename_noreplace = ModelInstallService._rename_noreplace

    def raise_exdev_for_original(src: Path, dst: Path) -> None:
        if src == moved:
            raise OSError(errno.EXDEV, "cross-device link")
        real_rename_noreplace(src, dst)

    monkeypatch.setattr(ModelInstallService, "_rename_noreplace", staticmethod(raise_exdev_for_original))

    ModelInstallService._restore_moved_path(moved, original)

    assert original.exists()
    assert not moved.exists()
    if is_directory:
        assert (original / "weights.safetensors").read_bytes() == b"model weights"
    else:
        assert original.read_bytes() == b"model weights"


def test_startup_cleanup_preserves_interrupted_exdev_restore_artifacts(
    tmp_path: Path,
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
) -> None:
    """Restore scratch has no durable owner record, so startup must leave it for manual inspection."""
    external_source_root = tmp_path / "external-source"
    managed_source_root = mm2_app_config.models_path / "local-source"
    managed_copy_root = mm2_app_config.models_path / "managed-copy"
    external_source_root.mkdir()
    managed_source_root.mkdir()
    managed_copy_root.mkdir()

    artifacts: list[tuple[Path, bytes]] = []
    for source_root in (external_source_root, managed_source_root):
        source = source_root / "model"
        source.write_bytes(b"original source")

        # A hard stop during copy can leave partial bytes, or a complete copy before publication.
        for name, content in (("partial", b"partial"), ("complete", b"complete copy"), ("lookalike", b"unknown owner")):
            scratch = source_root / f".model.restore-{name}-{uuid.uuid4().hex}"
            staged_copy = scratch / "restored"
            staged_copy.parent.mkdir()
            staged_copy.write_bytes(content)
            artifacts.append((staged_copy, content))

        # Publication precedes deletion of the managed copy. A crash can therefore leave both copies.
        managed_copy = managed_copy_root / source_root.name
        managed_copy.write_bytes(b"published copy")
        published = source_root / "published-model"
        published.write_bytes(b"published copy")
        artifacts.extend(((managed_copy, b"published copy"), (published, b"published copy")))

    _startup_cleanup(mm2_app_config.models_path)

    assert all(path.read_bytes() == expected for path, expected in artifacts)


def test_startup_cleanup_preserves_live_exdev_restore_lookalike(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = mm2_app_config.models_path / "local-source"
    managed_copy_root = mm2_app_config.models_path / "managed-copy"
    source_root.mkdir()
    managed_copy_root.mkdir()
    original = source_root / "model"
    moved = managed_copy_root / "model"
    moved.write_bytes(b"managed copy")

    copy_started = threading.Event()
    finish_copy = threading.Event()
    real_copy2 = shutil.copy2
    real_rename_noreplace = ModelInstallService._rename_noreplace

    def pause_copy(src: Path, dst: Path) -> Path:
        real_copy2(src, dst)
        copy_started.set()
        assert finish_copy.wait(timeout=10)
        return dst

    def raise_exdev_for_managed_copy(src: Path, dst: Path) -> None:
        if src == moved:
            raise OSError(errno.EXDEV, "cross-device link")
        real_rename_noreplace(src, dst)

    monkeypatch.setattr(model_install_default, "copy2", pause_copy)
    monkeypatch.setattr(ModelInstallService, "_rename_noreplace", staticmethod(raise_exdev_for_managed_copy))
    errors: list[Exception] = []

    def restore() -> None:
        try:
            ModelInstallService._restore_moved_path(moved, original)
        except Exception as error:
            errors.append(error)

    restore_thread = threading.Thread(target=restore)
    restore_thread.start()
    try:
        assert copy_started.wait(timeout=10)
        staging_dirs = list(source_root.glob(".model.restore-*"))
        assert len(staging_dirs) == 1
        staged_copy = staging_dirs[0] / "restored"

        _startup_cleanup(mm2_app_config.models_path)

        assert staged_copy.read_bytes() == b"managed copy"
        assert moved.read_bytes() == b"managed copy"
    finally:
        finish_copy.set()
        restore_thread.join(timeout=10)

    assert not restore_thread.is_alive()
    assert errors == []
    assert original.read_bytes() == b"managed copy"
    assert not moved.exists()


def test_directory_install_retries_windows_move_failures(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient handle failure must not strand a partially moved directory install."""
    expected_files = {
        path.relative_to(diffusers_dir): path.read_bytes() for path in diffusers_dir.rglob("*") if path.is_file()
    }
    real_move = shutil.move
    calls = 0

    def flaky_move(src: Path, dst: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("simulated Windows file-handle race")
        return real_move(src, dst)

    _fake_move(monkeypatch, flaky_move)

    key = mm2_installer.install_path(diffusers_dir)

    model_record = mm2_installer.record_store.get_model(key)
    installed_path = mm2_app_config.models_path / model_record.path
    installed_files = {
        path.relative_to(installed_path): path.read_bytes() for path in installed_path.rglob("*") if path.is_file()
    }
    assert installed_files == expected_files
    assert calls >= 3


def test_directory_install_failure_does_not_leave_partial_destination(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_paths = set(mm2_app_config.models_path.iterdir())

    def fail_first_move(_src: Path, _dst: Path):
        raise PermissionError("simulated Windows file-handle race")

    _fake_move(monkeypatch, fail_first_move)

    with pytest.raises(PermissionError, match="simulated Windows file-handle race"):
        mm2_installer.install_path(diffusers_dir)

    assert set(mm2_app_config.models_path.iterdir()) == existing_paths
    assert list(diffusers_dir.iterdir())
    assert mm2_installer.record_store.all_models() == []


def test_directory_install_late_move_failure_restores_source_files(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A locked later file must not strand files already removed from a user's import folder."""
    expected_files = {
        path.relative_to(diffusers_dir): path.read_bytes() for path in diffusers_dir.rglob("*") if path.is_file()
    }
    existing_paths = set(mm2_app_config.models_path.iterdir())
    first_item, locked_item, *_ = list(diffusers_dir.iterdir())
    real_move = shutil.move
    moved_first_item = False

    def move_with_locked_source(src: Path, dst: Path):
        nonlocal moved_first_item
        if src == locked_item:
            assert moved_first_item
            raise PermissionError("source file remains locked")
        result = real_move(src, dst)
        if src == first_item:
            moved_first_item = True
        return result

    _fake_move(monkeypatch, move_with_locked_source)

    with pytest.raises(PermissionError, match="source file remains locked"):
        mm2_installer.install_path(diffusers_dir)

    assert moved_first_item
    assert {
        path.relative_to(diffusers_dir): path.read_bytes() for path in diffusers_dir.rglob("*") if path.is_file()
    } == expected_files
    assert set(mm2_app_config.models_path.iterdir()) == existing_paths
    assert mm2_installer.record_store.all_models() == []


def test_directory_install_preserves_collision_during_rollback(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_paths = set(mm2_app_config.models_path.iterdir())
    first_item, locked_item, *_ = list(diffusers_dir.iterdir())
    first_item_is_dir = first_item.is_dir()
    real_move = shutil.move
    moved_first_item = False

    def move_with_recreated_source(src: Path, dst: Path):
        nonlocal moved_first_item
        if src == locked_item:
            recreated = diffusers_dir / first_item.name
            if first_item_is_dir:
                recreated.mkdir()
                (recreated / "new-user-file").write_text("new user file")
            else:
                recreated.write_text("new user file")
            raise PermissionError("later source remains locked")
        if src == first_item:
            if dst.exists():
                raise FileExistsError(dst)
            result = real_move(src, dst)
            moved_first_item = True
            return result
        if dst == first_item and dst.exists():
            raise FileExistsError(dst)
        return real_move(src, dst)

    _fake_move(monkeypatch, move_with_recreated_source)

    with pytest.raises(RuntimeError, match="recovery required") as exc_info:
        mm2_installer.install_path(diffusers_dir)

    assert moved_first_item
    assert "test-diffusers-main" in str(exc_info.value)
    recreated = diffusers_dir / first_item.name
    assert recreated.exists()
    if first_item_is_dir:
        assert (recreated / "new-user-file").read_text() == "new user file"
    else:
        assert recreated.read_text() == "new user file"
    assert str(mm2_app_config.models_path.resolve()) in str(exc_info.value)
    recovery_paths = [path for path in set(mm2_app_config.models_path.iterdir()) - existing_paths if path.is_dir()]
    assert len(recovery_paths) == 1
    recovery_dest = recovery_paths[0]
    preserved_item = recovery_dest / first_item.name
    assert preserved_item.exists()
    assert preserved_item.is_dir() == first_item_is_dir
    assert _recovery_path(recovery_dest).exists()


def test_queued_local_import_rejects_cancel_after_partial_transfer_requires_recovery(
    mm2_installer: ModelInstallServiceBase,
    embedding_file: Path,
    mm2_app_config: InvokeAIAppConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    source_path = tmp_path / embedding_file.name
    shutil.copy2(embedding_file, source_path)
    original_source_bytes = source_path.read_bytes()
    partial_destination_bytes = original_source_bytes[: max(1, len(original_source_bytes) // 2)]
    existing_model_paths = set(mm2_app_config.models_path.iterdir())

    def partial_copy_then_fail(src: Path, dst: Path) -> None:
        assert src == source_path
        dst.write_bytes(partial_destination_bytes)
        raise OSError(errno.ENOSPC, "simulated disk full")

    _fake_move(monkeypatch, partial_copy_then_fail)

    job = installer.import_model(LocalModelSource(path=source_path, inplace=False))
    installer.wait_for_job(job, timeout=10)

    assert job.errored
    assert job.error_type == "InstallRecoveryRequiredError"
    recovery_destinations = [
        path
        for path in set(mm2_app_config.models_path.iterdir()) - existing_model_paths
        if path.is_dir() and _recovery_path(path).exists()
    ]
    assert len(recovery_destinations) == 1
    recovery_file = recovery_destinations[0] / source_path.name
    assert recovery_file.read_bytes() == partial_destination_bytes
    assert source_path.read_bytes() == original_source_bytes

    with pytest.raises(InstallRecoveryRequiredError, match="requires recovery"):
        installer.cancel_job(job)

    assert job.status is InstallStatus.ERROR
    assert job.errored
    assert not job.cancelled
    assert _recovery_path(recovery_destinations[0]).exists()
    assert recovery_file.read_bytes() == partial_destination_bytes
    assert source_path.read_bytes() == original_source_bytes


def test_directory_install_rollback_race_preserves_both_source_artifacts(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_iterdir = Path.iterdir

    def files_first(path: Path):
        items = real_iterdir(path)
        if path == diffusers_dir:
            # Move a file before the locked directory, regardless of filesystem enumeration order.
            return iter(sorted(items, key=lambda item: (not item.is_file(), item.name)))
        return items

    monkeypatch.setattr(Path, "iterdir", files_first)
    items = list(diffusers_dir.iterdir())
    first_item = next(path for path in items if path.is_file())
    locked_item = items[-1]
    original_bytes = first_item.read_bytes()
    real_move = shutil.move
    real_rename_noreplace = ModelInstallService._rename_noreplace

    def fail_later_move(src: Path, dst: Path):
        if src == locked_item:
            raise PermissionError("later source remains locked")
        return real_move(src, dst)

    def recreate_after_absence_check(src: Path, dst: Path) -> None:
        if dst == first_item:
            dst.write_bytes(b"new user bytes")
        real_rename_noreplace(src, dst)

    _fake_move(monkeypatch, fail_later_move)
    monkeypatch.setattr(ModelInstallService, "_rename_noreplace", staticmethod(recreate_after_absence_check))

    with pytest.raises(model_install_default.InstallRecoveryRequiredError, match="recovery required"):
        mm2_installer.install_path(diffusers_dir)

    assert first_item.read_bytes() == b"new user bytes"
    recovery_dirs = [path for path in mm2_app_config.models_path.iterdir() if path.is_dir() and path.name != "tmp"]
    preserved = [path / first_item.name for path in recovery_dirs if (path / first_item.name).exists()]
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == original_bytes


@pytest.mark.parametrize(
    "unsupported_errno",
    [None, errno.ENOSYS, errno.EOPNOTSUPP],
    ids=["primitive-unavailable", "not-implemented", "operation-unsupported"],
)
def test_directory_install_preserves_recreated_source_when_no_replace_rename_fails(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
    unsupported_errno: int | None,
) -> None:
    real_iterdir = Path.iterdir

    def files_first(path: Path):
        items = real_iterdir(path)
        if path == diffusers_dir:
            return iter(sorted(items, key=lambda item: (not item.is_file(), item.name)))
        return items

    monkeypatch.setattr(Path, "iterdir", files_first)
    first_item = next(path for path in diffusers_dir.iterdir() if path.is_file())
    locked_item = list(diffusers_dir.iterdir())[-1]
    original_bytes = first_item.read_bytes()
    existing_paths = set(mm2_app_config.models_path.iterdir())

    def fail_later_move(src: Path, dst: Path):
        if src == locked_item:
            raise PermissionError("later source remains locked")
        return shutil.move(src, dst)

    def fail_no_replace_after_source_recreation(src: Path, dst: Path) -> None:
        assert src.name == first_item.name
        assert src.parent != diffusers_dir
        assert dst == first_item
        dst.write_bytes(b"concurrently recreated source")
        if unsupported_errno is None:
            raise OSError("atomic no-replace rename is unavailable")
        raise OSError(unsupported_errno, "atomic no-replace rename unsupported", str(dst))

    _fake_move(monkeypatch, fail_later_move)
    monkeypatch.setattr(
        ModelInstallService,
        "_rename_noreplace",
        staticmethod(fail_no_replace_after_source_recreation),
    )

    with pytest.raises(InstallRecoveryRequiredError, match="recovery required") as exc_info:
        mm2_installer.install_path(diffusers_dir)

    assert "atomic no-replace rename" in str(exc_info.value)
    assert first_item.read_bytes() == b"concurrently recreated source"
    recovery_dirs = [path for path in set(mm2_app_config.models_path.iterdir()) - existing_paths if path.is_dir()]
    assert len(recovery_dirs) == 1
    recovery_destination = recovery_dirs[0]
    assert _recovery_path(recovery_destination).exists()
    assert (recovery_destination / first_item.name).read_bytes() == original_bytes


def test_directory_install_does_not_retry_partial_copy(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_item = next(diffusers_dir.iterdir())
    calls = 0

    def partial_copy_then_fail(src: Path, dst: Path):
        nonlocal calls
        calls += 1
        if src == first_item:
            dst.mkdir(parents=True)
            (dst / "unknown-partial").write_text("partial")
            raise PermissionError("copy failed after destination creation")
        return shutil.move(src, dst)

    _fake_move(monkeypatch, partial_copy_then_fail)

    with pytest.raises(RuntimeError, match="recovery required") as exc_info:
        mm2_installer.install_path(diffusers_dir)

    assert calls == 1
    assert first_item.exists()
    assert "recovery required" in str(exc_info.value)


def test_directory_install_retains_unowned_destination_artifact_after_rollback(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_paths = set(mm2_app_config.models_path.iterdir())
    _, locked_item, *_ = list(diffusers_dir.iterdir())

    def fail_after_unowned_destination_artifact(src: Path, dst: Path):
        if src == locked_item:
            (dst.parent / "unowned-artifact").write_text("created concurrently")
            raise PermissionError("later source remains locked")
        return shutil.move(src, dst)

    _fake_move(monkeypatch, fail_after_unowned_destination_artifact)

    with pytest.raises(RuntimeError, match="recovery required") as exc_info:
        mm2_installer.install_path(diffusers_dir)

    recovery_paths = [path for path in set(mm2_app_config.models_path.iterdir()) - existing_paths if path.is_dir()]
    assert len(recovery_paths) == 1
    recovery_dest = recovery_paths[0]
    assert _recovery_path(recovery_dest).exists()
    assert (recovery_dest / "unowned-artifact").read_text() == "created concurrently"
    assert str(diffusers_dir.resolve()) in str(exc_info.value)
    assert str(recovery_dest.resolve()) in str(exc_info.value)
    assert locked_item.exists()


def test_remote_install_recovery_survives_cleanup_and_restart(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)
    existing_paths = set(mm2_app_config.models_path.iterdir())
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}recovery-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    shutil.copytree(diffusers_dir, tmpdir, dirs_exist_ok=True)
    failure_file = tmpdir / "zz_partial_failure.txt"
    failure_file.write_bytes(b"source bytes")
    source_files = {path.name: path.read_bytes() for path in tmpdir.iterdir() if path.is_file()}
    source = URLModelSource(url=Url("https://example.com/model.bin"))
    job = ModelInstallJob(
        id=991,
        source=source,
        config_in=ModelRecordChanges(key="tmpinstall_recovery"),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADS_DONE,
    )
    _stage(job, tmpdir)
    _write_v1_marker(job, InstallStatus.DOWNLOADS_DONE)

    def fail_after_partial_transfer(src: Path, dst: Path):
        if src == failure_file:
            dst.write_bytes(b"unrecognized partial destination bytes")
            raise PermissionError("simulated transfer failure after partial destination creation")
        return shutil.move(src, dst)

    real_iterdir = Path.iterdir

    def sorted_source_files(path: Path):
        entries = list(real_iterdir(path))
        return iter(sorted(entries, key=lambda entry: entry.name)) if path == tmpdir else iter(entries)

    monkeypatch.setattr(Path, "iterdir", sorted_source_files)
    _fake_move(monkeypatch, fail_after_partial_transfer)
    real_persist = InstallStaging.persist

    def fail_recovery_marker(staging: InstallStaging, **kwargs: Any) -> None:
        if kwargs["recovery_required"]:
            raise OSError("simulated marker write failure")
        real_persist(staging, **kwargs)

    real_delete_marker = InstallStaging.delete_marker

    def assert_recovery_state_before_marker_delete(staging: InstallStaging) -> None:
        if staging.tmpdir == tmpdir:
            assert _recovery_path(tmpdir).exists() is job._recovery_required
        real_delete_marker(staging)

    monkeypatch.setattr(InstallStaging, "persist", fail_recovery_marker)
    monkeypatch.setattr(InstallStaging, "delete_marker", assert_recovery_state_before_marker_delete)
    mm2_installer._install_jobs.append(job)
    mm2_installer._put_in_queue(job)
    mm2_installer.wait_for_job(job, timeout=10)

    assert job.errored
    assert job.error
    assert job in mm2_installer.list_jobs()
    assert _recovery_path(tmpdir).exists()
    assert INSTALL_RECOVERY_SENTINEL not in {path.name for path in real_iterdir(tmpdir)}
    assert {path.name: path.read_bytes() for path in real_iterdir(tmpdir) if path.is_file()} == source_files
    recovery_files = [
        path
        for path in set(mm2_app_config.models_path.iterdir()) - existing_paths
        if path != tmpdir and path.is_dir() and (path / failure_file.name).exists()
    ]
    assert len(recovery_files) == 1
    recovery_dir = recovery_files[0]
    assert recovery_dir.name == "tmpinstall_recovery"
    assert _recovery_path(recovery_dir).exists()
    assert (recovery_dir / failure_file.name).read_bytes() == b"unrecognized partial destination bytes"

    # Pruning removes terminal job diagnostics from the live service, but it must not remove the recovery roots.
    mm2_installer.prune_jobs()
    assert job not in mm2_installer.list_jobs()
    assert tmpdir.exists()
    assert recovery_dir.exists()

    dangling_tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}dangling-{uuid.uuid4().hex}"
    dangling_tmpdir.mkdir()
    (dangling_tmpdir / "ordinary.tmp").write_bytes(b"discard me")

    restarted_installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_installer.record_store,
        download_queue=mm2_download_queue,
        session=mm2_session,
    )
    restarted_installer.start()
    try:
        assert restarted_installer._wait_for_restore_complete(timeout=10)
        assert tmpdir.exists()
        assert {path.name: path.read_bytes() for path in real_iterdir(tmpdir) if path.is_file()} == source_files
        assert recovery_dir.exists()
        assert (recovery_dir / failure_file.name).read_bytes() == b"unrecognized partial destination bytes"
        assert dangling_tmpdir.exists() is False
        assert restarted_installer.list_jobs() == []
    finally:
        restarted_installer.stop()
    shutil.rmtree(tmpdir)


def test_cancel_recovery_required_install_preserves_recovery_data(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}recovery-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    downloaded_file = tmpdir / "model.safetensors"
    downloaded_file.write_bytes(b"recoverable source")
    job = ModelInstallJob(
        id=992,
        source=URLModelSource(url=Url("https://example.com/model.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=downloaded_file,
        status=InstallStatus.ERROR,
    )
    _stage(job, tmpdir)
    job._recovery_required = True
    _recovery_path(tmpdir).write_text("preserve", encoding="utf-8")

    with pytest.raises(InstallRecoveryRequiredError, match="requires recovery"):
        mm2_installer.cancel_job(job)

    assert not job.cancelled
    assert downloaded_file.read_bytes() == b"recoverable source"
    assert _recovery_path(tmpdir).exists()


def test_cancelling_leaves_staging_another_owner_claimed_untouched(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}claimed-elsewhere-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    partial = tmpdir / "model.safetensors.downloading"
    partial.write_bytes(b"another owner's download")
    job = ModelInstallJob(
        id=9930,
        source=URLModelSource(url=Url("https://example.com/model.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.PAUSED,
    )
    _stage(job, tmpdir)
    _write_v1_marker(job, InstallStatus.PAUSED)
    marker = (tmpdir / INSTALL_MARKER_FILENAME).read_text(encoding="utf-8")

    with HeldClaim(tmpdir):
        mm2_installer.cancel_job(job)

        assert job.cancelled
        assert (tmpdir / INSTALL_MARKER_FILENAME).read_text(encoding="utf-8") == marker
        assert partial.read_bytes() == b"another owner's download"
    shutil.rmtree(tmpdir)


def test_cancel_recovery_sentinel_protects_data_before_job_flag_is_set(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}recovery-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    downloaded_file = tmpdir / "model.safetensors"
    downloaded_file.write_bytes(b"recoverable source")
    job = ModelInstallJob(
        id=993,
        source=URLModelSource(url=Url("https://example.com/model.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=downloaded_file,
        status=InstallStatus.RUNNING,
    )
    _stage(job, tmpdir)
    _recovery_path(tmpdir).write_text("preserve", encoding="utf-8")

    with pytest.raises(InstallRecoveryRequiredError, match="requires recovery"):
        mm2_installer.cancel_job(job)

    assert not job.cancelled
    assert downloaded_file.read_bytes() == b"recoverable source"
    assert _recovery_path(tmpdir).exists()


def _queue_downloaded_job(
    installer: ModelInstallService,
    tmpdir: Path,
    job_id: int,
) -> ModelInstallJob:
    job = ModelInstallJob(
        id=job_id,
        source=URLModelSource(url=Url("https://example.com/model.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADS_DONE,
    )
    _stage(job, tmpdir)
    installer._install_jobs.append(job)
    installer._install_queue.put(job)
    return job


def _write_remote_install_marker(
    installer: ModelInstallService,
    tmpdir: Path,
    job_id: int,
    status: InstallStatus,
) -> Path:
    tmpdir.mkdir(parents=True, exist_ok=True)
    download_path = tmpdir / (
        "weights.safetensors" if status == InstallStatus.DOWNLOADS_DONE else "weights.safetensors.downloading"
    )
    download_path.write_bytes(b"downloaded model" if status == InstallStatus.DOWNLOADS_DONE else b"partial download")
    job = ModelInstallJob(
        id=job_id,
        source=URLModelSource(url=Url("https://example.com/weights.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=status,
    )
    _stage(job, tmpdir)
    _write_v1_marker(job, status)
    return download_path


def test_restore_metadata_timeout_preserves_partial_download_for_retry(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}restore-timeout-{uuid.uuid4().hex}"
    tmpdir.mkdir(parents=True)
    partial_path = tmpdir / "weights.safetensors.downloading"
    partial_path.write_bytes(b"partial model")
    source = HFModelSource(repo_id="stabilityai/sdxl-turbo", variant=ModelRepoVariant.Default)
    job = ModelInstallJob(
        id=9906,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADING,
    )
    _stage(job, tmpdir)
    _write_v1_marker(job, InstallStatus.DOWNLOADING)

    def raise_metadata_timeout(*args: Any, **kwargs: Any) -> None:
        raise Timeout("Hugging Face metadata request timed out")

    assert mm2_installer._session is not None
    monkeypatch.setattr(mm2_installer._session, "get", raise_metadata_timeout)

    mm2_installer._restore_incomplete_installs()

    restored_job = mm2_installer.get_job_by_source(source)[-1]
    assert restored_job.status == InstallStatus.PAUSED
    assert partial_path.read_bytes() == b"partial model"
    marker = _read_marker(tmpdir)
    assert marker is not None
    assert marker["status"] == InstallStatus.PAUSED.value
    assert _claim_path(tmpdir).exists()


@pytest.mark.parametrize("status_code", [429, 503])
def test_restore_model_info_http_error_preserves_partial_download_for_retry(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)
    installer = mm2_installer
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}restore-model-info-{uuid.uuid4().hex}"
    partial_path = tmpdir / "text_encoder" / "model.safetensors.downloading"
    partial_path.parent.mkdir(parents=True)
    partial_path.write_bytes(b"partial model")
    source = HFModelSource(repo_id="stabilityai/sdxl-turbo", variant=ModelRepoVariant.Default)
    weights_url = "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/text_encoder/model.safetensors"
    resume_metadata = {
        weights_url: {
            "url": weights_url,
            "canonical_url": weights_url,
            "etag": "retained-etag",
            "last_modified": "retained-last-modified",
            "expected_total_bytes": 12,
            "final_url": weights_url,
            "download_path": partial_path.as_posix(),
            "resume_required": True,
            "resume_message": "Resuming previous download.",
        }
    }
    job = ModelInstallJob(
        id=9907,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADING,
    )
    _stage(job, tmpdir)
    job._resume_metadata = resume_metadata
    _write_v1_marker(job, InstallStatus.DOWNLOADING)

    api_url = "https://huggingface.co/api/models/stabilityai/sdxl-turbo"
    api_body = b'{"id":"stabilityai/sdxl-turbo","siblings":[{"rfilename":"text_encoder/model.safetensors","size":12}]}'
    real_get = installer._session.get
    api_attempts = 0

    def _response(url: str, code: int, body: bytes = b"") -> Response:
        response = Response()
        response.status_code = code
        response.url = url
        response._content = body
        response.request = Request("GET", url).prepare()
        return response

    def _get(url: str, *args: Any, **kwargs: Any) -> Response:
        nonlocal api_attempts
        if str(url) == api_url:
            api_attempts += 1
            if api_attempts == 1:
                return _response(api_url, status_code)
            return _response(api_url, 200, api_body)
        return real_get(url, *args, **kwargs)

    assert installer._session is not None
    monkeypatch.setattr(installer._session, "get", _get)

    installer._restore_incomplete_installs()

    restored_job = installer.get_job_by_source(source)[-1]
    assert restored_job.status == InstallStatus.PAUSED
    assert partial_path.read_bytes() == b"partial model"
    marker = _read_marker(tmpdir)
    assert marker is not None
    assert marker["status"] == InstallStatus.PAUSED.value
    assert marker["files"][0]["etag"] == "retained-etag"
    assert _claim_path(tmpdir).exists()

    queued: dict[str, Any] = {}

    def _capture_download(**kwargs: Any) -> ModelInstallJob:
        queued.update(kwargs)
        return restored_job

    monkeypatch.setattr(installer._session, "get", _get)
    monkeypatch.setattr(installer, "_enqueue_remote_download", _capture_download)
    installer.resume_job(restored_job)

    assert api_attempts == 2
    assert queued["resume_metadata"] == restored_job._resume_metadata
    assert queued["resume_metadata"][weights_url]["download_path"] == partial_path.as_posix()
    assert partial_path.read_bytes() == b"partial model"


def test_queue_rejection_during_shutdown_preserves_completed_download(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}queue-stop-{uuid.uuid4().hex}"
    partial_path = _write_remote_install_marker(mm2_installer, tmpdir, 9905, InstallStatus.DOWNLOADS_DONE)
    job = ModelInstallJob(
        id=9905,
        source=URLModelSource(url=Url("https://example.com/weights.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADS_DONE,
    )
    _stage(job, tmpdir)
    mm2_installer._install_jobs.append(job)
    mm2_installer._stop_event.set()

    mm2_installer._put_in_queue(job)

    assert job.status == InstallStatus.DOWNLOADS_DONE
    assert partial_path.read_bytes() == b"downloaded model"
    assert _read_marker(tmpdir) is not None


def test_restore_leaves_unvisited_markers_untouched_when_stopping(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)
    first_tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}restore-stop-first-{uuid.uuid4().hex}"
    second_tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}restore-stop-second-{uuid.uuid4().hex}"
    first_partial = _write_remote_install_marker(mm2_installer, first_tmpdir, 9901, InstallStatus.DOWNLOADING)
    second_partial = _write_remote_install_marker(mm2_installer, second_tmpdir, 9902, InstallStatus.DOWNLOADING)
    mm2_installer._stop_event.set()

    mm2_installer._restore_incomplete_installs()

    assert first_partial.read_bytes() == b"partial download"
    assert second_partial.read_bytes() == b"partial download"
    assert _read_marker(first_tmpdir) is not None
    assert _read_marker(second_tmpdir) is not None


def test_stop_during_startup_restore_preserves_the_current_download(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}restore-stop-active-{uuid.uuid4().hex}"
    partial_path = _write_remote_install_marker(mm2_installer, tmpdir, 9903, InstallStatus.DOWNLOADING)
    restore_started = threading.Event()
    resume_remote_download = mm2_installer._resume_remote_download

    def wait_until_stopping(job: ModelInstallJob, *, operation_reserved: bool = False) -> None:
        restore_started.set()
        assert mm2_installer._stop_event.wait(timeout=5)
        resume_remote_download(job, operation_reserved=operation_reserved)

    monkeypatch.setattr(mm2_installer, "_resume_remote_download", wait_until_stopping)
    mm2_installer._restore_incomplete_installs_async()
    assert restore_started.wait(timeout=5)

    mm2_installer.stop()

    marker = _read_marker(tmpdir)
    assert marker is not None
    assert marker["status"] == InstallStatus.PAUSED.value
    assert partial_path.read_bytes() == b"partial download"
    assert not _claim_path(tmpdir).exists()
    assert mm2_installer._restore_completed_event.is_set()


def test_metadata_chain_falls_back_to_default_revision_after_stop_requested(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The metadata call chain has no stop checkpoint before the variant fallback request."""
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    assert installer._wait_for_restore_complete(timeout=10)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}restore-stop-metadata-{uuid.uuid4().hex}"
    partial_path = tmpdir / "text_encoder" / "model.safetensors.downloading"
    partial_path.parent.mkdir(parents=True)
    partial_path.write_bytes(b"partial model")
    source = HFModelSource(repo_id="org/repo", variant=ModelRepoVariant.FP16)

    metadata_started = threading.Event()
    release_variant_request = threading.Event()
    requests: list[tuple[str, float]] = []
    metadata_results: list[tuple[list[RemoteModelFile], Any]] = []
    metadata_errors: list[BaseException] = []

    def response(status_code: int, content: bytes = b"", headers: dict[str, str] | None = None) -> Response:
        result = Response()
        result.status_code = status_code
        result.url = "https://huggingface.co/api/models/org/repo"
        result._content = content
        result.headers.update(headers or {})
        result.request = Request("GET", result.url).prepare()
        return result

    def scripted_get(url: str, *, params: dict[str, str] | None = None, timeout: float) -> Response:
        requests.append((url, timeout))
        if len(requests) == 1:
            metadata_started.set()
            assert release_variant_request.wait(timeout=5)
            return response(404, headers={"X-Error-Code": "RevisionNotFound"})
        if len(requests) == 2:
            assert installer._stop_event.is_set()
            return response(
                200,
                content=(b'{"id":"org/repo","siblings":[{"rfilename":"model.safetensors","size":10}]}'),
            )
        raise AssertionError(f"Unexpected metadata request: {url}")

    assert installer._session is not None
    monkeypatch.setattr(installer._session, "get", scripted_get)

    def fetch_metadata() -> None:
        try:
            metadata_results.append(installer._remote_files_from_source(source))
        except BaseException as error:
            metadata_errors.append(error)

    metadata_thread = threading.Thread(target=fetch_metadata)
    metadata_thread.start()
    assert metadata_started.wait(timeout=5)

    try:
        installer._stop_event.set()
    finally:
        release_variant_request.set()
        metadata_thread.join(timeout=10)

    assert not metadata_thread.is_alive()
    assert metadata_errors == []
    assert len(metadata_results) == 1
    assert len(requests) == 2
    assert "/revision/" in requests[0][0]
    assert "/revision/" not in requests[1][0]
    assert all(timeout == HF_METADATA_REQUEST_TIMEOUT_SECONDS for _, timeout in requests)
    assert partial_path.read_bytes() == b"partial model"


def test_shutdown_restore_policy_persists_current_and_resumes_unvisited_on_next_start(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    mm2_session: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown pauses the current restore, leaves unvisited markers alone, and startup resumes only non-paused."""
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    assert installer._wait_for_restore_complete(timeout=10)

    markers: dict[Path, tuple[InstallStatus, bytes]] = {}
    for name, status in (
        ("restore-policy-a", InstallStatus.DOWNLOADING),
        ("restore-policy-b", InstallStatus.DOWNLOADING),
        ("restore-policy-user-paused", InstallStatus.PAUSED),
    ):
        tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}{name}-{uuid.uuid4().hex}"
        partial_path = tmpdir / "weights.safetensors.downloading"
        tmpdir.mkdir()
        partial_path.write_bytes(name.encode())
        source = URLModelSource(url=Url(f"https://example.com/{name}.safetensors"))
        job = ModelInstallJob(
            id=9910 + len(markers),
            source=source,
            config_in=ModelRecordChanges(),
            local_path=tmpdir,
            status=status,
        )
        _stage(job, tmpdir)
        _write_v1_marker(job, status)
        markers[tmpdir] = (status, partial_path.read_bytes())

    ordered_active_markers = [
        path
        for path in mm2_app_config.models_path.glob(f"{TMPDIR_PREFIX}*")
        if path in markers and markers[path][0] != InstallStatus.PAUSED
    ]
    assert len(ordered_active_markers) == 2
    interrupted_tmpdir = ordered_active_markers[0]
    interrupted_restore_started = threading.Event()
    original_resume = installer._resume_remote_download

    def interrupt_first_restore(job: ModelInstallJob, *, operation_reserved: bool = False) -> None:
        if job._staging is not None and job._staging.tmpdir == interrupted_tmpdir:
            interrupted_restore_started.set()
            assert installer._stop_event.wait(timeout=5)
        original_resume(job, operation_reserved=operation_reserved)

    monkeypatch.setattr(installer, "_resume_remote_download", interrupt_first_restore)
    installer._restore_incomplete_installs_async()
    assert interrupted_restore_started.wait(timeout=5)
    installer.stop()

    for tmpdir, (original_status, expected_bytes) in markers.items():
        marker = _read_marker(tmpdir)
        assert marker is not None
        assert (tmpdir / "weights.safetensors.downloading").read_bytes() == expected_bytes
        assert not _claim_path(tmpdir).exists()
        if tmpdir == interrupted_tmpdir:
            assert marker["status"] == InstallStatus.PAUSED.value
        else:
            assert marker["status"] == original_status.value

    next_installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=installer.record_store,
        download_queue=installer._download_queue,
        event_bus=installer.event_bus,
        session=mm2_session,
    )
    resumed_sources: list[str] = []

    def observe_resume(job: ModelInstallJob, *, operation_reserved: bool = False) -> None:
        resumed_sources.append(str(job.source))

    monkeypatch.setattr(next_installer, "_resume_remote_download", observe_resume)
    next_installer.start()
    try:
        assert next_installer._wait_for_restore_complete(timeout=10)
        explicitly_paused_tmpdir = next(path for path, (status, _) in markers.items() if status == InstallStatus.PAUSED)
        unvisited_tmpdir = next(path for path in ordered_active_markers if path != interrupted_tmpdir)
        assert resumed_sources == [json.loads((unvisited_tmpdir / INSTALL_MARKER_FILENAME).read_text())["source"]]
        restored = {str(job.source): job for job in next_installer.list_jobs()}
        paused_source = json.loads((explicitly_paused_tmpdir / INSTALL_MARKER_FILENAME).read_text())["source"]
        interrupted_source = json.loads((interrupted_tmpdir / INSTALL_MARKER_FILENAME).read_text())["source"]
        assert restored[paused_source].status == InstallStatus.PAUSED
        assert restored[interrupted_source].status == InstallStatus.PAUSED
        assert _read_marker(explicitly_paused_tmpdir)["status"] == InstallStatus.PAUSED.value
        assert _read_marker(interrupted_tmpdir)["status"] == InstallStatus.PAUSED.value
        assert _read_marker(unvisited_tmpdir)["status"] == InstallStatus.DOWNLOADING.value
        for tmpdir, (_, expected_bytes) in markers.items():
            assert (tmpdir / "weights.safetensors.downloading").read_bytes() == expected_bytes
    finally:
        next_installer.stop()


@pytest.mark.timeout(timeout=30, method="thread")
def test_stop_waits_for_startup_restore_exception_handler_before_pending_cleanup(
    mm2_installer: ModelInstallServiceBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    handler_started = threading.Event()
    release_handler = threading.Event()
    restore_join_started = threading.Event()
    cleanup_started = threading.Event()
    stop_finished = threading.Event()
    real_error = installer._logger.error

    def _fail_restore() -> None:
        raise RuntimeError("restore failed")

    def _blocked_error(message: str, *args: Any, **kwargs: Any) -> None:
        handler_started.set()
        assert release_handler.wait(timeout=5)
        real_error(message, *args, **kwargs)

    def _observe_cleanup() -> None:
        cleanup_started.set()

    monkeypatch.setattr(installer, "_restore_incomplete_installs", _fail_restore)
    monkeypatch.setattr(installer._logger, "error", _blocked_error)
    monkeypatch.setattr(installer, "_clear_pending_jobs", _observe_cleanup)

    installer._restore_incomplete_installs_async()
    restore_thread = installer._restore_thread
    assert restore_thread is not None
    real_join = restore_thread.join

    def _observe_join(timeout: float | None = None) -> None:
        restore_join_started.set()
        real_join(timeout)

    monkeypatch.setattr(restore_thread, "join", _observe_join)
    try:
        assert handler_started.wait(timeout=5)
        stop_thread = threading.Thread(target=lambda: (installer.stop(), stop_finished.set()))
        stop_thread.start()
        assert restore_join_started.wait(timeout=5)
        assert not cleanup_started.is_set()
        assert not stop_finished.is_set()
        release_handler.set()
        assert cleanup_started.wait(timeout=5)
        assert stop_finished.wait(timeout=5)
        stop_thread.join(timeout=5)
        assert not stop_thread.is_alive()
    finally:
        release_handler.set()


@pytest.mark.parametrize("status", [InstallStatus.PAUSED, InstallStatus.DOWNLOADS_DONE])
def test_stop_preserves_restored_downloads_without_a_multifile_job(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
    status: InstallStatus,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}restore-stop-{status.value}-{uuid.uuid4().hex}"
    partial_path = _write_remote_install_marker(mm2_installer, tmpdir, 9904, status)
    resume_metadata = [
        {
            "url": "https://example.com/weights.safetensors",
            "canonical_url": "https://cdn.example.com/weights-v1.safetensors",
            "etag": '"model-v1"',
            "last_modified": "Wed, 01 Jan 2025 00:00:00 GMT",
            "expected_total_bytes": 4096,
            "final_url": "https://cdn.example.com/weights-v1.safetensors",
            "download_path": (tmpdir / "weights.safetensors").as_posix(),
            "resume_required": False,
            "resume_refused": False,
            "resume_message": None,
        }
    ]
    if status == InstallStatus.PAUSED:
        marker_path = tmpdir / INSTALL_MARKER_FILENAME
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["files"] = resume_metadata
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
    queued_jobs: list[ModelInstallJob] = []
    monkeypatch.setattr(mm2_installer, "_put_in_queue", queued_jobs.append)

    mm2_installer._restore_incomplete_installs()
    jobs = mm2_installer.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].status == status
    if status == InstallStatus.DOWNLOADS_DONE:
        assert queued_jobs == jobs

    mm2_installer.stop()

    marker = _read_marker(tmpdir)
    assert marker is not None
    assert marker["status"] == status.value
    if status == InstallStatus.PAUSED:
        assert marker["files"] == resume_metadata
    expected_payload = b"downloaded model" if status == InstallStatus.DOWNLOADS_DONE else b"partial download"
    assert partial_path.read_bytes() == expected_payload
    assert not _claim_path(tmpdir).exists()


def test_service_stop_waits_for_active_install_without_cancelling_it(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    embedding_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}shutdown-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    shutil.copy2(embedding_file, tmpdir / embedding_file.name)
    job = _queue_downloaded_job(installer, tmpdir, 996)

    transfer_started = threading.Event()
    release_transfer = threading.Event()
    clear_started = threading.Event()
    stop_finished = threading.Event()
    stop_errors: list[BaseException] = []
    sentinel_during_transfer: list[bool] = []
    real_move_with_retries = installer._move_with_retries
    real_clear_pending_jobs = installer._clear_pending_jobs

    def blocked_move_with_retries(src: Path, dst: Path) -> None:
        sentinel_during_transfer.append(_recovery_path(tmpdir).exists())
        transfer_started.set()
        assert release_transfer.wait(timeout=5)
        real_move_with_retries(src, dst)

    def observed_clear_pending_jobs() -> None:
        clear_started.set()
        real_clear_pending_jobs()

    def stop_installer() -> None:
        try:
            installer.stop()
        except BaseException as e:
            stop_errors.append(e)
        finally:
            stop_finished.set()

    monkeypatch.setattr(installer, "_move_with_retries", blocked_move_with_retries)
    monkeypatch.setattr(installer, "_clear_pending_jobs", observed_clear_pending_jobs)
    stop_thread = threading.Thread(target=stop_installer)

    try:
        assert transfer_started.wait(timeout=5)
        stop_thread.start()
        assert not stop_finished.wait(timeout=0.05)
        assert not clear_started.is_set()
    finally:
        release_transfer.set()
        if stop_thread.ident is not None:
            stop_thread.join(timeout=10)

    assert not stop_thread.is_alive()
    assert not stop_errors
    assert installer._running is False
    assert job.complete
    assert sentinel_during_transfer == [True]
    assert not tmpdir.exists()


def test_service_stop_preserves_a_job_between_dequeue_and_worker_activation(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    embedding_file: Path,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}shutdown-dequeued-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    shutil.copy2(embedding_file, tmpdir / embedding_file.name)
    installer._lock.acquire()
    job = _queue_downloaded_job(installer, tmpdir, 998)
    stop_finished = threading.Event()
    stop_errors: list[BaseException] = []

    def stop_installer() -> None:
        try:
            installer.stop()
        except BaseException as e:
            stop_errors.append(e)
        finally:
            stop_finished.set()

    stop_thread = threading.Thread(target=stop_installer)

    try:
        deadline = time.monotonic() + 5
        while not installer._install_queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert installer._install_queue.empty()
        installer._stop_event.set()
        stop_thread.start()
        assert not stop_finished.wait(timeout=0.05)
        assert tmpdir.exists()
        assert (tmpdir / embedding_file.name).exists()
    finally:
        installer._lock.release()
        if stop_thread.ident is not None:
            stop_thread.join(timeout=10)

    assert not stop_thread.is_alive()
    assert not stop_errors
    assert stop_finished.is_set()
    assert job.downloads_done
    assert tmpdir.exists()
    assert (tmpdir / embedding_file.name).exists()
    marker = _read_marker(tmpdir)
    assert marker is not None
    assert marker["status"] == InstallStatus.DOWNLOADS_DONE.value


def test_cancel_during_install_preflight_waits_for_probe_then_cleans_safely(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    embedding_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}cancel-preflight-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    shutil.copy2(embedding_file, tmpdir / embedding_file.name)
    job = _queue_downloaded_job(installer, tmpdir, 997)

    probe_started = threading.Event()
    release_probe = threading.Event()
    real_probe = installer._probe

    def blocked_probe(path: Path, config: ModelRecordChanges):
        probe_started.set()
        assert release_probe.wait(timeout=5)
        return real_probe(path, config)

    monkeypatch.setattr(installer, "_probe", blocked_probe)

    try:
        assert probe_started.wait(timeout=5)
        installer.cancel_job(job)
        assert tmpdir.exists()
        assert (tmpdir / embedding_file.name).exists()
        assert not _recovery_path(tmpdir).exists()
    finally:
        release_probe.set()

    assert installer.wait_for_job(job, timeout=10).cancelled
    assert not tmpdir.exists()


@pytest.mark.parametrize("operation", ["restart_failed", "restart_file"])
def test_restart_refuses_recovery_sentinel(
    operation: str,
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}recovery-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    downloaded_file = tmpdir / "model.safetensors"
    downloaded_file.write_bytes(b"recoverable source")
    job = ModelInstallJob(
        id=994,
        source=URLModelSource(url=Url("https://example.com/model.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.ERROR,
    )
    _stage(job, tmpdir)
    _recovery_path(tmpdir).write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        mm2_installer, "_remote_files_from_source", lambda *_args, **_kwargs: pytest.fail("restart reached source")
    )

    with pytest.raises(InstallRecoveryRequiredError, match="requires recovery"):
        if operation == "restart_failed":
            mm2_installer.restart_failed(job)
        else:
            mm2_installer.restart_file(job, str(job.source))

    assert job.status is InstallStatus.ERROR
    assert downloaded_file.read_bytes() == b"recoverable source"
    assert _recovery_path(tmpdir).exists()


def test_registering_recovery_root_removes_its_sentinel(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    recovery_root = mm2_app_config.models_path / "recovered-model"
    recovery_root.mkdir()
    shutil.copy2(embedding_file, recovery_root / embedding_file.name)
    sentinel = _recovery_path(recovery_root)
    sentinel.write_text("preserve until registered", encoding="utf-8")

    key = mm2_installer.register_path(recovery_root)

    assert mm2_installer.record_store.get_model(key).path == "recovered-model"
    assert not sentinel.exists()


def test_installing_recovery_root_clears_source_sentinel_after_transfer(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    recovery_root = mm2_app_config.models_path / "recovery-source"
    recovery_root.mkdir()
    shutil.copy2(embedding_file, recovery_root / embedding_file.name)
    sentinel = _recovery_path(recovery_root)
    sentinel.write_text("preserve until transferred", encoding="utf-8")

    key = mm2_installer.install_path(recovery_root)

    assert mm2_installer.record_store.get_model(key).path == key
    assert (mm2_app_config.models_path / key / embedding_file.name).exists()
    assert not sentinel.exists()


def test_startup_scan_skips_recovery_protected_root(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    recovery_root = mm2_app_config.models_path / "recovered-model"
    recovery_root.mkdir()
    shutil.copy2(embedding_file, recovery_root / embedding_file.name)
    sentinel = _recovery_path(recovery_root)
    sentinel.write_text("preserve until manually registered", encoding="utf-8")

    mm2_installer._register_orphaned_models()

    assert mm2_installer.record_store.all_models() == []
    assert sentinel.exists()
    assert (recovery_root / embedding_file.name).exists()


def test_startup_scan_skips_a_model_directory_holding_a_claimed_root(
    mm2_installer: ModelInstallServiceBase, diffusers_dir: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    pipeline = mm2_app_config.models_path / "pipeline-being-written"
    shutil.copytree(diffusers_dir, pipeline)
    claimed_child = next(path for path in pipeline.iterdir() if path.is_dir())
    _claim_path(claimed_child).write_text("2147483647 1.0\n", encoding="ascii")

    mm2_installer._register_orphaned_models()

    assert mm2_installer.record_store.all_models() == []
    assert _claim_path(claimed_child).exists()


def test_registered_model_with_temporary_prefix_survives_startup_cleanup(
    mm2_installer: ModelInstallServiceBase,
    embedding_file: Path,
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue: Any,
) -> None:
    model_key = f"{TMPDIR_PREFIX}registered-model"

    installed_key = mm2_installer.install_path(embedding_file, config=ModelRecordChanges(key=model_key))
    installed_path = mm2_app_config.models_path / mm2_installer.record_store.get_model(installed_key).path

    restarted_installer = ModelInstallService(
        app_config=mm2_app_config, record_store=mm2_installer.record_store, download_queue=mm2_download_queue
    )
    restarted_installer.start()
    restarted_installer.stop()

    assert installed_path.exists()
    assert mm2_installer.record_store.exists(model_key)


def test_install_registration_failure_preserves_complete_recoverable_files(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
) -> None:
    """A database made read-only after startup can reject registration after file moves succeed."""
    expected_files = {
        path.relative_to(diffusers_dir): path.read_bytes() for path in diffusers_dir.rglob("*") if path.is_file()
    }
    existing_paths = set(mm2_app_config.models_path.iterdir())

    store = mm2_installer.record_store
    assert isinstance(store, ModelRecordServiceSQL)
    database = store._queries._database
    # Exercise a genuine SQLite write rejection without changing permissions on any real user database. The
    # pragma is set in read transactions: a write transaction could not even begin while it is on.
    with database.begin(write=False) as conn:
        conn.exec_driver_sql("PRAGMA query_only = ON")
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly database"):
            mm2_installer.install_path(diffusers_dir)
    finally:
        with database.begin(write=False) as conn:
            conn.exec_driver_sql("PRAGMA query_only = OFF")

    assert mm2_installer.record_store.all_models() == []
    new_paths = set(mm2_app_config.models_path.iterdir()) - existing_paths
    # A rollback to the source or a complete unregistered managed folder both preserve recovery.
    recoverable_roots = [diffusers_dir, *new_paths]
    complete_copies = [
        root
        for root in recoverable_roots
        if {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == expected_files
    ]
    assert len(complete_copies) == 1
    recovered_key = mm2_installer.register_path(complete_copies[0])
    assert mm2_installer.record_store.get_model(recovered_key).key == recovered_key


@pytest.mark.parametrize(
    ("failure_target", "error_number"),
    [
        pytest.param(
            "directory",
            errno.EINVAL,
            marks=pytest.mark.skipif(
                model_install_default.os.name == "nt", reason="installer skips directory fsync on Windows"
            ),
        ),
        ("file", errno.EIO),
    ],
)
def test_install_fsync_failure_preserves_source_before_transfer(
    failure_target: str,
    error_number: int,
    mm2_installer: ModelInstallServiceBase,
    embedding_file: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    source_bytes = embedding_file.read_bytes()
    destination_key = f"fsync-{failure_target}"
    real_fsync = model_install_default.os.fsync

    def failing_fsync(fd: int) -> None:
        is_directory = stat.S_ISDIR(model_install_default.os.fstat(fd).st_mode)
        if (failure_target == "directory" and is_directory) or (failure_target == "file" and not is_directory):
            raise OSError(error_number, "scripted fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(model_install_default.os, "fsync", failing_fsync)

    with pytest.raises(OSError) as error:
        mm2_installer.install_path(embedding_file, ModelRecordChanges(key=destination_key))

    destination = mm2_app_config.models_path / destination_key
    assert error.value.errno == error_number
    assert embedding_file.read_bytes() == source_bytes
    assert not mm2_installer.record_store.exists(destination_key)
    assert not destination.exists()
    assert not _recovery_path(destination).exists()


def test_rename(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record.path.endswith(f"{key}/test_embedding.safetensors")
    new_model_record = store.update_model(
        key,
        ModelRecordChanges(name="new model name", base=BaseModelType.StableDiffusion2),
        allow_class_change=True,
    )
    # Renaming the model record shouldn't rename the file
    assert new_model_record.name == "new model name"
    assert model_record.path.endswith(f"{key}/test_embedding.safetensors")


@pytest.mark.parametrize(
    "fixture_name,size,key,destination",
    [
        ("embedding_file", 15440, "foo", "foo/test_embedding.safetensors"),
        ("diffusers_dir", 8241 if OS == "Windows" else 7907, "bar", "bar"),  # EOL chars
    ],
)
def test_background_install(
    mm2_installer: ModelInstallServiceBase,
    fixture_name: str,
    key: str,
    size: int,
    destination: str,
    mm2_app_config: InvokeAIAppConfig,
    request: pytest.FixtureRequest,
) -> None:
    """Note: may want to break this down into several smaller unit tests."""
    path: Path = request.getfixturevalue(fixture_name)
    description = "Test of metadata assignment"
    source = LocalModelSource(path=path, inplace=False)
    job = mm2_installer.import_model(source, config=ModelRecordChanges(key=key, description=description))
    assert job is not None
    assert isinstance(job, ModelInstallJob)

    # See if job is registered properly
    assert job in mm2_installer.get_job_by_source(source)

    # test that the job object tracked installation correctly
    jobs = mm2_installer.wait_for_installs()
    assert len(jobs) > 0
    my_job = [x for x in jobs if x.source == source]
    assert len(my_job) == 1
    assert job == my_job[0]
    assert job.status == InstallStatus.COMPLETED
    assert job.total_bytes == size

    # test that the expected events were issued
    bus: TestEventService = mm2_installer.event_bus
    assert bus
    assert hasattr(bus, "events")

    assert len(bus.events) == 2
    assert isinstance(bus.events[0], ModelInstallStartedEvent)
    assert isinstance(bus.events[1], ModelInstallCompleteEvent)
    assert Path(bus.events[0].source.path) == source
    assert Path(bus.events[1].source.path) == source
    key = bus.events[1].key
    assert key is not None

    # see if the thing actually got installed at the expected location
    model_record = mm2_installer.record_store.get_model(key)
    assert model_record is not None
    assert model_record.path.endswith(destination)
    assert (mm2_app_config.models_path / model_record.path).exists()

    # see if metadata was properly passed through
    assert model_record.description == description

    # see if job filtering works
    assert mm2_installer.get_job_by_source(source)[0] == job

    # see if prune works properly
    mm2_installer.prune_jobs()
    assert not mm2_installer.get_job_by_source(source)


def test_not_inplace_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    # An non in-place install will/should call `register_path()` internally
    source = LocalModelSource(path=embedding_file, inplace=False)
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs()
    assert job is not None
    assert job.config_out is not None
    # Non in-place install should _move_ the model from the original location to the models path
    # The model config's path should be different from the original file
    assert Path(job.config_out.path) != embedding_file
    # Original file should _not_ exist after install
    assert not embedding_file.exists()
    assert (mm2_app_config.models_path / job.config_out.path).exists()


def test_inplace_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    # An in-place install will/should call `install_path()` internally
    source = LocalModelSource(path=embedding_file, inplace=True)
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs()
    assert job is not None
    assert job.config_out is not None
    # In-place install should not touch the model file, just register it
    # The model config's path should be the same as the original file
    assert Path(job.config_out.path) == embedding_file
    # Model file should still exist after install
    assert embedding_file.exists()
    assert Path(job.config_out.path).exists()


def test_external_install(mm2_installer: ModelInstallServiceBase) -> None:
    config = ModelRecordChanges(name="ChatGPT Image", description="External model", key="chatgpt_image")
    job = mm2_installer.heuristic_import("external://openai/gpt-image-1", config=config)

    mm2_installer.wait_for_installs()

    assert job.status == InstallStatus.COMPLETED
    assert job.config_out is not None
    assert isinstance(job.config_out, ExternalApiModelConfig)
    assert job.config_out.provider_id == "openai"
    assert job.config_out.provider_model_id == "gpt-image-1"
    assert job.config_out.base == BaseModelType.External
    assert job.config_out.type == ModelType.ExternalImageGenerator
    assert job.config_out.source_type == ModelSourceType.External


def test_external_install_is_idempotent(mm2_installer: ModelInstallServiceBase) -> None:
    first_job = mm2_installer.heuristic_import(
        "external://openai/gpt-image-1",
        config=ModelRecordChanges(name="Initial name"),
    )
    mm2_installer.wait_for_installs()

    second_job = mm2_installer.heuristic_import(
        "external://openai/gpt-image-1",
        config=ModelRecordChanges(name="Updated name"),
    )
    mm2_installer.wait_for_installs()

    assert first_job.status == InstallStatus.COMPLETED
    assert second_job.status == InstallStatus.COMPLETED
    assert first_job.config_out is not None
    assert second_job.config_out is not None
    assert first_job.config_out.key == second_job.config_out.key

    external_models = mm2_installer.record_store.search_by_attr(
        base_model=BaseModelType.External,
        model_type=ModelType.ExternalImageGenerator,
    )
    assert len(external_models) == 1
    assert isinstance(external_models[0], ExternalApiModelConfig)
    assert external_models[0].name == "Updated name"


def test_delete_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)  # non in-place install
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert not embedding_file.exists()
    # ensure file handles are released on Windows
    gc.collect()
    mm2_installer.delete(key)
    # after deletion, installed copy should not exist
    assert not (mm2_app_config.models_path / model_record.path).exists()
    with pytest.raises(UnknownModelException):
        store.get_model(key)


def test_delete_register(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(embedding_file)  # in-place install
    model_record = store.get_model(key)
    assert Path(model_record.path).exists()
    assert embedding_file.exists()
    mm2_installer.delete(key)
    assert Path(model_record.path).exists()
    with pytest.raises(UnknownModelException):
        store.get_model(key)


@pytest.mark.timeout(timeout=10, method="thread")
def test_simple_download(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    bus: TestEventService = mm2_installer.event_bus
    store = mm2_installer.record_store
    assert store is not None
    assert bus is not None
    assert hasattr(bus, "events")  # the dummy event service has this

    job = mm2_installer.import_model(source)
    assert job.source == source
    job_list = mm2_installer.wait_for_installs(timeout=10)
    assert len(job_list) == 1
    assert job.complete
    assert job.config_out

    key = job.config_out.key
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()

    assert len(bus.events) == 5
    assert isinstance(bus.events[0], ModelInstallDownloadStartedEvent)  # download starts
    assert isinstance(bus.events[1], ModelInstallDownloadProgressEvent)  # download progresses
    assert isinstance(bus.events[2], ModelInstallDownloadsCompleteEvent)  # download completed
    assert isinstance(bus.events[3], ModelInstallStartedEvent)  # install started
    assert isinstance(bus.events[4], ModelInstallCompleteEvent)  # install completed


@pytest.mark.timeout(timeout=10, method="thread")
def test_wait_for_installs_waits_for_download_completion_to_enqueue_install(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not report all installs done between download-cache removal and install enqueue."""
    assert isinstance(mm2_installer, ModelInstallService)
    job = ModelInstallJob(
        id=123,
        source=URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors")),
        local_path=tmp_path,
    )
    _stage(job, None, tmp_path)
    with mm2_installer._lock:
        mm2_installer._download_cache[job.id] = job
    monkeypatch.setattr(mm2_installer, "_signal_job_downloads_done", lambda install_job: None)
    assert mm2_installer._wait_for_restore_complete(timeout=2)

    wait_finished = threading.Event()

    def wait_for_installs() -> None:
        mm2_installer.wait_for_installs(timeout=2)
        wait_finished.set()

    wait_thread = threading.Thread(target=wait_for_installs)
    wait_thread.start()

    enqueue_started = threading.Event()
    release_enqueue = threading.Event()

    def blocked_enqueue(install_job: ModelInstallJob) -> bool:
        assert install_job is job
        enqueue_started.set()
        assert release_enqueue.wait(timeout=2)
        return True

    monkeypatch.setattr(mm2_installer, "_queue_install_job_locked", blocked_enqueue, raising=False)
    callback_thread = threading.Thread(
        target=mm2_installer._download_complete_callback,
        args=(SimpleNamespace(id=job.id),),
    )
    callback_thread.start()
    assert enqueue_started.wait(timeout=2)
    assert not wait_finished.wait(timeout=0.5)

    release_enqueue.set()
    callback_thread.join(timeout=2)
    wait_thread.join(timeout=2)
    assert not callback_thread.is_alive()
    assert not wait_thread.is_alive()
    assert wait_finished.is_set()


@pytest.mark.timeout(timeout=10, method="thread")
def test_import_waits_for_startup_restore(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    restore_started = threading.Event()
    release_restore = threading.Event()
    import_waiting = threading.Event()
    imported = threading.Event()
    imported_jobs: list[ModelInstallJob] = []
    source = URLModelSource(url=Url("https://www.test.foo/download/interrupted.safetensors"))
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}interrupted"
    tmpdir.mkdir()
    interrupted_job = ModelInstallJob(
        id=99999,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
    )
    _stage(interrupted_job, tmpdir)
    _write_v1_marker(interrupted_job, InstallStatus.DOWNLOADING)
    restore = installer._restore_incomplete_installs
    wait_for_restore = installer._wait_for_restore_complete

    def _blocked_restore() -> None:
        restore_started.set()
        assert release_restore.wait(timeout=5)
        restore()

    def _import() -> None:
        imported_jobs.append(installer.import_model(source))
        imported.set()

    def _observed_wait_for_restore() -> bool:
        import_waiting.set()
        return wait_for_restore()

    monkeypatch.setattr(installer, "_restore_incomplete_installs", _blocked_restore)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job, *, operation_reserved=False: None)

    try:
        assert not installer._restore_completed_event.is_set()
        installer.start()
        assert restore_started.wait(timeout=5)
        with pytest.raises(TimeoutError):
            installer.wait_for_installs(timeout=0.1)

        monkeypatch.setattr(installer, "_wait_for_restore_complete", _observed_wait_for_restore)
        import_thread = threading.Thread(target=_import)
        import_thread.start()
        assert import_waiting.wait(timeout=5)
        assert not imported.is_set()

        release_restore.set()
        import_thread.join(timeout=5)
        assert imported.is_set()
        jobs = installer.get_job_by_source(source)
        assert len(jobs) == 1
        assert imported_jobs == jobs
    finally:
        release_restore.set()
        installer.stop()


@pytest.mark.timeout(timeout=30, method="thread")
def test_concurrent_imports_of_same_source_return_one_job(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    assert mm2_installer._restore_completed_event.wait(timeout=5)

    # Both imports can reuse this interrupted install directory. Hold the first helper after its duplicate check so the
    # second import can reach the same post-restore race window.
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}reusable"
    tmpdir.mkdir()
    interrupted_job = ModelInstallJob(
        id=99998,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
    )
    _stage(interrupted_job, tmpdir)
    _write_v1_marker(interrupted_job, InstallStatus.DOWNLOADING)

    first_import_ready = threading.Event()
    second_import_waiting = threading.Event()
    second_helper_entered = threading.Event()
    release_first_import = threading.Event()
    helper_calls = 0
    helper_calls_lock = threading.Lock()
    import_from_url = mm2_installer._import_from_url
    imported_jobs: list[ModelInstallJob] = []
    import_errors: list[BaseException] = []
    condition_wait = mm2_installer._install_condition.wait

    def _observed_condition_wait(timeout: float | None = None) -> bool:
        second_import_waiting.set()
        return condition_wait(timeout)

    def _synchronized_import_from_url(
        import_source: URLModelSource, config: ModelRecordChanges | None = None
    ) -> ModelInstallJob:
        nonlocal helper_calls
        with helper_calls_lock:
            helper_calls += 1
            is_first_import = helper_calls == 1
        if is_first_import:
            first_import_ready.set()
            assert release_first_import.wait(timeout=5)
        else:
            second_helper_entered.set()
        return import_from_url(import_source, config)

    def _import() -> None:
        try:
            imported_jobs.append(mm2_installer.import_model(source))
        except BaseException as error:
            import_errors.append(error)

    monkeypatch.setattr(mm2_installer._install_condition, "wait", _observed_condition_wait)
    monkeypatch.setattr(mm2_installer, "_import_from_url", _synchronized_import_from_url)

    first_import_thread = threading.Thread(target=_import)
    second_import_thread = threading.Thread(target=_import)
    first_import_thread.start()
    assert first_import_ready.wait(timeout=5)
    second_import_thread.start()
    assert second_import_waiting.wait(timeout=5)
    assert not second_helper_entered.is_set()
    release_first_import.set()

    import_threads = [first_import_thread, second_import_thread]
    for import_thread in import_threads:
        import_thread.join(timeout=20)

    assert all(not import_thread.is_alive() for import_thread in import_threads)
    assert not import_errors
    jobs = mm2_installer.get_job_by_source(source)
    assert len(jobs) == 1
    assert len(imported_jobs) == 2
    assert imported_jobs[0] is imported_jobs[1] is jobs[0]


@pytest.mark.timeout(timeout=10, method="thread")
def test_failed_import_releases_source_reservation(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    import_attempts = 0

    def _import_from_url(import_source: URLModelSource, config: ModelRecordChanges | None = None) -> ModelInstallJob:
        nonlocal import_attempts
        import_attempts += 1
        if import_attempts == 1:
            raise RuntimeError("metadata request failed")
        job = ModelInstallJob(
            id=99997,
            source=import_source,
            config_in=config or ModelRecordChanges(),
            local_path=mm2_app_config.models_path,
        )
        _stage(job, None, mm2_app_config.models_path)
        return job

    monkeypatch.setattr(mm2_installer, "_import_from_url", _import_from_url)

    with pytest.raises(RuntimeError, match="metadata request failed"):
        mm2_installer.import_model(source)

    job = mm2_installer.import_model(source)
    assert job.source == source
    assert mm2_installer.get_job_by_source(source) == [job]


@pytest.mark.timeout(timeout=30, method="thread")
def test_waiting_import_prefers_a_live_job_over_a_terminal_one(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    """A waiter can be released into a state where the source has BOTH a terminal job registered while it
    waited and a live one. It must return the live job, not the dead one."""
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    source_key = str(source)
    condition = mm2_installer._install_condition
    assert mm2_installer._restore_completed_event.wait(timeout=10)

    terminal_job = ModelInstallJob(
        id=88001, source=source, config_in=ModelRecordChanges(), local_path=mm2_app_config.models_path
    )
    terminal_job.status = InstallStatus.ERROR
    live_job = ModelInstallJob(
        id=88002, source=source, config_in=ModelRecordChanges(), local_path=mm2_app_config.models_path
    )
    live_job.status = InstallStatus.DOWNLOADING
    _stage(live_job, None, mm2_app_config.models_path)

    # Stand in for an owner that has reserved the source but not yet registered its job.
    with condition:
        mm2_installer._pending_sources.add(source_key)

    waiter_result: list[ModelInstallJob] = []
    waiter = threading.Thread(target=lambda: waiter_result.append(mm2_installer.import_model(source)))
    waiter.start()

    # Wait until the importer is actually parked on the condition rather than sleeping a fixed interval.
    deadline = time.time() + 10
    while not condition._waiters and time.time() < deadline:
        time.sleep(0.01)
    assert condition._waiters, "importer never parked on the install condition"

    # Everything the waiter can observe happens in one critical section: an earlier attempt that ended in ERROR
    # and a later one that is still downloading. The waiter must not see an intermediate state.
    with condition:
        mm2_installer._install_jobs.append(terminal_job)
        mm2_installer._install_jobs.append(live_job)
        mm2_installer._pending_sources.discard(source_key)
        condition.notify_all()

    waiter.join(timeout=10)
    assert not waiter.is_alive()
    assert waiter_result and waiter_result[0] is live_job


@pytest.mark.timeout(timeout=30, method="thread")
def test_prune_jobs_cannot_drop_a_concurrent_registration(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    """prune_jobs() filters and reassigns _install_jobs. Unlocked, an import_model() registration landing between
    the two is dropped, leaving a live install invisible to the duplicate check."""
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    assert mm2_installer._restore_completed_event.wait(timeout=10)

    entered = threading.Event()
    release = threading.Event()

    class _BlockingJob:
        """prune_jobs() only reads .in_terminal_state; block there to suspend it mid-prune."""

        id = -1
        source = "blocking"

        @property
        def in_terminal_state(self) -> bool:
            entered.set()
            assert release.wait(timeout=20)
            return True

    blocking_job = _BlockingJob()
    mm2_installer._install_jobs.append(blocking_job)  # type: ignore[arg-type]
    importer_done = threading.Event()
    imported: list[ModelInstallJob] = []

    def _import() -> None:
        imported.append(mm2_installer.import_model(source))
        importer_done.set()

    pruner = threading.Thread(target=mm2_installer.prune_jobs)
    importer = threading.Thread(target=_import)

    try:
        pruner.start()
        assert entered.wait(timeout=10)

        # prune_jobs() must hold the installer lock across the whole filter-and-reassign, so no registration can
        # land in between. Assert that directly rather than inferring it from the importer's timing.
        lock_was_free = mm2_installer._lock.acquire(timeout=0.5)
        if lock_was_free:
            mm2_installer._lock.release()
        assert not lock_was_free, "prune_jobs() ran its filter without holding the lock"

        importer.start()
        assert not importer_done.wait(timeout=1), "import_model registered a job while prune_jobs was mid-prune"
    finally:
        # Always release, or _BlockingJob survives in _install_jobs and blocks fixture teardown too.
        release.set()
        for thread in (pruner, importer):
            if thread.ident is not None:  # an early assertion may have fired before importer.start()
                thread.join(timeout=10)
        if blocking_job in mm2_installer._install_jobs:
            mm2_installer._install_jobs.remove(blocking_job)  # type: ignore[arg-type]

    assert not pruner.is_alive() and not importer.is_alive()
    assert imported and mm2_installer.get_job_by_source(source) == imported


@pytest.mark.timeout(timeout=10, method="thread")
def test_waiting_import_returns_its_new_terminal_job_when_no_live_job_exists(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    source_key = str(source)
    condition = mm2_installer._install_condition
    assert mm2_installer._restore_completed_event.wait(timeout=5)

    with condition:
        mm2_installer._pending_sources.add(source_key)

    importer_waiting = threading.Event()
    condition_wait = condition.wait

    def _observed_condition_wait(timeout: float | None = None) -> bool:
        importer_waiting.set()
        return condition_wait(timeout)

    monkeypatch.setattr(condition, "wait", _observed_condition_wait)
    imported_jobs: list[ModelInstallJob] = []
    importer = threading.Thread(target=lambda: imported_jobs.append(mm2_installer.import_model(source)))
    importer.start()
    assert importer_waiting.wait(timeout=5)

    terminal_job = ModelInstallJob(
        id=99001,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
    )
    terminal_job.status = InstallStatus.ERROR
    with condition:
        mm2_installer._install_jobs.append(terminal_job)
        mm2_installer._pending_sources.remove(source_key)
        condition.notify_all()

    importer.join(timeout=5)
    assert not importer.is_alive()
    assert imported_jobs == [terminal_job]


def test_prune_jobs_rebind_preserves_existing_iterators(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    jobs = [
        ModelInstallJob(
            id=99002 + index,
            source=URLModelSource(url=Url(f"https://www.test.foo/download/model-{index}.safetensors")),
            config_in=ModelRecordChanges(),
            local_path=mm2_app_config.models_path,
            status=InstallStatus.COMPLETED,
        )
        for index in range(3)
    ]
    active_job = ModelInstallJob(
        id=99010,
        source=URLModelSource(url=Url("https://www.test.foo/download/active.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
        status=InstallStatus.RUNNING,
    )
    _stage(active_job, None, mm2_app_config.models_path)
    mm2_installer._install_jobs.extend([*jobs, active_job])
    existing_iterator = iter(mm2_installer.list_jobs())
    assert next(existing_iterator) is jobs[0]

    mm2_installer.prune_jobs()

    assert list(existing_iterator) == [*jobs[1:], active_job]
    assert all(job not in mm2_installer.list_jobs() for job in jobs)
    assert mm2_installer.list_jobs() == [active_job]


@pytest.mark.timeout(timeout=30, method="thread")
def test_prune_jobs_waits_for_the_installer_lock(mm2_installer: ModelInstallServiceBase) -> None:
    lock_held = threading.Event()
    release_lock = threading.Event()
    prune_completed = threading.Event()

    def _hold_lock() -> None:
        with mm2_installer._lock:
            lock_held.set()
            assert release_lock.wait(timeout=3)

    lock_holder = threading.Thread(target=_hold_lock)
    pruner = threading.Thread(target=lambda: (mm2_installer.prune_jobs(), prune_completed.set()))
    try:
        lock_holder.start()
        assert lock_held.wait(timeout=3)
        pruner.start()
        assert not prune_completed.wait(timeout=0.25)
    finally:
        release_lock.set()
        for thread in (lock_holder, pruner):
            if thread.ident is not None:
                thread.join(timeout=3)

    assert not lock_holder.is_alive() and not pruner.is_alive()
    assert prune_completed.is_set()


@pytest.mark.timeout(timeout=30, method="thread")
def test_import_and_wait_for_installs_fail_before_start(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    embedding_file: Path,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )

    with pytest.raises(RuntimeError, match="not running"):
        installer.import_model(LocalModelSource(path=embedding_file))
    with pytest.raises(RuntimeError, match="not running"):
        installer.wait_for_installs(timeout=0.1)


@pytest.mark.timeout(timeout=30, method="thread")
def test_import_fails_after_startup_failure(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    embedding_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )

    def _fail_startup() -> None:
        raise RuntimeError("startup failed")

    monkeypatch.setattr(installer, "_migrate_yaml", _fail_startup)

    try:
        with pytest.raises(RuntimeError, match="startup failed"):
            installer.start()

        with pytest.raises(RuntimeError, match="Model install service failed to start"):
            installer.import_model(LocalModelSource(path=embedding_file))
    finally:
        installer.stop()


@pytest.mark.timeout(timeout=30, method="thread")
def test_base_exception_during_startup_releases_import_waiters(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    embedding_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )

    def _interrupt_startup() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(installer, "_migrate_yaml", _interrupt_startup)

    try:
        with pytest.raises(KeyboardInterrupt):
            installer.start()

        assert installer._restore_completed_event.is_set()
        with pytest.raises(RuntimeError, match="Model install service failed to start"):
            installer.import_model(LocalModelSource(path=embedding_file))
    finally:
        installer.stop()


def test_huggingface_blob_url_uses_resolve_download_url(mm2_installer: ModelInstallServiceBase) -> None:
    source = URLModelSource(
        url=Url("https://huggingface.co/h94/IP-Adapter/blob/main/sdxl_models/ip-adapter.safetensors")
    )

    assert isinstance(mm2_installer, ModelInstallService)
    files, metadata = mm2_installer._remote_files_from_source(source)

    assert metadata is None
    assert len(files) == 1
    assert str(files[0].url) == "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter.safetensors"


@pytest.mark.timeout(timeout=10, method="thread")
def test_huggingface_install(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://huggingface.co/stabilityai/sdxl-turbo"))

    bus: TestEventService = mm2_installer.event_bus
    store = mm2_installer.record_store
    assert isinstance(bus, EventServiceBase)
    assert store is not None

    job = mm2_installer.import_model(source)
    job_list = mm2_installer.wait_for_installs(timeout=10)
    assert len(job_list) == 1
    assert job.complete
    assert job.config_out

    key = job.config_out.key
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert model_record.type == ModelType.Main
    assert model_record.format == ModelFormat.Diffusers

    assert any(isinstance(x, ModelInstallStartedEvent) for x in bus.events)
    assert any(isinstance(x, ModelInstallDownloadProgressEvent) for x in bus.events)
    assert any(isinstance(x, ModelInstallCompleteEvent) for x in bus.events)
    assert len(bus.events) >= 3


@pytest.mark.timeout(timeout=10, method="thread")
def test_huggingface_repo_id(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = HFModelSource(repo_id="stabilityai/sdxl-turbo", variant=ModelRepoVariant.Default)

    bus = mm2_installer.event_bus
    store = mm2_installer.record_store
    assert isinstance(bus, EventServiceBase)
    assert store is not None

    job = mm2_installer.import_model(source)
    job_list = mm2_installer.wait_for_installs(timeout=10)
    assert len(job_list) == 1
    assert job.complete
    assert job.config_out

    key = job.config_out.key
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert model_record.type == ModelType.Main
    assert model_record.format == ModelFormat.Diffusers

    assert hasattr(bus, "events")  # the dummyeventservice has this
    assert len(bus.events) >= 3
    event_types = [type(x) for x in bus.events]
    assert all(
        x in event_types
        for x in [
            ModelInstallDownloadProgressEvent,
            ModelInstallDownloadsCompleteEvent,
            ModelInstallStartedEvent,
            ModelInstallCompleteEvent,
        ]
    )

    completed_events = [x for x in bus.events if isinstance(x, ModelInstallCompleteEvent)]
    downloading_events = [x for x in bus.events if isinstance(x, ModelInstallDownloadProgressEvent)]
    assert completed_events[0].total_bytes == downloading_events[-1].bytes
    assert job.total_bytes == completed_events[0].total_bytes
    print(downloading_events[-1])
    print(job.download_parts)
    assert job.total_bytes == sum(x["total_bytes"] for x in downloading_events[-1].parts)


def test_restore_paused_hf_install_preserves_access_token(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)

    access_token = "hf_test_access_token"
    tmpdir = mm2_app_config.models_path / f"tmpinstall_resume_token_{uuid.uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    try:
        paused_job = ModelInstallJob(
            id=99999,
            source=HFModelSource(
                repo_id="stabilityai/sdxl-turbo",
                variant=ModelRepoVariant.Default,
                access_token=access_token,
            ),
            config_in=ModelRecordChanges(),
            local_path=tmpdir,
        )
        _stage(paused_job, tmpdir)
        paused_job.status = InstallStatus.PAUSED

        _write_v1_marker(paused_job, InstallStatus.PAUSED)

        marker = _read_marker(tmpdir)
        assert marker is not None
        assert marker["access_token"] == access_token

        restored_installer = ModelInstallService(
            app_config=mm2_app_config,
            record_store=mm2_installer.record_store,
            download_queue=mm2_download_queue,
            session=mm2_session,
        )
        restored_installer._restore_incomplete_installs()
        restored_jobs = restored_installer.list_jobs()
        assert len(restored_jobs) == 1

        restored_job = restored_jobs[0]
        assert restored_job.paused
        assert isinstance(restored_job.source, HFModelSource)
        assert restored_job.source.access_token == access_token
        assert _claim_path(tmpdir).exists()

        captured: dict[str, str | None] = {}

        def _capture_resume(job: ModelInstallJob, *, operation_reserved: bool = False) -> None:
            assert isinstance(job.source, HFModelSource)
            captured["access_token"] = job.source.access_token

        monkeypatch.setattr(restored_installer, "_resume_remote_download", _capture_resume)
        restored_installer.resume_job(restored_job)
        assert captured["access_token"] == access_token
    finally:
        _claim_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


_HF_EMBEDDING_SOURCE = "InvokeAI-test/textual_inversion_tests::learned_embeds-steps-1000.safetensors"
_HF_EMBEDDING_URL = (
    "https://huggingface.co/InvokeAI-test/textual_inversion_tests/resolve/main/learned_embeds-steps-1000.safetensors"
)
_URL_EMBEDDING_URL = "https://www.test.foo/download/test_embedding.safetensors"


class _AuthorizationAdapter(TestAdapter):
    """Records each request's Authorization header; with `required_token`, refuses others like a gated HF repo."""

    def __init__(self, data: bytes, required_token: str | None = None) -> None:
        super().__init__(data, headers={"Content-Length": len(data)})
        self.required_token = required_token
        self.authorizations: list[str | None] = []

    def send(self, request, **kwargs):
        authorization = request.headers.get("Authorization")
        self.authorizations.append(authorization)
        if self.required_token is not None and authorization != f"Bearer {self.required_token}":
            return TestAdapter(b"", status=401).send(request, **kwargs)
        return super().send(request, **kwargs)


@pytest.mark.parametrize("login_at_failure", [None, "hf_revoked"], ids=["no_login", "outdated_login"])
@pytest.mark.timeout(timeout=10, method="thread")
def test_restart_failed_uses_hf_login_made_after_the_failure(
    mm2_installer: ModelInstallServiceBase,
    mm2_session,
    embedding_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    login_at_failure: str | None,
) -> None:
    hf_login = {"token": login_at_failure}
    monkeypatch.setattr(model_install_default, "hf_get_token", lambda: hf_login["token"])
    mm2_session.mount(
        _HF_EMBEDDING_URL, _AuthorizationAdapter(embedding_file.read_bytes(), required_token="hf_granted")
    )

    job = mm2_installer.heuristic_import(
        source=_HF_EMBEDDING_SOURCE, config=ModelRecordChanges(type=ModelType.TextualInversion)
    )
    mm2_installer.wait_for_installs(timeout=10)
    assert job.errored
    assert job.error_type == "HTTPError"
    assert job.error is not None and "UNAUTHORIZED" in job.error

    hf_login["token"] = "hf_granted"
    mm2_installer.restart_failed(job)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error


@pytest.mark.parametrize(
    ("source", "download_url", "explicit_token", "expected_authorization"),
    [
        (_HF_EMBEDDING_SOURCE, _HF_EMBEDDING_URL, "hf_explicit", "Bearer hf_explicit"),
        (_URL_EMBEDDING_URL, _URL_EMBEDDING_URL, None, None),
    ],
    ids=["explicit_token_wins_over_login", "url_source_never_gets_login"],
)
@pytest.mark.timeout(timeout=10, method="thread")
def test_download_authorization_while_logged_in_to_hf(
    mm2_installer: ModelInstallServiceBase,
    mm2_session,
    embedding_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    download_url: str,
    explicit_token: str | None,
    expected_authorization: str | None,
) -> None:
    monkeypatch.setattr(model_install_default, "hf_get_token", lambda: "hf_login")
    adapter = _AuthorizationAdapter(embedding_file.read_bytes())
    mm2_session.mount(download_url, adapter)

    job = mm2_installer.heuristic_import(
        source=source, config=ModelRecordChanges(type=ModelType.TextualInversion), access_token=explicit_token
    )
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert adapter.authorizations == [expected_authorization]


_SDXL_TURBO_SOURCE = HFModelSource(repo_id="stabilityai/sdxl-turbo", variant=ModelRepoVariant.Default)
_SDXL_TURBO_FILES_URL = "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/"
# Downloads run in URL order, so every other file of the repo is complete before this one starts. The file list
# assumes full-precision weights, which `sdxl_turbo_server` pins.
_SDXL_TURBO_LAST_FILE = "vae/diffusion_pytorch_model.safetensors"


class _RepoFileServer:
    """Serves the sdxl-turbo repo files, records every download, and scripts failures per file.

    A "fail" attempt answers HTTP 500, which errors the install. A "short" attempt announces one byte more than it
    sends, which pauses the install with a resume-required part. (The repo's weight files are empty.)

    Each file carries an ETag naming its revision, unless `validators` is off. A Range request is recorded in
    `resumed` with its If-Range; it gets the rest of the file (HTTP 206) unless If-Range names another revision, which
    gets the whole current file (HTTP 200), as RFC 9110 specifies.
    """

    def __init__(self, session, repo_dir: Path) -> None:
        self.downloads: list[str] = []
        self.resumed: list[tuple[str, str, str | None]] = []
        self.scripted: dict[str, list[str]] = {}
        self.validators = True
        # Called with each downloaded file's name before the server answers.
        self.on_download: Callable[[str], None] | None = None
        self.files: dict[str, bytes] = {}
        self._revisions: dict[str, int] = {}
        for path in sorted(repo_dir.rglob("*")):
            if path.is_file():
                name = path.relative_to(repo_dir).as_posix()
                self.files[name] = path.read_bytes()
                session.mount(_SDXL_TURBO_FILES_URL + name, _RepoFileAdapter(self, name))

    def etag(self, name: str) -> str:
        return f'"{name}@{self._revisions.get(name, 1)}"'

    def publish(self, name: str, data: bytes) -> None:
        """Replace a file upstream with a new revision."""
        self.files[name] = data
        self._revisions[name] = self._revisions.get(name, 1) + 1

    def take_downloads(self) -> list[str]:
        downloads, self.downloads = self.downloads, []
        return downloads


class _RepoFileAdapter(TestAdapter):
    def __init__(self, server: _RepoFileServer, name: str) -> None:
        super().__init__(b"")
        self.server = server
        self.name = name

    def send(self, request, stream=False, **kwargs):
        data = self.server.files[self.name]
        headers: dict[str, Any] = {"Content-Length": len(data)}
        if self.server.validators:
            headers["ETag"] = self.server.etag(self.name)
        if stream:  # a download, not a metadata read of model_index.json
            self.server.downloads.append(self.name)
            if self.server.on_download is not None:
                self.server.on_download(self.name)
            attempts = self.server.scripted.get(self.name)
            attempt = attempts.pop(0) if attempts else None
            if attempt == "fail":
                return TestAdapter(b"", status=500).send(request, stream=stream, **kwargs)
            if attempt == "short":
                headers["Content-Length"] = len(data) + 1
            elif byte_range := request.headers.get("Range"):
                if_range = request.headers.get("If-Range")
                self.server.resumed.append((self.name, byte_range, if_range))
                if if_range is None or if_range == headers.get("ETag"):
                    start = int(byte_range.removeprefix("bytes=").removesuffix("-"))
                    headers["Content-Length"] = len(data) - start
                    headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
                    return TestAdapter(data[start:], status=206, headers=headers).send(request, stream=stream, **kwargs)
        return TestAdapter(data, headers=headers).send(request, stream=stream, **kwargs)


@pytest.fixture
def sdxl_turbo_server(mm2_session, diffusers_dir: Path, monkeypatch: pytest.MonkeyPatch) -> _RepoFileServer:
    """Serve the sdxl-turbo repo to an installer that picks its full-precision weights on any device.

    The installer reads `ModelRepoVariant.Default` (an empty string) as no variant and guesses one from the torch dtype:
    fp16 weights on CUDA or MPS, full-precision ones on CPU. Pinning the dtype fixes the files these tests script.
    """
    import torch

    monkeypatch.setattr(TorchDevice, "choose_torch_dtype", staticmethod(lambda device=None: torch.float32))
    return _RepoFileServer(mm2_session, diffusers_dir)


def _stop_halfway(job: ModelInstallJob, server: _RepoFileServer, names: list[str]) -> None:
    """Turn finished staged files back into half-written partials, as a pause leaves them."""
    for name in names:
        staged = job.local_path / name
        staged.unlink()
        staged.with_name(staged.name + ".downloading").write_bytes(server.files[name][: len(server.files[name]) // 2])


def _installed_files(app_config: InvokeAIAppConfig, job: ModelInstallJob) -> set[str]:
    assert job.config_out is not None
    model_dir = app_config.models_path / job.config_out.path
    return {path.relative_to(model_dir).as_posix() for path in model_dir.rglob("*") if path.is_file()}


@pytest.mark.parametrize("failures", [1, 2], ids=["one_failure", "repeated_failure"])
@pytest.mark.timeout(timeout=20, method="thread")
def test_restart_failed_refetches_files_deleted_by_the_failed_attempt(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
    failures: int,
) -> None:
    server = sdxl_turbo_server
    server.scripted[_SDXL_TURBO_LAST_FILE] = ["fail"] * failures

    job = mm2_installer.import_model(_SDXL_TURBO_SOURCE)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.errored
    repo_files = server.take_downloads()
    assert repo_files[-1] == _SDXL_TURBO_LAST_FILE and len(repo_files) > 1
    assert job._staging is not None
    staging = job._staging.tmpdir
    assert staging is not None and not staging.exists()

    for _ in range(failures - 1):
        mm2_installer.restart_failed(job)
        mm2_installer.wait_for_installs(timeout=10)
        assert job.errored
        assert sorted(server.take_downloads()) == sorted(repo_files)
        assert not staging.exists()

    mm2_installer.restart_failed(job)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert sorted(server.take_downloads()) == sorted(repo_files)
    assert _installed_files(mm2_app_config, job) == set(repo_files)
    assert not staging.exists()
    assert not _claim_path(staging).exists()


@pytest.mark.parametrize("operation", ["restart_failed", "restart_file"])
@pytest.mark.timeout(timeout=20, method="thread")
def test_restart_of_a_paused_install_fetches_only_files_missing_from_staging(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
    operation: str,
) -> None:
    server = sdxl_turbo_server
    server.scripted[_SDXL_TURBO_LAST_FILE] = ["short"]

    job = mm2_installer.import_model(_SDXL_TURBO_SOURCE)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED
    assert any(part.resume_required for part in job.download_parts)
    repo_files = server.take_downloads()
    assert repo_files[-1] == _SDXL_TURBO_LAST_FILE and len(repo_files) > 1

    if operation == "restart_failed":
        mm2_installer.restart_failed(job)
    else:
        mm2_installer.restart_file(job, _SDXL_TURBO_FILES_URL + _SDXL_TURBO_LAST_FILE)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert server.take_downloads() == [_SDXL_TURBO_LAST_FILE]
    assert _installed_files(mm2_app_config, job) == set(repo_files)


@pytest.mark.timeout(timeout=20, method="thread")
def test_retry_after_a_partial_restart_failed_fetches_every_file(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
) -> None:
    # The pause keeps staging, so the first restart fetches one file; its failure then deletes staging while
    # job.download_parts lists only that file.
    server = sdxl_turbo_server
    server.scripted[_SDXL_TURBO_LAST_FILE] = ["short", "fail"]

    job = mm2_installer.import_model(_SDXL_TURBO_SOURCE)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED
    repo_files = server.take_downloads()

    mm2_installer.restart_failed(job)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.errored
    assert server.take_downloads() == [_SDXL_TURBO_LAST_FILE]
    assert [str(part.source) for part in job.download_parts] == [_SDXL_TURBO_FILES_URL + _SDXL_TURBO_LAST_FILE]

    mm2_installer.restart_failed(job)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert sorted(server.take_downloads()) == sorted(repo_files)
    assert _installed_files(mm2_app_config, job) == set(repo_files)


@pytest.mark.parametrize(
    ("operation", "validators", "expected_resumed"),
    [
        ("restart_file", True, ["scheduler/scheduler_config.json"]),
        ("restart_failed", True, ["scheduler/scheduler_config.json", "unet/config.json"]),
        ("restart_failed", False, []),
    ],
    ids=["restart_file", "restart_failed", "restart_failed_without_validators"],
)
@pytest.mark.timeout(timeout=20, method="thread")
def test_restart_of_a_paused_install_resumes_partials_it_does_not_restart(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
    diffusers_dir: Path,
    operation: str,
    validators: bool,
    expected_resumed: list[str],
) -> None:
    server = sdxl_turbo_server
    server.validators = validators
    server.scripted[_SDXL_TURBO_LAST_FILE] = ["short"]
    job = mm2_installer.import_model(_SDXL_TURBO_SOURCE)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED
    repo_files = server.take_downloads()
    partials = ["scheduler/scheduler_config.json", "unet/config.json"]
    _stop_halfway(job, server, partials)

    if operation == "restart_file":
        mm2_installer.restart_file(job, _SDXL_TURBO_FILES_URL + "unet/config.json")
    else:
        mm2_installer.restart_failed(job)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert sorted(server.take_downloads()) == sorted([*partials, _SDXL_TURBO_LAST_FILE])
    assert server.resumed == [
        (name, f"bytes={len(server.files[name]) // 2}-", server.etag(name)) for name in expected_resumed
    ]
    assert _installed_files(mm2_app_config, job) == set(repo_files)
    assert job.config_out is not None
    installed = mm2_app_config.models_path / job.config_out.path
    for name in partials:
        assert (installed / name).read_bytes() == (diffusers_dir / name).read_bytes()


@pytest.mark.timeout(timeout=20, method="thread")
def test_restart_never_splices_a_partial_onto_a_newer_upstream_revision(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
) -> None:
    server = sdxl_turbo_server
    server.scripted[_SDXL_TURBO_LAST_FILE] = ["short"]
    job = mm2_installer.import_model(_SDXL_TURBO_SOURCE)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED
    changed = "scheduler/scheduler_config.json"
    _stop_halfway(job, server, [changed])
    old_etag = server.etag(changed)
    # Same size, so only the validator can tell the revisions apart.
    new_revision = server.files[changed].replace(b"scaled_linear", b"SCALED_LINEAR")
    assert new_revision != server.files[changed] and len(new_revision) == len(server.files[changed])
    server.publish(changed, new_revision)

    mm2_installer.restart_file(job, _SDXL_TURBO_FILES_URL + _SDXL_TURBO_LAST_FILE)
    mm2_installer.wait_for_installs(timeout=10)

    # The server refused the stale range, so the part needs a restart instead of being completed from two revisions.
    assert server.resumed == [(changed, f"bytes={len(new_revision) // 2}-", old_etag)]
    assert job.status is InstallStatus.PAUSED
    assert [str(part.source) for part in job.download_parts if part.resume_required] == [
        _SDXL_TURBO_FILES_URL + changed
    ]

    server.take_downloads()
    mm2_installer.restart_failed(job)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    # The refused part paused the job before the requested file, which follows it in URL order, was fetched.
    assert server.take_downloads() == [changed, _SDXL_TURBO_LAST_FILE]
    assert job.config_out is not None
    assert (mm2_app_config.models_path / job.config_out.path / changed).read_bytes() == new_revision


@pytest.mark.timeout(timeout=20, method="thread")
def test_resume_never_splices_a_partial_onto_a_newer_upstream_revision(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
) -> None:
    server = sdxl_turbo_server
    server.scripted[_SDXL_TURBO_LAST_FILE] = ["short"]
    job = mm2_installer.import_model(_SDXL_TURBO_SOURCE)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED
    changed = "scheduler/scheduler_config.json"
    _stop_halfway(job, server, [changed])
    old_etag = server.etag(changed)
    new_revision = server.files[changed].replace(b"scaled_linear", b"SCALED_LINEAR")
    assert new_revision != server.files[changed] and len(new_revision) == len(server.files[changed])
    server.publish(changed, new_revision)

    mm2_installer.resume_job(job)
    mm2_installer.wait_for_installs(timeout=10)

    assert server.resumed == [(changed, f"bytes={len(new_revision) // 2}-", old_etag)]
    assert job.status is InstallStatus.PAUSED
    assert [str(part.source) for part in job.download_parts if part.resume_required] == [
        _SDXL_TURBO_FILES_URL + changed
    ]

    mm2_installer.restart_failed(job)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert job.config_out is not None
    assert (mm2_app_config.models_path / job.config_out.path / changed).read_bytes() == new_revision


@pytest.mark.timeout(timeout=20, method="thread")
def test_resume_restarts_a_partial_without_validators_from_zero(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
) -> None:
    server = sdxl_turbo_server
    server.validators = False
    server.scripted[_SDXL_TURBO_LAST_FILE] = ["short"]
    job = mm2_installer.import_model(_SDXL_TURBO_SOURCE)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED
    changed = "scheduler/scheduler_config.json"
    _stop_halfway(job, server, [changed])

    mm2_installer.resume_job(job)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert server.resumed == []
    assert job.config_out is not None
    assert (mm2_app_config.models_path / job.config_out.path / changed).read_bytes() == server.files[changed]


class _ShortOnceAdapter(TestAdapter):
    """Serves a file without validators, and records each download's Range header. With `short_first`, the first
    download announces one byte more than it sends, which pauses the install with the whole body left as a partial."""

    def __init__(self, data: bytes, short_first: bool = True) -> None:
        super().__init__(data, headers={"Content-Length": len(data)})
        self.short_first = short_first
        self.ranges: list[str | None] = []

    def send(self, request, stream=False, **kwargs):
        self.ranges.append(request.headers.get("Range"))
        if self.short_first and len(self.ranges) == 1:
            short = TestAdapter(self.stream, headers={"Content-Length": len(self.stream) + 1})
            return short.send(request, stream=stream, **kwargs)
        return super().send(request, stream=stream, **kwargs)


@pytest.mark.parametrize("operation", ["resume_job", "restart_failed"])
@pytest.mark.timeout(timeout=20, method="thread")
def test_url_source_partial_restarted_from_zero_is_deleted_and_fetched_whole(
    mm2_installer: ModelInstallServiceBase,
    mm2_session,
    embedding_file: Path,
    operation: str,
) -> None:
    # A single-file URL source downloads into its staging directory and names the file from the response, so the
    # partial is not at the part's destination.
    url = "https://www.test.foo/download/unvalidated_embedding.safetensors"
    adapter = _ShortOnceAdapter(embedding_file.read_bytes())
    mm2_session.mount(url, adapter)
    job = mm2_installer.heuristic_import(source=url, config=ModelRecordChanges(type=ModelType.TextualInversion))
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED
    assert job._staging is not None and job._staging.tmpdir is not None
    assert [path.name for path in job._staging.tmpdir.glob("*.downloading")] == [
        "unvalidated_embedding.safetensors.downloading"
    ]

    getattr(mm2_installer, operation)(job)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert adapter.ranges == [None, None]


@pytest.mark.timeout(timeout=20, method="thread")
def test_restored_download_path_outside_staging_is_never_used(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    mm2_session,
    embedding_file: Path,
    tmp_path: Path,
) -> None:
    # A marker from a moved or copied root can name a download path in some other folder.
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)
    url = "https://www.test.foo/download/outside_embedding.safetensors"
    adapter = _ShortOnceAdapter(embedding_file.read_bytes(), short_first=False)
    mm2_session.mount(url, adapter)
    elsewhere = tmp_path / "elsewhere" / "outside_embedding.safetensors"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"someone else's file")
    elsewhere_partial = elsewhere.with_name(elsewhere.name + ".downloading")
    elsewhere_partial.write_bytes(b"someone else's partial")
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}outside-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    job = ModelInstallJob(
        id=9910,
        source=URLModelSource(url=Url(url)),
        config_in=ModelRecordChanges(type=ModelType.TextualInversion),
        local_path=tmpdir,
        status=InstallStatus.PAUSED,
    )
    _stage(job, tmpdir)
    job._resume_metadata = {
        url: {"url": url, "canonical_url": url, "etag": '"v1"', "download_path": elsewhere.as_posix()}
    }
    mm2_installer._write_install_marker(job, status=InstallStatus.PAUSED)
    mm2_installer._restore_incomplete_installs()
    restored_job = mm2_installer.get_job_by_source(URLModelSource(url=Url(url)))[-1]
    assert restored_job.status is InstallStatus.PAUSED

    mm2_installer.resume_job(restored_job)
    mm2_installer.wait_for_installs(timeout=10)

    assert restored_job.complete, restored_job.error
    assert adapter.ranges == [None]
    assert elsewhere.read_bytes() == b"someone else's file"
    assert elsewhere_partial.read_bytes() == b"someone else's partial"


@pytest.mark.parametrize("recorded_path", ["outside_staging", "missing"])
@pytest.mark.timeout(timeout=20, method="thread")
def test_restored_url_source_partial_without_validators_restarts_from_zero(
    mm2_installer: ModelInstallServiceBase,
    mm2_session,
    mm2_app_config: InvokeAIAppConfig,
    embedding_file: Path,
    tmp_path: Path,
    recorded_path: str,
) -> None:
    # A single-file URL source names its file from the response, so its partial is found inside the staging folder,
    # not at a recorded path that a moved root or an interrupted marker write left unusable.
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)
    url = "https://www.test.foo/download/unvalidated_embedding.safetensors"
    data = embedding_file.read_bytes()
    adapter = _ShortOnceAdapter(data, short_first=False)
    mm2_session.mount(url, adapter)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}restored-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    (tmpdir / "unvalidated_embedding.safetensors.downloading").write_bytes(data[: len(data) // 2])
    elsewhere = tmp_path / "elsewhere" / "unvalidated_embedding.safetensors"
    job = ModelInstallJob(
        id=9911,
        source=URLModelSource(url=Url(url)),
        config_in=ModelRecordChanges(type=ModelType.TextualInversion),
        local_path=tmpdir,
    )
    _stage(job, tmpdir)
    job._resume_metadata = {
        url: {
            "url": url,
            "canonical_url": url,
            "download_path": elsewhere.as_posix() if recorded_path == "outside_staging" else None,
        }
    }
    mm2_installer._write_install_marker(job, status=InstallStatus.PAUSED)
    mm2_installer._restore_incomplete_installs()
    restored_job = mm2_installer.get_job_by_source(URLModelSource(url=Url(url)))[-1]

    mm2_installer.resume_job(restored_job)
    [part] = restored_job.download_parts
    assert part.resume_from_scratch and part.resume_message
    mm2_installer.wait_for_installs(timeout=10)

    assert restored_job.complete, restored_job.error
    assert adapter.ranges == [None]
    assert restored_job.config_out is not None
    assert (mm2_app_config.models_path / restored_job.config_out.path).read_bytes() == data


@pytest.mark.parametrize("case", ["refused", "unvalidated"])
@pytest.mark.timeout(timeout=20, method="thread")
def test_resume_never_sends_an_unguarded_range_for_a_partial_it_cannot_delete(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    # A partial that must restart but is held open (a Windows lock) cannot be deleted. Resuming it without If-Range
    # would splice: in the refused case upstream is known to have changed, in the unvalidated one nothing would tell.
    server = sdxl_turbo_server
    server.validators = case == "refused"
    server.scripted[_SDXL_TURBO_LAST_FILE] = ["short"]
    job = mm2_installer.import_model(_SDXL_TURBO_SOURCE)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED
    changed = "scheduler/scheduler_config.json"
    _stop_halfway(job, server, [changed])
    old_revision = server.files[changed]
    new_revision = old_revision.replace(b"scaled_linear", b"SCALED_LINEAR")
    server.publish(changed, new_revision)
    if case == "refused":
        mm2_installer.resume_job(job)
        mm2_installer.wait_for_installs(timeout=10)
        assert job.status is InstallStatus.PAUSED
    resumed_before = list(server.resumed)
    partial = job.local_path / f"{changed}.downloading"
    real_unlink = Path.unlink

    def locked_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == partial:
            raise PermissionError(13, "The process cannot access the file", str(path))
        real_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", locked_unlink)
        with pytest.raises(InstallDownloadConflictError, match="scheduler_config.json"):
            mm2_installer.resume_job(job)
        mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED
    assert server.resumed == resumed_before
    assert partial.read_bytes() == old_revision[: len(old_revision) // 2]

    mm2_installer.resume_job(job)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert server.resumed == resumed_before
    assert job.config_out is not None
    assert (mm2_app_config.models_path / job.config_out.path / changed).read_bytes() == new_revision


@pytest.mark.timeout(timeout=20, method="thread")
def test_resume_after_a_refused_range_restarts_the_partial_from_zero(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
) -> None:
    server = sdxl_turbo_server
    server.scripted[_SDXL_TURBO_LAST_FILE] = ["short"]
    job = mm2_installer.import_model(_SDXL_TURBO_SOURCE)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED
    changed = "scheduler/scheduler_config.json"
    _stop_halfway(job, server, [changed])
    old_revision, old_etag = server.files[changed], server.etag(changed)
    new_revision = old_revision.replace(b"scaled_linear", b"SCALED_LINEAR")
    server.publish(changed, new_revision)

    mm2_installer.resume_job(job)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status is InstallStatus.PAUSED

    # The refusal recorded no validator of the new revision, so a second resume cannot pass If-Range with it.
    mm2_installer.resume_job(job)
    changed_part = next(part for part in job.download_parts if str(part.source) == _SDXL_TURBO_FILES_URL + changed)
    assert changed_part.resume_from_scratch and "refused" in (changed_part.resume_message or "")
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert server.resumed == [(changed, f"bytes={len(old_revision) // 2}-", old_etag)]
    assert job.config_out is not None
    assert (mm2_app_config.models_path / job.config_out.path / changed).read_bytes() == new_revision


_CRASHED_PART = "unet/config.json"


def _stage_a_crash_during(
    installer: ModelInstallService,
    app_config: InvokeAIAppConfig,
    server: _RepoFileServer,
    interrupted: str,
    case: str,
) -> tuple[Path, Path, bytes]:
    """Leave a staging folder as a crash while `interrupted` downloads leaves it.

    Returns the staging folder, the staged path of `interrupted` and that file's data.

    A crash leaves the marker as the last part start wrote it, with that part half written. Parts run in URL order.
    Copying the staging folder when the part after `interrupted` is requested captures the marker and finished files
    as such a crash leaves them; `interrupted` is then cut back to a half partial, or removed for "vanished". The
    original install then fails, which removes its own staging.
    """
    assert installer._wait_for_restore_complete(timeout=10)
    server.validators = case != "unvalidated"
    following = "unet/diffusion_pytorch_model.safetensors"
    crashed = app_config.models_path / f"{TMPDIR_PREFIX}crashed-{uuid.uuid4().hex}"
    jobs: list[ModelInstallJob] = []

    def crash_before(name: str) -> None:
        if name == following:
            assert jobs[0]._staging is not None and jobs[0]._staging.tmpdir is not None
            shutil.copytree(jobs[0]._staging.tmpdir, crashed)

    server.on_download = crash_before
    server.scripted[following] = ["fail"]
    jobs.append(installer.import_model(_SDXL_TURBO_SOURCE))
    installer.wait_for_installs(timeout=10)
    assert jobs[0].errored
    server.on_download = None
    server.take_downloads()
    assert jobs[0]._staging is not None and jobs[0]._staging.tmpdir is not None
    staged = crashed / jobs[0].local_path.relative_to(jobs[0]._staging.tmpdir) / interrupted
    data = server.files[interrupted]
    staged.unlink()
    if case != "vanished":
        staged.with_name(staged.name + ".downloading").write_bytes(data[: len(data) // 2])
    if case.startswith("refused"):
        marker_path = crashed / INSTALL_MARKER_FILENAME
        marker = json.loads(marker_path.read_text())
        for entry in marker["files"]:
            if entry["url"] == _SDXL_TURBO_FILES_URL + interrupted:
                entry["resume_required"] = True
                entry["resume_message"] = "Resume refused by server. Restart required."
                if case == "refused":
                    entry["resume_refused"] = True
                else:
                    del entry["resume_refused"]
        marker_path.write_text(json.dumps(marker))
    return crashed, staged, data


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("validated", None),
        ("refused", "refused"),
        ("refused_by_an_older_build", "refused"),
        ("unvalidated", "could not be checked"),
        ("vanished", "missing"),
    ],
)
@pytest.mark.timeout(timeout=30, method="thread")
def test_install_restored_after_a_crash_resumes_only_partials_its_marker_validates(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
    case: str,
    expected_message: str | None,
) -> None:
    """The "refused" cases are a part the server already refused to resume, as its marker entry records (an older
    build recorded only the message); its validator still matches, so only that record keeps it from resuming."""
    assert isinstance(mm2_installer, ModelInstallService)
    crashed, _, data = _stage_a_crash_during(mm2_installer, mm2_app_config, sdxl_turbo_server, _CRASHED_PART, case)
    server = sdxl_turbo_server
    interrupted = _CRASHED_PART

    mm2_installer._restore_incomplete_installs()
    restored = mm2_installer.get_job_by_source(_SDXL_TURBO_SOURCE)[-1]
    assert restored._staging is not None and restored._staging.tmpdir == crashed
    mm2_installer.wait_for_installs(timeout=10)

    assert restored.complete, restored.error
    if case == "validated":
        assert server.resumed == [(interrupted, f"bytes={len(data) // 2}-", server.etag(interrupted))]
    else:
        assert server.resumed == []
    # Only the interrupted part is reported; the finished files it skipped, and those never started, are not.
    reported = {str(part.source): part.resume_message for part in restored.download_parts if part.resume_from_scratch}
    if expected_message is None:
        assert reported == {}
    else:
        assert list(reported) == [_SDXL_TURBO_FILES_URL + interrupted]
        assert expected_message in (reported[_SDXL_TURBO_FILES_URL + interrupted] or "")
    assert restored.config_out is not None
    assert (mm2_app_config.models_path / restored.config_out.path / interrupted).read_bytes() == data


@pytest.mark.parametrize("case", ["refused", "unvalidated"])
@pytest.mark.timeout(timeout=30, method="thread")
def test_restore_keeps_a_partial_it_cannot_delete_and_leaves_the_install_paused(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    # The partial must restart, but another process holds it open. Startup must neither resume it without a validator
    # nor treat the failure as an install error, which would delete the staging folder.
    assert isinstance(mm2_installer, ModelInstallService)
    staging_dir, staged, data = _stage_a_crash_during(
        mm2_installer, mm2_app_config, sdxl_turbo_server, _CRASHED_PART, case
    )
    partial = staged.with_name(staged.name + ".downloading")
    real_unlink = Path.unlink

    def locked_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == partial:
            raise PermissionError(13, "The process cannot access the file", str(path))
        real_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", locked_unlink)
        mm2_installer._restore_incomplete_installs()
        mm2_installer.wait_for_installs(timeout=10)

    restored = mm2_installer.get_job_by_source(_SDXL_TURBO_SOURCE)[-1]
    assert restored._staging is not None and restored._staging.tmpdir == staging_dir
    assert restored.status is InstallStatus.PAUSED
    assert json.loads((staging_dir / INSTALL_MARKER_FILENAME).read_text())["status"] == "paused"
    assert partial.read_bytes() == data[: len(data) // 2]
    assert sdxl_turbo_server.take_downloads() == []
    assert sdxl_turbo_server.resumed == []


@pytest.mark.timeout(timeout=20, method="thread")
def test_restart_file_after_an_error_also_fetches_every_file_deleted_with_the_staging(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    sdxl_turbo_server: _RepoFileServer,
) -> None:
    server = sdxl_turbo_server
    server.scripted[_SDXL_TURBO_LAST_FILE] = ["fail"]

    job = mm2_installer.import_model(_SDXL_TURBO_SOURCE)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.errored
    repo_files = server.take_downloads()

    mm2_installer.restart_file(job, _SDXL_TURBO_FILES_URL + _SDXL_TURBO_LAST_FILE)
    mm2_installer.wait_for_installs(timeout=10)

    assert job.complete, job.error
    assert sorted(server.take_downloads()) == sorted(repo_files)
    assert _installed_files(mm2_app_config, job) == set(repo_files)


def test_restore_claims_downloads_done_staging_before_queueing(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}restore-complete-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    staging_job = ModelInstallJob(
        id=99998,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADS_DONE,
    )
    _stage(staging_job, tmpdir)
    (tmpdir / "model.safetensors").write_bytes(b"downloaded model")
    _write_v1_marker(staging_job, InstallStatus.DOWNLOADS_DONE)

    restored_installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_installer.record_store,
        download_queue=mm2_download_queue,
        session=mm2_session,
    )
    queued_jobs: list[ModelInstallJob] = []

    def capture_queued_job(job: ModelInstallJob) -> None:
        assert _claim_path(tmpdir).exists()
        queued_jobs.append(job)

    monkeypatch.setattr(restored_installer, "_put_in_queue", capture_queued_job)

    try:
        restored_installer._restore_incomplete_installs()
        assert len(queued_jobs) == 1
        assert queued_jobs[0].downloads_done
        assert queued_jobs[0]._staging.claimed
    finally:
        _claim_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_restart_failed_uses_parts_created_before_download_started(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(
        source=source.url,
        dest=tmp_path / "model.safetensors",
        canonical_url="https://cdn.example.com/model.safetensors",
        etag='"model-etag"',
        expected_total_bytes=8,
    )
    download_job = MultiFileDownloadJob(id=123, dest=tmp_path, download_parts={part})
    job = ModelInstallJob(id=99999, source=source, config_in=ModelRecordChanges(), local_path=tmp_path)
    _stage(job, tmp_path)
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=8)

    monkeypatch.setattr(mm2_installer, "_multifile_download", lambda **_: download_job)
    submit_multifile_download = MagicMock()
    monkeypatch.setattr(mm2_installer._download_queue, "submit_multifile_download", submit_multifile_download)
    mm2_installer._enqueue_remote_download(
        job=job,
        source=source,
        remote_files=[remote_file],
        metadata=None,
    )

    part.resume_required = True
    mm2_installer._download_cancelled_callback(download_job)

    assert job.status == InstallStatus.PAUSED
    assert job.model_dump(mode="json")["download_parts"][0]["resume_required"] is True
    marker = _read_marker(tmp_path)
    assert marker is not None
    assert marker["files"][0]["etag"] == '"model-etag"'
    assert marker["files"][0]["resume_required"] is True
    submit_multifile_download.assert_called_once_with(download_job)

    monkeypatch.setattr(mm2_installer, "_remote_files_from_source", lambda _: ([remote_file], None))
    enqueue_remote_download = MagicMock()
    monkeypatch.setattr(mm2_installer, "_enqueue_remote_download", enqueue_remote_download)
    mm2_installer.restart_failed(job)

    assert job.status == InstallStatus.WAITING
    enqueue_remote_download.assert_called_once()
    assert enqueue_remote_download.call_args.kwargs["restart_from_zero"] == {str(source.url)}


@pytest.mark.parametrize("callback_name", ["_download_error_callback", "_download_cancelled_callback"])
def test_download_terminal_callbacks_preserve_recovery_protected_staging(
    callback_name: str, mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}callback-recovery-{callback_name}"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors"
    payload.write_bytes(b"recovery data")
    _recovery_path(tmpdir).write_text("preserve", encoding="utf-8")

    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=payload)
    download_job = MultiFileDownloadJob(id=456, dest=tmpdir, download_parts={part})
    install_job = ModelInstallJob(
        id=456,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADING,
    )
    _stage(install_job, tmpdir).claim()
    install_job._multifile_job = download_job
    installer._download_cache[download_job.id] = install_job
    monkeypatch.setattr(installer._download_queue, "cancel_job", MagicMock())

    if callback_name == "_download_error_callback":
        installer._download_error_callback(download_job, RuntimeError("download failed"))
    else:
        installer._download_cancelled_callback(download_job)

    assert _recovery_path(tmpdir).exists()
    assert payload.read_bytes() == b"recovery data"
    # The recovery mark protects the data from here on; a leftover claim would block cleanup until restart.
    assert not _claim_path(tmpdir).exists()


def test_download_completion_during_shutdown_preserves_staging_for_restore(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}shutdown-download-complete"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors"
    payload.write_bytes(b"downloaded model")

    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=payload)
    part.status = DownloadJobStatus.COMPLETED
    download_job = MultiFileDownloadJob(id=457, dest=tmpdir, download_parts={part})
    download_job.status = DownloadJobStatus.COMPLETED
    install_job = ModelInstallJob(
        id=457,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADING,
    )
    _stage(install_job, tmpdir)
    install_job._multifile_job = download_job
    install_job._staging.claim()
    installer._install_jobs.append(install_job)
    installer._download_cache[download_job.id] = install_job
    monkeypatch.setattr(installer._download_queue, "cancel_job", MagicMock())
    installer._stop_event.set()

    callback_errors: list[BaseException] = []
    callback_done = threading.Event()

    def complete_download() -> None:
        try:
            installer._download_complete_callback(download_job)
        except BaseException as e:
            callback_errors.append(e)
        finally:
            callback_done.set()

    callback_thread = threading.Thread(target=complete_download, daemon=True)
    callback_thread.start()
    callback_thread.join(timeout=2)

    try:
        assert callback_done.is_set(), "download completion deadlocked while cancelling during shutdown"
        assert not callback_thread.is_alive()
        assert not callback_errors
        assert install_job.downloads_done
        assert tmpdir.exists()
        marker = _read_marker(tmpdir)
        assert marker is not None
        assert marker["status"] == InstallStatus.DOWNLOADS_DONE.value
        assert not _claim_path(tmpdir).exists()
    finally:
        # A deadlocked daemon callback is expected to survive a failed assertion in the regression case; keep fixture
        # teardown from waiting on the same broken lock path.
        installer._running = False
        if installer._install_thread is not None:
            installer._install_thread.join(timeout=2)


def test_stop_releases_claim_for_completed_downloads_waiting_to_install(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}shutdown-completed-download"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors"
    payload.write_bytes(b"downloaded model")
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=payload)
    part.status = DownloadJobStatus.COMPLETED
    download_job = MultiFileDownloadJob(id=458, dest=tmpdir, download_parts={part})
    download_job.status = DownloadJobStatus.COMPLETED
    install_job = ModelInstallJob(
        id=458,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADS_DONE,
    )
    _stage(install_job, tmpdir)
    install_job._staging.claim()
    install_job._multifile_job = download_job
    installer._install_jobs.append(install_job)

    installer.stop()

    try:
        assert install_job.downloads_done
        assert tmpdir.exists()
        marker = _read_marker(tmpdir)
        assert marker is not None
        assert marker["status"] == InstallStatus.DOWNLOADS_DONE.value
        assert not _claim_path(tmpdir).exists()
    finally:
        _claim_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_paused_remote_install_stays_hidden_from_orphan_cleanup(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    from invokeai.app.services.orphaned_models import OrphanedModelsService

    installer = mm2_installer
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}paused-install-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors"
    payload.write_bytes(b"paused model")
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=payload)
    part.pause()
    part.status = DownloadJobStatus.PAUSED
    download_job = MultiFileDownloadJob(id=461, dest=tmpdir, download_parts={part})
    download_job.pause()
    download_job.status = DownloadJobStatus.PAUSED
    job = ModelInstallJob(
        id=461,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.PAUSED,
    )
    _stage(job, tmpdir)
    job._staging.claim()
    job._multifile_job = download_job
    installer._download_cache[download_job.id] = job
    orphan_service = OrphanedModelsService(config=mm2_app_config, store=installer.record_store)

    try:
        installer._download_cancelled_callback(download_job)

        assert _claim_path(tmpdir).exists()
        assert tmpdir.name not in {orphan.path for orphan in orphan_service.find_orphaned_models()}
        result = orphan_service.delete_orphaned_models([tmpdir.name])[tmpdir.name]
        assert "reserved" in result
        assert payload.exists()
    finally:
        _claim_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_cancel_active_remote_install_waits_for_download_queue_cleanup(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}cancel-active-download"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors.downloading"
    payload.write_bytes(b"partial download")
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=tmpdir / "model.safetensors")
    part.status = DownloadJobStatus.RUNNING
    download_job = MultiFileDownloadJob(id=460, dest=tmpdir, download_parts={part})
    download_job.status = DownloadJobStatus.RUNNING
    job = ModelInstallJob(
        id=460,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADING,
    )
    _stage(job, tmpdir)
    job._staging.claim()
    job._multifile_job = download_job
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    cancel_download = MagicMock()
    monkeypatch.setattr(installer._download_queue, "cancel_job", cancel_download)

    installer.cancel_job(job)

    try:
        assert cancel_download.call_args.args == (download_job,)
        assert tmpdir.exists()
        assert payload.exists()
        assert _claim_path(tmpdir).exists()
        assert installer._download_cache.get(download_job.id) is job

        # Cancellation may race an interrupted part being marked resumable by the queue.
        part.resume_required = True
        installer._download_cancelled_callback(download_job)

        assert job.cancelled
        assert not tmpdir.exists()
        assert not _claim_path(tmpdir).exists()
        assert download_job.id not in installer._download_cache
    finally:
        _claim_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize("terminal_callback", ["complete", "error"])
def test_cancel_waits_for_pending_terminal_download_callback(
    terminal_callback: str,
    mm2_installer: ModelInstallServiceBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}cancel-pending-{terminal_callback}"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors"
    payload.write_bytes(b"downloaded model")
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=payload)
    part.status = DownloadJobStatus.COMPLETED if terminal_callback == "complete" else DownloadJobStatus.ERROR
    download_job = MultiFileDownloadJob(id=4587, dest=tmpdir, download_parts={part})
    download_job.status = part.status
    job = ModelInstallJob(
        id=4587,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADING,
    )
    _stage(job, tmpdir)
    job._staging.claim()
    job._multifile_job = download_job
    job.download_parts = download_job.download_parts
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    cancel_download = MagicMock()
    monkeypatch.setattr(installer._download_queue, "cancel_job", cancel_download)

    try:
        installer.cancel_job(job)

        assert job.cancelled
        assert tmpdir.exists()
        assert payload.exists()
        assert _claim_path(tmpdir).exists()
        cancel_download.assert_called_once_with(download_job)

        if terminal_callback == "complete":
            installer._download_complete_callback(download_job)
        else:
            installer._download_error_callback(download_job, RuntimeError("late download error"))

        assert job.cancelled
        assert download_job.id not in installer._download_cache
        assert installer._install_queue.qsize() == 0
        assert not tmpdir.exists()
        assert not _claim_path(tmpdir).exists()
    finally:
        _claim_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize("terminal_callback", ["complete", "error"])
def test_pause_survives_pending_terminal_download_callback(
    terminal_callback: str,
    mm2_installer: ModelInstallServiceBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}pause-pending-{terminal_callback}"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors"
    payload.write_bytes(b"downloaded model")
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=payload)
    part.status = DownloadJobStatus.COMPLETED if terminal_callback == "complete" else DownloadJobStatus.ERROR
    download_job = MultiFileDownloadJob(id=4588, dest=tmpdir, download_parts={part})
    download_job.status = part.status
    job = ModelInstallJob(
        id=4588,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADING,
    )
    _stage(job, tmpdir)
    job._staging.claim()
    job._multifile_job = download_job
    job.download_parts = download_job.download_parts
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    monkeypatch.setattr(installer._download_queue, "pause_job", MagicMock())

    try:
        installer.pause_job(job)

        if terminal_callback == "complete":
            installer._download_complete_callback(download_job)
        else:
            installer._download_error_callback(download_job, RuntimeError("late download error"))

        assert job.paused
        assert download_job.id not in installer._download_cache
        assert installer._install_queue.qsize() == 0
        assert tmpdir.exists()
        assert payload.exists()
        assert _claim_path(tmpdir).exists()
        marker = _read_marker(tmpdir)
        assert marker is not None and marker["status"] == InstallStatus.PAUSED.value
    finally:
        installer.cancel_job(job)
        _claim_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_stop_preserves_staging_claim_until_download_pause_callback(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}shutdown-active-download"
    tmpdir.mkdir()
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=tmpdir / "model.safetensors")
    part.status = DownloadJobStatus.RUNNING
    download_job = MultiFileDownloadJob(id=459, dest=tmpdir, download_parts={part})
    job = ModelInstallJob(
        id=459,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADING,
    )
    _stage(job, tmpdir)
    job._staging.claim()
    job._multifile_job = download_job
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job

    installer.stop()

    try:
        assert job.paused
        assert installer._download_cache.get(download_job.id) is job
        assert _claim_path(tmpdir).exists()

        installer._download_cancelled_callback(download_job)

        assert download_job.id not in installer._download_cache
        assert not _claim_path(tmpdir).exists()
        assert tmpdir.exists()
    finally:
        _claim_path(tmpdir).unlink(missing_ok=True)


def test_shutdown_releases_claim_when_pause_callback_already_ran(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}shutdown-after-pause-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=tmpdir / "model.safetensors")
    part.status = DownloadJobStatus.PAUSED
    download_job = MultiFileDownloadJob(id=4581, dest=tmpdir, download_parts={part})
    download_job.status = DownloadJobStatus.PAUSED
    job = ModelInstallJob(
        id=4581,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.PAUSED,
    )
    _stage(job, tmpdir)
    job._staging.claim()
    job._multifile_job = download_job
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job

    try:
        # The pause callback proves the queue has stopped touching staging, but a live user-paused job
        # intentionally keeps its claim until shutdown.
        installer._download_cancelled_callback(download_job)
        assert download_job.id not in installer._download_cache
        assert _claim_path(tmpdir).exists()

        installer.stop()

        assert not _claim_path(tmpdir).exists()
        marker = json.loads((tmpdir / INSTALL_MARKER_FILENAME).read_text(encoding="utf-8"))
        assert marker["status"] == InstallStatus.PAUSED.value
        assert tmpdir.exists()
    finally:
        _claim_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_cancel_after_pause_callback_cleans_staging(mm2_installer: ModelInstallServiceBase, tmp_path: Path) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}cancel-after-pause"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors.downloading"
    payload.write_bytes(b"partial model")
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=tmpdir / "model.safetensors")
    part.status = DownloadJobStatus.PAUSED
    download_job = MultiFileDownloadJob(id=4583, dest=tmpdir, download_parts={part})
    download_job.status = DownloadJobStatus.PAUSED
    job = ModelInstallJob(
        id=4583,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.PAUSED,
    )
    _stage(job, tmpdir)
    job._staging.claim()
    job._multifile_job = download_job
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job

    try:
        installer._download_cancelled_callback(download_job)
        assert _claim_path(tmpdir).exists()

        installer.cancel_job(job)

        assert job.cancelled
        assert not tmpdir.exists()
        assert not _claim_path(tmpdir).exists()
    finally:
        _claim_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize("operation", ["resume", "restart_failed", "restart_file"])
def test_remote_download_replacement_waits_for_previous_callback(
    operation: str,
    mm2_installer: ModelInstallServiceBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    target = tmp_path / "model.safetensors"
    partial = target.with_name(target.name + ".downloading")
    partial.write_bytes(b"partial data")
    part = DownloadJob(source=source.url, dest=target)
    part.bytes = len(b"partial data")
    part.download_path = target
    part.resume_required = True
    old_download = MultiFileDownloadJob(id=4582, dest=tmp_path, download_parts={part})
    old_download.status = DownloadJobStatus.RUNNING
    job = ModelInstallJob(
        id=4582,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmp_path,
        status=InstallStatus.PAUSED if operation == "resume" else InstallStatus.ERROR,
    )
    _stage(job, tmp_path)
    job._multifile_job = old_download
    job.download_parts = old_download.download_parts
    installer._download_cache[old_download.id] = job
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=64)
    monkeypatch.setattr(installer, "_remote_files_from_source", lambda _: ([remote_file], None))
    monkeypatch.setattr(installer._download_queue, "submit_multifile_download", MagicMock())

    with pytest.raises(InstallDownloadConflictError, match="previous download"):
        if operation == "resume":
            installer.resume_job(job)
        elif operation == "restart_failed":
            installer.restart_failed(job)
        else:
            installer.restart_file(job, str(source.url))

    assert job._multifile_job is old_download
    assert installer._download_cache[old_download.id] is job
    assert partial.read_bytes() == b"partial data"


def test_concurrent_restart_requests_are_serialized(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    job = ModelInstallJob(
        id=4584,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmp_path,
        status=InstallStatus.ERROR,
    )
    _stage(job, tmp_path)
    job._staging.claim()
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=64)
    metadata_started = threading.Event()
    release_metadata = threading.Event()

    def blocked_metadata(_source: URLModelSource) -> tuple[list[RemoteModelFile], None]:
        metadata_started.set()
        assert release_metadata.wait(timeout=5)
        return [remote_file], None

    monkeypatch.setattr(installer, "_remote_files_from_source", blocked_metadata)
    submit_download = MagicMock()
    monkeypatch.setattr(installer._download_queue, "submit_multifile_download", submit_download)
    errors: list[BaseException] = []

    def restart() -> None:
        try:
            installer.restart_file(job, str(source.url))
        except BaseException as error:
            errors.append(error)

    first_request = threading.Thread(target=restart)
    first_request.start()
    try:
        assert metadata_started.wait(timeout=5)
        with pytest.raises(InstallDownloadConflictError, match="previous download"):
            installer.restart_file(job, str(source.url))
    finally:
        release_metadata.set()
        first_request.join(timeout=5)

    assert not first_request.is_alive()
    assert errors == []
    submit_download.assert_called_once()
    job._staging.release_claim()
    (tmp_path / INSTALL_MARKER_FILENAME).unlink(missing_ok=True)


def test_concurrent_resume_requests_are_serialized(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    job = ModelInstallJob(
        id=4586,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmp_path,
        status=InstallStatus.PAUSED,
    )
    _stage(job, tmp_path)
    job._staging.claim()
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=64)
    metadata_started = threading.Event()
    release_metadata = threading.Event()

    def blocked_metadata(_source: URLModelSource) -> tuple[list[RemoteModelFile], None]:
        metadata_started.set()
        assert release_metadata.wait(timeout=5)
        return [remote_file], None

    monkeypatch.setattr(installer, "_remote_files_from_source", blocked_metadata)
    submit_download = MagicMock()
    monkeypatch.setattr(installer._download_queue, "submit_multifile_download", submit_download)
    errors: list[BaseException] = []

    def resume() -> None:
        try:
            installer.resume_job(job)
        except BaseException as error:
            errors.append(error)

    first_request = threading.Thread(target=resume)
    first_request.start()
    try:
        assert metadata_started.wait(timeout=5)
        with pytest.raises(InstallDownloadConflictError, match="previous download"):
            installer.resume_job(job)
    finally:
        release_metadata.set()
        first_request.join(timeout=5)

    assert not first_request.is_alive()
    assert errors == []
    submit_download.assert_called_once()
    installer._download_cache.pop(job._multifile_job.id)
    job._staging.release_claim()
    (tmp_path / INSTALL_MARKER_FILENAME).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("operation", "initial_status"),
    [
        ("resume", InstallStatus.PAUSED),
        ("restart_failed", InstallStatus.ERROR),
        ("restart_file", InstallStatus.ERROR),
    ],
)
def test_remote_download_preparation_failure_preserves_retryable_state(
    operation: str,
    initial_status: InstallStatus,
    mm2_installer: ModelInstallServiceBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=tmp_path / "model.safetensors")
    part.resume_required = True
    job = ModelInstallJob(
        id=4589,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmp_path,
        status=initial_status,
        download_parts={part},
    )
    _stage(job, tmp_path)
    job._multifile_job = MultiFileDownloadJob(id=4589, dest=tmp_path, download_parts={part})
    monkeypatch.setattr(
        installer,
        "_remote_files_from_source",
        MagicMock(side_effect=RuntimeError("transient metadata failure")),
    )

    with pytest.raises(RuntimeError, match="transient metadata failure"):
        if operation == "resume":
            installer.resume_job(job)
        elif operation == "restart_failed":
            installer.restart_failed(job)
        else:
            installer.restart_file(job, str(source.url))

    assert job.status == initial_status
    assert job.id not in installer._remote_download_operations
    assert not installer._download_cache


@pytest.mark.parametrize(
    ("operation", "initial_status"),
    [("resume", InstallStatus.PAUSED), ("restart_failed", InstallStatus.ERROR)],
)
def test_remote_download_retry_with_no_matching_files_preserves_retryable_state(
    operation: str,
    initial_status: InstallStatus,
    mm2_installer: ModelInstallServiceBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=tmp_path / "model.safetensors")
    part.resume_required = True
    job = ModelInstallJob(
        id=4590,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmp_path,
        status=initial_status,
        download_parts={part},
    )
    _stage(job, tmp_path)
    job._multifile_job = MultiFileDownloadJob(id=4590, dest=tmp_path, download_parts={part})
    monkeypatch.setattr(installer, "_remote_files_from_source", lambda _: ([], None))

    with pytest.raises(RuntimeError, match="No remote files are available to download"):
        if operation == "resume":
            installer.resume_job(job)
        else:
            installer.restart_failed(job)

    assert job.status == initial_status
    assert job.id not in installer._remote_download_operations
    assert not installer._download_cache


@pytest.mark.parametrize("install_worker_started", [False, True])
def test_pause_conflicts_after_download_install_handoff(
    install_worker_started: bool,
    mm2_installer: ModelInstallServiceBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}pause-install-handoff-{install_worker_started}"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors"
    payload.write_bytes(b"downloaded model")
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=payload)
    part.status = DownloadJobStatus.COMPLETED
    download_job = MultiFileDownloadJob(id=4587, dest=tmpdir, download_parts={part})
    download_job.status = DownloadJobStatus.COMPLETED
    job = ModelInstallJob(
        id=4587,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADING,
    )
    _stage(job, tmpdir)
    job._staging.claim()
    job._multifile_job = download_job
    job.download_parts = download_job.download_parts
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    queue_install_job = MagicMock(return_value=True)
    monkeypatch.setattr(installer, "_queue_install_job_locked", queue_install_job)

    try:
        installer._download_complete_callback(download_job)
        expected_status = InstallStatus.DOWNLOADS_DONE
        if install_worker_started:
            job.status = InstallStatus.WAITING
            job._install_phase = "preflight"
            installer._active_install_job = job
            expected_status = InstallStatus.WAITING

        with pytest.raises(InstallDownloadConflictError, match="cannot be paused"):
            installer.pause_job(job)

        assert job.status == expected_status
        queue_install_job.assert_called_once_with(job)
        assert tmpdir.exists()
        assert _claim_path(tmpdir).exists()
    finally:
        installer._active_install_job = None
        job._install_phase = None
        job.status = InstallStatus.DOWNLOADS_DONE
        installer.cancel_job(job)
        _claim_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize("action", ["cancel", "pause"])
def test_install_state_changes_conflict_during_remote_download_handoff(
    action: str,
    mm2_installer: ModelInstallServiceBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}handoff-{action}"
    tmpdir.mkdir()
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=tmpdir / "model.safetensors")
    part.status = DownloadJobStatus.PAUSED
    download_job = MultiFileDownloadJob(id=4585, dest=tmpdir, download_parts={part})
    download_job.status = DownloadJobStatus.PAUSED
    job = ModelInstallJob(
        id=4585,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.PAUSED,
    )
    _stage(job, tmpdir)
    job._staging.claim()
    job._multifile_job = download_job
    job.download_parts = download_job.download_parts
    installer._install_jobs.append(job)
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=64)
    metadata_started = threading.Event()
    release_metadata = threading.Event()

    def blocked_metadata(_source: URLModelSource) -> tuple[list[RemoteModelFile], None]:
        metadata_started.set()
        assert release_metadata.wait(timeout=5)
        return [remote_file], None

    monkeypatch.setattr(installer, "_remote_files_from_source", blocked_metadata)
    submit_download = MagicMock()
    monkeypatch.setattr(installer._download_queue, "submit_multifile_download", submit_download)
    operation_errors: list[BaseException] = []

    def resume() -> None:
        try:
            installer.resume_job(job)
        except BaseException as error:
            operation_errors.append(error)

    operation_thread = threading.Thread(target=resume)
    operation_thread.start()
    try:
        assert metadata_started.wait(timeout=5)
        with pytest.raises(InstallDownloadConflictError, match="being prepared"):
            if action == "cancel":
                installer.cancel_job(job)
            else:
                installer.pause_job(job)
        assert tmpdir.exists()
        assert _claim_path(tmpdir).exists()
    finally:
        release_metadata.set()
        operation_thread.join(timeout=5)

    assert not operation_thread.is_alive()
    assert operation_errors == []
    submit_download.assert_called_once()
    assert not job.cancelled
    installer._download_cache.pop(job._multifile_job.id)
    job._staging.release_claim()
    (tmpdir / INSTALL_MARKER_FILENAME).unlink(missing_ok=True)
    _claim_path(tmpdir).unlink(missing_ok=True)


def test_shutdown_waits_for_remote_download_handoff_and_preserves_pause(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}shutdown-handoff-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    part = DownloadJob(source=source.url, dest=tmpdir / "model.safetensors")
    part.status = DownloadJobStatus.PAUSED
    download_job = MultiFileDownloadJob(id=4586, dest=tmpdir, download_parts={part})
    download_job.status = DownloadJobStatus.PAUSED
    job = ModelInstallJob(
        id=4586,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.PAUSED,
    )
    _stage(job, tmpdir)
    job._staging.claim()
    job._multifile_job = download_job
    job.download_parts = download_job.download_parts
    installer._install_jobs.append(job)
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=64)
    metadata_started = threading.Event()
    release_metadata = threading.Event()
    stop_finished = threading.Event()

    def blocked_metadata(_source: URLModelSource) -> tuple[list[RemoteModelFile], None]:
        metadata_started.set()
        assert release_metadata.wait(timeout=5)
        return [remote_file], None

    monkeypatch.setattr(installer, "_remote_files_from_source", blocked_metadata)
    submit_download = MagicMock()
    monkeypatch.setattr(installer._download_queue, "submit_multifile_download", submit_download)
    operation_errors: list[BaseException] = []

    def resume() -> None:
        try:
            installer.resume_job(job)
        except BaseException as error:
            operation_errors.append(error)

    operation_thread = threading.Thread(target=resume)
    stop_thread = threading.Thread(target=lambda: (installer.stop(), stop_finished.set()))
    operation_thread.start()
    try:
        assert metadata_started.wait(timeout=5)
        stop_thread.start()
        assert installer._stop_event.wait(timeout=5)
    finally:
        release_metadata.set()
        operation_thread.join(timeout=5)
        if stop_thread.ident is not None:
            stop_thread.join(timeout=5)

    assert not operation_thread.is_alive()
    assert not stop_thread.is_alive()
    assert stop_finished.is_set()
    assert len(operation_errors) == 1
    assert isinstance(operation_errors[0], InstallDownloadConflictError)
    submit_download.assert_not_called()
    assert job.paused
    assert tmpdir.exists()
    assert not _claim_path(tmpdir).exists()
    marker = json.loads((tmpdir / INSTALL_MARKER_FILENAME).read_text(encoding="utf-8"))
    assert marker["status"] == InstallStatus.PAUSED.value
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_remote_staging_is_claimed_before_download_submission(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}claim-before-download"
    tmpdir.mkdir()
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    payload = tmpdir / "model.safetensors"
    part = DownloadJob(source=source.url, dest=payload)
    download_job = MultiFileDownloadJob(id=457, dest=tmpdir, download_parts={part})
    job = ModelInstallJob(id=457, source=source, config_in=ModelRecordChanges(), local_path=tmpdir)
    _stage(job, tmpdir)
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=8)
    monkeypatch.setattr(installer, "_multifile_download", lambda **_: download_job)

    def assert_claimed_before_submit(submitted_job: MultiFileDownloadJob) -> None:
        assert submitted_job is download_job
        assert _claim_path(tmpdir).exists()
        assert job._staging.claimed

    monkeypatch.setattr(installer._download_queue, "submit_multifile_download", assert_claimed_before_submit)

    installer._enqueue_remote_download(
        job=job,
        source=source,
        remote_files=[remote_file],
        metadata=None,
    )

    assert _claim_path(tmpdir).exists()
    # The fixture does not run a download worker for this synthetic job.
    installer._download_cache.pop(download_job.id)
    job._staging.release(preserve_recovery=False, completed=False)


def test_install_worker_preserves_remote_staging_when_claim_conflicts(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}claim-conflict"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors"
    payload.write_bytes(b"owned by another install")
    HeldClaim(tmpdir)

    job = ModelInstallJob(
        id=458,
        source=URLModelSource(url=Url("https://example.com/model.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADS_DONE,
    )
    _stage(job, tmpdir)
    installer._install_queue.put(job)

    def attempt_transfer(queued_job: ModelInstallJob) -> None:
        installer._begin_install_transfer(queued_job)

    try:
        monkeypatch.setattr(installer, "_register_or_install", attempt_transfer)
        installer._install_queue.put(job)
        installer._install_queue.join()

        assert job.errored
        assert payload.read_bytes() == b"owned by another install"
        assert _claim_path(tmpdir).exists()
    finally:
        # This sidecar belongs to the simulated competing operation.
        _claim_path(tmpdir).unlink(missing_ok=True)


def test_resume_reports_restart_from_scratch_on_fresh_parts(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming replaces download_parts; a vanished partial file must still be reported on the new parts."""
    assert isinstance(mm2_installer, ModelInstallService)
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    old_part = DownloadJob(source=source.url, dest=tmp_path / "model.safetensors")
    old_part.bytes = 4
    old_part.download_path = tmp_path / "model.safetensors"  # no .downloading file on disk
    job = ModelInstallJob(id=4242, source=source, config_in=ModelRecordChanges(), local_path=tmp_path)
    _stage(job, tmp_path)
    job.download_parts = {old_part}
    job.status = InstallStatus.PAUSED
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=8)
    monkeypatch.setattr(mm2_installer, "_remote_files_from_source", lambda _: ([remote_file], None))
    submit_multifile_download = MagicMock()
    monkeypatch.setattr(mm2_installer._download_queue, "submit_multifile_download", submit_multifile_download)

    mm2_installer.resume_job(job)

    submit_multifile_download.assert_called_once()
    assert old_part not in job.download_parts
    parts = job.model_dump(mode="json")["download_parts"]
    assert len(parts) == 1
    assert parts[0]["resume_from_scratch"] is True
    assert "Partial file missing" in parts[0]["resume_message"]


@pytest.mark.parametrize("etag", ['"v1"', None], ids=["validated", "unvalidated"])
def test_resume_reports_restart_from_scratch_only_for_a_partial_it_cannot_validate(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, etag: str | None
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    old_part = DownloadJob(source=source.url, dest=tmp_path / "model.safetensors", etag=etag)
    old_part.bytes = 4
    old_part.download_path = tmp_path / "model.safetensors"
    partial = tmp_path / "model.safetensors.downloading"
    partial.write_bytes(b"1234")
    job = ModelInstallJob(id=4243, source=source, config_in=ModelRecordChanges(), local_path=tmp_path)
    _stage(job, tmp_path)
    job.download_parts = {old_part}
    job.status = InstallStatus.PAUSED
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=8)
    monkeypatch.setattr(mm2_installer, "_remote_files_from_source", lambda _: ([remote_file], None))
    monkeypatch.setattr(mm2_installer._download_queue, "submit_multifile_download", MagicMock())

    mm2_installer.resume_job(job)

    parts = job.model_dump(mode="json")["download_parts"]
    assert len(parts) == 1
    if etag is not None:
        assert parts[0]["resume_from_scratch"] is False
        assert parts[0]["etag"] == etag
        assert partial.read_bytes() == b"1234"
    else:
        assert parts[0]["resume_from_scratch"] is True
        assert parts[0]["resume_message"]
        assert not partial.exists()


def test_404_download(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://test.com/missing_model.safetensors"))
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status == InstallStatus.ERROR
    assert job.errored
    assert job.error_type == "HTTPError"
    assert job.error
    assert "NOT FOUND" in job.error
    assert job.error_traceback is not None
    assert job.error_traceback.startswith("Traceback")
    bus = mm2_installer.event_bus
    assert bus is not None
    assert hasattr(bus, "events")  # the dummyeventservice has this
    event_types = [type(x) for x in bus.events]
    assert ModelInstallErrorEvent in event_types


def test_other_error_during_install(
    monkeypatch: pytest.MonkeyPatch, mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(
        "invokeai.app.services.model_install.model_install_default.ModelInstallService._register_or_install",
        raise_runtime_error,
    )
    source = LocalModelSource(path=Path("tests/data/embedding/test_embedding.safetensors"))
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status == InstallStatus.ERROR
    assert job.errored
    assert job.error_type == "RuntimeError"
    assert job.error == "Test error"


@pytest.mark.parametrize(
    "model_params",
    [
        # SDXL, Lora
        {
            "repo_id": "InvokeAI-test/textual_inversion_tests::learned_embeds-steps-1000.safetensors",
            "name": "test_lora",
            "type": "embedding",
        },
        # SDXL, Lora - incorrect type
        {
            "repo_id": "InvokeAI-test/textual_inversion_tests::learned_embeds-steps-1000.safetensors",
            "name": "test_lora",
            "type": "lora",
        },
    ],
)
@pytest.mark.timeout(timeout=10, method="thread")
def test_heuristic_import_with_type(mm2_installer: ModelInstallServiceBase, model_params: Dict[str, str]):
    """Test whether or not type is respected on configs when passed to heuristic import."""
    assert "name" in model_params and "type" in model_params
    config1: Dict[str, Any] = {
        "name": f"{model_params['name']}_1",
        "type": model_params["type"],
        "hash": "placeholder1",
    }
    config2: Dict[str, Any] = {
        "name": f"{model_params['name']}_2",
        "type": ModelType(model_params["type"]),
        "hash": "placeholder2",
    }
    assert "repo_id" in model_params
    install_job1 = mm2_installer.heuristic_import(source=model_params["repo_id"], config=config1)
    mm2_installer.wait_for_job(install_job1, timeout=10)
    if model_params["type"] != "embedding":
        assert install_job1.errored
        assert install_job1.error_type == "InvalidModelConfigException"
        return
    assert install_job1.complete
    assert install_job1.config_out if model_params["type"] == "embedding" else not install_job1.config_out

    install_job2 = mm2_installer.heuristic_import(source=model_params["repo_id"], config=config2)
    mm2_installer.wait_for_job(install_job2, timeout=10)
    assert install_job2.complete
    assert install_job2.config_out if model_params["type"] == "embedding" else not install_job2.config_out


def test_multifile_download_layout_with_explicit_files(mm2_installer: ModelInstallServiceBase, tmp_path: Path) -> None:
    """Explicit file entries in a multi-subfolder source keep their repo-relative paths: the root
    pipeline index lands at the model root and transformer/config.json stays inside transformer/
    (naive relative_to() matching would flatten it to the root), while plain subfolder entries keep
    the pre-existing one-directory-per-subfolder layout."""
    from invokeai.backend.model_manager.metadata.metadata_base import RemoteModelFile

    remote_files = [
        RemoteModelFile(url="https://example.com/root_index", path=Path("MiniMax-H3/modular_model_index.json")),
        RemoteModelFile(url="https://example.com/transformer_config", path=Path("MiniMax-H3/transformer/config.json")),
        RemoteModelFile(url="https://example.com/vae_config", path=Path("MiniMax-H3/vae/config.json")),
        RemoteModelFile(
            url="https://example.com/vae_weights", path=Path("MiniMax-H3/vae/diffusion_pytorch_model.safetensors")
        ),
    ]
    job = mm2_installer._multifile_download(  # pyright: ignore[reportAttributeAccessIssue]
        remote_files=remote_files,
        dest=tmp_path,
        subfolders=[Path("modular_model_index.json"), Path("transformer/config.json"), Path("vae")],
        submit_job=False,
    )
    top = Path("MiniMax-H3_modular_model_index_config_vae")
    assert {part.dest.relative_to(tmp_path.resolve()) for part in job.download_parts} == {
        top / "modular_model_index.json",
        top / "transformer" / "config.json",
        top / "vae" / "config.json",
        top / "vae" / "diffusion_pytorch_model.safetensors",
    }


def test_restore_keeps_a_legacy_marker_whose_key_predates_key_validation(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue,
    mm2_session,
) -> None:
    """A marker written before install keys were checked can carry a key that is not a plain filename. It must
    still be restored: dropping it strands the partial download forever, because no job is created to clean up
    the tmpdir and startup cleanup keeps any tmpdir whose marker is readable and non-terminal.
    Containment does not depend on this - `install_path()` checks the key at the join and errors the job."""
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)

    tmpdirs: list[Path] = []
    try:
        for repo_id, config_in in [
            ("stabilityai/legacy-key", ModelRecordChanges.model_construct(key="../escaped")),
            ("stabilityai/ordinary", ModelRecordChanges()),
        ]:
            tmpdir = mm2_app_config.models_path / f"tmpinstall_legacy_{uuid.uuid4().hex}"
            tmpdir.mkdir(parents=True, exist_ok=True)
            tmpdirs.append(tmpdir)
            job = ModelInstallJob(
                id=99999,
                source=HFModelSource(repo_id=repo_id, variant=ModelRepoVariant.Default),
                config_in=config_in,
                local_path=tmpdir,
            )
            _stage(job, tmpdir)
            job.status = InstallStatus.PAUSED
            _write_v1_marker(job, InstallStatus.PAUSED)

        restored_installer = ModelInstallService(
            app_config=mm2_app_config,
            record_store=mm2_installer.record_store,
            download_queue=mm2_download_queue,
            session=mm2_session,
        )
        restored_installer._restore_incomplete_installs()

        restored = {str(job.source) for job in restored_installer.list_jobs()}
        assert restored == {"stabilityai/legacy-key", "stabilityai/ordinary"}
    finally:
        for tmpdir in tmpdirs:
            _claim_path(tmpdir).unlink(missing_ok=True)
            shutil.rmtree(tmpdir, ignore_errors=True)


def test_startup_restores_staging_whose_claim_a_crashed_process_left_behind(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue: Any,
    mm2_session: Any,
) -> None:
    """A process killed mid-download never released its claim; startup must prove it stale, then resume the install."""
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}crashed-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    (tmpdir / "weights.safetensors.downloading").write_bytes(b"partial download")
    source = URLModelSource(url=Url("https://example.com/crashed.safetensors"))
    job = ModelInstallJob(id=99996, source=source, config_in=ModelRecordChanges(), local_path=tmpdir)
    _stage(job, tmpdir)
    _write_v1_marker(job, InstallStatus.PAUSED)
    _claim_path(tmpdir).write_text("2147483647 1.0\n", encoding="ascii")

    restarted_installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_installer.record_store,
        download_queue=mm2_download_queue,
        session=mm2_session,
    )
    restarted_installer.start()
    try:
        assert restarted_installer._wait_for_restore_complete(timeout=10)
        [restored] = restarted_installer.get_job_by_source(source)
        assert restored.status == InstallStatus.PAUSED
        assert _claim_path(tmpdir).read_text(encoding="ascii").split()[0] == str(os.getpid())
    finally:
        restarted_installer.stop()
    assert (tmpdir / "weights.safetensors.downloading").read_bytes() == b"partial download"
    shutil.rmtree(tmpdir)


def test_restore_skips_an_unparseable_marker_without_abandoning_the_rest(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue,
    mm2_session,
) -> None:
    """A marker whose stored config no longer validates must skip that one marker, not raise out of the loop and
    leave every interrupted install after it unrestored."""
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)

    tmpdirs: list[Path] = []
    try:
        bad_tmpdir = mm2_app_config.models_path / f"tmpinstall_bad_{uuid.uuid4().hex}"
        bad_tmpdir.mkdir(parents=True, exist_ok=True)
        tmpdirs.append(bad_tmpdir)
        bad_payload = bad_tmpdir / "weights.safetensors.downloading"
        bad_payload.write_bytes(b"partial download")
        bad_marker = json.dumps(
            {
                "version": INSTALL_MARKER_VERSION,
                "source": "stabilityai/unparseable",
                "config_in": {"base": "not-a-real-base"},
                "status": InstallStatus.PAUSED.value,
            }
        )
        (bad_tmpdir / INSTALL_MARKER_FILENAME).write_text(bad_marker)

        good_tmpdir = mm2_app_config.models_path / f"tmpinstall_good_{uuid.uuid4().hex}"
        good_tmpdir.mkdir(parents=True, exist_ok=True)
        tmpdirs.append(good_tmpdir)
        job = ModelInstallJob(
            id=99998,
            source=HFModelSource(repo_id="stabilityai/ordinary", variant=ModelRepoVariant.Default),
            config_in=ModelRecordChanges(),
            local_path=good_tmpdir,
        )
        _stage(job, good_tmpdir)
        job.status = InstallStatus.PAUSED
        _write_v1_marker(job, InstallStatus.PAUSED)

        restored_installer = ModelInstallService(
            app_config=mm2_app_config,
            record_store=mm2_installer.record_store,
            download_queue=mm2_download_queue,
            session=mm2_session,
        )
        restored_installer._restore_incomplete_installs()

        assert [str(job.source) for job in restored_installer.list_jobs()] == ["stabilityai/ordinary"]
        assert not _claim_path(bad_tmpdir).exists(), "a skipped marker must not be claimed"
        assert (bad_tmpdir / INSTALL_MARKER_FILENAME).read_text() == bad_marker
        assert bad_payload.read_bytes() == b"partial download"
    finally:
        for tmpdir in tmpdirs:
            _claim_path(tmpdir).unlink(missing_ok=True)
            shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize("invalid_status", ["not-a-valid-status", ["not-a-valid-status"]])
def test_restore_skips_a_marker_with_an_invalid_status_without_claiming_it_or_abandoning_the_rest(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue,
    mm2_session,
    invalid_status: str | list[str],
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    assert mm2_installer._wait_for_restore_complete(timeout=10)

    bad_tmpdir = mm2_app_config.models_path / f"tmpinstall_bad_status_{uuid.uuid4().hex}"
    good_tmpdir = mm2_app_config.models_path / f"tmpinstall_good_{uuid.uuid4().hex}"
    bad_tmpdir.mkdir(parents=True, exist_ok=True)
    good_tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        (bad_tmpdir / INSTALL_MARKER_FILENAME).write_text(
            json.dumps(
                {
                    "version": INSTALL_MARKER_VERSION,
                    "source": "stabilityai/invalid-status",
                    "config_in": {},
                    "status": invalid_status,
                }
            )
        )

        job = ModelInstallJob(
            id=99997,
            source=HFModelSource(repo_id="stabilityai/ordinary", variant=ModelRepoVariant.Default),
            config_in=ModelRecordChanges(),
            local_path=good_tmpdir,
        )
        _stage(job, good_tmpdir)
        job.status = InstallStatus.PAUSED
        _write_v1_marker(job, InstallStatus.PAUSED)

        restored_installer = ModelInstallService(
            app_config=mm2_app_config,
            record_store=mm2_installer.record_store,
            download_queue=mm2_download_queue,
            session=mm2_session,
        )
        restored_installer._restore_incomplete_installs()

        assert [str(job.source) for job in restored_installer.list_jobs()] == ["stabilityai/ordinary"]
        assert not _claim_path(bad_tmpdir).exists()
    finally:
        for tmpdir in (bad_tmpdir, good_tmpdir):
            _claim_path(tmpdir).unlink(missing_ok=True)
            shutil.rmtree(tmpdir, ignore_errors=True)


def test_installed_pipeline_components_resolve_inside_the_installed_model(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    """Pipelines are identified where they are staged (a `tmpinstall_*` folder for downloads) and
    then moved into the models dir; the stored record must lead loading to the moved components."""
    from invokeai.backend.model_manager.load.load_default import resolve_submodel_path
    from invokeai.backend.model_manager.taxonomy import SubModelType
    from tests.backend.model_manager.configs.test_sdnq_flux1_submodel_discovery import _make_flux1_pipeline

    staging = mm2_app_config.models_path / "tmpinstall_test" / "pipeline"
    _make_flux1_pipeline(staging, component_keys={"text_encoder": "clip_encoder"})

    record = mm2_installer.record_store.get_model(mm2_installer.install_path(staging))
    # Stored relative to the model, so nothing in the record points back at the staging folder.
    assert record.submodels is not None
    assert record.submodels[SubModelType.TextEncoder].path_or_prefix == "clip_encoder"

    # The loader makes `path` absolute against the models dir before resolving components.
    record.path = (mm2_app_config.models_path / record.path).resolve().as_posix()
    text_encoder = resolve_submodel_path(record, SubModelType.TextEncoder, Path(record.path) / "text_encoder")
    assert text_encoder.parent == Path(record.path)
    assert (text_encoder / "config.json").is_file()
