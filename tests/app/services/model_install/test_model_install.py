"""
Test the model installer
"""

import errno
import gc
import json
import platform
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from pydantic_core import Url

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
from invokeai.app.services.model_install.model_install_common import (
    INSTALL_RECOVERY_SENTINEL,
    InstallCancellationConflictError,
    InstallDownloadConflictError,
    InstallRecoveryRequiredError,
    InstallStatus,
    InvalidModelConfigException,
    LocalModelSource,
    ModelInstallJob,
    URLModelSource,
    active_install_sentinel_path,
    create_active_install_sentinel,
    has_active_install_sentinel,
)
from invokeai.app.services.model_install.model_install_default import (
    INSTALL_MARKER_FILENAME,
    INSTALL_MARKER_VERSION,
    TMPDIR_PREFIX,
)
from invokeai.app.services.model_records import ModelRecordChanges, ModelRecordServiceSQL, UnknownModelException
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig
from invokeai.backend.model_manager.metadata import RemoteModelFile
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelSourceType,
    ModelType,
)
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403
from tests.test_nodes import TestEventService

OS = platform.uname().system


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


def test_retain_recovery_destination_does_not_log_when_sentinel_already_exists(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    dest_dir = tmp_path / "model"
    dest_dir.mkdir()
    installer._write_recovery_sentinel(dest_dir)

    with caplog.at_level("ERROR"):
        installer._retain_recovery_destination(dest_dir)

    assert installer._has_recovery_sentinel(dest_dir)
    assert not any("Failed to persist destination recovery sentinel" in record.message for record in caplog.records)


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

    monkeypatch.setattr(model_install_default, "move", copy_then_fail_unlink)

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
    from invokeai.app.services.model_install.model_install_common import (
        has_active_install_sentinel,
        has_recovery_sentinel,
    )
    from invokeai.app.services.orphaned_models import OrphanedModelsService

    orphan_service = OrphanedModelsService(config=mm2_app_config, db=mm2_installer.record_store._db)
    observations: list[tuple[str, bool, set[str]]] = []
    real_move = shutil.move

    def move_then_scan(src: Path, dst: Path):
        result = real_move(src, dst)
        destination_root = dst.parent
        assert has_active_install_sentinel(destination_root)
        orphan_keys = {orphan.path for orphan in orphan_service.find_orphaned_models()}
        observations.append((destination_root.name, has_recovery_sentinel(destination_root), orphan_keys))
        return result

    monkeypatch.setattr(model_install_default, "move", move_then_scan)

    mm2_installer.install_path(embedding_file)

    assert len(observations) == 1
    destination_key, has_sentinel, orphan_keys = observations[0]
    assert has_sentinel
    assert destination_key not in orphan_keys
    assert not has_recovery_sentinel(mm2_app_config.models_path / destination_key)


def test_managed_local_source_is_hidden_from_orphan_cleanup_during_install(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invokeai.app.services.model_install.model_install_common import (
        has_active_install_sentinel,
        has_recovery_sentinel,
    )
    from invokeai.app.services.orphaned_models import OrphanedModelsService

    source_root = mm2_app_config.models_path / f"local-source-{uuid.uuid4().hex}"
    shutil.copytree(diffusers_dir, source_root)
    orphan_service = OrphanedModelsService(config=mm2_app_config, db=mm2_installer.record_store._db)
    observations: list[tuple[bool, set[str], str]] = []
    real_probe = mm2_installer._probe
    real_move_with_retries = mm2_installer._move_with_retries

    def observe_source_during_operation(path: Path, config: ModelRecordChanges):
        assert path == source_root
        protected = has_active_install_sentinel(source_root)
        orphan_keys = {orphan.path for orphan in orphan_service.find_orphaned_models()}
        delete_result = orphan_service.delete_orphaned_models([source_root.name])[source_root.name]
        observations.append((protected, orphan_keys, delete_result))
        return real_probe(path, config)

    def observe_source_during_transfer(src: Path, dst: Path) -> None:
        assert has_active_install_sentinel(source_root)
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
    assert not has_active_install_sentinel(source_root)
    assert not has_recovery_sentinel(source_root)


def test_orphan_delete_claim_prevents_a_local_install_from_starting(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invokeai.app.services.model_install.model_install_common import has_active_install_sentinel
    from invokeai.app.services.orphaned_models import OrphanedModelsService, orphaned_models_service

    source_root = mm2_app_config.models_path / f"local-source-{uuid.uuid4().hex}"
    shutil.copytree(diffusers_dir, source_root)
    orphan_service = OrphanedModelsService(config=mm2_app_config, db=mm2_installer.record_store._db)
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
        assert has_active_install_sentinel(source_root)
        with pytest.raises(InstallCancellationConflictError, match="another install or orphan cleanup"):
            mm2_installer.install_path(source_root)
        assert source_root.exists()
    finally:
        release_deletion.set()
        delete_thread.join(timeout=10)

    assert not delete_thread.is_alive()
    assert results[source_root.name] == "deleted"
    assert not source_root.exists()


def test_start_removes_stale_but_preserves_live_install_claims(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    from invokeai.app.services.model_install.model_install_common import (
        active_install_sentinel_path,
        create_active_install_sentinel,
    )

    assert isinstance(mm2_installer, ModelInstallService)
    stale_root = mm2_app_config.models_path / "stale-claim"
    live_root = mm2_app_config.models_path / "live-claim"
    stale_root.mkdir()
    live_root.mkdir()
    stale_claim = active_install_sentinel_path(stale_root)
    stale_claim.write_text("2147483647 1.0\n", encoding="ascii")
    create_active_install_sentinel(live_root)

    mm2_installer._remove_stale_install_source_claims()

    assert not stale_claim.exists()
    assert active_install_sentinel_path(live_root).exists()
    from invokeai.app.services.model_install.model_install_common import delete_active_install_sentinel

    delete_active_install_sentinel(live_root)


def test_startup_cleanup_preserves_claimed_remote_staging(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    from invokeai.app.services.model_install.model_install_common import (
        active_install_sentinel_path,
        create_active_install_sentinel,
    )

    assert isinstance(mm2_installer, ModelInstallService)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}live-download-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors.downloading"
    payload.write_bytes(b"partial download")
    create_active_install_sentinel(tmpdir)

    try:
        mm2_installer._remove_dangling_install_dirs()
        assert payload.read_bytes() == b"partial download"
        assert active_install_sentinel_path(tmpdir).exists()
    finally:
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)


def test_unregistered_destination_remains_protected_after_registration_failure(
    mm2_installer: ModelInstallServiceBase,
    embedding_file: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invokeai.app.services.model_install.model_install_common import has_recovery_sentinel
    from invokeai.app.services.orphaned_models import OrphanedModelsService

    orphan_service = OrphanedModelsService(config=mm2_app_config, db=mm2_installer.record_store._db)

    def fail_recording(_config: Any) -> None:
        raise RuntimeError("simulated model record failure")

    monkeypatch.setattr(mm2_installer.record_store, "add_model", fail_recording)

    with pytest.raises(RuntimeError, match="simulated model record failure"):
        mm2_installer.install_path(embedding_file)

    protected_roots = [
        path for path in mm2_app_config.models_path.iterdir() if path.is_dir() and has_recovery_sentinel(path)
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

    monkeypatch.setattr(model_install_default, "move", flaky_move)

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

    monkeypatch.setattr(model_install_default, "move", fail_first_move)

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

    monkeypatch.setattr(model_install_default, "move", move_with_locked_source)

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

    monkeypatch.setattr(model_install_default, "move", move_with_recreated_source)

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
    assert mm2_installer._has_recovery_sentinel(recovery_dest)


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

    monkeypatch.setattr(model_install_default, "move", partial_copy_then_fail)

    job = installer.import_model(LocalModelSource(path=source_path, inplace=False))
    installer.wait_for_job(job, timeout=10)

    assert job.errored
    assert job.error_type == "InstallRecoveryRequiredError"
    recovery_destinations = [
        path
        for path in set(mm2_app_config.models_path.iterdir()) - existing_model_paths
        if path.is_dir() and installer._has_recovery_sentinel(path)
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
    assert installer._has_recovery_sentinel(recovery_destinations[0])
    assert recovery_file.read_bytes() == partial_destination_bytes
    assert source_path.read_bytes() == original_source_bytes


def test_directory_install_rollback_race_preserves_both_source_artifacts(
    mm2_installer: ModelInstallServiceBase,
    diffusers_dir: Path,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = list(diffusers_dir.iterdir())
    locked_item = items[-1]
    first_item = next(path for path in items[:-1] if path.is_file())
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

    monkeypatch.setattr(model_install_default, "move", fail_later_move)
    monkeypatch.setattr(ModelInstallService, "_rename_noreplace", staticmethod(recreate_after_absence_check))

    with pytest.raises(model_install_default.InstallRecoveryRequiredError, match="recovery required"):
        mm2_installer.install_path(diffusers_dir)

    assert first_item.read_bytes() == b"new user bytes"
    recovery_dirs = [path for path in mm2_app_config.models_path.iterdir() if path.is_dir() and path.name != "tmp"]
    preserved = [path / first_item.name for path in recovery_dirs if (path / first_item.name).exists()]
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == original_bytes


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

    monkeypatch.setattr(model_install_default, "move", partial_copy_then_fail)

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

    monkeypatch.setattr(model_install_default, "move", fail_after_unowned_destination_artifact)

    with pytest.raises(RuntimeError, match="recovery required") as exc_info:
        mm2_installer.install_path(diffusers_dir)

    recovery_paths = [path for path in set(mm2_app_config.models_path.iterdir()) - existing_paths if path.is_dir()]
    assert len(recovery_paths) == 1
    recovery_dest = recovery_paths[0]
    assert mm2_installer._has_recovery_sentinel(recovery_dest)
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
    job._install_tmpdir = tmpdir
    mm2_installer._write_install_marker(job, status=InstallStatus.DOWNLOADS_DONE)

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
    monkeypatch.setattr(model_install_default, "move", fail_after_partial_transfer)
    real_write_install_marker = mm2_installer._write_install_marker

    def fail_recovery_marker(job_arg, *args, **kwargs):
        if job_arg._recovery_required:
            raise OSError("simulated marker write failure")
        return real_write_install_marker(job_arg, *args, **kwargs)

    real_delete_install_marker = mm2_installer._delete_install_marker

    def assert_recovery_state_before_marker_delete(tmpdir_arg: Path) -> None:
        assert mm2_installer._has_recovery_sentinel(tmpdir_arg) is job._recovery_required
        real_delete_install_marker(tmpdir_arg)

    monkeypatch.setattr(mm2_installer, "_write_install_marker", fail_recovery_marker)
    monkeypatch.setattr(mm2_installer, "_delete_install_marker", assert_recovery_state_before_marker_delete)
    mm2_installer._put_in_queue(job)
    mm2_installer.wait_for_job(job, timeout=10)

    assert job.errored
    assert mm2_installer._has_recovery_sentinel(tmpdir)
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
    assert mm2_installer._has_recovery_sentinel(recovery_dir)
    assert (recovery_dir / failure_file.name).read_bytes() == b"unrecognized partial destination bytes"

    dangling_tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}dangling-{uuid.uuid4().hex}"
    dangling_tmpdir.mkdir()
    (dangling_tmpdir / "ordinary.tmp").write_bytes(b"discard me")

    restarted_installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_installer.record_store,
        download_queue=mm2_download_queue,
        session=mm2_session,
    )
    restarted_installer._remove_dangling_install_dirs()
    restarted_installer._restore_incomplete_installs()

    assert tmpdir.exists()
    assert {path.name: path.read_bytes() for path in real_iterdir(tmpdir) if path.is_file()} == source_files
    assert recovery_dir.exists()
    assert (recovery_dir / failure_file.name).read_bytes() == b"unrecognized partial destination bytes"
    assert dangling_tmpdir.exists() is False
    assert restarted_installer.list_jobs() == []
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
    job._install_tmpdir = tmpdir
    job._recovery_required = True
    mm2_installer._write_recovery_sentinel(tmpdir)

    with pytest.raises(InstallRecoveryRequiredError, match="requires recovery"):
        mm2_installer.cancel_job(job)

    assert not job.cancelled
    assert downloaded_file.read_bytes() == b"recoverable source"
    assert mm2_installer._has_recovery_sentinel(tmpdir)


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
    job._install_tmpdir = tmpdir
    mm2_installer._write_recovery_sentinel(tmpdir)

    with pytest.raises(InstallRecoveryRequiredError, match="requires recovery"):
        mm2_installer.cancel_job(job)

    assert not job.cancelled
    assert downloaded_file.read_bytes() == b"recoverable source"
    assert mm2_installer._has_recovery_sentinel(tmpdir)


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
    job._install_tmpdir = tmpdir
    installer._install_jobs.append(job)
    installer._install_queue.put(job)
    return job


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
        sentinel_during_transfer.append(installer._has_recovery_sentinel(tmpdir))
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


def test_service_stop_does_not_delete_a_job_between_dequeue_and_worker_activation(
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
    assert job.cancelled
    assert not tmpdir.exists()


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
        assert not installer._has_recovery_sentinel(tmpdir)
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
    job._install_tmpdir = tmpdir
    mm2_installer._write_recovery_sentinel(tmpdir)
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
    assert mm2_installer._has_recovery_sentinel(tmpdir)


def test_recovery_required_marker_round_trips_and_preserves_staging_dir(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}recovery-{uuid.uuid4().hex}"
    tmpdir.mkdir()
    downloaded_file = tmpdir / "model.safetensors"
    downloaded_file.write_bytes(b"recoverable source")
    job = ModelInstallJob(
        id=995,
        source=URLModelSource(url=Url("https://example.com/model.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.ERROR,
    )
    job._install_tmpdir = tmpdir
    job._recovery_required = True

    mm2_installer._write_install_marker(job)
    marker = json.loads((tmpdir / INSTALL_MARKER_FILENAME).read_text(encoding="utf-8"))
    mm2_installer._remove_dangling_install_dirs()

    assert marker["recovery_required"] is True
    assert downloaded_file.read_bytes() == b"recoverable source"
    assert tmpdir.exists()


def test_registering_recovery_root_removes_its_sentinel(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    recovery_root = mm2_app_config.models_path / "recovered-model"
    recovery_root.mkdir()
    shutil.copy2(embedding_file, recovery_root / embedding_file.name)
    sentinel = mm2_installer._recovery_sentinel_path(recovery_root)
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
    sentinel = mm2_installer._recovery_sentinel_path(recovery_root)
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
    sentinel = mm2_installer._recovery_sentinel_path(recovery_root)
    sentinel.write_text("preserve until manually registered", encoding="utf-8")

    mm2_installer._register_orphaned_models()

    assert mm2_installer.record_store.all_models() == []
    assert sentinel.exists()
    assert (recovery_root / embedding_file.name).exists()


def test_registered_model_with_temporary_prefix_survives_startup_cleanup(
    mm2_installer: ModelInstallServiceBase,
    embedding_file: Path,
    mm2_app_config: InvokeAIAppConfig,
) -> None:
    model_key = f"{TMPDIR_PREFIX}registered-model"

    installed_key = mm2_installer.install_path(embedding_file, config=ModelRecordChanges(key=model_key))
    installed_path = mm2_app_config.models_path / mm2_installer.record_store.get_model(installed_key).path

    mm2_installer._remove_dangling_install_dirs()

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
    # Exercise a genuine SQLite write rejection without changing permissions on any real user database.
    with store._db.transaction() as cursor:
        cursor.execute("PRAGMA query_only = ON")
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly database"):
            mm2_installer.install_path(diffusers_dir)
    finally:
        with store._db.transaction() as cursor:
            cursor.execute("PRAGMA query_only = OFF")

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
    interrupted_job._install_tmpdir = tmpdir
    installer._write_install_marker(interrupted_job, status=InstallStatus.DOWNLOADING)
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
    interrupted_job._install_tmpdir = tmpdir
    mm2_installer._write_install_marker(interrupted_job, status=InstallStatus.DOWNLOADING)

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
        return ModelInstallJob(
            id=99997,
            source=import_source,
            config_in=config or ModelRecordChanges(),
            local_path=mm2_app_config.models_path,
        )

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
    mm2_installer._install_jobs.extend(jobs)
    existing_iterator = iter(mm2_installer.list_jobs())
    assert next(existing_iterator) is jobs[0]

    mm2_installer.prune_jobs()

    assert list(existing_iterator) == jobs[1:]
    assert all(job not in mm2_installer.list_jobs() for job in jobs)


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
        paused_job._install_tmpdir = tmpdir
        paused_job.status = InstallStatus.PAUSED

        mm2_installer._write_install_marker(paused_job, status=InstallStatus.PAUSED)

        marker = mm2_installer._read_install_marker(tmpdir)
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
        assert has_active_install_sentinel(tmpdir)

        captured: dict[str, str | None] = {}

        def _capture_resume(job: ModelInstallJob, *, operation_reserved: bool = False) -> None:
            assert isinstance(job.source, HFModelSource)
            captured["access_token"] = job.source.access_token

        monkeypatch.setattr(restored_installer, "_resume_remote_download", _capture_resume)
        restored_installer.resume_job(restored_job)
        assert captured["access_token"] == access_token
    finally:
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


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
    staging_job._install_tmpdir = tmpdir
    (tmpdir / "model.safetensors").write_bytes(b"downloaded model")
    mm2_installer._write_install_marker(staging_job, status=InstallStatus.DOWNLOADS_DONE)

    restored_installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_installer.record_store,
        download_queue=mm2_download_queue,
        session=mm2_session,
    )
    queued_jobs: list[ModelInstallJob] = []

    def capture_queued_job(job: ModelInstallJob) -> None:
        assert has_active_install_sentinel(tmpdir)
        queued_jobs.append(job)

    monkeypatch.setattr(restored_installer, "_put_in_queue", capture_queued_job)

    try:
        restored_installer._restore_incomplete_installs()
        assert len(queued_jobs) == 1
        assert queued_jobs[0].downloads_done
        assert queued_jobs[0]._install_tmpdir_active_sentinel_created
    finally:
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
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
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=8)

    monkeypatch.setattr(mm2_installer, "_multifile_download", lambda **_: download_job)
    submit_multifile_download = MagicMock()
    monkeypatch.setattr(mm2_installer._download_queue, "submit_multifile_download", submit_multifile_download)
    mm2_installer._enqueue_remote_download(
        job=job,
        source=source,
        remote_files=[remote_file],
        metadata=None,
        destdir=tmp_path,
    )

    part.resume_required = True
    mm2_installer._download_cancelled_callback(download_job)

    assert job.status == InstallStatus.PAUSED
    assert job.model_dump(mode="json")["download_parts"][0]["resume_required"] is True
    marker = mm2_installer._read_install_marker(tmp_path)
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
    assert enqueue_remote_download.call_args.kwargs["clear_partials"] is True


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
    installer._write_recovery_sentinel(tmpdir)

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
    install_job._install_tmpdir = tmpdir
    install_job._multifile_job = download_job
    installer._download_cache[download_job.id] = install_job
    monkeypatch.setattr(installer._download_queue, "cancel_job", MagicMock())

    if callback_name == "_download_error_callback":
        installer._download_error_callback(download_job, RuntimeError("download failed"))
    else:
        installer._download_cancelled_callback(download_job)

    assert installer._has_recovery_sentinel(tmpdir)
    assert payload.read_bytes() == b"recovery data"


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
    install_job._install_tmpdir = tmpdir
    install_job._multifile_job = download_job
    install_job._install_tmpdir_active_sentinel_created = True
    installer._install_jobs.append(install_job)
    installer._download_cache[download_job.id] = install_job
    create_active_install_sentinel(tmpdir)
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
        marker = installer._read_install_marker(tmpdir)
        assert marker is not None
        assert marker["status"] == InstallStatus.DOWNLOADS_DONE.value
        assert not has_active_install_sentinel(tmpdir)
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
    install_job._install_tmpdir = tmpdir
    install_job._install_tmpdir_active_sentinel_created = True
    install_job._multifile_job = download_job
    installer._install_jobs.append(install_job)
    create_active_install_sentinel(tmpdir)

    installer.stop()

    try:
        assert install_job.downloads_done
        assert tmpdir.exists()
        marker = installer._read_install_marker(tmpdir)
        assert marker is not None
        assert marker["status"] == InstallStatus.DOWNLOADS_DONE.value
        assert not has_active_install_sentinel(tmpdir)
    finally:
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
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
    job._install_tmpdir = tmpdir
    job._install_tmpdir_active_sentinel_created = True
    job._multifile_job = download_job
    installer._download_cache[download_job.id] = job
    create_active_install_sentinel(tmpdir)
    orphan_service = OrphanedModelsService(config=mm2_app_config, db=installer.record_store._db)

    try:
        installer._download_cancelled_callback(download_job)

        assert has_active_install_sentinel(tmpdir)
        assert tmpdir.name not in {orphan.path for orphan in orphan_service.find_orphaned_models()}
        result = orphan_service.delete_orphaned_models([tmpdir.name])[tmpdir.name]
        assert "reserved" in result
        assert payload.exists()
    finally:
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
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
    job._install_tmpdir = tmpdir
    job._install_tmpdir_active_sentinel_created = True
    job._multifile_job = download_job
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    create_active_install_sentinel(tmpdir)
    cancel_download = MagicMock()
    monkeypatch.setattr(installer._download_queue, "cancel_job", cancel_download)

    installer.cancel_job(job)

    try:
        assert cancel_download.call_args.args == (download_job,)
        assert tmpdir.exists()
        assert payload.exists()
        assert has_active_install_sentinel(tmpdir)
        assert installer._download_cache.get(download_job.id) is job

        # Cancellation may race an interrupted part being marked resumable by the queue.
        part.resume_required = True
        installer._download_cancelled_callback(download_job)

        assert job.cancelled
        assert not tmpdir.exists()
        assert not has_active_install_sentinel(tmpdir)
        assert download_job.id not in installer._download_cache
    finally:
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
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
    job._install_tmpdir = tmpdir
    job._install_tmpdir_active_sentinel_created = True
    job._multifile_job = download_job
    job.download_parts = download_job.download_parts
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    create_active_install_sentinel(tmpdir)
    cancel_download = MagicMock()
    monkeypatch.setattr(installer._download_queue, "cancel_job", cancel_download)

    try:
        installer.cancel_job(job)

        assert job.cancelled
        assert tmpdir.exists()
        assert payload.exists()
        assert has_active_install_sentinel(tmpdir)
        cancel_download.assert_called_once_with(download_job)

        if terminal_callback == "complete":
            installer._download_complete_callback(download_job)
        else:
            installer._download_error_callback(download_job, RuntimeError("late download error"))

        assert job.cancelled
        assert download_job.id not in installer._download_cache
        assert installer._install_queue.qsize() == 0
        assert not tmpdir.exists()
        assert not has_active_install_sentinel(tmpdir)
    finally:
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
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
    job._install_tmpdir = tmpdir
    job._install_tmpdir_active_sentinel_created = True
    job._multifile_job = download_job
    job.download_parts = download_job.download_parts
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    create_active_install_sentinel(tmpdir)
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
        assert has_active_install_sentinel(tmpdir)
        marker = installer._read_install_marker(tmpdir)
        assert marker is not None and marker["status"] == InstallStatus.PAUSED.value
    finally:
        installer.cancel_job(job)
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
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
    job._install_tmpdir = tmpdir
    job._install_tmpdir_active_sentinel_created = True
    job._multifile_job = download_job
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    create_active_install_sentinel(tmpdir)

    installer.stop()

    try:
        assert job.paused
        assert installer._download_cache.get(download_job.id) is job
        assert has_active_install_sentinel(tmpdir)

        installer._download_cancelled_callback(download_job)

        assert download_job.id not in installer._download_cache
        assert not has_active_install_sentinel(tmpdir)
        assert tmpdir.exists()
    finally:
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)


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
    job._install_tmpdir = tmpdir
    job._install_tmpdir_active_sentinel_created = True
    job._multifile_job = download_job
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    create_active_install_sentinel(tmpdir)

    try:
        # The pause callback proves the queue has stopped touching staging, but a live user-paused job
        # intentionally keeps its claim until shutdown.
        installer._download_cancelled_callback(download_job)
        assert download_job.id not in installer._download_cache
        assert has_active_install_sentinel(tmpdir)

        installer.stop()

        assert not has_active_install_sentinel(tmpdir)
        marker = json.loads((tmpdir / INSTALL_MARKER_FILENAME).read_text(encoding="utf-8"))
        assert marker["status"] == InstallStatus.PAUSED.value
        assert tmpdir.exists()
    finally:
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
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
    job._install_tmpdir = tmpdir
    job._install_tmpdir_active_sentinel_created = True
    job._multifile_job = download_job
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    create_active_install_sentinel(tmpdir)

    try:
        installer._download_cancelled_callback(download_job)
        assert has_active_install_sentinel(tmpdir)

        installer.cancel_job(job)

        assert job.cancelled
        assert not tmpdir.exists()
        assert not has_active_install_sentinel(tmpdir)
    finally:
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
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
    job._install_tmpdir = tmp_path
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
    job._install_tmpdir = tmp_path
    job._install_tmpdir_active_sentinel_created = True
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
    installer._release_install_tmpdir_claim(job)
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
    job._install_tmpdir = tmp_path
    job._install_tmpdir_active_sentinel_created = True
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
    installer._release_install_tmpdir_claim(job)
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
    job._install_tmpdir = tmp_path
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
    job._install_tmpdir = tmp_path
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
    job._install_tmpdir = tmpdir
    job._install_tmpdir_active_sentinel_created = True
    job._multifile_job = download_job
    job.download_parts = download_job.download_parts
    installer._install_jobs.append(job)
    installer._download_cache[download_job.id] = job
    create_active_install_sentinel(tmpdir)
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
        assert has_active_install_sentinel(tmpdir)
    finally:
        installer._active_install_job = None
        job._install_phase = None
        job.status = InstallStatus.DOWNLOADS_DONE
        installer.cancel_job(job)
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
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
    job._install_tmpdir = tmpdir
    job._install_tmpdir_active_sentinel_created = True
    job._multifile_job = download_job
    job.download_parts = download_job.download_parts
    installer._install_jobs.append(job)
    create_active_install_sentinel(tmpdir)
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
        assert has_active_install_sentinel(tmpdir)
    finally:
        release_metadata.set()
        operation_thread.join(timeout=5)

    assert not operation_thread.is_alive()
    assert operation_errors == []
    submit_download.assert_called_once()
    assert not job.cancelled
    installer._download_cache.pop(job._multifile_job.id)
    installer._release_install_tmpdir_claim(job)
    (tmpdir / INSTALL_MARKER_FILENAME).unlink(missing_ok=True)
    active_install_sentinel_path(tmpdir).unlink(missing_ok=True)


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
    job._install_tmpdir = tmpdir
    job._install_tmpdir_active_sentinel_created = True
    job._multifile_job = download_job
    job.download_parts = download_job.download_parts
    installer._install_jobs.append(job)
    create_active_install_sentinel(tmpdir)
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
    assert not has_active_install_sentinel(tmpdir)
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
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=8)
    monkeypatch.setattr(installer, "_multifile_download", lambda **_: download_job)

    def assert_claimed_before_submit(submitted_job: MultiFileDownloadJob) -> None:
        assert submitted_job is download_job
        assert has_active_install_sentinel(tmpdir)
        assert job._install_tmpdir_active_sentinel_created

    monkeypatch.setattr(installer._download_queue, "submit_multifile_download", assert_claimed_before_submit)

    installer._enqueue_remote_download(
        job=job,
        source=source,
        remote_files=[remote_file],
        metadata=None,
        destdir=tmpdir,
    )

    assert has_active_install_sentinel(tmpdir)
    assert active_install_sentinel_path(tmpdir).exists()
    # The fixture does not run a download worker for this synthetic job.
    installer._download_cache.pop(download_job.id)
    installer._release_job_source_protection(job)


def test_install_worker_preserves_remote_staging_when_claim_conflicts(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    installer = mm2_installer
    tmpdir = tmp_path / f"{TMPDIR_PREFIX}claim-conflict"
    tmpdir.mkdir()
    payload = tmpdir / "model.safetensors"
    payload.write_bytes(b"owned by another install")
    create_active_install_sentinel(tmpdir)

    job = ModelInstallJob(
        id=458,
        source=URLModelSource(url=Url("https://example.com/model.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
        status=InstallStatus.DOWNLOADS_DONE,
    )
    job._install_tmpdir = tmpdir
    installer._install_queue.put(job)

    def attempt_transfer(queued_job: ModelInstallJob) -> None:
        installer._begin_install_transfer(queued_job)

    try:
        monkeypatch.setattr(installer, "_register_or_install", attempt_transfer)
        installer._install_queue.put(job)
        installer._install_queue.join()

        assert job.errored
        assert payload.read_bytes() == b"owned by another install"
        assert has_active_install_sentinel(tmpdir)
    finally:
        # This sidecar belongs to the simulated competing operation.
        active_install_sentinel_path(tmpdir).unlink(missing_ok=True)


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
    job._install_tmpdir = tmp_path
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


def test_resume_does_not_report_restart_from_scratch_when_partial_exists(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)
    source = URLModelSource(url=Url("https://example.com/model.safetensors"))
    old_part = DownloadJob(source=source.url, dest=tmp_path / "model.safetensors")
    old_part.bytes = 4
    old_part.download_path = tmp_path / "model.safetensors"
    (tmp_path / "model.safetensors.downloading").write_bytes(b"1234")
    job = ModelInstallJob(id=4243, source=source, config_in=ModelRecordChanges(), local_path=tmp_path)
    job._install_tmpdir = tmp_path
    job.download_parts = {old_part}
    job.status = InstallStatus.PAUSED
    remote_file = RemoteModelFile(url=source.url, path=Path("model.safetensors"), size=8)
    monkeypatch.setattr(mm2_installer, "_remote_files_from_source", lambda _: ([remote_file], None))
    monkeypatch.setattr(mm2_installer._download_queue, "submit_multifile_download", MagicMock())

    mm2_installer.resume_job(job)

    parts = job.model_dump(mode="json")["download_parts"]
    assert len(parts) == 1
    assert parts[0]["resume_from_scratch"] is False


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
    the tmpdir and `_remove_dangling_install_dirs` keeps any tmpdir whose marker is readable and non-terminal.
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
            job._install_tmpdir = tmpdir
            job.status = InstallStatus.PAUSED
            mm2_installer._write_install_marker(job, status=InstallStatus.PAUSED)

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
            active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
            shutil.rmtree(tmpdir, ignore_errors=True)


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
        (bad_tmpdir / INSTALL_MARKER_FILENAME).write_text(
            json.dumps(
                {
                    "version": INSTALL_MARKER_VERSION,
                    "source": "stabilityai/unparseable",
                    "config_in": {"base": "not-a-real-base"},
                    "status": InstallStatus.PAUSED.value,
                }
            )
        )

        good_tmpdir = mm2_app_config.models_path / f"tmpinstall_good_{uuid.uuid4().hex}"
        good_tmpdir.mkdir(parents=True, exist_ok=True)
        tmpdirs.append(good_tmpdir)
        job = ModelInstallJob(
            id=99998,
            source=HFModelSource(repo_id="stabilityai/ordinary", variant=ModelRepoVariant.Default),
            config_in=ModelRecordChanges(),
            local_path=good_tmpdir,
        )
        job._install_tmpdir = good_tmpdir
        job.status = InstallStatus.PAUSED
        mm2_installer._write_install_marker(job, status=InstallStatus.PAUSED)

        restored_installer = ModelInstallService(
            app_config=mm2_app_config,
            record_store=mm2_installer.record_store,
            download_queue=mm2_download_queue,
            session=mm2_session,
        )
        restored_installer._restore_incomplete_installs()

        assert [str(job.source) for job in restored_installer.list_jobs()] == ["stabilityai/ordinary"]
    finally:
        for tmpdir in tmpdirs:
            active_install_sentinel_path(tmpdir).unlink(missing_ok=True)
            shutil.rmtree(tmpdir, ignore_errors=True)
