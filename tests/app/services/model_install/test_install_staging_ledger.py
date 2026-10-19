"""The install staging ledger on a real models directory: claims, markers, recovery marks and startup recovery.

Expected on-disk names are spelled out here rather than taken from the ledger: they are the format version 1 contract
that markers and sidecars written by earlier releases rely on.
"""

import errno
import gc
import json
import os
import platform
import stat
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import pytest

from invokeai.app.services.model_install import install_staging_ledger
from invokeai.app.services.model_install.install_staging_ledger import (
    INSTALL_ACTIVE_SCAN_LOCK,
    MAX_ACTIVE_CLAIM_DIAGNOSTICS,
    InstallStaging,
    InstallStagingLedger,
)
from invokeai.app.services.model_install.model_install_common import (
    InstallCancellationConflictError,
    InstallStatus,
    URLModelSource,
)
from invokeai.app.services.model_records import ModelRecordChanges
from invokeai.backend.util.logging import InvokeAILogger

SOURCE = "https://example.com/weights.safetensors"
# A PID above any real pid_max, so its owner is reliably gone.
DEAD_OWNER = "2147483647 1.0\n"


@pytest.fixture(autouse=True)
def _collect_only_this_tests_objects():
    """Removing a staging tree runs gc.collect() to release file handles. A full collection walks the worker's whole
    heap; freezing what existed before the test leaves each collection to the objects the test created."""
    gc.freeze()
    yield
    gc.unfreeze()


@pytest.fixture
def models_path(tmp_path: Path) -> Path:
    path = tmp_path / "models"
    path.mkdir()
    return path


@pytest.fixture
def ledger(models_path: Path) -> InstallStagingLedger:
    return InstallStagingLedger(models_path, InvokeAILogger.get_logger("InstallStagingLedgerTest"))


def _claim_path(root: Path) -> Path:
    return root.parent / f".{root.name}.invokeai_install_active"


def _recovery_path(root: Path) -> Path:
    return root.parent / f".{root.name}.invokeai_install_recovery_required"


def _marker_path(tmpdir: Path) -> Path:
    return tmpdir / ".invokeai_install.json"


def _staging_dir(models_path: Path, name: str, payload: bytes = b"partial download") -> Path:
    tmpdir = models_path / f"tmpinstall_{name}"
    tmpdir.mkdir()
    (tmpdir / "weights.safetensors.downloading").write_bytes(payload)
    return tmpdir


def _write_v1_marker(tmpdir: Path, status: str = "downloading", **fields: Any) -> dict[str, Any]:
    """Write a marker as releases with format version 1 wrote it."""
    marker: dict[str, Any] = {
        "version": 1,
        "source": SOURCE,
        "access_token": None,
        "config_in": {},
        "status": status,
        "updated_at": "2026-01-01T00:00:00",
        "files": [],
        **fields,
    }
    _marker_path(tmpdir).write_text(json.dumps(marker), encoding="utf-8")
    return marker


def _persist(staging: InstallStaging, status: InstallStatus, **overrides: Any) -> None:
    arguments: dict[str, Any] = {
        "source": URLModelSource(url=SOURCE),
        "config_in": ModelRecordChanges(),
        "status": status,
        "download_parts": [],
        "resume_files": None,
        "recovery_required": False,
        **overrides,
    }
    staging.persist(**arguments)


def _read_marker(tmpdir: Path) -> Optional[dict[str, Any]]:
    path = _marker_path(tmpdir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------------------------------------------------
# Markers and startup recovery
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("recovery_field", [{}, {"recovery_required": False}], ids=["no-flag", "flag-false"])
def test_a_v1_marker_from_an_earlier_release_is_restored_as_written(
    ledger: InstallStagingLedger, models_path: Path, recovery_field: dict[str, Any]
) -> None:
    tmpdir = _staging_dir(models_path, "legacy")
    files = [{"url": SOURCE, "etag": '"v1"', "download_path": (tmpdir / "weights.safetensors").as_posix()}]
    _write_v1_marker(
        tmpdir, status="paused", access_token="secret", config_in={"name": "kept"}, files=files, **recovery_field
    )

    [entry] = ledger.recoverable_installs()

    assert entry.staging.tmpdir == tmpdir
    assert not entry.staging.claimed
    assert entry.status is InstallStatus.PAUSED
    assert (entry.source, entry.access_token, entry.config_in, entry.files) == (
        SOURCE,
        "secret",
        {"name": "kept"},
        files,
    )


def test_a_marker_without_status_restores_as_waiting(ledger: InstallStagingLedger, models_path: Path) -> None:
    _write_v1_marker(_staging_dir(models_path, "no-status"), status="")

    [entry] = ledger.recoverable_installs()

    assert entry.status is InstallStatus.WAITING


def test_persisted_state_round_trips_through_startup_recovery(ledger: InstallStagingLedger) -> None:
    staging = ledger.create_staging(SOURCE)
    assert staging.tmpdir is not None and staging.tmpdir.name.startswith("tmpinstall_")
    resume_files = {SOURCE: {"url": SOURCE, "etag": '"v2"', "resume_required": True}}

    _persist(
        staging,
        InstallStatus.DOWNLOADING,
        source=URLModelSource(url=SOURCE, access_token="token"),
        config_in=ModelRecordChanges(name="my model"),
        resume_files=resume_files,
    )

    marker = _read_marker(staging.tmpdir)
    assert marker is not None
    assert marker["version"] == 1
    assert marker["status"] == "downloading"
    assert marker["access_token"] == "token"
    assert "recovery_required" not in marker
    [entry] = ledger.recoverable_installs()
    assert entry.staging.tmpdir == staging.tmpdir
    assert entry.status is InstallStatus.DOWNLOADING
    assert entry.source == SOURCE
    assert entry.access_token == "token"
    assert entry.config_in["name"] == "my model"
    assert entry.files == list(resume_files.values())


@pytest.mark.parametrize("status", ["completed", "error", "cancelled"])
def test_terminal_staging_is_removed_at_startup_and_never_restored(
    ledger: InstallStagingLedger, models_path: Path, status: str
) -> None:
    tmpdir = _staging_dir(models_path, status)
    _write_v1_marker(tmpdir, status=status)

    assert list(ledger.recoverable_installs()) == []
    ledger.remove_dangling_staging([])

    assert not tmpdir.exists()
    assert not _claim_path(tmpdir).exists()


def test_startup_cleanup_removes_markerless_staging_and_keeps_resumable_and_registered_dirs(
    ledger: InstallStagingLedger, models_path: Path
) -> None:
    markerless = _staging_dir(models_path, "markerless")
    resumable = _staging_dir(models_path, "resumable")
    _write_v1_marker(resumable)
    registered = _staging_dir(models_path, "registered-model")

    ledger.remove_dangling_staging([(registered / "weights.safetensors.downloading").resolve()])

    assert not markerless.exists()
    assert resumable.exists()
    assert registered.exists()


def test_recovery_required_marker_is_neither_restored_nor_removed_nor_reused(
    ledger: InstallStagingLedger, models_path: Path
) -> None:
    staging = InstallStaging(ledger, _staging_dir(models_path, "recovery", b"recoverable source"))
    assert staging.tmpdir is not None

    _persist(staging, InstallStatus.ERROR, recovery_required=True)
    ledger.remove_dangling_staging([])

    assert _read_marker(staging.tmpdir)["recovery_required"] is True  # type: ignore[index]
    assert (staging.tmpdir / "weights.safetensors.downloading").read_bytes() == b"recoverable source"
    _write_v1_marker(staging.tmpdir, status="downloading", recovery_required=True)
    assert list(ledger.recoverable_installs()) == []
    assert ledger.create_staging(SOURCE).tmpdir != staging.tmpdir


def test_reuse_picks_the_newest_resumable_staging_of_the_same_source(
    ledger: InstallStagingLedger, models_path: Path
) -> None:
    older = _staging_dir(models_path, "older")
    _write_v1_marker(older, updated_at="2026-01-01T00:00:00")
    newer = _staging_dir(models_path, "newer")
    _write_v1_marker(newer, updated_at="2026-02-01T00:00:00")
    other_source = _staging_dir(models_path, "other")
    _write_v1_marker(other_source, source="https://example.com/other.safetensors", updated_at="2026-03-01T00:00:00")
    finished = _staging_dir(models_path, "finished")
    _write_v1_marker(finished, status="completed", updated_at="2026-04-01T00:00:00")
    claimed = _staging_dir(models_path, "claimed")
    _write_v1_marker(claimed, updated_at="2026-05-01T00:00:00")

    with ledger.claim(claimed):
        assert ledger.create_staging(SOURCE).tmpdir == newer


def test_restore_skips_claimed_and_recovery_marked_staging(ledger: InstallStagingLedger, models_path: Path) -> None:
    claimed = _staging_dir(models_path, "claimed")
    _write_v1_marker(claimed)
    marked = _staging_dir(models_path, "marked")
    _write_v1_marker(marked)
    ledger.mark_recovery_required(marked)

    with ledger.claim(claimed):
        assert list(ledger.recoverable_installs()) == []
        ledger.remove_dangling_staging([])

    assert claimed.exists()
    assert marked.exists()


def test_staging_claimed_by_a_crashed_process_is_recovered_after_the_sweep(
    ledger: InstallStagingLedger, models_path: Path
) -> None:
    """A process that died while it held its claim never released it; the sweep proves the owner gone."""
    tmpdir = _staging_dir(models_path, "crashed-owner")
    _write_v1_marker(tmpdir)
    _claim_path(tmpdir).write_text(DEAD_OWNER, encoding="ascii")

    assert list(ledger.recoverable_installs()) == []
    ledger.sweep_stale_claims()
    [entry] = ledger.recoverable_installs()

    assert entry.staging.tmpdir == tmpdir
    entry.staging.claim()
    assert entry.staging.claimed
    assert _claim_path(tmpdir).read_text(encoding="ascii").split()[0] == str(os.getpid())
    entry.staging.release_claim()
    assert not _claim_path(tmpdir).exists()


def test_a_crash_during_transfer_leaves_staging_protected_across_restart(
    ledger: InstallStagingLedger, models_path: Path
) -> None:
    """The recovery mark is durable before any file moves, so a crash mid-transfer never exposes the tree."""
    tmpdir = _staging_dir(models_path, "mid-transfer")
    _write_v1_marker(tmpdir, status="running")
    staging = InstallStaging(ledger, tmpdir)
    staging.begin_transfer()
    # The process dies here: its claim stays behind until the next startup sweeps it.
    _claim_path(tmpdir).write_text(DEAD_OWNER, encoding="ascii")

    ledger.sweep_stale_claims()
    ledger.remove_dangling_staging([])

    assert tmpdir.exists()
    assert _recovery_path(tmpdir).exists()
    assert list(ledger.recoverable_installs()) == []
    assert ledger.is_protected(tmpdir)


def test_discard_without_a_claim_spares_a_tree_another_owner_holds(
    ledger: InstallStagingLedger, models_path: Path
) -> None:
    tmpdir = _staging_dir(models_path, "owned-elsewhere")

    with ledger.claim(tmpdir):
        InstallStaging(ledger, tmpdir).discard()
        assert tmpdir.exists()

    InstallStaging(ledger, tmpdir).discard()
    assert not tmpdir.exists()
    assert not _claim_path(tmpdir).exists()


def test_discard_keeps_recovery_data_and_releases_the_claim(ledger: InstallStagingLedger, models_path: Path) -> None:
    claimed = InstallStaging(ledger, _staging_dir(models_path, "kept-claimed", b"recovery data"))
    claimed.claim()
    unclaimed = InstallStaging(ledger, _staging_dir(models_path, "kept-unclaimed", b"recovery data"))
    for staging in (claimed, unclaimed):
        assert staging.tmpdir is not None
        ledger.mark_recovery_required(staging.tmpdir)

        staging.discard()

        assert (staging.tmpdir / "weights.safetensors.downloading").read_bytes() == b"recovery data"
        assert not staging.claimed
        assert not _claim_path(staging.tmpdir).exists()


@pytest.mark.parametrize("change", ["marker-written", "recovery-marked"])
def test_startup_cleanup_rechecks_a_staging_dir_once_it_holds_the_claim(
    ledger: InstallStagingLedger, models_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Another process may change the directory between cleanup's decision and its claim."""
    tmpdir = _staging_dir(models_path, "changed-before-claim")
    real_publish = install_staging_ledger._publish_claim

    def change_then_publish(root: Path) -> None:
        if root == tmpdir:
            if change == "marker-written":
                _write_v1_marker(tmpdir)
            else:
                _recovery_path(tmpdir).write_text("preserve", encoding="utf-8")
        real_publish(root)

    monkeypatch.setattr(install_staging_ledger, "_publish_claim", change_then_publish)

    ledger.remove_dangling_staging([])

    assert (tmpdir / "weights.safetensors.downloading").exists()
    assert not _claim_path(tmpdir).exists()


def test_a_claimed_tree_is_discarded_and_its_claim_released(ledger: InstallStagingLedger, models_path: Path) -> None:
    staging = InstallStaging(ledger, _staging_dir(models_path, "discarded"))
    staging.claim()
    assert staging.tmpdir is not None

    staging.discard()

    assert not staging.tmpdir.exists()
    assert not _claim_path(staging.tmpdir).exists()
    assert not staging.claimed


def test_a_staging_tree_cannot_be_discarded_mid_transfer(ledger: InstallStagingLedger, models_path: Path) -> None:
    tmpdir = _staging_dir(models_path, "mid-transfer-discard")
    staging = InstallStaging(ledger, tmpdir)
    staging.begin_transfer()

    with pytest.raises(AssertionError, match="end the transfer"):
        staging.discard()
    assert tmpdir.exists()

    staging.end_transfer()
    staging.discard()
    assert not tmpdir.exists()


def test_after_a_claim_conflict_the_other_owner_keeps_the_tree_and_marker(
    ledger: InstallStagingLedger, models_path: Path
) -> None:
    tmpdir = _staging_dir(models_path, "conflict-cleanup")
    _write_v1_marker(tmpdir)
    contender = InstallStaging(ledger, tmpdir)

    with ledger.claim(tmpdir):
        with pytest.raises(InstallCancellationConflictError):
            contender.claim()
        contender.delete_marker()
        contender.discard()

        assert _marker_path(tmpdir).exists()
        assert _claim_path(tmpdir).exists()


def test_a_second_claim_on_staging_is_refused_and_recorded(ledger: InstallStagingLedger, models_path: Path) -> None:
    tmpdir = _staging_dir(models_path, "contended")
    owner = InstallStaging(ledger, tmpdir)
    owner.claim()
    contender = InstallStaging(ledger, tmpdir)

    with pytest.raises(InstallCancellationConflictError, match="another install or orphan cleanup"):
        contender.claim()

    assert contender.claim_conflict
    assert not contender.claimed
    owner.release_claim()
    contender.claim()
    assert contender.claimed and not contender.claim_conflict
    contender.release_claim()


@pytest.mark.parametrize("root_kind", ["staging", "source"])
def test_release_drops_claims_before_it_clears_recovery_marks(
    ledger: InstallStagingLedger, models_path: Path, monkeypatch: pytest.MonkeyPatch, root_kind: str
) -> None:
    """A crash between the two steps must leave recovery marks, which preserve the data, not claims."""
    if root_kind == "staging":
        root = _staging_dir(models_path, "release-order")
        staging = InstallStaging(ledger, root)
    else:
        root = models_path / "local-source"
        root.mkdir()
        staging = InstallStaging(ledger)
        staging.protect_source(root)
    staging.begin_transfer()
    assert _recovery_path(root).exists()
    unlinked: list[str] = []
    real_unlink = Path.unlink

    def record_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        unlinked.append(path.name)
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", record_unlink)

    staging.release(preserve_recovery=False, completed=True)

    assert unlinked == [_claim_path(root).name, _recovery_path(root).name]


def test_release_preserving_recovery_keeps_the_marks(ledger: InstallStagingLedger, models_path: Path) -> None:
    tmpdir = _staging_dir(models_path, "preserved")
    staging = InstallStaging(ledger, tmpdir)
    staging.begin_transfer()

    staging.release(preserve_recovery=True, completed=False)

    assert not _claim_path(tmpdir).exists()
    assert _recovery_path(tmpdir).exists()


def test_source_protection_claims_the_top_level_root_and_clears_only_resolved_recovery(
    ledger: InstallStagingLedger, models_path: Path, tmp_path: Path
) -> None:
    source_root = models_path / "local-source"
    nested = source_root / "nested"
    nested.mkdir(parents=True)
    ledger.mark_recovery_required(source_root)
    staging = InstallStaging(ledger)

    staging.protect_source(nested)

    assert _claim_path(source_root).exists()
    staging.release(preserve_recovery=False, completed=False)
    assert not _claim_path(source_root).exists()
    assert _recovery_path(source_root).exists(), "an unfinished install must keep recovery data it did not create"

    registered = InstallStaging(ledger)
    registered.protect_source(nested)
    registered.release(preserve_recovery=False, completed=True)
    assert not _recovery_path(source_root).exists(), "completing resolves the recovery data the source held"

    outside = InstallStaging(ledger)
    outside.protect_source(tmp_path)
    assert not _claim_path(tmp_path).exists()
    outside.release(preserve_recovery=False, completed=True)


def test_an_unfinished_job_keeps_the_marks_of_a_source_that_already_held_recovery_data(
    ledger: InstallStagingLedger, models_path: Path
) -> None:
    source_root = models_path / "local-source"
    recovered = source_root / "recovered"
    recovered.mkdir(parents=True)
    ledger.mark_recovery_required(recovered)
    staging = InstallStaging(ledger)
    staging.protect_source(recovered)
    staging.begin_transfer()

    staging.finish(recovery_required=False, completed=False)

    assert not _claim_path(source_root).exists()
    assert _recovery_path(source_root).exists(), "the mark this job made stays until the job completes"
    assert _recovery_path(recovered).exists()

    completed = InstallStaging(ledger)
    completed.protect_source(recovered)
    completed.finish(recovery_required=False, completed=True)
    assert not _recovery_path(source_root).exists()
    assert not _claim_path(source_root).exists()


def test_source_protection_refuses_a_root_held_by_another_owner(
    ledger: InstallStagingLedger, models_path: Path
) -> None:
    source_root = models_path / "local-source"
    nested = source_root / "nested"
    nested.mkdir(parents=True)

    with ledger.claim(nested):
        with pytest.raises(InstallCancellationConflictError, match="another install or orphan cleanup"):
            InstallStaging(ledger).protect_source(nested)
    with ledger.claim(source_root):
        with pytest.raises(InstallCancellationConflictError, match="another install or orphan cleanup"):
            InstallStaging(ledger).protect_source(nested)

    assert not _claim_path(source_root).exists()


def test_is_protected_covers_the_path_its_ancestors_and_its_contents(
    ledger: InstallStagingLedger, models_path: Path
) -> None:
    group = models_path / "group"
    recovered = group / "recovered"
    recovered.mkdir(parents=True)
    ordinary = models_path / "ordinary"
    ordinary.mkdir()
    ledger.mark_recovery_required(recovered)

    assert ledger.is_protected(recovered)
    assert ledger.is_protected(recovered / "weights.safetensors")
    assert ledger.is_protected(group), "deleting the parent would delete the recovery root"
    assert not ledger.is_protected(ordinary)

    with ledger.claim(ordinary):
        assert ledger.is_protected(ordinary / "sub")
    assert not ledger.is_protected(ordinary)

    with ledger.claim(ordinary / "child"):
        assert ledger.is_protected(ordinary), "deleting the parent would delete the claimed child"
    assert not ledger.is_protected(ordinary)


def test_retain_recovery_marks_an_existing_root_once_without_logging(
    ledger: InstallStagingLedger, models_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    dest_dir = models_path / "model"
    dest_dir.mkdir()
    ledger.mark_recovery_required(dest_dir)

    with caplog.at_level("ERROR"):
        ledger.retain_recovery(dest_dir)
        ledger.retain_recovery(models_path / "missing")

    assert _recovery_path(dest_dir).exists()
    assert not _recovery_path(models_path / "missing").exists()
    assert "Failed to persist destination recovery sentinel" not in caplog.text


@pytest.mark.parametrize("error_number", [errno.EINVAL, errno.EIO])
@pytest.mark.skipif(os.name == "nt", reason="directory fsync is skipped on Windows")
def test_clearing_recovery_logs_a_directory_fsync_failure(
    ledger: InstallStagingLedger,
    models_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_number: int,
) -> None:
    root = models_path / "recovery-root"
    root.mkdir()
    _recovery_path(root).write_text("preserve until cleanup", encoding="utf-8")
    real_fsync = os.fsync

    def failing_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(error_number, "scripted directory fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(install_staging_ledger.os, "fsync", failing_directory_fsync)

    ledger.clear_recovery(root)

    assert not _recovery_path(root).exists()
    assert "Failed to remove install recovery sentinel" in caplog.text


# --------------------------------------------------------------------------------------------------------------------
# Active claims and the startup sweep
# --------------------------------------------------------------------------------------------------------------------


def test_sweep_removes_stale_but_preserves_live_claims(ledger: InstallStagingLedger, models_path: Path) -> None:
    stale_root = models_path / "stale-claim"
    live_root = models_path / "live-claim"
    stale_root.mkdir()
    live_root.mkdir()
    _claim_path(stale_root).write_text(DEAD_OWNER, encoding="ascii")

    with ledger.claim(live_root):
        ledger.sweep_stale_claims()

        assert not _claim_path(stale_root).exists()
        assert _claim_path(live_root).exists()
    assert not _claim_path(live_root).exists()


def test_sweep_preserves_ambiguous_legacy_claim_after_large_start_time_drift(
    ledger: InstallStagingLedger, models_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claimed_root = models_path / "claim-time-drift"
    claimed_root.mkdir()
    claim = _claim_path(claimed_root)
    claim.write_text("12345 100.0\n", encoding="ascii")
    observed_start_time = [100.75]

    class Process:
        def __init__(self, _pid: int) -> None:
            pass

        def status(self) -> str:
            return install_staging_ledger.psutil.STATUS_RUNNING

        def create_time(self) -> float:
            return observed_start_time[0]

    monkeypatch.setattr(install_staging_ledger.psutil, "Process", Process)

    ledger.sweep_stale_claims()
    assert claim.exists()

    observed_start_time[0] = 102.0
    ledger.sweep_stale_claims()
    assert claim.exists()


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux process start ticks are used for stable claim identity")
def test_sweep_reclaims_linux_claim_when_pid_has_been_reused(
    ledger: InstallStagingLedger, models_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = models_path / "claim-pid-reuse"
    root.mkdir()
    claim = _claim_path(root)
    current_identity = install_staging_ledger._linux_process_start_identity(os.getpid())
    boot_id = current_identity.rsplit(":", 1)[0]
    claim.write_text(f"{os.getpid()} 100.0 {boot_id}:111\n", encoding="ascii")

    class LiveProcess:
        def __init__(self, _pid: int) -> None:
            pass

        def status(self) -> str:
            return install_staging_ledger.psutil.STATUS_RUNNING

    monkeypatch.setattr(install_staging_ledger.psutil, "Process", LiveProcess)
    monkeypatch.setattr(install_staging_ledger, "_linux_process_start_identity", lambda _pid: f"{boot_id}:222")

    ledger.sweep_stale_claims()

    assert not claim.exists()


def test_sweep_preserves_claim_while_its_payload_is_being_published(
    ledger: InstallStagingLedger, models_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = models_path / "claim-being-published"
    root.mkdir()
    claim = _claim_path(root)
    claim_created = threading.Event()
    release_publish = threading.Event()
    creator_errors: list[BaseException] = []
    real_process = install_staging_ledger.psutil.Process

    class BlockedProcess:
        def __init__(self) -> None:
            self._process = real_process()

        @property
        def pid(self) -> int:
            return self._process.pid

        def create_time(self) -> float:
            if threading.current_thread().name == "blocked-claim-creator":
                claim_created.set()
                assert release_publish.wait(timeout=5)
            return self._process.create_time()

    monkeypatch.setattr(install_staging_ledger.psutil, "Process", BlockedProcess)

    def create_claim() -> None:
        try:
            ledger.claim(root)
        except BaseException as e:
            creator_errors.append(e)

    creator = threading.Thread(target=create_claim, name="blocked-claim-creator")
    creator.start()
    try:
        assert claim_created.wait(timeout=5)
        assert not claim.exists()
        ledger.sweep_stale_claims()
        ledger.claim(root)
        assert len(claim.read_text(encoding="ascii").split()) >= 2
    finally:
        release_publish.set()
        creator.join(timeout=10)

    assert not creator.is_alive()
    assert len(creator_errors) == 1
    assert isinstance(creator_errors[0], FileExistsError)
    assert claim.read_text(encoding="ascii").split()
    claim.unlink(missing_ok=True)


def test_claim_creation_interrupted_before_publication_leaves_no_final_sidecar(
    ledger: InstallStagingLedger, models_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = models_path / "interrupted-claim-publication"
    root.mkdir()

    def exit_before_publication() -> None:
        raise SystemExit("simulated process exit before claim publication")

    monkeypatch.setattr(install_staging_ledger.psutil, "Process", exit_before_publication)

    with pytest.raises(SystemExit, match="simulated process exit"):
        ledger.claim(root)

    assert not _claim_path(root).exists()
    assert list(models_path.iterdir()) == [root]


@pytest.mark.parametrize(
    "payload",
    [
        "2147483647 100.0 linux:malformed\n",
        "2147483647 100.0 windows:unsupported\n",
        "not-a-pid 100.0 linux:11111111-1111-1111-1111-111111111111:1\n",
        "2147483647 nan linux:11111111-1111-1111-1111-111111111111:1\n",
    ],
)
def test_sweep_preserves_malformed_claim_before_inspecting_owner(
    ledger: InstallStagingLedger,
    models_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    payload: str,
) -> None:
    root = models_path / "malformed-claim"
    root.mkdir()
    claim = _claim_path(root)
    claim.write_text(payload, encoding="ascii")
    inspected_pids: list[int] = []

    def inspect_process(pid: int) -> Any:
        inspected_pids.append(pid)
        raise install_staging_ledger.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(install_staging_ledger.psutil, "Process", inspect_process)

    ledger.sweep_stale_claims()

    assert claim.exists()
    assert inspected_pids == []
    assert "could not verify" in caplog.text.lower()


def test_claim_publication_survives_temporary_unlink_failure(
    ledger: InstallStagingLedger, models_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = models_path / "claim-temp-cleanup-failure"
    root.mkdir()
    claim = _claim_path(root)
    temp_paths: list[Path] = []
    real_link = os.link
    real_unlink = Path.unlink

    def publish(temp_path: Path, claim_path: Path) -> None:
        temp_paths.append(Path(temp_path))
        real_link(temp_path, claim_path)

    def fail_temp_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if temp_paths and path == temp_paths[0]:
            raise OSError("simulated temporary sidecar cleanup failure")
        real_unlink(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(install_staging_ledger.os, "link", publish)
            patch.setattr(Path, "unlink", fail_temp_unlink)
            held = ledger.claim(root)
        assert claim.exists()
        assert len(claim.read_text(encoding="ascii").split()) >= 2
        held.release()
        assert not claim.exists()
    finally:
        for temp_path in temp_paths:
            real_unlink(temp_path, missing_ok=True)


def test_sweep_bounds_unknown_claim_diagnostics(
    ledger: InstallStagingLedger, models_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    claims = []
    for index in range(MAX_ACTIVE_CLAIM_DIAGNOSTICS + 2):
        root = models_path / f"malformed-claim-{index}"
        root.mkdir()
        claim = _claim_path(root)
        claim.write_text("unknown owner", encoding="ascii")
        claims.append(claim)

    ledger.sweep_stale_claims()

    claim_warnings = [
        record.message for record in caplog.records if "Could not verify active install claim" in record.message
    ]
    assert len(claim_warnings) == MAX_ACTIVE_CLAIM_DIAGNOSTICS
    assert any("2 additional active install claims" in record.message for record in caplog.records)
    assert all(claim.exists() for claim in claims)


def test_sweep_does_not_delete_replacement_claim_after_process_inspection(
    ledger: InstallStagingLedger, models_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = models_path / "replaced-claim"
    root.mkdir()
    claim = _claim_path(root)
    claim.write_text(DEAD_OWNER, encoding="ascii")
    replacement = "2147483646 2.0\n"
    replacement_path = root / "replacement-claim"
    replacement_path.write_text(replacement, encoding="ascii")

    class DeadProcess:
        def __init__(self, _pid: int) -> None:
            claim.unlink()
            replacement_path.replace(claim)
            raise install_staging_ledger.psutil.NoSuchProcess(pid=_pid)

    monkeypatch.setattr(install_staging_ledger.psutil, "Process", DeadProcess)

    ledger.sweep_stale_claims()

    assert claim.read_text(encoding="ascii") == replacement


def test_serialized_stale_claim_sweeps_preserve_a_claim_created_after_reap(
    ledger: InstallStagingLedger, models_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = models_path / "serialized-claim-scan"
    root.mkdir()
    claim = _claim_path(root)
    claim.write_text(DEAD_OWNER, encoding="ascii")
    final_unlink_started = threading.Event()
    release_first_unlink = threading.Event()
    first_claim_unlinked = threading.Event()
    release_first_scan = threading.Event()
    second_scan_waiting_for_lock = threading.Event()
    second_scan_finished = threading.Event()
    scan_errors: list[Exception] = []
    condition = threading.Condition()
    real_unlink = Path.unlink
    real_claim_scan_lock = install_staging_ledger._claim_scan_lock

    def signal(event: threading.Event) -> None:
        event.set()
        with condition:
            condition.notify_all()

    def blocked_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == claim and threading.current_thread().name == "first-stale-scan":
            signal(final_unlink_started)
            assert release_first_unlink.wait(timeout=5)
            real_unlink(path, *args, **kwargs)
            signal(first_claim_unlinked)
            assert release_first_scan.wait(timeout=5)
            return
        real_unlink(path, *args, **kwargs)

    class ObservedClaimScanLock:
        def __init__(self, models_path: Path) -> None:
            self._lock = real_claim_scan_lock(models_path)

        def __enter__(self) -> "ObservedClaimScanLock":
            if threading.current_thread().name == "second-stale-scan":
                signal(second_scan_waiting_for_lock)
            self._lock.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._lock.__exit__(*args)

    monkeypatch.setattr(Path, "unlink", blocked_unlink)
    monkeypatch.setattr(install_staging_ledger, "_claim_scan_lock", ObservedClaimScanLock)

    def run_scan() -> None:
        try:
            ledger.sweep_stale_claims()
        except Exception as e:
            scan_errors.append(e)

    def run_second_scan() -> None:
        run_scan()
        signal(second_scan_finished)

    first_scan = threading.Thread(target=run_scan, name="first-stale-scan")
    second_scan = threading.Thread(target=run_second_scan, name="second-stale-scan")
    first_scan.start()
    try:
        assert final_unlink_started.wait(timeout=5)
        second_scan.start()
        with condition:
            assert condition.wait_for(
                lambda: second_scan_finished.is_set() or second_scan_waiting_for_lock.is_set(), timeout=5
            )

        if second_scan_finished.is_set():
            # An uncoordinated sweep reaps the observed claim while sweep A still holds its stale view.
            ledger.claim(root)
            release_first_unlink.set()
            assert first_claim_unlinked.wait(timeout=5)
        else:
            # A owns the sweep lock. Let it unlink, publish a new owner before releasing that lock, then let B sweep.
            release_first_unlink.set()
            assert first_claim_unlinked.wait(timeout=5)
            ledger.claim(root)

        release_first_scan.set()
        first_scan.join(timeout=10)
        second_scan.join(timeout=10)
        assert not first_scan.is_alive()
        assert not second_scan.is_alive()
        assert scan_errors == []

        assert claim.read_text(encoding="ascii").split()[0] == str(os.getpid())
        ledger.sweep_stale_claims()
        assert claim.exists()
    finally:
        release_first_unlink.set()
        release_first_scan.set()
        first_scan.join(timeout=10)
        if second_scan.ident is not None:
            second_scan.join(timeout=10)
        claim.unlink(missing_ok=True)


def test_sweep_lock_is_released_after_inspection_error(
    ledger: InstallStagingLedger, models_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = models_path / "claim-scan-lock-lifecycle"
    root.mkdir()
    claim = _claim_path(root)
    claim.write_text(DEAD_OWNER, encoding="ascii")
    real_process = install_staging_ledger.psutil.Process

    def fail_process_inspection(_pid: int) -> None:
        raise RuntimeError("simulated process inspection failure")

    monkeypatch.setattr(install_staging_ledger.psutil, "Process", fail_process_inspection)
    with pytest.raises(RuntimeError, match="simulated process inspection failure"):
        ledger.sweep_stale_claims()

    lock_path = models_path / INSTALL_ACTIVE_SCAN_LOCK
    assert lock_path.name == ".invokeai_install_claims.lock"
    assert lock_path.exists()
    monkeypatch.setattr(install_staging_ledger.psutil, "Process", real_process)
    scan_thread = threading.Thread(target=ledger.sweep_stale_claims)
    scan_thread.start()
    scan_thread.join(timeout=10)

    assert not scan_thread.is_alive()
    assert not claim.exists()
    assert lock_path.exists()


def test_windows_sweep_lock_retries_past_crt_finite_lock_attempts(
    models_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    class ContendedWindowsLock:
        LK_NBLCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(_fd: int, mode: int, _nbytes: int) -> None:
            calls.append(mode)
            if mode == ContendedWindowsLock.LK_NBLCK and calls.count(ContendedWindowsLock.LK_NBLCK) <= 12:
                raise OSError(errno.EACCES, "simulated lock contention")

    monkeypatch.setattr(install_staging_ledger.os, "name", "nt")
    monkeypatch.setitem(sys.modules, "msvcrt", ContendedWindowsLock)
    monkeypatch.setattr(install_staging_ledger.time, "sleep", lambda _seconds: None)

    with install_staging_ledger._claim_scan_lock(models_path):
        pass

    assert calls == [*([ContendedWindowsLock.LK_NBLCK] * 13), ContendedWindowsLock.LK_UNLCK]


def test_sweep_preserves_access_denied_claim_then_reclaims_confirmed_dead_owner(
    ledger: InstallStagingLedger,
    models_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root = models_path / "inaccessible-claim"
    root.mkdir()
    claim = _claim_path(root)
    claim.write_text("12345 100.0\n", encoding="ascii")

    class InaccessibleProcess:
        def __init__(self, _pid: int) -> None:
            raise install_staging_ledger.psutil.AccessDenied(pid=_pid)

    monkeypatch.setattr(install_staging_ledger.psutil, "Process", InaccessibleProcess)
    ledger.sweep_stale_claims()
    assert claim.exists()
    assert "could not verify" in caplog.text.lower()

    class LiveProcess:
        def __init__(self, _pid: int) -> None:
            pass

        def status(self) -> str:
            return install_staging_ledger.psutil.STATUS_RUNNING

        def create_time(self) -> float:
            return 100.0

    monkeypatch.setattr(install_staging_ledger.psutil, "Process", LiveProcess)
    ledger.sweep_stale_claims()
    assert claim.exists()

    class DeadProcess:
        def __init__(self, _pid: int) -> None:
            raise install_staging_ledger.psutil.NoSuchProcess(pid=_pid)

    monkeypatch.setattr(install_staging_ledger.psutil, "Process", DeadProcess)
    ledger.sweep_stale_claims()
    assert not claim.exists()
