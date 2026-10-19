"""The install staging ledger: what model installs keep on disk under the models directory, and its lifecycle.

An install's on-disk state must survive a crash or restart, and orphan cleanup must never delete files an install still
needs. The ledger owns every piece of that state (paths relative to the models directory):

- ``tmpinstall_*/``: a remote install's staging directory. Downloads land here before the install moves them.
- ``tmpinstall_*/.invokeai_install.json``: the staging directory's marker (format version 1), what a restart needs to
  resume the install: source, access token (Hugging Face and URL sources only), requested config, status, timestamp,
  per-file resume metadata, and ``recovery_required`` once a failed transfer left data that needs a person. Each
  write replaces the marker atomically (temporary file, fsync, rename, directory fsync off Windows).
- ``.<name>.invokeai_install_active`` beside a root: an active claim, ``"<pid> <create-time>[ linux:<boot-id>:<start-
  ticks>]"``. A live install or orphan deletion owns the root. Claims are published atomically and refused when taken.
- ``.<name>.invokeai_install_recovery_required`` beside a root: the root holds data a failed transfer could not roll
  back. Nothing deletes, reuses or restores it; a person resolves it and registers the result.
- ``.invokeai_install_claims.lock``: serializes the startup sweep that removes claims whose process has exited.

Sidecars sit beside the root they protect rather than inside it, so they never travel with the files an install moves.
"""

import errno
import gc
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from types import TracebackType
from typing import Any, Callable, Iterable, Iterator, Optional

import psutil

from invokeai.app.services.download import DownloadJob
from invokeai.app.services.model_install.model_install_common import (
    HFModelSource,
    InstallCancellationConflictError,
    InstallStatus,
    ModelSource,
    URLModelSource,
)
from invokeai.app.services.model_records import ModelRecordChanges
from invokeai.app.util.misc import get_iso_timestamp

TMPDIR_PREFIX = "tmpinstall_"
INSTALL_MARKER_FILENAME = ".invokeai_install.json"
INSTALL_MARKER_VERSION = 1
INSTALL_RECOVERY_SENTINEL = ".invokeai_install_recovery_required"
INSTALL_ACTIVE_SENTINEL = ".invokeai_install_active"
INSTALL_ACTIVE_SCAN_LOCK = ".invokeai_install_claims.lock"
MAX_ACTIVE_CLAIM_DIAGNOSTICS = 10
ACTIVE_CLAIM_SCAN_LOCK_RETRY_INTERVAL_SECONDS = 0.05
_MARKER_REPLACE_ATTEMPTS = 5
_MARKER_REPLACE_RETRY_SECONDS = 0.05

_TERMINAL_STATUSES = {InstallStatus.COMPLETED, InstallStatus.ERROR, InstallStatus.CANCELLED}
_active_claim_scan_thread_lock = threading.Lock()


def _recovery_sentinel_path(root: Path) -> Path:
    return root.parent / f".{root.name}{INSTALL_RECOVERY_SENTINEL}"


def _active_claim_path(root: Path) -> Path:
    return root.parent / f".{root.name}{INSTALL_ACTIVE_SENTINEL}"


def _is_claimed(root: Path) -> bool:
    return _active_claim_path(root).exists()


def _requires_recovery(root: Path) -> bool:
    return _recovery_sentinel_path(root).exists()


def _fsync_directory(path: Path) -> None:
    # Windows cannot open a directory for fsync; its directory entries are not synced separately.
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _linux_process_start_identity(pid: int) -> str:
    """Return process identity fields unaffected by wall-clock changes."""
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    stat = Path(f"/proc/{pid}/stat").read_bytes()
    fields = stat.rsplit(b")", 1)[1].split()
    if len(fields) <= 19:
        raise ValueError(f"Unexpected /proc/{pid}/stat format")
    start_ticks = fields[19].decode("ascii")
    return f"linux:{boot_id}:{start_ticks}"


def _publish_claim(root: Path) -> None:
    """Atomically claim root; raises FileExistsError when another install or orphan deletion holds it."""
    path = _active_claim_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            process = psutil.Process()
            payload = f"{process.pid} {process.create_time()!r}"
            if sys.platform == "linux":
                payload += f" {_linux_process_start_identity(process.pid)}"
            f.write(f"{payload}\n".encode("ascii"))
            f.flush()
            os.fsync(f.fileno())
        # A hard link publishes the complete payload atomically and fails if another owner won.
        os.link(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    else:
        # Once linked, cleanup cannot make publication fail for the owner.
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _drop_claim(root: Path) -> None:
    try:
        _active_claim_path(root).unlink()
    except FileNotFoundError:
        pass


def _write_recovery_sentinel(root: Path) -> None:
    """Durably mark root as recovery data; raises FileExistsError when it is already marked."""
    path = _recovery_sentinel_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(b"Install transfer recovery required. Preserve this directory.\n")
        f.flush()
        os.fsync(f.fileno())
    # Persist the directory entry before any source data moves.
    _fsync_directory(path.parent)


def _delete_recovery_sentinel(root: Path, logger: logging.Logger) -> None:
    try:
        _recovery_sentinel_path(root).unlink()
        _fsync_directory(root.parent)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning(f"Failed to remove install recovery sentinel in {root}: {e}")


def _has_terminal_status(marker: Optional[dict[str, Any]]) -> bool:
    return marker is not None and marker.get("status") in {status.value for status in _TERMINAL_STATUSES}


def _write_marker(path: Path, marker: dict[str, Any], logger: logging.Logger) -> None:
    """Replace a marker atomically: readers and a crash see the previous marker or the new one, never a torn one."""
    fd, temp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wt", encoding="utf-8") as f:
            json.dump(marker, f)
            f.flush()
            os.fsync(f.fileno())
        # Windows refuses to replace a file another handle has open, e.g. a concurrent reader; retry briefly.
        delay = _MARKER_REPLACE_RETRY_SECONDS
        for attempts_left in range(_MARKER_REPLACE_ATTEMPTS - 1, -1, -1):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                if attempts_left == 0:
                    raise
                time.sleep(delay)
                delay *= 2
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    try:
        _fsync_directory(path.parent)
    except OSError as e:
        # The new marker is in place; only its survival across a crash is uncertain.
        logger.warning(f"Could not sync install marker directory {path.parent}: {e}")


def _remove_tree(path: Path, logger: logging.Logger) -> None:
    """Remove a staging tree, retrying the file-handle races Windows reports as PermissionError.

    A failure is logged, not raised: startup removes the leftover tree once its marker and claim are gone.
    """
    max_retries = 3
    retry_delay = 0.5  # seconds
    for attempt in range(max_retries):
        try:
            # Release lingering references to files in the tree before removing it.
            gc.collect()
            rmtree(path)
            return
        except PermissionError as e:
            if attempt < max_retries - 1 and sys.platform == "win32":
                logger.warning(
                    f"Failed to remove {path} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Failed to remove temporary directory {path}: {e}")
                return
        except Exception as e:
            logger.error(f"Unexpected error removing {path}: {e}")
            return


@contextmanager
def _claim_scan_lock(models_path: Path) -> Iterator[None]:
    """Serialize stale-claim sweeps across threads and processes using a stable lock file."""
    with _active_claim_scan_thread_lock:
        lock_path = models_path / INSTALL_ACTIVE_SCAN_LOCK
        with open(lock_path, "a+b") as lock_file:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                # LK_LOCK gives up after a fixed number of attempts, which a long sweep can exceed.
                while True:
                    try:
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as e:
                        if e.errno not in {errno.EACCES, errno.EDEADLK, errno.EAGAIN}:
                            raise
                        time.sleep(ACTIVE_CLAIM_SCAN_LOCK_RETRY_INTERVAL_SECONDS)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class HeldClaim:
    """An active claim this process published on a root; leaving the ``with`` block releases it."""

    def __init__(self, root: Path) -> None:
        _publish_claim(root)
        self._root = root
        self._released = False

    def release(self) -> None:
        """Release the claim; later calls do nothing, so a claim published since by another owner survives."""
        if not self._released:
            self._released = True
            _drop_claim(self._root)

    def __enter__(self) -> "HeldClaim":
        return self

    def __exit__(
        self, exc_type: Optional[type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]
    ) -> None:
        self.release()


@dataclass
class _SourceClaim:
    root: Path
    recovery_preexisting: bool
    claimed: bool = True
    owns_recovery_sentinel: bool = False


@dataclass(frozen=True)
class RecoverableInstall:
    """A staging directory startup may resume: unclaimed, not under recovery, with a readable non-terminal marker.

    The marker's other fields are as written; the installer validates them before it trusts them.
    """

    staging: "InstallStaging"
    status: InstallStatus
    source: Any
    access_token: Any
    config_in: Any
    files: list[Any]


class InstallStaging:
    """One install's footprint under the models directory: its staging directory and the local source it moves.

    A remote install has a staging directory; a local install of a source under the models directory protects that
    source's top-level root. Either part may be absent, and operations on an absent part do nothing. The installer
    creates the handle with its job and keeps it for the job's life.

    The handle has no lock. The installer reaches it from its worker, download callbacks and API calls, and relies on the
    job's lifecycle instead: one transfer at a time, and download callbacks quiesced before the staging tree is cleaned.
    """

    def __init__(self, ledger: "InstallStagingLedger", tmpdir: Optional[Path] = None) -> None:
        self._ledger = ledger
        self._tmpdir = tmpdir
        self._claimed = False
        self._claim_conflict = False
        self._owns_recovery_sentinel = False
        self._transferring = False
        self._source: Optional[_SourceClaim] = None

    @property
    def tmpdir(self) -> Optional[Path]:
        return self._tmpdir

    @property
    def claimed(self) -> bool:
        return self._claimed

    @property
    def claim_conflict(self) -> bool:
        """Whether the last claim attempt found another owner, which then owns the staging tree's cleanup."""
        return self._claim_conflict

    def claim(self) -> None:
        """Claim the staging directory against orphan cleanup and other installs; does nothing if already held."""
        if self._tmpdir is None or self._claimed:
            return
        try:
            _publish_claim(self._tmpdir)
        except FileExistsError as e:
            self._claim_conflict = True
            raise InstallCancellationConflictError(
                f"Cannot use install staging path {self._tmpdir}: another install or orphan cleanup is using it."
            ) from e
        self._claimed = True
        self._claim_conflict = False

    def release_claim(self) -> None:
        """Release the staging claim, leaving the tree for a later resume or startup restore."""
        if self._tmpdir is not None and self._claimed:
            _drop_claim(self._tmpdir)
            self._claimed = False

    def protect_source(self, model_path: Path) -> None:
        """Claim the top-level root under the models directory that contains a local source.

        A source outside the models directory, or the models directory itself, is not claimed. A source holding
        recovery data may be used, which is how a person registers what they recovered. Raises
        InstallCancellationConflictError when another install or orphan deletion holds the source or a root above it.
        """
        assert self._source is None, "the source is already protected"
        models_root = self._ledger._models_path.resolve()
        source_dir = (model_path if model_path.is_dir() else model_path.parent).resolve()
        if source_dir == models_root or not source_dir.is_relative_to(models_root):
            return
        source_root = models_root / source_dir.relative_to(models_root).parts[0]
        has_recovery = False
        current = model_path.resolve() if model_path.is_dir() else model_path.resolve().parent
        while current != models_root and current.is_relative_to(models_root):
            if _is_claimed(current):
                raise InstallCancellationConflictError(
                    f"Cannot use local model source {current}: another install or orphan cleanup is using it."
                )
            has_recovery = has_recovery or _requires_recovery(current)
            current = current.parent
        try:
            _publish_claim(source_root)
        except FileExistsError as e:
            raise InstallCancellationConflictError(
                f"Cannot use local model source {source_root}: another install or orphan cleanup is using it."
            ) from e
        self._source = _SourceClaim(root=source_root, recovery_preexisting=has_recovery)

    def persist(
        self,
        *,
        source: ModelSource,
        config_in: ModelRecordChanges,
        status: InstallStatus,
        download_parts: Iterable[DownloadJob],
        resume_files: Optional[dict[str, dict[str, Any]]],
        recovery_required: bool,
    ) -> None:
        """Write the staging directory's marker so a restart can resume the install."""
        if self._tmpdir is None:
            return
        files: list[dict[str, Any]] = [
            {
                "url": str(part.source),
                "canonical_url": part.canonical_url,
                "etag": part.etag,
                "last_modified": part.last_modified,
                "expected_total_bytes": part.expected_total_bytes,
                "final_url": part.final_url,
                "download_path": part.download_path.as_posix() if part.download_path else None,
                "resume_required": part.resume_required,
                "resume_message": part.resume_message,
            }
            for part in download_parts
        ]
        if not files and resume_files:
            files.extend(dict(metadata) for metadata in resume_files.values())
        marker: dict[str, Any] = {
            "version": INSTALL_MARKER_VERSION,
            "source": str(source),
            "access_token": source.access_token if isinstance(source, (HFModelSource, URLModelSource)) else None,
            "config_in": config_in.model_dump(),
            "status": status.value,
            "updated_at": get_iso_timestamp(),
            "files": files,
        }
        if recovery_required:
            marker["recovery_required"] = True
        self._tmpdir.mkdir(parents=True, exist_ok=True)
        _write_marker(self._tmpdir / INSTALL_MARKER_FILENAME, marker, self._ledger._logger)

    def delete_marker(self) -> None:
        """Remove the marker once the install no longer needs a restart to resume it.

        Does nothing after a claim conflict: the staging directory's other owner keeps its marker.
        """
        if self._tmpdir is None or self._claim_conflict:
            return
        path = self._tmpdir / INSTALL_MARKER_FILENAME
        # A process killed mid-write leaves its temporary file; it must not travel with the files an install moves.
        for leftover in [path, *self._tmpdir.glob(f"{INSTALL_MARKER_FILENAME}.*.tmp")]:
            if leftover.exists():
                try:
                    leftover.unlink()
                except Exception as e:
                    self._ledger._logger.warning(f"Failed to remove install marker {leftover}: {e}")

    def requires_recovery(self) -> bool:
        """Whether the staging directory is marked as holding recovery data."""
        return self._tmpdir is not None and _requires_recovery(self._tmpdir)

    def discard(self) -> None:
        """Remove the staging tree under this install's claim, then release the claim.

        A tree marked as recovery data is kept. Without a held claim, the tree is claimed for the removal first and kept
        if another owner holds it. Does nothing after a claim conflict, since the other owner then owns the tree.
        """
        if self._tmpdir is None or self._claim_conflict:
            return
        assert not self._transferring, "end the transfer before discarding its staging tree"
        if self.requires_recovery():
            self.release_claim()
        elif self._claimed:
            _remove_tree(self._tmpdir, self._ledger._logger)
            self.release_claim()
        else:
            self._ledger._remove_unclaimed_staging(self._tmpdir)

    def begin_transfer(self) -> None:
        """Mark both roots as recovery data before any file moves, so an interrupted transfer is preserved.

        Raises when the staging claim or a sentinel cannot be made durable; nothing has moved yet.
        """
        if self._tmpdir is not None:
            self.claim()
            _write_recovery_sentinel(self._tmpdir)
            self._owns_recovery_sentinel = True
            self._transferring = True
        source = self._source
        if source is not None and source.claimed and not _requires_recovery(source.root):
            _write_recovery_sentinel(source.root)
            source.owns_recovery_sentinel = True

    def end_transfer(self) -> None:
        """Clear the staging directory's recovery mark once its transfer finished or rolled back cleanly."""
        if self._tmpdir is not None:
            _delete_recovery_sentinel(self._tmpdir, self._ledger._logger)
            self._transferring = False

    def finish(self, *, recovery_required: bool, completed: bool) -> None:
        """Release an install job's claims and the recovery marks it no longer needs.

        A job whose source already held recovery data keeps every mark until it completes. ``install_path()`` uses
        ``release()`` directly and clears a mark it made even then.
        """
        source_recovery_preexisting = self._source is not None and self._source.recovery_preexisting
        self.release(
            preserve_recovery=recovery_required or (source_recovery_preexisting and not completed),
            completed=completed,
        )

    def release(self, *, preserve_recovery: bool, completed: bool) -> None:
        """Release every claim this install holds, then clear the recovery marks it no longer needs.

        Claims go first: a crash in between leaves recovery marks, which preserve the data, rather than claims.
        Unless ``preserve_recovery``, the marks this install made are cleared, and a mark the source already had is
        cleared once the install ``completed``.
        """
        self.release_claim()
        source = self._source
        if source is not None and source.claimed:
            _drop_claim(source.root)
            source.claimed = False
        if preserve_recovery:
            return
        if self._tmpdir is not None and self._owns_recovery_sentinel:
            _delete_recovery_sentinel(self._tmpdir, self._ledger._logger)
            self._owns_recovery_sentinel = False
        if source is not None and (source.owns_recovery_sentinel or (source.recovery_preexisting and completed)):
            _delete_recovery_sentinel(source.root, self._ledger._logger)
            source.owns_recovery_sentinel = False


class InstallStagingLedger:
    """The models directory's install staging state; see the module docstring for the on-disk format."""

    def __init__(self, models_path: Path, logger: logging.Logger) -> None:
        self._models_path = models_path
        self._logger = logger

    def create_staging(self, source: str) -> InstallStaging:
        """Return an unclaimed handle on a staging directory for a new remote install of source.

        Reuses the newest resumable staging directory left by an earlier install of the same source, so its partial
        downloads resume; otherwise creates a new one.
        """
        candidates: list[tuple[str, Path]] = []
        for tmpdir in self._models_path.glob(f"{TMPDIR_PREFIX}*"):
            if _is_claimed(tmpdir) or _requires_recovery(tmpdir):
                continue
            marker = self._read_marker(tmpdir)
            if not marker or marker.get("source") != source or marker.get("recovery_required") is True:
                continue
            if marker.get("status") in {status.value for status in _TERMINAL_STATUSES}:
                continue
            candidates.append((marker.get("updated_at", ""), tmpdir))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return InstallStaging(self, candidates[0][1])
        return InstallStaging(self, Path(tempfile.mkdtemp(dir=self._models_path, prefix=TMPDIR_PREFIX)))

    def claim(self, root: Path) -> HeldClaim:
        """Claim root for an operation; raises FileExistsError when another install or orphan deletion holds it."""
        return HeldClaim(root)

    def mark_recovery_required(self, root: Path) -> None:
        """Durably mark root as recovery data; raises if the mark cannot be made durable or already exists."""
        _write_recovery_sentinel(root)

    def retain_recovery(self, root: Path) -> None:
        """Mark an existing root as recovery data if it is not already; failures are logged, not raised."""
        try:
            if root.exists() and not _requires_recovery(root):
                _write_recovery_sentinel(root)
        except Exception as sentinel_error:
            self._logger.error(f"Failed to persist destination recovery sentinel in {root}: {sentinel_error}")

    def clear_recovery(self, root: Path) -> None:
        """Remove root's recovery mark, if any; failures are logged."""
        _delete_recovery_sentinel(root, self._logger)

    def is_protected(self, path: Path) -> bool:
        """Whether deleting path could destroy data an install owns.

        True when path or an ancestor below the models directory is claimed or marked as recovery data, or when
        anything beneath path is.
        """
        boundary = self._models_path.resolve()
        current = path.resolve()
        while current != boundary and current.is_relative_to(boundary):
            if _requires_recovery(current) or _is_claimed(current):
                return True
            current = current.parent
        if not path.is_dir():
            return False
        suffixes = (INSTALL_RECOVERY_SENTINEL, INSTALL_ACTIVE_SENTINEL)
        return any(entry.name.endswith(suffixes) for entry in path.rglob("*.invokeai_install_*"))

    def sweep_stale_claims(self) -> None:
        """Remove active claims whose owning process has exited.

        Sweeps are serialized across threads and processes, so no sweep can delete a claim published after another
        sweep inspected its predecessor. A claim whose owner cannot be proven gone is kept and reported.
        """
        if not self._models_path.is_dir():
            return
        with _claim_scan_lock(self._models_path):
            self._sweep_stale_claims_locked()

    def _sweep_stale_claims_locked(self) -> None:
        diagnostics: list[tuple[Path, str]] = []
        omitted_diagnostics = 0

        def record_diagnostic(claim_path: Path, detail: str) -> None:
            nonlocal omitted_diagnostics
            if len(diagnostics) < MAX_ACTIVE_CLAIM_DIAGNOSTICS:
                diagnostics.append((claim_path, detail))
            else:
                omitted_diagnostics += 1

        for claim_path in self._models_path.glob(f"*{INSTALL_ACTIVE_SENTINEL}"):
            try:
                original_stat = claim_path.stat(follow_symlinks=False)
                owner_data = claim_path.read_text(encoding="ascii").split()
                if len(owner_data) not in {2, 3}:
                    raise ValueError("expected legacy PID/start-time or PID/start-time/identity payload")
                pid_text, create_time_text = owner_data[:2]
                if not re.fullmatch(r"[0-9]+", pid_text) or not re.fullmatch(
                    r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", create_time_text
                ):
                    raise ValueError("malformed PID or process start time")
                pid = int(pid_text)
                create_time = float(create_time_text)
                if pid <= 0 or not math.isfinite(create_time) or create_time < 0:
                    raise ValueError("invalid PID or process start time")
                if len(owner_data) == 3:
                    identity = owner_data[2]
                    if not re.fullmatch(r"[a-z][a-z0-9_-]*:.+", identity):
                        raise ValueError("malformed process identity")
                    if identity.startswith("linux:") and not re.fullmatch(
                        r"linux:[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}:[0-9]+", identity
                    ):
                        raise ValueError("malformed Linux process identity")
                    if sys.platform != "linux" or not identity.startswith("linux:"):
                        record_diagnostic(claim_path, "its process identity is unsupported here; preserving it")
                        continue
                process = psutil.Process(pid)
                if process.status() == psutil.STATUS_ZOMBIE:
                    stale = True
                elif len(owner_data) == 2:
                    # A legacy wall-clock timestamp cannot distinguish PID reuse from a clock adjustment.
                    stale = False
                else:
                    stale = _linux_process_start_identity(pid) != owner_data[2]
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                stale = True
            except psutil.AccessDenied as e:
                record_diagnostic(
                    claim_path,
                    f"process inspection was denied ({e}); preserving it. Retry on next startup or remove it "
                    "manually only after confirming the owner is gone",
                )
                continue
            except (UnicodeError, ValueError) as e:
                record_diagnostic(
                    claim_path,
                    f"its owner payload is malformed ({e}); preserving it. Retry on next startup or remove it "
                    "manually only after confirming no install or orphan deletion is active",
                )
                continue
            except OSError as e:
                record_diagnostic(claim_path, f"inspection failed ({e}); preserving it")
                continue

            if stale:
                try:
                    current_stat = claim_path.stat(follow_symlinks=False)
                    if (original_stat.st_dev, original_stat.st_ino) != (current_stat.st_dev, current_stat.st_ino):
                        continue
                    claim_path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as e:
                    record_diagnostic(claim_path, f"stale claim removal failed ({e})")

        for claim_path, detail in diagnostics:
            self._logger.warning(f"Could not verify active install claim {claim_path}: {detail}.")
        if omitted_diagnostics:
            self._logger.warning(f"Could not verify {omitted_diagnostics} additional active install claims.")

    def remove_dangling_staging(self, registered_paths: Iterable[Path]) -> None:
        """Remove staging directories no install can resume: no marker, or a terminal one.

        Keeps claimed directories, directories holding a registered model (resolved paths), and recovery data.
        """
        registered = list(registered_paths)
        for tmpdir in self._models_path.glob(f"{TMPDIR_PREFIX}*"):
            if _is_claimed(tmpdir):
                self._logger.debug(f"Preserving active install directory {tmpdir}")
                continue
            resolved_tmpdir = tmpdir.resolve()
            if any(path == resolved_tmpdir or path.is_relative_to(resolved_tmpdir) for path in registered):
                self._logger.debug(f"Preserving registered model directory {tmpdir}")
                continue
            if _requires_recovery(tmpdir):
                self._logger.warning(f"Preserving install recovery data in {tmpdir}")
                continue
            marker = self._read_marker(tmpdir)
            if marker is None:
                self._logger.info(f"Removing dangling temporary directory {tmpdir}")
                self._remove_unclaimed_staging(tmpdir, still_removable=lambda marker: marker is None)
                continue
            if marker.get("recovery_required") is True:
                self._logger.warning(f"Preserving install recovery data in {tmpdir}")
                continue
            if marker.get("status") in {status.value for status in _TERMINAL_STATUSES}:
                self._logger.info(f"Removing completed/errored temporary directory {tmpdir}")
                self._remove_unclaimed_staging(tmpdir, still_removable=_has_terminal_status)

    def recoverable_installs(self) -> Iterator[RecoverableInstall]:
        """Yield the staging directories startup should resume, reading each one only when the caller asks for it."""
        for tmpdir in self._models_path.glob(f"{TMPDIR_PREFIX}*"):
            if _is_claimed(tmpdir):
                self._logger.debug(f"Skipping active install directory {tmpdir}")
                continue
            if _requires_recovery(tmpdir):
                self._logger.warning(f"Preserving install recovery data in {tmpdir}")
                continue
            marker = self._read_marker(tmpdir)
            if not marker:
                continue
            if marker.get("recovery_required") is True:
                self._logger.warning(f"Preserving install recovery data in {tmpdir}")
                continue
            raw_status = marker.get("status")
            try:
                status = InstallStatus(raw_status) if raw_status else InstallStatus.WAITING
            except Exception as e:
                self._logger.warning(f"Skipping install marker in {tmpdir}: {e}")
                continue
            if status in _TERMINAL_STATUSES:
                continue
            yield RecoverableInstall(
                staging=InstallStaging(self, tmpdir),
                status=status,
                source=marker.get("source"),
                access_token=marker.get("access_token"),
                config_in=marker.get("config_in"),
                files=marker.get("files") or [],
            )

    def _read_marker(self, tmpdir: Path) -> Optional[dict[str, Any]]:
        path = tmpdir / INSTALL_MARKER_FILENAME
        if not path.exists():
            return None
        try:
            with open(path, "rt", encoding="utf-8") as f:
                marker = json.load(f)
            if marker.get("version") != INSTALL_MARKER_VERSION:
                return None
            return marker
        except Exception as e:
            self._logger.warning(f"Invalid install marker in {tmpdir}: {e}")
            return None

    def _remove_unclaimed_staging(
        self, tmpdir: Path, still_removable: Callable[[Optional[dict[str, Any]]], bool] = lambda marker: True
    ) -> None:
        """Remove a staging tree only after atomically excluding active install work.

        Under the claim, the tree is checked again: it is kept if it gained recovery data, or if its marker no longer
        allows ``still_removable``, since another owner may have changed it between the decision and the claim.
        """
        try:
            claim = HeldClaim(tmpdir)
        except FileExistsError:
            self._logger.debug(f"Preserving active install directory {tmpdir}")
            return
        with claim:
            marker = self._read_marker(tmpdir)
            recovery_marked = _requires_recovery(tmpdir) or (
                marker is not None and marker.get("recovery_required") is True
            )
            if recovery_marked or not still_removable(marker):
                self._logger.debug(f"Preserving install directory {tmpdir}: it changed before it could be removed")
                return
            _remove_tree(tmpdir, self._logger)
