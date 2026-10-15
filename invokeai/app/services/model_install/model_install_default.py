"""Model installation class."""

import ctypes
import errno
import filecmp
import gc
import json
import locale
import os
import re
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue
from shutil import copy2, copytree, move, rmtree
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

import psutil
import torch
import yaml
from huggingface_hub import get_token as hf_get_token
from pydantic.networks import AnyHttpUrl
from pydantic_core import Url
from requests import Session

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadQueueServiceBase, MultiFileDownloadJob
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_install.model_install_base import ModelInstallServiceBase
from invokeai.app.services.model_install.model_install_common import (
    INSTALL_ACTIVE_SENTINEL,
    MODEL_SOURCE_TO_TYPE_MAP,
    ExternalModelSource,
    HFModelSource,
    InstallCancellationConflictError,
    InstallDownloadConflictError,
    InstallRecoveryRequiredError,
    InstallStatus,
    InvalidModelConfigException,
    LocalModelSource,
    ModelInstallJob,
    ModelSource,
    StringLikeSource,
    URLModelSource,
    create_active_install_sentinel,
    delete_active_install_sentinel,
    has_active_install_sentinel,
    has_recovery_sentinel,
    is_recovery_protected_path,
    recovery_sentinel_path,
)
from invokeai.app.services.model_records import DuplicateModelException, ModelRecordServiceBase, UnknownModelException
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.app.util.path_safety import is_plain_filename
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelConfig,
    ExternalApiModelDefaultSettings,
    ExternalModelCapabilities,
)
from invokeai.backend.model_manager.configs.factory import (
    AnyModelConfig,
    ModelConfigFactory,
)
from invokeai.backend.model_manager.configs.unknown import Unknown_Config
from invokeai.backend.model_manager.metadata import (
    AnyModelRepoMetadata,
    HuggingFaceMetadataFetch,
    ModelMetadataFetchBase,
    ModelMetadataUnavailableError,
    ModelMetadataWithFiles,
    RemoteModelFile,
)
from invokeai.backend.model_manager.metadata.metadata_base import HuggingFaceMetadata
from invokeai.backend.model_manager.search import ModelSearch
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelSourceType,
    ModelType,
)
from invokeai.backend.model_manager.util.lora_metadata_extractor import apply_lora_metadata
from invokeai.backend.util import InvokeAILogger
from invokeai.backend.util.catch_sigint import catch_sigint
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.util import slugify

if TYPE_CHECKING:
    from invokeai.app.services.events.events_base import EventServiceBase


TMPDIR_PREFIX = "tmpinstall_"
# Marker file used to resume or pause remote model installs across restarts.
INSTALL_MARKER_FILENAME = ".invokeai_install.json"
INSTALL_MARKER_VERSION = 1


class _InstallCancelledBeforeTransfer(Exception):
    """Raised when a cancellation request wins before file transfer becomes protected."""


# Filesystems cap a single path component at 255 bytes. A source that lists many explicit files
# (an LTX-2 component folder names eight) would otherwise produce a folder name that cannot be
# created; the combined name is only a label, so it is shortened past this point with a count.
_MAX_COMBINED_SUBFOLDER_NAME = 96


def _combined_subfolder_name(subfolder_names: List[str]) -> str:
    combined = "_".join(subfolder_names)
    if len(combined) <= _MAX_COMBINED_SUBFOLDER_NAME:
        return combined
    kept: List[str] = []
    for name in subfolder_names:
        candidate = "_".join([*kept, name])
        if kept and len(candidate) > _MAX_COMBINED_SUBFOLDER_NAME - 16:
            break
        kept.append(name)
    remaining = len(subfolder_names) - len(kept)
    return "_".join(kept) + (f"_and_{remaining}_more" if remaining else "")


class ModelInstallService(ModelInstallServiceBase):
    """class for InvokeAI model installation."""

    def __init__(
        self,
        app_config: InvokeAIAppConfig,
        record_store: ModelRecordServiceBase,
        download_queue: DownloadQueueServiceBase,
        event_bus: Optional["EventServiceBase"] = None,
        session: Optional[Session] = None,
    ):
        """
        Initialize the installer object.

        :param app_config: InvokeAIAppConfig object
        :param record_store: Previously-opened ModelRecordService database
        :param event_bus: Optional EventService object
        """
        self._app_config = app_config
        self._record_store = record_store
        self._event_bus = event_bus
        self._logger = InvokeAILogger.get_logger(name=self.__class__.__name__)
        self._install_jobs: List[ModelInstallJob] = []
        self._install_queue: Queue[ModelInstallJob] = Queue()
        self._lock = threading.Lock()
        self._active_install_job: Optional[ModelInstallJob] = None
        self._stop_event = threading.Event()
        self._downloads_changed_event = threading.Event()
        self._install_completed_event = threading.Event()
        # Imports must not begin until startup restoration has completed. Leave this unset until
        # _restore_incomplete_installs_async() finishes so an import racing start() cannot pass the barrier early.
        self._restore_completed_event = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._restore_thread: Optional[threading.Thread] = None
        self._download_queue = download_queue
        self._download_cache: Dict[int, ModelInstallJob] = {}
        self._remote_download_operations: set[int] = set()
        self._remote_download_condition = threading.Condition(self._lock)
        # Per-source locks serializing download_and_cache_model() so parallel (multi-GPU) sessions
        # that need the same remote model (e.g. the LaMa infill model) don't race to download into
        # the same cache directory. _download_cache_locks_guard protects the dict itself.
        self._download_cache_locks: Dict[str, threading.Lock] = {}
        self._download_cache_locks_guard = threading.Lock()
        # Import helpers may call into the download queue, so they must run without _lock held. Reserve sources under
        # this condition instead, preventing concurrent imports from creating jobs for the same source.
        self._install_condition = threading.Condition(self._lock)
        self._pending_sources: set[str] = set()
        self._running = False
        self._session = session
        self._install_thread: Optional[threading.Thread] = None
        self._next_job_id = 0

    def _marker_path(self, tmpdir: Path) -> Path:
        return tmpdir / INSTALL_MARKER_FILENAME

    def _recovery_sentinel_path(self, tmpdir: Path) -> Path:
        return recovery_sentinel_path(tmpdir)

    def _has_recovery_sentinel(self, tmpdir: Path) -> bool:
        return has_recovery_sentinel(tmpdir)

    def _write_recovery_sentinel(self, tmpdir: Path) -> None:
        # Keep recovery state outside the tree being transferred into the managed model directory.
        path = self._recovery_sentinel_path(tmpdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as f:
            f.write(b"Install transfer recovery required. Preserve this directory.\n")
            f.flush()
            os.fsync(f.fileno())
        # Persist the directory entry before moving any source data where directory fsync is supported.
        if os.name != "nt":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def _delete_recovery_sentinel(self, tmpdir: Path) -> None:
        try:
            self._recovery_sentinel_path(tmpdir).unlink()
            if os.name != "nt":
                fd = os.open(tmpdir.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except FileNotFoundError:
            pass
        except OSError as e:
            self._logger.warning(f"Failed to remove install recovery sentinel in {tmpdir}: {e}")

    def _protect_managed_source_path(
        self, model_path: Path, *, allow_recovery: bool = False
    ) -> tuple[Optional[Path], bool, bool]:
        """Claim the top-level orphan-scan root containing a local source."""
        source_root = model_path if model_path.is_dir() else model_path.parent
        models_root = self.app_config.models_path.resolve()
        source_root = source_root.resolve()
        if source_root == models_root or not source_root.is_relative_to(models_root):
            return None, False, False
        source_relative = source_root.relative_to(models_root)
        source_root = models_root / source_relative.parts[0]
        has_recovery = False
        current = model_path.resolve() if model_path.is_dir() else model_path.resolve().parent
        while current != models_root and current.is_relative_to(models_root):
            if has_active_install_sentinel(current):
                raise InstallCancellationConflictError(
                    f"Cannot use local model source {current}: another install or orphan cleanup is using it."
                )
            has_recovery = has_recovery or self._has_recovery_sentinel(current)
            current = current.parent
        if has_recovery and not allow_recovery:
            raise InstallRecoveryRequiredError(
                f"Cannot use local model source {source_root}: it contains install recovery data."
            )
        try:
            create_active_install_sentinel(source_root)
        except FileExistsError as e:
            raise InstallCancellationConflictError(
                f"Cannot use local model source {source_root}: another install or orphan cleanup is using it."
            ) from e
        return source_root, True, has_recovery

    def _release_job_source_protection(self, job: ModelInstallJob, *, preserve_recovery: bool = False) -> None:
        self._release_install_tmpdir_claim(job)
        if job._source_active_sentinel_created:
            assert job._source_protection_root is not None
            delete_active_install_sentinel(job._source_protection_root)
            job._source_active_sentinel_created = False
        if not preserve_recovery:
            if job._install_tmpdir is not None and job._install_tmpdir_recovery_sentinel_created:
                self._delete_recovery_sentinel(job._install_tmpdir)
                job._install_tmpdir_recovery_sentinel_created = False
            if job._source_recovery_sentinel_created or (job._source_recovery_sentinel_preexisting and job.complete):
                assert job._source_protection_root is not None
                self._delete_recovery_sentinel(job._source_protection_root)
                job._source_recovery_sentinel_created = False

    @staticmethod
    def _release_install_tmpdir_claim(job: ModelInstallJob) -> None:
        if job._install_tmpdir is not None and job._install_tmpdir_active_sentinel_created:
            delete_active_install_sentinel(job._install_tmpdir)
            job._install_tmpdir_active_sentinel_created = False

    def _claim_install_tmpdir(self, job: ModelInstallJob) -> None:
        if job._install_tmpdir is None or job._install_tmpdir_active_sentinel_created:
            return
        try:
            create_active_install_sentinel(job._install_tmpdir)
        except FileExistsError as e:
            job._install_tmpdir_claim_conflict = True
            raise InstallCancellationConflictError(
                f"Cannot use install staging path {job._install_tmpdir}: another install or orphan cleanup is using it."
            ) from e
        job._install_tmpdir_active_sentinel_created = True
        job._install_tmpdir_claim_conflict = False

    def _remove_unclaimed_install_tmpdir(self, tmpdir: Path) -> bool:
        """Remove a stale staging tree only after atomically excluding active install work."""
        try:
            create_active_install_sentinel(tmpdir)
        except FileExistsError:
            self._logger.debug(f"Preserving active install directory {tmpdir}")
            return False
        try:
            self._safe_rmtree(tmpdir, self._logger)
        finally:
            delete_active_install_sentinel(tmpdir)
        return True

    def _remove_stale_install_source_claims(self) -> None:
        """Remove active-claim sidecars left by processes that exited before releasing them."""
        for claim_path in self.app_config.models_path.glob(f"*{INSTALL_ACTIVE_SENTINEL}"):
            try:
                pid_text, create_time_text = claim_path.read_text(encoding="ascii").split()
                pid = int(pid_text)
                create_time = float(create_time_text)
                process = psutil.Process(pid)
                # On Linux, psutil derives process start time from boot time plus kernel ticks. A small boot-time
                # adjustment can change that value for a live process, so exact float equality can remove its claim.
                if process.status() == psutil.STATUS_ZOMBIE or abs(process.create_time() - create_time) > 1.0:
                    claim_path.unlink(missing_ok=True)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                claim_path.unlink(missing_ok=True)
            except psutil.AccessDenied:
                continue
            except ValueError:
                claim_path.unlink(missing_ok=True)
            except OSError:
                # Permission errors mean the owner process may still be active. Unknown failures are conservative too.
                continue

    def _retain_recovery_destination(self, dest_dir: Path) -> None:
        try:
            if dest_dir.exists() and not self._has_recovery_sentinel(dest_dir):
                self._write_recovery_sentinel(dest_dir)
        except Exception as sentinel_error:
            self._logger.error(f"Failed to persist destination recovery sentinel in {dest_dir}: {sentinel_error}")

    def _write_install_marker(self, job: ModelInstallJob, status: Optional[InstallStatus] = None) -> None:
        if job._install_tmpdir is None:
            return
        files: list[dict] = []
        if job.download_parts:
            for part in job.download_parts:
                files.append(
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
                )
        elif job._resume_metadata:
            files.extend(dict(metadata) for metadata in job._resume_metadata.values())
        marker = {
            "version": INSTALL_MARKER_VERSION,
            "source": str(job.source),
            "access_token": (
                job.source.access_token if isinstance(job.source, (HFModelSource, URLModelSource)) else None
            ),
            "config_in": job.config_in.model_dump(),
            "status": (status or job.status).value,
            "updated_at": get_iso_timestamp(),
            "files": files,
        }
        if job._recovery_required:
            marker["recovery_required"] = True
        path = self._marker_path(job._install_tmpdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wt", encoding="utf-8") as f:
            json.dump(marker, f)

    def _read_install_marker(self, tmpdir: Path) -> Optional[dict]:
        path = self._marker_path(tmpdir)
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

    def _delete_install_marker(self, tmpdir: Path) -> None:
        path = self._marker_path(tmpdir)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                self._logger.warning(f"Failed to remove install marker {path}: {e}")

    def _find_reusable_tmpdir(self, source: ModelSource) -> Optional[Path]:
        path = self._app_config.models_path
        source_str = str(source)
        candidates: list[tuple[str, Path]] = []
        for tmpdir in path.glob(f"{TMPDIR_PREFIX}*"):
            if has_active_install_sentinel(tmpdir) or self._has_recovery_sentinel(tmpdir):
                continue
            marker = self._read_install_marker(tmpdir)
            if not marker:
                continue
            if marker.get("source") != source_str:
                continue
            status = marker.get("status")
            if marker.get("recovery_required") is True:
                continue
            if status in {InstallStatus.COMPLETED.value, InstallStatus.ERROR.value, InstallStatus.CANCELLED.value}:
                continue
            candidates.append((marker.get("updated_at", ""), tmpdir))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _restore_incomplete_installs(self) -> None:
        path = self._app_config.models_path
        seen_sources: set[str] = set()
        # Collect sources already tracked by active jobs (including those being downloaded right now).
        # We must not re-queue these or delete their tmpdirs.
        with self._lock:
            active_sources = {str(j.source) for j in self._install_jobs if not j.in_terminal_state}
            active_sources.update(str(j.source) for j in self._download_cache.values() if not j.in_terminal_state)
        for tmpdir in path.glob(f"{TMPDIR_PREFIX}*"):
            if self._stop_event.is_set():
                return
            if has_active_install_sentinel(tmpdir):
                self._logger.debug(f"Skipping active install directory {tmpdir}")
                continue
            if self._has_recovery_sentinel(tmpdir):
                self._logger.warning(f"Preserving install recovery data in {tmpdir}")
                continue
            marker = self._read_install_marker(tmpdir)
            if not marker:
                continue
            status = marker.get("status")
            if marker.get("recovery_required") is True:
                self._logger.warning(f"Preserving install recovery data in {tmpdir}")
                continue
            try:
                parsed_status = InstallStatus(status) if status else InstallStatus.WAITING
                if parsed_status in {InstallStatus.COMPLETED, InstallStatus.ERROR, InstallStatus.CANCELLED}:
                    continue
                source_str = marker.get("source")
                if not isinstance(source_str, str):
                    raise ValueError("Missing source in install marker")
                source = self._guess_source(source_str)
                access_token = marker.get("access_token")
                if isinstance(source, (HFModelSource, URLModelSource)) and isinstance(access_token, str):
                    source.access_token = access_token
                if source_str in active_sources:
                    # This tmpdir belongs to an install already in progress; leave it alone.
                    self._logger.debug(f"Skipping restore for {source_str} - already being tracked")
                    continue
                if source_str in seen_sources:
                    self._logger.info(f"Removing duplicate temporary directory {tmpdir}")
                    self._remove_unclaimed_install_tmpdir(tmpdir)
                    continue
                # Inside the `try`: a marker written by an older version can hold a config that no longer
                # validates, and that must skip this marker, not abort the restore of every marker after it. Note
                # that an unsafe *key* is no longer such a case - `ModelRecordChanges` deliberately accepts one so
                # the install is restored and then fails at the join in `install_path()`, which errors the job and
                # reclaims its tmpdir. Skipping it here would strand the partial download instead: no job is
                # created to clean up, and `_remove_dangling_install_dirs` keeps any readable non-terminal marker.
                config_in = ModelRecordChanges(**(marker.get("config_in") or {}))
                seen_sources.add(source_str)
            except Exception as e:
                self._logger.warning(f"Skipping install marker in {tmpdir}: {e}")
                continue

            job = ModelInstallJob(
                id=self._next_id(),
                source=source,
                config_in=config_in,
                local_path=tmpdir,
            )
            job._install_tmpdir = tmpdir
            try:
                # Protect paused and completed downloads as soon as they are restored, before any
                # later orphan scan can mistake finalized staging files for unregistered models.
                self._claim_install_tmpdir(job)
            except InstallCancellationConflictError as e:
                self._logger.info(f"Skipping restore of claimed install directory {tmpdir}: {e}")
                continue
            files_meta = marker.get("files") or []
            if files_meta:
                job._resume_metadata = {f.get("url"): f for f in files_meta if f.get("url")}
            job.status = parsed_status
            self._install_jobs.append(job)

            if job.paused:
                continue

            if job.status in [InstallStatus.DOWNLOADS_DONE, InstallStatus.RUNNING]:
                job.status = InstallStatus.DOWNLOADS_DONE
                self._put_in_queue(job)
            else:
                try:
                    self._resume_remote_download(job)
                except ModelMetadataUnavailableError as e:
                    self._logger.warning(f"Could not resume install {source_str} because metadata is unavailable: {e}")
                    job.status = InstallStatus.PAUSED
                    self._write_install_marker(job, status=InstallStatus.PAUSED)
                    if self._stop_event.is_set():
                        return
                except Exception as e:
                    if self._stop_event.is_set():
                        self._logger.info(f"Leaving interrupted install in {job._install_tmpdir} for next startup")
                        job.status = InstallStatus.PAUSED
                        return
                    self._set_error(job, e)
                    if job._install_tmpdir is not None and not job._install_tmpdir_claim_conflict:
                        if job._install_tmpdir_active_sentinel_created:
                            self._safe_rmtree(job._install_tmpdir, self._logger)
                            self._release_install_tmpdir_claim(job)
                        else:
                            self._remove_unclaimed_install_tmpdir(job._install_tmpdir)

    def _restore_incomplete_installs_async(self) -> None:
        self._restore_completed_event.clear()

        def _run() -> None:
            try:
                self._logger.info("Restoring incomplete installs")
                self._restore_incomplete_installs()
                self._logger.info("Finished restoring incomplete installs")
            except Exception as e:
                self._logger.error(f"Failed to restore incomplete installs: {e}")
            finally:
                self._restore_completed_event.set()

        self._restore_thread = threading.Thread(target=_run, daemon=True)
        self._restore_thread.start()

    def _wait_for_restore_complete(self, timeout: Optional[float] = None) -> bool:
        deadline = time.monotonic() + timeout if timeout is not None else None
        if deadline is None:
            self._lock.acquire()
        elif not self._lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            return False
        try:
            if not self._running and not self._restore_completed_event.is_set():
                raise RuntimeError("Model install service is not running")
            startup_error = self._startup_error
        finally:
            self._lock.release()
        if startup_error is not None:
            raise RuntimeError("Model install service failed to start") from startup_error
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        return self._restore_completed_event.wait(timeout=remaining)

    def _resume_remote_download(self, job: ModelInstallJob, *, operation_reserved: bool = False) -> None:
        if not operation_reserved:
            self._begin_remote_download_operation(job)
        previous_status = job.status
        try:
            # Sources whose partial file has vanished. _enqueue_remote_download replaces job.download_parts
            # with fresh parts, so the flag must be carried onto them or the resume response loses it.
            restarted_from_scratch: set[str] = set()
            if job.download_parts:
                for part in job.download_parts:
                    if part.complete or part.bytes <= 0:
                        continue
                    if not part.download_path:
                        continue
                    in_progress_path = part.download_path.with_name(part.download_path.name + ".downloading")
                    if not in_progress_path.exists():
                        part.bytes = 0
                        part.resume_from_scratch = True
                        part.resume_message = "Partial file missing. Restarted download from the beginning."
                        restarted_from_scratch.add(str(part.source))
                job.bytes = sum(p.bytes for p in job.download_parts)
            remote_files, metadata = self._remote_files_from_source(job.source)
            if not remote_files:
                raise RuntimeError("No remote files are available to download")
            subfolders = job.source.subfolders if isinstance(job.source, HFModelSource) else []
            job.status = InstallStatus.WAITING
            self._enqueue_remote_download(
                job=job,
                source=job.source,
                remote_files=remote_files,
                metadata=metadata,
                destdir=job._install_tmpdir or job.local_path,
                subfolder=job.source.subfolder
                if isinstance(job.source, HFModelSource) and len(subfolders) <= 1
                else None,
                subfolders=subfolders if len(subfolders) > 1 else None,
                resume_metadata=job._resume_metadata,
                restarted_from_scratch=restarted_from_scratch,
            )
        except BaseException:
            job.status = previous_status
            raise
        finally:
            if not operation_reserved:
                self._end_remote_download_operation(job)

    def _begin_remote_download_operation(self, job: ModelInstallJob, *, require_paused: bool = False) -> bool:
        """Serialize resume/restart handoffs and reject replacements before old callbacks finish."""
        with self._remote_download_condition:
            if self._stop_event.is_set():
                raise InstallDownloadConflictError("The model install service is stopping; retry after it starts.")
            download_active = job.id in self._remote_download_operations or any(
                cached_job is job for cached_job in self._download_cache.values()
            )
            if require_paused:
                if download_active:
                    raise InstallDownloadConflictError(
                        "A previous download is still active; wait for it to finish before resuming or restarting."
                    )
                if not job.paused:
                    return False
            if job.cancelled or job.complete:
                raise InstallDownloadConflictError(
                    "The install is already terminal and cannot be resumed or restarted."
                )
            if download_active:
                raise InstallDownloadConflictError(
                    "A previous download is still active; wait for it to finish before resuming or restarting."
                )
            self._remote_download_operations.add(job.id)
            return True

    def _end_remote_download_operation(self, job: ModelInstallJob) -> None:
        with self._remote_download_condition:
            self._remote_download_operations.discard(job.id)
            self._remote_download_condition.notify_all()

    @property
    def app_config(self) -> InvokeAIAppConfig:  # noqa D102
        return self._app_config

    @property
    def record_store(self) -> ModelRecordServiceBase:  # noqa D102
        return self._record_store

    @property
    def event_bus(self) -> Optional["EventServiceBase"]:  # noqa D102
        return self._event_bus

    # make the invoker optional here because we don't need it and it
    # makes the installer harder to use outside the web app
    def start(self, invoker: Optional[Invoker] = None) -> None:
        """Start the installer thread."""

        with self._lock:
            if self._running:
                raise Exception("Attempt to start the installer service twice")
            self._startup_error = None
            self._restore_completed_event.clear()
            try:
                self._remove_stale_install_source_claims()
                self._start_installer_thread()
                self._remove_dangling_install_dirs()
                self._migrate_yaml()
                # In normal use, we do not want to scan the models directory - it should never have orphaned models.
                # We should only do the scan when the flag is set (which should only be set when testing).
                if self.app_config.scan_models_on_startup:
                    with catch_sigint():
                        self._register_orphaned_models()

                # Check all models' paths and confirm they exist. A model could be missing if it was installed on a volume
                # that isn't currently mounted. In this case, we don't want to delete the model from the database, but we do
                # want to alert the user.
                for model in self._scan_for_missing_models():
                    self._logger.warning(f"Missing model file: {model.name} at {model.path}")

                self._write_invoke_managed_models_dir_readme()
                self._restore_incomplete_installs_async()
            except BaseException as error:
                self._startup_error = error
                self._restore_completed_event.set()
                raise

    def stop(self, invoker: Optional[Invoker] = None) -> None:
        """Stop the installer thread; after this the object can be deleted and garbage collected."""
        if not self._running:
            return
        self._logger.debug("calling stop_event.set()")
        with self._lock:
            self._stop_event.set()
        assert self._install_thread is not None
        try:
            # Let the worker finish (or leave a dequeued job untouched) before cleaning pending jobs. Otherwise
            # shutdown can mistake a job between Queue.get() and _active_install_job assignment for pending work.
            self._install_thread.join()
            restore_thread = self._restore_thread
            if restore_thread is not None and restore_thread is not threading.current_thread():
                restore_thread.join()
        finally:
            try:
                with self._remote_download_condition:
                    self._remote_download_condition.wait_for(lambda: not self._remote_download_operations)
                self._clear_pending_jobs()
            finally:
                with self._lock:
                    # Download cancellation is asynchronous. Keep these job references until their
                    # callbacks have stopped touching staging paths and release their active claims.
                    self._running = False

    def _write_invoke_managed_models_dir_readme(self) -> None:
        """Write a README file to the Invoke-managed models directory warning users to not fiddle with it."""
        readme_path = self.app_config.models_path / "README.txt"
        with open(readme_path, "wt", encoding=locale.getpreferredencoding()) as f:
            f.write(
                "This directory is managed by Invoke. Do not add, delete or move files in this directory.\n\nTo manage models, use the web interface.\n"
            )

    def _clear_pending_jobs(self) -> None:
        for job in list(self.list_jobs()):
            with self._lock:
                if job.in_terminal_state or self._active_install_job is job:
                    continue
                if job._recovery_required or (
                    job._install_tmpdir is not None and self._has_recovery_sentinel(job._install_tmpdir)
                ):
                    self._logger.warning(f"Preserving recovery data for job {job.id} during shutdown")
                    continue
                multifile_job = job._multifile_job
                download_callback_pending = any(cached_job is job for cached_job in self._download_cache.values())
                downloads_complete = multifile_job is not None and multifile_job.complete
                preserve_download_status = (
                    job.status
                    if multifile_job is None and job.status in {InstallStatus.PAUSED, InstallStatus.DOWNLOADS_DONE}
                    else None
                )
                if preserve_download_status is None and multifile_job is None:
                    job.cancel()
                elif preserve_download_status is None and downloads_complete:
                    job.status = InstallStatus.DOWNLOADS_DONE
                elif preserve_download_status is None:
                    job.status = InstallStatus.PAUSED

            if preserve_download_status is not None:
                self._write_install_marker(job, status=preserve_download_status)
                if not download_callback_pending:
                    self._release_install_tmpdir_claim(job)
            elif multifile_job is not None and downloads_complete:
                self._write_install_marker(job, status=InstallStatus.DOWNLOADS_DONE)
                if not download_callback_pending:
                    self._release_install_tmpdir_claim(job)
            elif multifile_job is not None:
                self._logger.warning(f"Pausing job {job.id}")
                for part in multifile_job.download_parts:
                    self._download_queue.pause_job(part)
                self._write_install_marker(job, status=InstallStatus.PAUSED)
                if self._stop_event.is_set() and not download_callback_pending:
                    # The pause callback already ran, so no download worker can still touch staging.
                    self._release_install_tmpdir_claim(job)
            elif job._install_tmpdir is not None:
                self._logger.warning(f"Cancelling job {job.id}")
                if not job._install_tmpdir_claim_conflict:
                    self._write_install_marker(job, status=InstallStatus.CANCELLED)
                    self._delete_install_marker(job._install_tmpdir)
                    self._safe_rmtree(job._install_tmpdir, self._logger)
                    self._release_install_tmpdir_claim(job)
        while True:
            try:
                job = self._install_queue.get(block=False)
                self._install_queue.task_done()
            except Empty:
                break

    def _put_in_queue(self, job: ModelInstallJob) -> None:
        with self._lock:
            queued = self._queue_install_job_locked(job)
        if not queued:
            if self._stop_event.is_set():
                # A completed download that races shutdown remains resumable on the next start.
                job.status = InstallStatus.DOWNLOADS_DONE
            else:
                self.cancel_job(job)

    def _queue_install_job_locked(self, job: ModelInstallJob) -> bool:
        """Queue an install while holding _lock, preserving shutdown and wait_for_installs ordering."""
        if self._stop_event.is_set():
            return False
        self._install_queue.put(job)
        return True

    def register_path(
        self,
        model_path: Union[Path, str],
        config: Optional[ModelRecordChanges] = None,
    ) -> str:  # noqa D102
        model_path = Path(model_path)
        config = config or ModelRecordChanges()
        if not config.source:
            config.source = model_path.resolve().as_posix()
        config.source_type = ModelSourceType.Path
        source_root, active_sentinel_created, had_recovery_sentinel = self._protect_managed_source_path(
            model_path, allow_recovery=True
        )
        had_recovery_sentinel = source_root is not None and had_recovery_sentinel
        registered = False
        try:
            model_id = self._register(model_path, config)
            registered = True
            return model_id
        finally:
            if active_sentinel_created and source_root is not None:
                delete_active_install_sentinel(source_root)
            if registered and had_recovery_sentinel and source_root is not None:
                self._delete_recovery_sentinel(source_root)

    # TODO: Replace this with a proper fix for underlying problem of Windows holding open
    # the file when it needs to be moved.
    @staticmethod
    def _move_with_retries(src: Path, dst: Path, attempts: int = 5, delay: float = 0.5) -> None:
        """Workaround for Windows file-handle issues when moving files."""
        for tries_left in range(attempts, 0, -1):
            try:
                move(src, dst)
                return
            except PermissionError:
                if dst.exists() or dst.is_symlink():
                    # On Windows, shutil.move may copy a file successfully and then fail unlinking its
                    # source. Accept that state only when both regular files have identical contents.
                    if (
                        src.is_file()
                        and not src.is_symlink()
                        and dst.is_file()
                        and not dst.is_symlink()
                        and filecmp.cmp(src, dst, shallow=False)
                    ):
                        try:
                            src.unlink()
                            return
                        except PermissionError:
                            pass
                    else:
                        raise
                gc.collect()
                if tries_left == 1:
                    raise
                time.sleep(delay)
                delay *= 2  # Exponential backoff

    @staticmethod
    def _rename_noreplace(src: Path, dst: Path) -> None:
        """Atomically restore a path only if its destination is still absent."""
        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = getattr(libc, "renameat2", None)
            if renameat2 is None:
                raise OSError("atomic no-replace rename is unavailable")
            renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            renameat2.restype = ctypes.c_int
            result = renameat2(-100, os.fsencode(src), -100, os.fsencode(dst), 1)  # AT_FDCWD, RENAME_NOREPLACE
            if result != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(dst))
            return
        if sys.platform == "darwin":
            libc = ctypes.CDLL(None, use_errno=True)
            renamex_np = getattr(libc, "renamex_np", None)
            if renamex_np is None:
                raise OSError("atomic no-replace rename is unavailable")
            renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            renamex_np.restype = ctypes.c_int
            result = renamex_np(os.fsencode(src), os.fsencode(dst), 0x00000004)  # RENAME_EXCL
            if result != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(dst))
            return
        if os.name == "nt":
            # Windows rename fails when the destination already exists.
            os.rename(src, dst)
            return
        raise OSError("atomic no-replace rename is unavailable on this platform")

    @classmethod
    def _restore_moved_path(cls, src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(f"Refusing to overwrite recreated source {dst}")
        try:
            cls._rename_noreplace(src, dst)
        except OSError as e:
            if e.errno != errno.EXDEV:
                raise
            # Local imports can span filesystems. Stage a full copy beside the original source, then
            # atomically claim its name with a no-replace rename before removing the managed copy.
            staging_dir = Path(mkdtemp(prefix=f".{dst.name}.restore-", dir=dst.parent))
            staged_path = staging_dir / "restored"
            try:
                if src.is_symlink():
                    os.symlink(os.readlink(src), staged_path)
                elif src.is_dir():
                    copytree(src, staged_path, symlinks=True)
                elif src.is_file():
                    copy2(src, staged_path)
                else:
                    raise OSError(f"Cannot restore unsupported filesystem object {src}")
                cls._rename_noreplace(staged_path, dst)
                if src.is_dir() and not src.is_symlink():
                    rmtree(src)
                else:
                    src.unlink()
            finally:
                rmtree(staging_dir, ignore_errors=True)

    def install_path(
        self,
        model_path: Union[Path, str],
        config: Optional[ModelRecordChanges] = None,
    ) -> str:
        model_path = Path(model_path)
        config = config or ModelRecordChanges()
        source_root, active_sentinel_created, had_recovery_sentinel = self._protect_managed_source_path(
            model_path, allow_recovery=True
        )
        source_recovery_sentinel_created = False
        preserve_source = False
        installed = False
        try:
            info: AnyModelConfig = self._probe(Path(model_path), config)  # type: ignore

            def begin_transfer() -> bool:
                nonlocal source_recovery_sentinel_created
                if source_root is not None and active_sentinel_created and not self._has_recovery_sentinel(source_root):
                    self._write_recovery_sentinel(source_root)
                    source_recovery_sentinel_created = True
                return True

            model_id = self._install_path_with_info(model_path, config, info, before_transfer=begin_transfer)
            installed = True
            return model_id
        except InstallRecoveryRequiredError:
            preserve_source = True
            raise
        finally:
            if active_sentinel_created and source_root is not None:
                delete_active_install_sentinel(source_root)
            if (
                active_sentinel_created
                and source_root is not None
                and (installed or (source_recovery_sentinel_created and not preserve_source))
            ):
                self._delete_recovery_sentinel(source_root)

    def _install_path_with_info(
        self,
        model_path: Path,
        config: ModelRecordChanges,
        info: AnyModelConfig,
        before_transfer: Optional[Callable[[], bool]] = None,
    ) -> str:
        # The key names the directory the model is moved into. `ModelRecordChanges` validates a client-supplied key,
        # but a caller can build one without validation, so check again here - before anything is created or moved.
        if not is_plain_filename(info.key):
            raise ValueError(f"Invalid model key {info.key!r}: it must be a plain filename")
        dest_dir = self.app_config.models_path / info.key
        try:
            create_active_install_sentinel(dest_dir)
        except FileExistsError as e:
            raise InstallCancellationConflictError(
                f"Cannot install model to {dest_dir}: another install or orphan cleanup is using it."
            ) from e
        try:
            return self._install_path_with_info_claimed(model_path, config, info, before_transfer)
        finally:
            delete_active_install_sentinel(dest_dir)

    def _install_path_with_info_claimed(
        self,
        model_path: Path,
        config: ModelRecordChanges,
        info: AnyModelConfig,
        before_transfer: Optional[Callable[[], bool]],
    ) -> str:
        dest_dir = self.app_config.models_path / info.key
        moved: list[tuple[Path, Path]] = []
        source_is_directory = model_path.is_dir()
        destination_created = False
        pending_dest: Optional[Path] = None
        try:
            if dest_dir.exists():
                raise DuplicateModelException(
                    f"Cannot install model {model_path.name} to {dest_dir}: destination already exists"
                )
            dest_dir.mkdir(parents=True)
            destination_created = True
            # Protect every file in the destination until the model record is committed. An admin can run orphan
            # cleanup concurrently with this transfer, and the directory is not registered until _register().
            self._write_recovery_sentinel(dest_dir)
            if before_transfer is not None and not before_transfer():
                raise _InstallCancelledBeforeTransfer()
            dest_path = dest_dir / model_path.name if model_path.is_file() else dest_dir
            if model_path.is_file():
                try:
                    self._move_with_retries(model_path, dest_path)
                except Exception as move_error:
                    if dest_path.exists() or dest_path.is_symlink():
                        raise InstallRecoveryRequiredError(
                            f"Install recovery required after {move_error}. Source: {model_path.resolve()}; "
                            f"destination: {dest_dir.resolve()}. Unrecognized destination artifact: "
                            f"{dest_path.resolve()}"
                        ) from move_error
                    raise
            elif model_path.is_dir():
                # Move the contents of the directory, not the directory itself
                for item in model_path.iterdir():
                    item_dest = dest_dir / item.name
                    pending_dest = item_dest
                    self._move_with_retries(item, item_dest)
                    moved.append((item, item_dest))
                    pending_dest = None
        except InstallRecoveryRequiredError:
            # Recovery destinations can share TMPDIR_PREFIX with remote staging dirs. Keep their artifacts out of
            # startup's dangling-install cleanup even when the configured model key uses that prefix.
            self._retain_recovery_destination(dest_dir)
            raise
        except Exception as transfer_error:
            if not destination_created:
                if isinstance(transfer_error, FileExistsError):
                    raise DuplicateModelException(
                        f"A model named {model_path.name} is already installed at {dest_dir.as_posix()}"
                    ) from transfer_error
                raise
            if source_is_directory:
                rollback_errors: list[str] = []
                if pending_dest is not None and (pending_dest.exists() or pending_dest.is_symlink()):
                    rollback_errors.append(f"unrecognized destination artifact: {pending_dest.resolve()}")
                for moved_source, moved_dest in reversed(moved):
                    try:
                        if moved_source.exists() or moved_source.is_symlink():
                            raise FileExistsError(f"Refusing to overwrite recreated source {moved_source}")
                        self._restore_moved_path(moved_dest, moved_source)
                    except Exception as rollback_error:
                        rollback_errors.append(
                            f"could not restore {moved_dest.resolve()} to {moved_source.resolve()}: {rollback_error}"
                        )
                if rollback_errors:
                    details = "; ".join(rollback_errors)
                    self._retain_recovery_destination(dest_dir)
                    raise InstallRecoveryRequiredError(
                        f"Install recovery required after {transfer_error}. Source: {model_path.resolve()}; "
                        f"destination: {dest_dir.resolve()}. {details}"
                    ) from transfer_error
                moved.clear()
                try:
                    remaining = list(dest_dir.iterdir())
                    if remaining:
                        leftovers = ", ".join(str(path.resolve()) for path in remaining)
                        raise OSError(f"unexpected destination artifacts remain: {leftovers}")
                    dest_dir.rmdir()
                    self._delete_recovery_sentinel(dest_dir)
                except Exception as cleanup_error:
                    self._retain_recovery_destination(dest_dir)
                    raise InstallRecoveryRequiredError(
                        f"Install recovery required after {transfer_error}. Source: {model_path.resolve()}; "
                        f"destination: {dest_dir.resolve()}. Could not remove owned empty destination: "
                        f"{cleanup_error}"
                    ) from transfer_error
                raise
            if dest_dir.exists():
                try:
                    remaining = list(dest_dir.iterdir())
                    if remaining:
                        leftovers = ", ".join(str(path.resolve()) for path in remaining)
                        raise OSError(f"unexpected destination artifacts remain: {leftovers}")
                    dest_dir.rmdir()
                    self._delete_recovery_sentinel(dest_dir)
                except Exception as cleanup_error:
                    self._retain_recovery_destination(dest_dir)
                    raise InstallRecoveryRequiredError(
                        f"Install recovery required after {transfer_error}. Source: {model_path.resolve()}; "
                        f"destination: {dest_dir.resolve()}. Could not remove owned empty destination: "
                        f"{cleanup_error}"
                    ) from transfer_error
            raise

        return self._register(
            dest_path,
            config,
            info,
        )

    def heuristic_import(
        self,
        source: str,
        config: Optional[ModelRecordChanges] = None,
        access_token: Optional[str] = None,
        inplace: Optional[bool] = False,
    ) -> ModelInstallJob:
        """Install a model using pattern matching to infer the type of source."""
        source_obj = self._guess_source(source)
        if isinstance(source_obj, LocalModelSource):
            source_obj.inplace = inplace
        elif isinstance(source_obj, HFModelSource) or isinstance(source_obj, URLModelSource):
            source_obj.access_token = access_token
        return self.import_model(source_obj, config)

    def import_model(self, source: ModelSource, config: Optional[ModelRecordChanges] = None) -> ModelInstallJob:  # noqa D102
        self._wait_for_restore_complete()

        source_key = str(source)
        with self._install_condition:
            known_job_ids = {job.id for job in self._install_jobs if job.source == source}
            while source_key in self._pending_sources:
                self._install_condition.wait()

            # Prefer a live job. Waiting can leave this source with both a job that was registered while we waited
            # and has since gone terminal, and a live one; returning the dead one would report a failure for a
            # source that is actively installing.
            similar_jobs = [job for job in self._install_jobs if job.source == source and not job.in_terminal_state]
            if similar_jobs:
                self._logger.warning(f"There is already an active install job for {source}. Not enqueuing.")
                return similar_jobs[0]

            # No live job, but a concurrent owner may have registered one for us while we waited. Return it even if
            # it is already terminal - we asked at the same time it did, so we get the same answer.
            new_jobs = [job for job in self._install_jobs if job.source == source and job.id not in known_job_ids]
            if new_jobs:
                return new_jobs[0]
            self._pending_sources.add(source_key)

        try:
            if isinstance(source, LocalModelSource):
                install_job = self._import_local_model(source, config)
                self._put_in_queue(install_job)  # synchronously install
            elif isinstance(source, HFModelSource):
                install_job = self._import_from_hf(source, config)
            elif isinstance(source, URLModelSource):
                install_job = self._import_from_url(source, config)
            elif isinstance(source, ExternalModelSource):
                install_job = self._import_external_model(source, config)
                self._put_in_queue(install_job)
            else:
                raise ValueError(f"Unsupported model source: '{type(source)}'")
        except BaseException:
            with self._install_condition:
                self._pending_sources.remove(source_key)
                self._install_condition.notify_all()
            raise

        with self._install_condition:
            self._install_jobs.append(install_job)
            self._pending_sources.remove(source_key)
            self._install_condition.notify_all()
        return install_job

    def list_jobs(self) -> List[ModelInstallJob]:  # noqa D102
        return self._install_jobs

    def get_job_by_source(self, source: ModelSource) -> List[ModelInstallJob]:  # noqa D102
        return [x for x in self._install_jobs if x.source == source]

    def get_job_by_id(self, id: int) -> ModelInstallJob:  # noqa D102
        jobs = [x for x in self._install_jobs if x.id == id]
        if not jobs:
            raise ValueError(f"No job with id {id} known")
        assert len(jobs) == 1
        assert isinstance(jobs[0], ModelInstallJob)
        return jobs[0]

    def wait_for_job(self, job: ModelInstallJob, timeout: int = 0) -> ModelInstallJob:
        """Block until the indicated job has reached terminal state, or when timeout limit reached."""
        start = time.time()
        while not job.in_terminal_state:
            if self._install_completed_event.wait(timeout=5):  # in case we miss an event
                self._install_completed_event.clear()
            if timeout > 0 and time.time() - start > timeout:
                raise TimeoutError("Timeout exceeded")
        return job

    def wait_for_installs(self, timeout: int = 0) -> List[ModelInstallJob]:  # noqa D102
        """Block until all installation jobs are done."""
        start = time.time()
        restore_timeout = timeout if timeout > 0 else None
        if not self._wait_for_restore_complete(timeout=restore_timeout):
            raise TimeoutError("Timeout exceeded")

        while True:
            # The completion callback removes a download from this cache while holding
            # the same lock it uses to enqueue the install. Do not observe the cache
            # between those two operations.
            with self._lock:
                downloads_pending = bool(self._download_cache)
            if not downloads_pending:
                break
            if self._downloads_changed_event.wait(timeout=0.25):  # in case we miss an event
                self._downloads_changed_event.clear()
            if timeout > 0 and time.time() - start > timeout:
                raise TimeoutError("Timeout exceeded")
        self._install_queue.join()

        return self._install_jobs

    def cancel_job(self, job: ModelInstallJob) -> None:
        """Cancel the indicated job."""
        with self._lock:
            if job.id in self._remote_download_operations:
                raise InstallDownloadConflictError(
                    "A remote download is being prepared; wait for it to finish before cancelling."
                )
            if job.complete or job.cancelled:
                return
            if job._recovery_required or (
                job._install_tmpdir is not None and self._has_recovery_sentinel(job._install_tmpdir)
            ):
                raise InstallRecoveryRequiredError(
                    "Cannot cancel an install that requires recovery; preserve its files for manual recovery."
                )
            if self._active_install_job is job:
                if job._install_phase == "transferring":
                    raise InstallCancellationConflictError(
                        "Cannot cancel while install files are being moved; allow the transfer to finish."
                    )
                job._cancel_requested = True
                return
            job.cancel()
            download_callback_pending = any(cached_job is job for cached_job in self._download_cache.values())
        self._logger.warning(f"Cancelling {job.source}")
        if dj := job._multifile_job:
            if not dj.in_terminal_state or download_callback_pending:
                # The terminal queue callback still owns cleanup while the job remains in _download_cache.
                self._download_queue.cancel_job(dj)
                return
        self._cleanup_cancelled_install(job)

    def pause_job(self, job: ModelInstallJob) -> None:
        """Pause the indicated job, preserving partial downloads."""
        with self._lock:
            if job.id in self._remote_download_operations:
                raise InstallDownloadConflictError(
                    "A remote download is being prepared; wait for it to finish before pausing."
                )
            if job.in_terminal_state:
                return
            if job.downloads_done or job.running or self._active_install_job is job:
                raise InstallDownloadConflictError("The install has started and cannot be paused.")
            job.status = InstallStatus.PAUSED
        self._logger.warning(f"Pausing {job.source}")
        if dj := job._multifile_job:
            for part in dj.download_parts:
                self._download_queue.pause_job(part)
        self._write_install_marker(job, status=InstallStatus.PAUSED)

    def resume_job(self, job: ModelInstallJob) -> None:
        """Resume a previously paused job."""
        if not self._begin_remote_download_operation(job, require_paused=True):
            return
        self._logger.info(f"Resuming {job.source}")
        try:
            self._resume_remote_download(job, operation_reserved=True)
        finally:
            self._end_remote_download_operation(job)

    def restart_failed(self, job: ModelInstallJob) -> None:
        """Restart failed or non-resumable downloads for a job."""
        if job._recovery_required or (
            job._install_tmpdir is not None and self._has_recovery_sentinel(job._install_tmpdir)
        ):
            raise InstallRecoveryRequiredError("Cannot restart an install that requires recovery.")
        if not isinstance(job.source, (HFModelSource, URLModelSource)):
            return
        if not job.download_parts:
            return
        if not any(part.resume_required or part.errored for part in job.download_parts):
            return
        sources_to_restart = {str(part.source) for part in job.download_parts if not part.complete}
        if not sources_to_restart:
            return
        self._begin_remote_download_operation(job)
        previous_status = job.status
        try:
            remote_files, metadata = self._remote_files_from_source(job.source)
            remote_files = [rf for rf in remote_files if str(rf.url) in sources_to_restart]
            if not remote_files:
                raise RuntimeError("No remote files are available to download")
            subfolders = job.source.subfolders if isinstance(job.source, HFModelSource) else []
            job.status = InstallStatus.WAITING
            self._enqueue_remote_download(
                job=job,
                source=job.source,
                remote_files=remote_files,
                metadata=metadata,
                destdir=job._install_tmpdir or job.local_path,
                subfolder=job.source.subfolder
                if isinstance(job.source, HFModelSource) and len(subfolders) <= 1
                else None,
                subfolders=subfolders if len(subfolders) > 1 else None,
                clear_partials=True,
            )
        except BaseException:
            job.status = previous_status
            raise
        finally:
            self._end_remote_download_operation(job)

    def restart_file(self, job: ModelInstallJob, file_source: str) -> None:
        """Restart a specific file download for a job."""
        if job._recovery_required or (
            job._install_tmpdir is not None and self._has_recovery_sentinel(job._install_tmpdir)
        ):
            raise InstallRecoveryRequiredError("Cannot restart an install that requires recovery.")
        if not isinstance(job.source, (HFModelSource, URLModelSource)):
            return
        self._begin_remote_download_operation(job)
        previous_status = job.status
        try:
            remote_files, metadata = self._remote_files_from_source(job.source)
            remote_files = [rf for rf in remote_files if str(rf.url) == file_source]
            if not remote_files:
                return
            subfolders = job.source.subfolders if isinstance(job.source, HFModelSource) else []
            job.status = InstallStatus.WAITING
            self._enqueue_remote_download(
                job=job,
                source=job.source,
                remote_files=remote_files,
                metadata=metadata,
                destdir=job._install_tmpdir or job.local_path,
                subfolder=job.source.subfolder
                if isinstance(job.source, HFModelSource) and len(subfolders) <= 1
                else None,
                subfolders=subfolders if len(subfolders) > 1 else None,
                clear_partials=True,
            )
        except BaseException:
            job.status = previous_status
            raise
        finally:
            self._end_remote_download_operation(job)

    def prune_jobs(self) -> None:
        """Prune all completed and errored jobs."""
        # Filter and rebind under the condition. Unlocked, a registration made by a concurrent import_model()
        # between the two would be dropped, leaving a live install invisible to the duplicate check. Rebind rather
        # than mutating in place so that readers already iterating the old list are not silently truncated.
        with self._install_condition:
            self._install_jobs = [x for x in self._install_jobs if not x.in_terminal_state]

    def _migrate_yaml(self) -> None:
        db_models = self.record_store.all_models()

        legacy_models_yaml_path = (
            self._app_config.legacy_models_yaml_path or self._app_config.root_path / "configs" / "models.yaml"
        )

        # The old path may be relative to the root path
        if not legacy_models_yaml_path.exists():
            legacy_models_yaml_path = Path(self._app_config.root_path, legacy_models_yaml_path)

        if legacy_models_yaml_path.exists():
            with open(legacy_models_yaml_path, "rt", encoding=locale.getpreferredencoding()) as file:
                legacy_models_yaml = yaml.safe_load(file)

            yaml_metadata = legacy_models_yaml.pop("__metadata__")
            yaml_version = yaml_metadata.get("version")

            if yaml_version != "3.0.0":
                raise ValueError(
                    f"Attempted migration of unsupported `models.yaml` v{yaml_version}. Only v3.0.0 is supported. Exiting."
                )

            self._logger.info(
                f"Starting one-time migration of {len(legacy_models_yaml.items())} models from {str(legacy_models_yaml_path)}. This may take a few minutes."
            )

            if len(db_models) == 0 and len(legacy_models_yaml.items()) != 0:
                for model_key, stanza in legacy_models_yaml.items():
                    _, _, model_name = str(model_key).split("/")
                    model_path = Path(stanza["path"])
                    if not model_path.is_absolute():
                        model_path = self._app_config.models_path / model_path
                    model_path = model_path.resolve()

                    config = ModelRecordChanges(
                        name=model_name,
                        description=stanza.get("description"),
                    )
                    legacy_config_path = stanza.get("config")
                    if legacy_config_path:
                        # In v3, these paths were relative to the root. Migrate them to be relative to the legacy_conf_dir.
                        legacy_config_path = self._app_config.root_path / legacy_config_path
                        if legacy_config_path.is_relative_to(self._app_config.legacy_conf_path):
                            legacy_config_path = legacy_config_path.relative_to(self._app_config.legacy_conf_path)
                        config.config_path = str(legacy_config_path)
                    try:
                        id = self.register_path(model_path=model_path, config=config)
                        self._logger.info(f"Migrated {model_name} with id {id}")
                    except Exception as e:
                        self._logger.warning(f"Model at {model_path} could not be migrated: {e}")

            # Rename `models.yaml` to `models.yaml.bak` to prevent re-migration
            legacy_models_yaml_path.rename(legacy_models_yaml_path.with_suffix(".yaml.bak"))

        # Unset the path - we are done with it either way
        self._app_config.legacy_models_yaml_path = None

    def unregister(self, key: str) -> None:  # noqa D102
        self.record_store.del_model(key)

    def delete(self, key: str) -> None:  # noqa D102
        """Unregister the model. Delete its files only if they are within our models directory."""
        model = self.record_store.get_model(key)
        model_path = self.app_config.models_path / model.path

        if model_path.is_relative_to(self.app_config.models_path):
            # If the models is in the Invoke-managed models dir, we delete it
            self.unconditionally_delete(key)
        else:
            # Else we only unregister it, leaving the file in place
            self.unregister(key)

    def unconditionally_delete(self, key: str) -> None:  # noqa D102
        model = self.record_store.get_model(key)
        model_path = self.app_config.models_path / model.path
        # Models are stored in a directory named by their key. To delete the model on disk, we delete the entire
        # directory. However, the path we store in the model record may be either a file within the key directory,
        # or the directory itself. So we have to handle both cases.
        if model_path.is_file() or model_path.is_symlink():
            # Delete the individual model file, not the entire parent directory.
            # Other unrelated files may exist in the same directory.
            model_path.unlink()
            # Clean up the parent directory only if it is now empty
            if model_path.parent != self.app_config.models_path and not any(model_path.parent.iterdir()):
                model_path.parent.rmdir()
        elif model_path.is_dir():
            # Sanity check - folder models should be in their own directory under the models dir. The path should
            # not be the Invoke models dir itself!
            assert model_path != self.app_config.models_path
            rmtree(model_path)
        self.unregister(key)

    @classmethod
    def _download_cache_path(cls, source: Union[str, AnyHttpUrl], app_config: InvokeAIAppConfig) -> Path:
        escaped_source = slugify(str(source))
        return app_config.download_cache_path / escaped_source

    def download_and_cache_model(
        self,
        source: str | AnyHttpUrl,
    ) -> Path:
        """Download the model file located at source to the models cache and return its Path."""
        model_path = self._download_cache_path(str(source), self._app_config)

        # We expect the cache directory to contain one and only one downloaded file or directory.
        # We don't know the file's name in advance, as it is set by the download
        # content-disposition header.
        if model_path.exists():
            contents: List[Path] = list(model_path.iterdir())
            if len(contents) > 0:
                return contents[0]

        # Serialize concurrent downloads of the same source. Parallel multi-GPU sessions can each
        # request the same remote model (e.g. the LaMa infill model) at once; without this lock they
        # both download into the same cache directory and collide on the final rename, which fails on
        # Windows with "WinError 32: the file is being used by another process". The other waiters
        # find the completed download on the post-lock re-check below and skip downloading.
        with self._download_cache_lock(str(source)):
            if model_path.exists():
                contents = list(model_path.iterdir())
                if len(contents) > 0:
                    return contents[0]

            model_path.mkdir(parents=True, exist_ok=True)
            model_source = self._guess_source(str(source))
            remote_files, _ = self._remote_files_from_source(model_source)
            # Handle multiple subfolders for HFModelSource
            subfolders = model_source.subfolders if isinstance(model_source, HFModelSource) else []
            job = self._multifile_download(
                dest=model_path,
                remote_files=remote_files,
                subfolder=model_source.subfolder
                if isinstance(model_source, HFModelSource) and len(subfolders) <= 1
                else None,
                subfolders=subfolders if len(subfolders) > 1 else None,
            )
            files_string = "file" if len(remote_files) == 1 else "files"
            self._logger.info(f"Queuing model download: {source} ({len(remote_files)} {files_string})")
            self._download_queue.wait_for_job(job)
            if job.complete:
                assert job.download_path is not None
                return job.download_path
            else:
                raise Exception(job.error)

    def _download_cache_lock(self, source: str) -> threading.Lock:
        """Return the lock that serializes downloads for a given source, creating it on first use."""
        with self._download_cache_locks_guard:
            lock = self._download_cache_locks.get(source)
            if lock is None:
                lock = threading.Lock()
                self._download_cache_locks[source] = lock
            return lock

    def _remote_files_from_source(
        self, source: ModelSource
    ) -> Tuple[List[RemoteModelFile], Optional[AnyModelRepoMetadata]]:
        metadata = None
        if isinstance(source, HFModelSource):
            metadata = HuggingFaceMetadataFetch(self._session).from_id(source.repo_id, source.variant)
            assert isinstance(metadata, ModelMetadataWithFiles)
            # Use subfolders property which handles '+' separated multiple subfolders
            subfolders = source.subfolders
            return (
                metadata.download_urls(
                    variant=source.variant or self._guess_variant(),
                    subfolder=source.subfolder if len(subfolders) <= 1 else None,
                    subfolders=subfolders if len(subfolders) > 1 else None,
                    session=self._session,
                ),
                metadata,
            )

        if isinstance(source, URLModelSource):
            try:
                fetcher = self.get_fetcher_from_url(str(source.url))
                kwargs: dict[str, Any] = {"session": self._session}
                metadata = fetcher(**kwargs).from_url(source.url)
                assert isinstance(metadata, ModelMetadataWithFiles)
                return metadata.download_urls(session=self._session), metadata
            except ValueError:
                pass

            return [RemoteModelFile(url=self._normalize_huggingface_blob_url(source.url), path=Path("."), size=0)], None

        raise Exception(f"No files associated with {source}")

    def _guess_source(self, source: str) -> ModelSource:
        """Turn a source string into a ModelSource object."""
        variants = "|".join(ModelRepoVariant.__members__.values())
        hf_repoid_re = f"^([^/:]+/[^/:]+)(?::({variants})?(?::/?([^:]+))?)?$"
        source_obj: Optional[StringLikeSource] = None
        source_stripped = source.strip('"')

        if source_stripped.startswith("external://"):
            external_id = source_stripped.removeprefix("external://")
            provider_id, _, provider_model_id = external_id.partition("/")
            if not provider_id or not provider_model_id:
                raise ValueError(f"Invalid external model source: '{source_stripped}'")
            source_obj = ExternalModelSource(provider_id=provider_id, provider_model_id=provider_model_id)
        elif Path(source_stripped).exists():  # A local file or directory
            source_obj = LocalModelSource(path=Path(source_stripped))
        elif match := re.match(hf_repoid_re, source):
            source_obj = HFModelSource(
                repo_id=match.group(1),
                variant=ModelRepoVariant(match.group(2)) if match.group(2) else None,  # pass None rather than ''
                subfolder=Path(match.group(3)) if match.group(3) else None,
            )
        elif re.match(r"^https?://[^/]+", source):
            source_obj = URLModelSource(
                url=Url(source),
            )
        else:
            raise ValueError(f"Unsupported model source: '{source}'")
        return source_obj

    # --------------------------------------------------------------------------------------------
    # Internal functions that manage the installer threads
    # --------------------------------------------------------------------------------------------
    def _start_installer_thread(self) -> None:
        self._install_thread = threading.Thread(target=self._install_next_item, daemon=True)
        self._install_thread.start()
        self._running = True

    @staticmethod
    def _safe_rmtree(path: Path, logger: Any) -> None:
        """Remove a directory tree with retry logic for Windows file locking issues.

        On Windows, memory-mapped files may not be immediately released even after
        the file handle is closed. This function retries the removal with garbage
        collection to help release any lingering references.
        """
        max_retries = 3
        retry_delay = 0.5  # seconds

        for attempt in range(max_retries):
            try:
                # Force garbage collection to release any lingering file references
                gc.collect()
                rmtree(path)
                return
            except PermissionError as e:
                if attempt < max_retries - 1 and sys.platform == "win32":
                    logger.warning(
                        f"Failed to remove {path} (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to remove temporary directory {path}: {e}")
                    # On final failure, don't raise - the temp dir will be cleaned up on next startup
                    return
            except Exception as e:
                logger.error(f"Unexpected error removing {path}: {e}")
                return

    def _install_next_item(self) -> None:
        self._logger.debug(f"Installer thread {threading.get_ident()} starting")
        while True:
            if self._stop_event.is_set():
                break
            self._logger.debug(f"Installer thread {threading.get_ident()} polling")
            try:
                job = self._install_queue.get(timeout=1)
            except Empty:
                continue
            assert job.local_path is not None
            with self._lock:
                should_process = not self._stop_event.is_set()
                if should_process:
                    self._active_install_job = job
            try:
                if not should_process:
                    continue
                if job.cancelled or job._cancel_requested:
                    job.cancel()
                    self._signal_job_cancelled(job)

                elif job.errored:
                    self._signal_job_errored(job)

                elif job.waiting or job.downloads_done:
                    self._register_or_install(job)

            except Exception as e:
                # Expected errors include InvalidModelConfigException, DuplicateModelException, OSError, but we must
                # gracefully handle _any_ error here.
                self._set_error(job, e)
                if job._recovery_required:
                    try:
                        self._write_install_marker(job, status=InstallStatus.ERROR)
                    except Exception as marker_error:
                        self._logger.error(
                            f"Failed to persist install recovery marker in {job._install_tmpdir}: {marker_error}"
                        )

            finally:
                if should_process:
                    with self._lock:
                        if self._active_install_job is job:
                            self._active_install_job = None
                        job._install_phase = None
                    # Keep our staging claim until cleanup is finished so orphan cleanup cannot
                    # acquire the path in the gap before the directory is removed.
                    if (
                        job._install_tmpdir is not None
                        and not job._recovery_required
                        and not job._install_tmpdir_claim_conflict
                    ):
                        self._safe_rmtree(job._install_tmpdir, self._logger)
                    self._release_job_source_protection(
                        job,
                        preserve_recovery=job._recovery_required
                        or (job._source_recovery_sentinel_preexisting and not job.complete),
                    )
                self._install_completed_event.set()
                self._install_queue.task_done()
        self._logger.info(f"Installer thread {threading.get_ident()} exiting")

    def _begin_install_transfer(self, job: ModelInstallJob) -> bool:
        with self._lock:
            if job._cancel_requested:
                return False
            if job._install_tmpdir is not None:
                self._claim_install_tmpdir(job)
                self._write_recovery_sentinel(job._install_tmpdir)
                job._install_tmpdir_recovery_sentinel_created = True
            if job._source_protection_root is not None and job._source_active_sentinel_created:
                if not self._has_recovery_sentinel(job._source_protection_root):
                    self._write_recovery_sentinel(job._source_protection_root)
                    job._source_recovery_sentinel_created = True
            job._install_phase = "transferring"
            return True

    def _register_or_install(self, job: ModelInstallJob) -> None:
        with self._lock:
            if job._cancel_requested:
                job.cancel()
                cancel_before_start = True
            else:
                job._install_phase = "preflight"
                cancel_before_start = False
        if cancel_before_start:
            self._signal_job_cancelled(job)
            return

        # A restored completed download has no in-memory claim yet. Claim it before probing or
        # touching its marker so orphan cleanup cannot delete the staging tree concurrently.
        self._claim_install_tmpdir(job)

        if isinstance(job.source, LocalModelSource):
            source_root, active_sentinel_created, recovery_sentinel_preexisting = self._protect_managed_source_path(
                Path(job.source.path), allow_recovery=True
            )
            job._source_protection_root = source_root
            job._source_active_sentinel_created = active_sentinel_created
            job._source_recovery_sentinel_preexisting = recovery_sentinel_preexisting

        if isinstance(job.source, ExternalModelSource):
            with self._lock:
                if job._cancel_requested:
                    job.cancel()
                    cancel_before_registering = True
                else:
                    job._install_phase = "transferring"
                    cancel_before_registering = False
            if cancel_before_registering:
                self._signal_job_cancelled(job)
                return
            self._register_external_model(job)
            return
        # local jobs will be in waiting state, remote jobs will be downloading state
        job.total_bytes = self._stat_size(job.local_path)
        job.bytes = job.total_bytes
        self._signal_job_running(job)
        job.config_in.source = str(job.source)
        job.config_in.source_type = MODEL_SOURCE_TO_TYPE_MAP[job.source.__class__]
        # enter the metadata, if there is any
        if isinstance(job.source_metadata, (HuggingFaceMetadata)):
            job.config_in.source_api_response = job.source_metadata.api_response

        if job._install_tmpdir is not None:
            self._delete_install_marker(job._install_tmpdir)

        if job.inplace:
            with self._lock:
                if job._cancel_requested:
                    job.cancel()
                    cancel_before_start = True
                else:
                    job._install_phase = "transferring"
                    cancel_before_start = False
            if cancel_before_start:
                self._signal_job_cancelled(job)
                return
            key = self._register(job.local_path, job.config_in)
        else:
            try:
                info = self._probe(Path(job.local_path), job.config_in)  # type: ignore
                key = self._install_path_with_info(
                    Path(job.local_path),
                    job.config_in,
                    info,
                    before_transfer=lambda: self._begin_install_transfer(job),
                )
            except _InstallCancelledBeforeTransfer:
                with self._lock:
                    job._install_phase = None
                    job.cancel()
                self._signal_job_cancelled(job)
                return
            except InstallRecoveryRequiredError:
                job._recovery_required = True
                if job._install_tmpdir is not None:
                    try:
                        self._write_install_marker(job, status=job.status)
                    except Exception as marker_error:
                        self._logger.error(
                            f"Failed to persist install recovery marker in {job._install_tmpdir}: {marker_error}"
                        )
                raise
            except Exception:
                if job._install_tmpdir is not None:
                    self._delete_recovery_sentinel(job._install_tmpdir)
                raise
            if job._install_tmpdir is not None:
                self._delete_recovery_sentinel(job._install_tmpdir)
        job.config_out = self.record_store.get_model(key)
        self._signal_job_completed(job)

    def _register_external_model(self, job: ModelInstallJob) -> None:
        job.total_bytes = 0
        job.bytes = 0
        self._signal_job_running(job)
        job.config_in.source = str(job.source)
        job.config_in.source_type = MODEL_SOURCE_TO_TYPE_MAP[job.source.__class__]

        provider_id = job.source.provider_id
        provider_model_id = job.source.provider_model_id
        capabilities = job.config_in.capabilities or ExternalModelCapabilities()
        default_settings = (
            job.config_in.default_settings
            if isinstance(job.config_in.default_settings, ExternalApiModelDefaultSettings)
            else None
        )
        name = job.config_in.name or f"{provider_id} {provider_model_id}"
        key = job.config_in.key or slugify(f"{provider_id}-{provider_model_id}")
        # External registration builds its config directly, so it never passes through `_probe`'s check.
        if not is_plain_filename(key):
            raise InvalidModelConfigException(f"Invalid model key {key!r}: it must be a plain filename")

        existing_external = next(
            (
                model
                for model in self.record_store.search_by_attr(
                    base_model=BaseModelType.External, model_type=ModelType.ExternalImageGenerator
                )
                if isinstance(model, ExternalApiModelConfig)
                and model.provider_id == provider_id
                and model.provider_model_id == provider_model_id
            ),
            None,
        )

        if existing_external is not None:
            key = existing_external.key
        else:
            try:
                self.record_store.get_model(key)
                raise DuplicateModelException(
                    f"Model key '{key}' already exists. Provide a different key to install this external model."
                )
            except UnknownModelException:
                pass

        config = ExternalApiModelConfig(
            key=key,
            name=name,
            description=job.config_in.description,
            provider_id=provider_id,
            provider_model_id=provider_model_id,
            capabilities=capabilities,
            default_settings=default_settings,
            source=str(job.source),
            source_type=MODEL_SOURCE_TO_TYPE_MAP[job.source.__class__],
            path="",
            hash="",
            file_size=0,
        )

        if existing_external is not None:
            self.record_store.replace_model(existing_external.key, config)
        else:
            self.record_store.add_model(config)

        job.config_out = self.record_store.get_model(config.key)
        self._signal_job_completed(job)

    def _set_error(self, install_job: ModelInstallJob, excp: Exception) -> None:
        multifile_download_job = install_job._multifile_job
        if multifile_download_job and any(
            x.content_type is not None and "text/html" in x.content_type for x in multifile_download_job.download_parts
        ):
            install_job.set_error(
                ValueError(
                    f"At least one file in {install_job.local_path} is an HTML page, not a model. This can happen when an access token is required to download."
                )
            )
        else:
            install_job.set_error(excp)
        self._signal_job_errored(install_job)

    # --------------------------------------------------------------------------------------------
    # Internal functions that manage the models directory
    # --------------------------------------------------------------------------------------------
    def _remove_dangling_install_dirs(self) -> None:
        """Remove leftover tmpdirs from aborted installs."""
        path = self._app_config.models_path
        registered_model_paths = {
            (path / model.path).resolve() for model in self.record_store.all_models() if model.path
        }
        for tmpdir in path.glob(f"{TMPDIR_PREFIX}*"):
            if has_active_install_sentinel(tmpdir):
                self._logger.debug(f"Preserving active install directory {tmpdir}")
                continue
            resolved_tmpdir = tmpdir.resolve()
            if any(
                model_path == resolved_tmpdir or model_path.is_relative_to(resolved_tmpdir)
                for model_path in registered_model_paths
            ):
                self._logger.debug(f"Preserving registered model directory {tmpdir}")
                continue
            if self._has_recovery_sentinel(tmpdir):
                self._logger.warning(f"Preserving install recovery data in {tmpdir}")
                continue
            marker = self._read_install_marker(tmpdir)
            if marker is None:
                self._logger.info(f"Removing dangling temporary directory {tmpdir}")
                self._remove_unclaimed_install_tmpdir(tmpdir)
                continue
            status = marker.get("status")
            if marker.get("recovery_required") is True:
                self._logger.warning(f"Preserving install recovery data in {tmpdir}")
                continue
            if status in {InstallStatus.COMPLETED.value, InstallStatus.ERROR.value, InstallStatus.CANCELLED.value}:
                self._logger.info(f"Removing completed/errored temporary directory {tmpdir}")
                self._remove_unclaimed_install_tmpdir(tmpdir)

    def _scan_for_missing_models(self) -> list[AnyModelConfig]:
        """Scan the models directory for missing models and return a list of them."""
        missing_models: list[AnyModelConfig] = []
        for model_config in self.record_store.all_models():
            if model_config.base == BaseModelType.External or model_config.format == ModelFormat.ExternalApi:
                continue
            if not (self.app_config.models_path / model_config.path).resolve().exists():
                missing_models.append(model_config)
        return missing_models

    def _register_orphaned_models(self) -> None:
        """Scan the invoke-managed models directory for orphaned models and registers them.

        This is typically only used during testing with a new DB or when using the memory DB, because those are the
        only situations in which we may have orphaned models in the models directory.
        """
        installed_model_paths = {
            (self._app_config.models_path / x.path).resolve() for x in self.record_store.all_models()
        }
        models_path = self._app_config.models_path.resolve()

        # The bool returned by this callback determines if the model is added to the list of models found by the search
        def on_model_found(model_path: Path) -> bool:
            resolved_path = model_path.resolve()
            if is_recovery_protected_path(resolved_path, models_path):
                self._logger.warning(f"Skipping recovery-protected model path {model_path}")
                return False
            # Already registered models should be in the list of found models, but not re-registered.
            if resolved_path in installed_model_paths:
                return True
            # Skip core models entirely - these aren't registered with the model manager.
            for special_directory in [
                self.app_config.models_path / "core",
                self.app_config.convert_cache_dir,
                self.app_config.download_cache_dir,
            ]:
                if resolved_path.is_relative_to(special_directory):
                    return False
            try:
                model_id = self.register_path(model_path)
                self._logger.info(f"Registered {model_path.name} with id {model_id}")
            except DuplicateModelException:
                # In case a duplicate models sneaks by, we will ignore this error - we "found" the model
                pass
            except InvalidModelConfigException as e:
                # A file we cannot register at all - unidentifiable with allow_unknown_models off, or
                # recognised and rejected as unusable (e.g. a truncated checkpoint). ModelSearch already
                # contains anything this callback raises, so startup survives either way; handling it
                # here makes that a property of the scan rather than of the caller, and says which file
                # was skipped and why.
                self._logger.warning(f"Skipping {model_path.name}: {e}")
                return False
            return True

        self._logger.info(f"Scanning {self._app_config.models_path} for orphaned models")
        search = ModelSearch(on_model_found=on_model_found)
        found_models = search.search(self._app_config.models_path)
        self._logger.info(f"{len(found_models)} new models registered")

    def _probe(self, model_path: Path, config: Optional[ModelRecordChanges] = None):
        config = config or ModelRecordChanges()
        # A caller may name the key it wants, and the key names a directory under `models_path` and a file under
        # `model_images`. `ModelRecordChanges` deliberately does not validate it (it is the update body too, and is
        # re-parsed from old install markers), so assert it here - the one place a caller-supplied key becomes a
        # record's key, for in-place registration as well as for a move-in install.
        if config.key is not None and not is_plain_filename(config.key):
            raise InvalidModelConfigException(f"Invalid model key {config.key!r}: it must be a plain filename")
        hash_algo = self._app_config.hashing_algorithm
        fields = config.model_dump()

        result = ModelConfigFactory.from_model_on_disk(
            mod=model_path,
            override_fields=deepcopy(fields),
            hash_algo=hash_algo,
            allow_unknown=self.app_config.allow_unknown_models,
        )

        if result.config is None:
            # A model that was recognised and then rejected as unusable (e.g. a truncated checkpoint)
            # comes with a specific reason. Report that instead of "could not identify", which would be
            # both wrong and useless to whoever has to work out why the install failed.
            if invalid := result.invalid_matches:
                reason = f"Model at {model_path} cannot be used: {invalid[0]}"
            else:
                reason = f"Could not identify model for {model_path}"
            self._logger.error(f"{reason}, detailed results: {result.details}")
            raise InvalidModelConfigException(reason)
        elif isinstance(result.config, Unknown_Config):
            self._logger.error(f"Could not identify model for {model_path}, detailed results: {result.details}")

        return result.config

    def _register(
        self, model_path: Path, config: Optional[ModelRecordChanges] = None, info: Optional[AnyModelConfig] = None
    ) -> str:
        config = config or ModelRecordChanges()

        info = info or self._probe(model_path, config)

        # Apply LoRA metadata if applicable
        model_images_path = self.app_config.models_path / "model_images"
        apply_lora_metadata(info, model_path.resolve(), model_images_path)

        model_path = model_path.resolve()
        recovery_root = model_path if model_path.is_dir() else model_path.parent

        # Models in the Invoke-managed models dir should use relative paths.
        if model_path.is_relative_to(self.app_config.models_path):
            model_path = model_path.relative_to(self.app_config.models_path)

        info.path = model_path.as_posix()

        if isinstance(info, Checkpoint_Config_Base) and info.config_path is not None:
            # Checkpoints have a config file needed for conversion. Same handling as the model weights - if it's in the
            # invoke-managed legacy config dir, we use a relative path.
            legacy_config_path = self.app_config.legacy_conf_path / info.config_path
            if legacy_config_path.is_relative_to(self.app_config.legacy_conf_path):
                legacy_config_path = legacy_config_path.relative_to(self.app_config.legacy_conf_path)
            info.config_path = legacy_config_path.as_posix()
        self.record_store.add_model(info)
        if self._has_recovery_sentinel(recovery_root):
            self._delete_recovery_sentinel(recovery_root)
        return info.key

    def _next_id(self) -> int:
        with self._lock:
            id = self._next_job_id
            self._next_job_id += 1
        return id

    def _guess_variant(self) -> Optional[ModelRepoVariant]:
        """Guess the best HuggingFace variant type to download."""
        precision = TorchDevice.choose_torch_dtype()
        return ModelRepoVariant.FP16 if precision == torch.float16 else None

    def _import_local_model(
        self, source: LocalModelSource, config: Optional[ModelRecordChanges] = None
    ) -> ModelInstallJob:
        return ModelInstallJob(
            id=self._next_id(),
            source=source,
            config_in=config or ModelRecordChanges(),
            local_path=Path(source.path),
            inplace=source.inplace or False,
        )

    def _import_from_hf(
        self,
        source: HFModelSource,
        config: Optional[ModelRecordChanges] = None,
    ) -> ModelInstallJob:
        # Add user's cached access token to HuggingFace requests
        if source.access_token is None:
            source.access_token = hf_get_token()
        remote_files, metadata = self._remote_files_from_source(source)
        return self._import_remote_model(
            source=source,
            config=config,
            remote_files=remote_files,
            metadata=metadata,
        )

    def _import_from_url(
        self,
        source: URLModelSource,
        config: Optional[ModelRecordChanges] = None,
    ) -> ModelInstallJob:
        remote_files, metadata = self._remote_files_from_source(source)
        return self._import_remote_model(
            source=source,
            config=config,
            metadata=metadata,
            remote_files=remote_files,
        )

    def _import_external_model(
        self,
        source: ExternalModelSource,
        config: Optional[ModelRecordChanges] = None,
    ) -> ModelInstallJob:
        return ModelInstallJob(
            id=self._next_id(),
            source=source,
            config_in=config or ModelRecordChanges(),
            local_path=self._app_config.models_path,
            inplace=True,
        )

    def _import_remote_model(
        self,
        source: HFModelSource | URLModelSource,
        remote_files: List[RemoteModelFile],
        metadata: Optional[AnyModelRepoMetadata],
        config: Optional[ModelRecordChanges],
    ) -> ModelInstallJob:
        if len(remote_files) == 0:
            raise ValueError(f"{source}: No downloadable files found")
        destdir = self._find_reusable_tmpdir(source)
        if destdir is None:
            destdir = Path(
                mkdtemp(
                    dir=self._app_config.models_path,
                    prefix=TMPDIR_PREFIX,
                )
            )
        install_job = ModelInstallJob(
            id=self._next_id(),
            source=source,
            config_in=config or ModelRecordChanges(),
            source_metadata=metadata,
            local_path=destdir,  # local path may change once the download has started due to content-disposition handling
            bytes=0,
            total_bytes=0,
        )
        # remember the temporary directory for later removal
        install_job._install_tmpdir = destdir

        # Handle multiple subfolders for HFModelSource
        subfolders = source.subfolders if isinstance(source, HFModelSource) else []
        return self._enqueue_remote_download(
            job=install_job,
            source=source,
            remote_files=remote_files,
            metadata=metadata,
            destdir=destdir,
            subfolder=source.subfolder if isinstance(source, HFModelSource) and len(subfolders) <= 1 else None,
            subfolders=subfolders if len(subfolders) > 1 else None,
        )

    def _enqueue_remote_download(
        self,
        job: ModelInstallJob,
        source: HFModelSource | URLModelSource,
        remote_files: List[RemoteModelFile],
        metadata: Optional[AnyModelRepoMetadata],
        destdir: Path,
        subfolder: Optional[Path] = None,
        subfolders: Optional[List[Path]] = None,
        resume_metadata: Optional[dict] = None,
        clear_partials: bool = False,
        restarted_from_scratch: Optional[set[str]] = None,
    ) -> ModelInstallJob:
        previous_source_metadata = job.source_metadata
        previous_status = job.status
        previous_local_path = job.local_path
        previous_install_tmpdir = job._install_tmpdir
        previous_total_bytes = job.total_bytes
        previous_multifile_job = job._multifile_job
        previous_download_parts = job.download_parts
        job._install_tmpdir = destdir
        claim_was_already_held = job._install_tmpdir_active_sentinel_created
        self._claim_install_tmpdir(job)

        multifile_job: Optional[MultiFileDownloadJob] = None
        try:
            multifile_job = self._multifile_download(
                remote_files=remote_files,
                dest=destdir,
                subfolder=subfolder,
                subfolders=subfolders,
                access_token=source.access_token,
                submit_job=False,  # Important! Don't submit the job until we have set our _download_cache dict
            )
            if resume_metadata:
                for part in multifile_job.download_parts:
                    meta = resume_metadata.get(str(part.source))
                    if not meta:
                        continue
                    part.canonical_url = meta.get("canonical_url") or part.canonical_url
                    part.etag = meta.get("etag") or part.etag
                    part.last_modified = meta.get("last_modified") or part.last_modified
                    part.expected_total_bytes = meta.get("expected_total_bytes") or part.expected_total_bytes
                    part.final_url = meta.get("final_url") or part.final_url
                    if meta.get("download_path"):
                        part.download_path = Path(meta.get("download_path"))
            if restarted_from_scratch:
                for part in multifile_job.download_parts:
                    if str(part.source) in restarted_from_scratch:
                        part.resume_from_scratch = True
                        part.resume_message = "Partial file missing. Restarted download from the beginning."
            with self._lock:
                if self._stop_event.is_set():
                    raise InstallDownloadConflictError(
                        "The model install service is stopping; the download was not submitted."
                    )
                if job.cancelled or job.paused:
                    raise InstallDownloadConflictError(
                        "The install changed state before the download could be submitted; retry the operation."
                    )
                if clear_partials:
                    for part in multifile_job.download_parts:
                        target_path = part.dest
                        if target_path.exists():
                            try:
                                self._logger.info(f"Deleting partial file before restart: {target_path}")
                                target_path.unlink()
                            except Exception:
                                pass
                        in_progress_path = target_path.with_name(target_path.name + ".downloading")
                        if in_progress_path.exists():
                            try:
                                self._logger.info(f"Deleting partial file before restart: {in_progress_path}")
                                in_progress_path.unlink()
                            except Exception:
                                pass
                job.source_metadata = metadata
                job.local_path = destdir
                job._install_tmpdir = destdir
                job.total_bytes = sum((x.size or 0) for x in remote_files)
                job._multifile_job = multifile_job
                job.download_parts = multifile_job.download_parts
                job.status = InstallStatus.WAITING
                self._write_install_marker(job, status=InstallStatus.WAITING)
                self._download_cache[multifile_job.id] = job
        except BaseException:
            with self._lock:
                if multifile_job is not None:
                    if self._download_cache.get(multifile_job.id) is job:
                        self._download_cache.pop(multifile_job.id, None)
                job.source_metadata = previous_source_metadata
                job.status = previous_status
                job.local_path = previous_local_path
                job._install_tmpdir = previous_install_tmpdir
                job.total_bytes = previous_total_bytes
                job._multifile_job = previous_multifile_job
                job.download_parts = previous_download_parts
            if not claim_was_already_held:
                self._release_install_tmpdir_claim(job)
            raise

        files_string = "file" if len(remote_files) == 1 else "files"
        self._logger.info(f"Queueing model install: {source} ({len(remote_files)} {files_string})")
        self._logger.debug(f"remote_files={remote_files}")
        try:
            self._download_queue.submit_multifile_download(multifile_job)
        except BaseException:
            with self._lock:
                if self._download_cache.get(multifile_job.id) is job:
                    self._download_cache.pop(multifile_job.id, None)
                job.source_metadata = previous_source_metadata
                job.status = previous_status
                job.local_path = previous_local_path
                job._install_tmpdir = previous_install_tmpdir
                job.total_bytes = previous_total_bytes
                job._multifile_job = previous_multifile_job
                job.download_parts = previous_download_parts
            if not claim_was_already_held:
                self._release_install_tmpdir_claim(job)
            raise
        return job

    def _stat_size(self, path: Path) -> int:
        size = 0
        if path.is_file():
            size = path.stat().st_size
        elif path.is_dir():
            for root, _, files in os.walk(path):
                size += sum(self._stat_size(Path(root, x)) for x in files)
        return size

    def _multifile_download(
        self,
        remote_files: List[RemoteModelFile],
        dest: Path,
        subfolder: Optional[Path] = None,
        subfolders: Optional[List[Path]] = None,
        access_token: Optional[str] = None,
        submit_job: bool = True,
    ) -> MultiFileDownloadJob:
        # HuggingFace repo subfolders are a little tricky. If the name of the model is "sdxl-turbo", and
        # we are installing the "vae" subfolder, we do not want to create an additional folder level, such
        # as "sdxl-turbo/vae", nor do we want to put the contents of the vae folder directly into "sdxl-turbo".
        # So what we do is to synthesize a folder named "sdxl-turbo_vae" here.
        #
        # For multiple subfolders (e.g., text_encoder+tokenizer), we create a combined folder name
        # (e.g., sdxl-turbo_text_encoder_tokenizer) and keep each subfolder's contents in its own
        # subdirectory within the model folder.

        if subfolders and len(subfolders) > 1:
            # Multiple subfolders: create combined name and keep subfolder structure. Entries may
            # also be explicit files (e.g. "modular_model_index.json" or "transformer/config.json");
            # use their stems in the combined name so it stays a sane directory name.
            top = Path(remote_files[0].path.parts[0])  # e.g. "Z-Image-Turbo/"
            subfolder_names = [
                (sf.stem if sf.suffix else sf.name).replace("/", "_").replace("\\", "_") for sf in subfolders
            ]
            combined_name = _combined_subfolder_name(subfolder_names)
            path_to_add = Path(f"{top}_{combined_name}")

            parts: List[RemoteModelFile] = []
            for model_file in remote_files:
                assert model_file.size is not None
                # Determine which subfolder this file belongs to
                file_path = model_file.path
                new_path: Optional[Path] = None
                for sf in subfolders:
                    if file_path == top / sf:
                        # An explicit file entry: keep its repo-relative path so e.g.
                        # transformer/config.json stays inside transformer/. (relative_to() below
                        # would return "." here and flatten the file to the model root.)
                        new_path = path_to_add / sf
                        break
                    try:
                        # Try to get relative path from this subfolder
                        relative = file_path.relative_to(top / sf)
                        # Keep the subfolder name as a subdirectory
                        new_path = path_to_add / sf.name / relative
                        break
                    except ValueError:
                        continue

                if new_path is None:
                    # File doesn't match any subfolder, keep original path structure
                    new_path = path_to_add / file_path.relative_to(top)

                parts.append(RemoteModelFile(url=model_file.url, path=new_path))
        elif subfolder:
            # Single subfolder: flatten into renamed folder
            top = Path(remote_files[0].path.parts[0])  # e.g. "sdxl-turbo/"
            path_to_remove = top / subfolder  # sdxl-turbo/vae/
            subfolder_rename = subfolder.name.replace("/", "_").replace("\\", "_")
            path_to_add = Path(f"{top}_{subfolder_rename}")

            parts = []
            for model_file in remote_files:
                assert model_file.size is not None
                parts.append(
                    RemoteModelFile(
                        url=model_file.url,
                        path=path_to_add / model_file.path.relative_to(path_to_remove),
                    )
                )
        else:
            # No subfolder specified - pass through unchanged
            parts = []
            for model_file in remote_files:
                assert model_file.size is not None
                parts.append(RemoteModelFile(url=model_file.url, path=model_file.path))

        return self._download_queue.multifile_download(
            parts=parts,
            dest=dest,
            access_token=access_token,
            submit_job=submit_job,
            on_start=self._download_started_callback,
            on_progress=self._download_progress_callback,
            on_complete=self._download_complete_callback,
            on_error=self._download_error_callback,
            on_cancelled=self._download_cancelled_callback,
        )

    # ------------------------------------------------------------------
    # Callbacks are executed by the download queue in a separate thread
    # ------------------------------------------------------------------
    def _download_started_callback(self, download_job: MultiFileDownloadJob) -> None:
        with self._lock:
            if install_job := self._download_cache.get(download_job.id, None):
                if install_job.cancelled or install_job.paused:
                    return
                install_job.status = InstallStatus.DOWNLOADING

                if install_job.local_path == install_job._install_tmpdir:  # first time
                    assert download_job.download_path
                    install_job.local_path = download_job.download_path
                install_job.download_parts = download_job.download_parts
                install_job.bytes = sum(x.bytes for x in download_job.download_parts)
                total_parts = sum(x.total_bytes for x in download_job.download_parts)
                if total_parts > 0:
                    install_job.total_bytes = max(install_job.total_bytes or 0, total_parts)
                self._signal_job_download_started(install_job)

    def _download_progress_callback(self, download_job: MultiFileDownloadJob) -> None:
        with self._lock:
            if install_job := self._download_cache.get(download_job.id, None):
                if install_job.cancelled:  # This catches the case in which the caller directly calls job.cancel()
                    self._download_queue.cancel_job(download_job)
                else:
                    # update sizes
                    install_job.bytes = sum(x.bytes for x in download_job.download_parts)
                    total_parts = sum(x.total_bytes for x in download_job.download_parts)
                    if total_parts > 0:
                        install_job.total_bytes = max(install_job.total_bytes or 0, total_parts)
                    self._signal_job_downloading(install_job)

    def _download_complete_callback(self, download_job: MultiFileDownloadJob) -> None:
        install_job: Optional[ModelInstallJob] = None
        queued = False
        with self._lock:
            if install_job := self._download_cache.pop(download_job.id, None):
                if install_job.cancelled:
                    self._cleanup_cancelled_install(install_job)
                elif install_job.paused:
                    # A pause requested before this callback must not turn into an automatic install.
                    self._write_install_marker(install_job, status=InstallStatus.PAUSED)
                    if self._stop_event.is_set():
                        self._release_install_tmpdir_claim(install_job)
                else:
                    self._signal_job_downloads_done(install_job)
                    if self._stop_event.is_set():
                        # Every part is complete, so the staging tree is safe to recover without this process's claim.
                        self._release_install_tmpdir_claim(install_job)
                        queued = True
                    else:
                        queued = self._queue_install_job_locked(install_job)
            if install_job is not None:
                self._downloads_changed_event.set()
        if install_job is not None:
            if not queued and not install_job.cancelled and not install_job.paused:
                self.cancel_job(install_job)
            # Let other threads know that the number of downloads has changed.
            self._downloads_changed_event.set()

    def _download_error_callback(self, download_job: MultiFileDownloadJob, excp: Optional[Exception] = None) -> None:
        with self._lock:
            if install_job := self._download_cache.pop(download_job.id, None):
                assert excp is not None
                if install_job.cancelled:
                    self._cleanup_cancelled_install(install_job)
                elif install_job.paused:
                    # Preserve the explicit pause even if an already-terminal download reports its error late.
                    self._write_install_marker(install_job, status=InstallStatus.PAUSED)
                    if self._stop_event.is_set():
                        self._release_install_tmpdir_claim(install_job)
                else:
                    self._set_error(install_job, excp)
                    self._download_queue.cancel_job(download_job)
                    if install_job._install_tmpdir is not None and not install_job._install_tmpdir_claim_conflict:
                        if not self._has_recovery_sentinel(install_job._install_tmpdir):
                            self._safe_rmtree(install_job._install_tmpdir, self._logger)
                        self._release_install_tmpdir_claim(install_job)

                # Let other threads know that the number of downloads has changed
                self._downloads_changed_event.set()

    def _cleanup_cancelled_install(self, install_job: ModelInstallJob) -> None:
        """Remove cancelled staging only after its download callback has stopped using it."""
        if install_job._install_tmpdir is None or install_job._install_tmpdir_claim_conflict:
            return
        if not self._has_recovery_sentinel(install_job._install_tmpdir):
            self._write_install_marker(install_job, status=InstallStatus.CANCELLED)
            self._delete_install_marker(install_job._install_tmpdir)
            self._safe_rmtree(install_job._install_tmpdir, self._logger)
        self._release_install_tmpdir_claim(install_job)

    def _download_cancelled_callback(self, download_job: MultiFileDownloadJob) -> None:
        with self._lock:
            if install_job := self._download_cache.pop(download_job.id, None):
                self._downloads_changed_event.set()
                if install_job.cancelled:
                    self._cleanup_cancelled_install(install_job)
                    return
                if install_job.paused or any(part.resume_required for part in download_job.download_parts):
                    install_job.status = InstallStatus.PAUSED
                    self._write_install_marker(install_job, status=InstallStatus.PAUSED)
                    # A user-paused install remains protected from orphan cleanup until resumed or cancelled.
                    # Shutdown releases only after the download queue has quiesced this callback.
                    if self._stop_event.is_set():
                        self._release_install_tmpdir_claim(install_job)
                    self._downloads_changed_event.set()
                    return
                # if install job has already registered an error, then do not replace its status with cancelled
                if not install_job.errored and not install_job.paused:
                    install_job.cancel()
                    if install_job._install_tmpdir is not None and not install_job._install_tmpdir_claim_conflict:
                        # Mark cancelled before cleanup so we don't reuse the folder if deletion fails.
                        if not self._has_recovery_sentinel(install_job._install_tmpdir):
                            self._write_install_marker(install_job, status=InstallStatus.CANCELLED)
                            self._delete_install_marker(install_job._install_tmpdir)
                            self._safe_rmtree(install_job._install_tmpdir, self._logger)
                if not install_job._install_tmpdir_claim_conflict:
                    self._release_install_tmpdir_claim(install_job)

                # Let other threads know that the number of downloads has changed
                self._downloads_changed_event.set()

    # ------------------------------------------------------------------------------------------------
    # Internal methods that put events on the event bus
    # ------------------------------------------------------------------------------------------------
    def _signal_job_running(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.RUNNING
        self._logger.info(f"Model install started: {job.source}")
        self._write_install_marker(job, status=InstallStatus.RUNNING)
        if self._event_bus:
            self._event_bus.emit_model_install_started(job)

    def _signal_job_download_started(self, job: ModelInstallJob) -> None:
        if self._event_bus:
            assert job._multifile_job is not None
            assert job.bytes is not None
            assert job.total_bytes is not None
            self._event_bus.emit_model_install_download_started(job)
        self._write_install_marker(job, status=InstallStatus.DOWNLOADING)

    def _signal_job_downloading(self, job: ModelInstallJob) -> None:
        if self._event_bus:
            assert job._multifile_job is not None
            assert job.bytes is not None
            assert job.total_bytes is not None
            self._event_bus.emit_model_install_download_progress(job)

    def _signal_job_downloads_done(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.DOWNLOADS_DONE
        self._logger.info(f"Model download complete: {job.source}")
        self._write_install_marker(job, status=InstallStatus.DOWNLOADS_DONE)
        if self._event_bus:
            self._event_bus.emit_model_install_downloads_complete(job)

    def _signal_job_completed(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.COMPLETED
        assert job.config_out
        self._logger.info(f"Model install complete: {job.source}")
        self._logger.debug(f"{job.local_path} registered key {job.config_out.key}")
        if job._install_tmpdir is not None:
            self._delete_install_marker(job._install_tmpdir)
        if self._event_bus:
            assert job.local_path is not None
            assert job.config_out is not None
            self._event_bus.emit_model_install_complete(job)

    def _signal_job_errored(self, job: ModelInstallJob) -> None:
        self._logger.error(f"Model install error: {job.source}\n{job.error_type}: {job.error}")
        if job._install_tmpdir is not None and not job._install_tmpdir_claim_conflict:
            self._delete_install_marker(job._install_tmpdir)
        if self._event_bus:
            assert job.error_type is not None
            assert job.error is not None
            self._event_bus.emit_model_install_error(job)

    def _signal_job_cancelled(self, job: ModelInstallJob) -> None:
        self._logger.info(f"Model install canceled: {job.source}")
        if job._install_tmpdir is not None:
            self._delete_install_marker(job._install_tmpdir)
        if self._event_bus:
            self._event_bus.emit_model_install_cancelled(job)

    @staticmethod
    def get_fetcher_from_url(url: str) -> Type[ModelMetadataFetchBase]:
        """
        Return a metadata fetcher appropriate for provided url.

        This used to be more useful, but the number of supported model
        sources has been reduced to HuggingFace alone.
        """
        if re.match(r"^https?://huggingface.co/[^/]+/[^/]+$", url.lower()):
            return HuggingFaceMetadataFetch
        raise ValueError(f"Unsupported model source: '{url}'")

    @staticmethod
    def _normalize_huggingface_blob_url(url: AnyHttpUrl) -> Url:
        """Convert Hugging Face file page URLs to direct download URLs."""
        return Url(
            re.sub(
                r"^(https?://huggingface\.co/[^/]+/[^/]+)/blob/([^?#]+)([?#].*)?$",
                r"\1/resolve/\2\3",
                str(url),
                flags=re.IGNORECASE,
            )
        )
