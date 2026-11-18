"""Model installation class."""

import errno
import filecmp
import gc
import locale
import os
import re
import ssl
import threading
import time
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue
from shutil import copy2, copytree, move, rmtree
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union
from urllib.parse import urljoin, urlparse

import requests
import torch
import yaml
from huggingface_hub import constants as hf_constants
from huggingface_hub import get_token as hf_get_token
from pydantic.networks import AnyHttpUrl
from pydantic_core import Url
from requests import Session
from urllib3.util.ssl_match_hostname import CertificateError

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import (
    DownloadJob,
    DownloadQueueServiceBase,
    MultiFileDownloadJob,
    in_progress_path,
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_install.install_staging_ledger import (
    INSTALL_MARKER_FILENAME,
    InstallStaging,
    InstallStagingLedger,
    part_resume_metadata,
)
from invokeai.app.services.model_install.model_install_base import ModelInstallServiceBase
from invokeai.app.services.model_install.model_install_common import (
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
)
from invokeai.app.services.model_records import DuplicateModelException, ModelRecordServiceBase, UnknownModelException
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges
from invokeai.app.util.no_replace import rename_noreplace
from invokeai.app.util.path_safety import is_plain_filename
from invokeai.app.util.ssrf import UnsafeDownloadURLException
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
from invokeai.backend.model_manager.metadata.metadata_base import (
    HF_METADATA_REQUEST_TIMEOUT_SECONDS,
    HuggingFaceMetadata,
    ModelRevisionUnavailableError,
    is_commit_hash,
    revision_unavailable_message,
)
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


class _HFFileURL(NamedTuple):
    """A Hugging Face file URL, `{endpoint}{org}/{repo}/{resolve|blob}/{revision}/{file}`, split into its parts."""

    endpoint: str
    repo_id: str
    kind: str  # "resolve" or "blob"
    revision: str  # as written in the URL
    filename: str
    tail: str  # query and fragment

    def with_revision(self, revision: str) -> str:
        """This file's download URL at `revision`, every other character of the URL unchanged."""
        return f"{self.endpoint}{self.repo_id}/resolve/{revision}/{self.filename}{self.tail}"


# `refs/pr/<n>` and `refs/convert/<name>` are revisions with slashes that the hub accepts unencoded.
_HF_FILE_PATH = re.compile(
    r"^([^/?#]+/[^/?#]+)/(resolve|blob)/(refs/pr/\d+|refs/convert/[^/?#]+|[^/?#]+)/([^?#]+)(.*)$", re.DOTALL
)


def _hf_file_url_parts(url: str, trusted_endpoint: Optional[str] = None) -> Optional[_HFFileURL]:
    """Split a Hugging Face file URL on huggingface.co, the configured HF_ENDPOINT mirror, or `trusted_endpoint` (the
    endpoint an install recorded for its source); None for any other URL. Unknown hosts are never inferred from their
    path."""
    configured = hf_constants.HUGGINGFACE_CO_URL_TEMPLATE.split("{repo_id}", 1)[0]
    endpoints = ("https://huggingface.co/", "http://huggingface.co/", configured, trusted_endpoint)
    for endpoint in dict.fromkeys(item for item in endpoints if item):
        if url[: len(endpoint)].lower() == endpoint.lower() and (match := _HF_FILE_PATH.match(url[len(endpoint) :])):
            return _HFFileURL(url[: len(endpoint)], *match.groups())
    return None


_HF_HEAD_MAX_REDIRECTS = 5


class _DownloadCallbackPendingError(InstallDownloadConflictError):
    """The job's previous download has ended but its callback has not yet run; a retry once it has can succeed."""


class _InstallCancelledBeforeTransfer(Exception):
    """Raised when a cancellation request wins before file transfer becomes protected."""


# Filesystems cap a single path component at 255 bytes. A source that lists many explicit files
# (an LTX-2 component folder names eight) would otherwise produce a folder name that cannot be
# created; the combined name is only a label, so it is shortened past this point with a count.
_MAX_COMBINED_SUBFOLDER_NAME = 96

# How long the idle install thread blocks on its queue before rechecking the stop event, which bounds how long
# stop() waits.
_INSTALL_QUEUE_POLL_SECONDS = 1.0

# How long wait_for_job() sleeps between status checks when no job-state signal wakes it first.
_JOB_WAIT_POLL_SECONDS = 5.0

# First wait before retrying a move that hit a locked file; each further retry doubles it.
_MOVE_RETRY_DELAY_SECONDS = 0.5

_UNVALIDATED_PARTIAL_MESSAGE = (
    "The partial file could not be checked against the server's current file. Restarted download from the beginning."
)
_REFUSED_PARTIAL_MESSAGE = "The server refused to continue the partial file. Restarted download from the beginning."
_MISSING_PARTIAL_MESSAGE = "Partial file missing. Restarted download from the beginning."


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


_TRANSIENT_HTTP_STATUSES = frozenset({408, 429})


def _is_certificate_failure(excp: BaseException) -> bool:
    """Return True if a TLS error was caused by the server's certificate failing verification.

    requests and urllib3 wrap the ssl error as an exception argument (requests' and urllib3's SSLError) or as a
    MaxRetryError's reason.
    """
    pending: List[object] = [excp]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (ssl.SSLCertVerificationError, CertificateError)):
            return True
        pending.extend((getattr(current, "reason", None), *current.args))
    return False


def _is_transient_download_error(excp: BaseException) -> bool:
    """Return True for a download failure that a later resume or retry can plausibly get past.

    A paused install waits for someone to resume it, so anything not known to be transient (auth and missing-file
    responses, a full disk, an unsafe filename, TLS verification) is treated as permanent and errors the install.
    """
    if isinstance(excp, requests.exceptions.SSLError) and _is_certificate_failure(excp):
        return False
    if isinstance(excp, (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)):
        return True
    if isinstance(excp, requests.HTTPError):
        if excp.response is None:
            return False
        status = excp.response.status_code
        return status in _TRANSIENT_HTTP_STATUSES or 500 <= status < 600
    # The download queue promotes a part's `.downloading` sidecar with a rename; something else removing the
    # sidecar between its size check and that rename raises this. The part is fetched again on retry.
    return (
        isinstance(excp, FileNotFoundError)
        and excp.filename2 is not None
        and str(excp.filename).endswith(".downloading")
    )


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
        self._ledger = InstallStagingLedger(app_config.models_path, self._logger)

    @staticmethod
    def _staging_of(job: ModelInstallJob) -> InstallStaging:
        assert job._staging is not None, "every install job gets its staging handle when it is created"
        return job._staging

    def _write_install_marker(self, job: ModelInstallJob, status: Optional[InstallStatus] = None) -> None:
        self._staging_of(job).persist(
            source=job.source,
            config_in=job.config_in,
            status=status or job.status,
            download_parts=job.download_parts,
            resume_files=job._resume_metadata,
            recovery_required=job._recovery_required,
            revision=job._source_revision,
            hf_endpoint=job._source_hf_endpoint or self._hf_endpoint_of(job.source, []),
        )

    def _requires_recovery(self, job: ModelInstallJob) -> bool:
        return job._recovery_required or self._staging_of(job).requires_recovery()

    def _discard_cancelled_staging(self, job: ModelInstallJob) -> None:
        """Remove a cancelled install's staging tree unless it holds recovery data, and release its claim.

        Call only after the job's download callback has stopped using the tree.
        """
        staging = self._staging_of(job)
        if staging.claim_conflict:
            return
        if staging.requires_recovery():
            staging.release_claim()
            return
        try:
            # Touch neither the marker nor the tree of a staging directory another owner holds.
            staging.claim()
        except InstallCancellationConflictError:
            self._logger.debug(f"Leaving cancelled install directory {staging.tmpdir} to its current owner")
            return
        # Mark cancelled before cleanup so a tree whose removal fails is never reused.
        self._write_install_marker(job, status=InstallStatus.CANCELLED)
        staging.delete_marker()
        staging.discard()

    def _recorded_resume_metadata(self, job: ModelInstallJob) -> dict[str, dict[str, Any]]:
        """Each file's resume metadata as last recorded: by its part in this process, else by the install marker the
        job was restored from."""
        recorded = dict(job._resume_metadata or {})
        recorded.update((str(part.source), part_resume_metadata(part)) for part in job.download_parts)
        return recorded

    def _restore_incomplete_installs(self) -> None:
        seen_sources: set[str] = set()
        # Collect sources already tracked by active jobs (including those being downloaded right now).
        # We must not re-queue these or delete their tmpdirs.
        with self._lock:
            active_sources = {str(j.source) for j in self._install_jobs if not j.in_terminal_state}
            active_sources.update(str(j.source) for j in self._download_cache.values() if not j.in_terminal_state)
        for entry in self._ledger.recoverable_installs():
            if self._stop_event.is_set():
                return
            staging = entry.staging
            tmpdir = staging.tmpdir
            assert tmpdir is not None
            try:
                source_str = entry.source
                if not isinstance(source_str, str):
                    raise ValueError("Missing source in install marker")
                source = self._guess_source(source_str)
                if isinstance(source, (HFModelSource, URLModelSource)) and isinstance(entry.access_token, str):
                    source.access_token = entry.access_token
                if source_str in active_sources:
                    # This tmpdir belongs to an install already in progress; leave it alone.
                    self._logger.debug(f"Skipping restore for {source_str} - already being tracked")
                    continue
                if source_str in seen_sources:
                    self._logger.info(f"Removing duplicate temporary directory {tmpdir}")
                    staging.discard()
                    continue
                # Inside the `try`: a marker written by an older version can hold a config that no longer
                # validates, and that must skip this marker, not abort the restore of every marker after it. Note
                # that an unsafe *key* is no longer such a case - `ModelRecordChanges` deliberately accepts one so
                # the install is restored and then fails at the join in `install_path()`, which errors the job and
                # reclaims its tmpdir. Skipping it here would strand the partial download instead: no job is
                # created to clean up, and startup cleanup keeps any readable non-terminal marker.
                config_in = ModelRecordChanges(**(entry.config_in or {}))
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
            job._staging = staging
            job._source_revision = entry.revision
            # A marker written before endpoints were recorded names none; an endpoint recognized now is recorded.
            job._source_hf_endpoint = entry.hf_endpoint or self._hf_endpoint_of(source, [])
            try:
                # Protect paused and completed downloads as soon as they are restored, before any
                # later orphan scan can mistake finalized staging files for unregistered models.
                staging.claim()
            except InstallCancellationConflictError as e:
                self._logger.info(f"Skipping restore of claimed install directory {tmpdir}: {e}")
                continue
            # A Hugging Face download can only be resumed if its staged files are known to come from the commit it
            # resumes from. A marker that names no commit - written before installs were pinned, or pinned to a
            # branch because the hub reported no commit - may hold files from an older state of a branch that has
            # since moved; they would be accepted by size or appended to, mixing two commits. They are discarded and
            # the download starts over, pinned to the commit the resume fetches. Finished downloads are kept: they
            # need no more files.
            discard_staging = (
                not is_commit_hash(job._source_revision)
                and entry.status not in {InstallStatus.DOWNLOADS_DONE, InstallStatus.RUNNING}
                and self._is_hf_source(source, job._source_hf_endpoint)
            )
            if discard_staging:
                job._source_revision = None
            elif entry.files:
                job._resume_metadata = {f.get("url"): f for f in entry.files if f.get("url")}
            job.status = entry.status
            if discard_staging:
                # Before the job is listed, so no resume can start a download into the directory being emptied.
                self._logger.warning(
                    f"Discarding the partial download of {source_str} in {tmpdir}: the repository revision it came "
                    "from is unknown. It will download again from the beginning."
                )
                try:
                    staging.discard_files()
                    self._write_install_marker(job)
                except Exception as e:
                    # Errored before it is listed, so no resume can reach a job whose staging is being removed.
                    self._fail_restored_install(job, e)
                    self._install_jobs.append(job)
                    continue
            self._install_jobs.append(job)

            if job.paused:
                continue

            if job.status in [InstallStatus.DOWNLOADS_DONE, InstallStatus.RUNNING]:
                job.status = InstallStatus.DOWNLOADS_DONE
                self._put_in_queue(job)
            else:
                try:
                    self._resume_remote_download(job)
                except (ModelMetadataUnavailableError, InstallDownloadConflictError) as e:
                    # Unavailable metadata, or a partial file that must restart but is locked: keep it for a retry.
                    self._logger.warning(f"Could not resume install {source_str}: {e}")
                    job.status = InstallStatus.PAUSED
                    self._write_install_marker(job, status=InstallStatus.PAUSED)
                    self._signal_job_paused(job)
                    if self._stop_event.is_set():
                        return
                except Exception as e:
                    if self._stop_event.is_set():
                        self._logger.info(f"Leaving interrupted install in {tmpdir} for next startup")
                        job.status = InstallStatus.PAUSED
                        return
                    self._fail_restored_install(job, e)

    def _fail_restored_install(self, job: ModelInstallJob, error: Exception) -> None:
        if isinstance(error, ModelRevisionUnavailableError):
            job._pinned_revision_vanished = True
        self._set_error(job, error)
        self._staging_of(job).discard()

    def _is_hf_source(self, source: ModelSource, trusted_endpoint: Optional[str] = None) -> bool:
        """Whether the source's files come from a Hugging Face repository, whose files change between commits."""
        if isinstance(source, HFModelSource):
            return True
        if isinstance(source, URLModelSource):
            try:
                self.get_fetcher_from_url(str(source.url))
            except ValueError:
                return _hf_file_url_parts(str(source.url), trusted_endpoint) is not None
            return True
        return False

    @staticmethod
    def _hf_endpoint_of(source: ModelSource, remote_files: List[RemoteModelFile]) -> Optional[str]:
        """The Hugging Face endpoint a URL source, or else its file URLs, are on now; None if not Hugging Face.

        Recorded with the install, so a later HF_ENDPOINT change cannot hide that its files came from a repository.
        """
        if isinstance(source, URLModelSource) and (parts := _hf_file_url_parts(str(source.url))) is not None:
            return parts.endpoint
        return next((parts.endpoint for item in remote_files if (parts := _hf_file_url_parts(str(item.url)))), None)

    def _discard_unpinned_hf_staging(self, job: ModelInstallJob) -> bool:
        """Before a Hugging Face download that is not pinned to a commit is fetched again, delete everything it staged.

        Its branch may have moved since the previous transfer, and nothing tells which commit staged files came from.
        Returns whether the job was such a download.
        """
        if job._source_revision is not None or not self._is_hf_source(job.source, job._source_hf_endpoint):
            return False
        if job.download_parts or job._resume_metadata:
            self._staging_of(job).discard_files()
        job.download_parts = set()
        job._resume_metadata = None
        job.bytes = 0
        return True

    def _single_hf_file(self, source: ModelSource, trusted_endpoint: Optional[str] = None) -> Optional[_HFFileURL]:
        """The file URL of a source that names one file of a Hugging Face repository."""
        if not isinstance(source, URLModelSource):
            return None
        try:
            self.get_fetcher_from_url(str(source.url))
        except ValueError:
            return _hf_file_url_parts(str(source.url), trusted_endpoint)
        return None

    def _download_access_token(
        self, source: HFModelSource | URLModelSource, trusted_endpoint: Optional[str] = None
    ) -> Optional[str]:
        if isinstance(source, HFModelSource):
            return source.access_token or hf_get_token()
        if self._single_hf_file(source, trusted_endpoint) is not None:
            return self._single_hf_file_token(source)
        return source.access_token

    def _download_token_for(
        self, source: HFModelSource | URLModelSource, trusted_endpoint: Optional[str] = None
    ) -> Callable[[str], Optional[str]]:
        """The token a download of a URL of this source sends: `_download_access_token`, else the `remote_api_tokens`
        entry that the download queue matches against that URL. Requests made to list a repository's files send it
        too, so a gated repository that will serve the files also serves its index."""
        return lambda url: self._download_access_token(source, trusted_endpoint) or self._configured_api_token(url)

    def _single_hf_file_token(self, source: URLModelSource) -> Optional[str]:
        """The token a download of this source's file sends: the source's own, else the configured `remote_api_tokens`
        entry matching the URL as the user gave it. The download URL names a commit instead, so the token is resolved
        here, once, and sent explicitly."""
        return source.access_token or self._configured_api_token(str(source.url))

    def _configured_api_token(self, url: str) -> Optional[str]:
        """The token of the first `remote_api_tokens` entry whose pattern matches `url`."""
        for pair in self._app_config.remote_api_tokens or []:
            if re.search(pair.url_regex, url):
                return pair.token
        return None

    @staticmethod
    def _download_revision(
        metadata: Optional[AnyModelRepoMetadata],
        remote_files: List[RemoteModelFile],
        trusted_endpoint: Optional[str] = None,
    ) -> Optional[str]:
        """The commit `_remote_files_from_source` pinned these files to, if any; a branch is no pin."""
        if isinstance(metadata, HuggingFaceMetadata):
            return metadata.revision if is_commit_hash(metadata.revision) else None
        if (
            len(remote_files) == 1
            and (parts := _hf_file_url_parts(str(remote_files[0].url), trusted_endpoint)) is not None
        ):
            return parts.revision if is_commit_hash(parts.revision) else None
        return None

    def _pin_new_hf_file_download(
        self, source: ModelSource, remote_files: List[RemoteModelFile], trusted_endpoint: Optional[str] = None
    ) -> List[RemoteModelFile]:
        """Point a download of one Hugging Face file, about to start with nothing staged, at the commit its branch or
        tag names now, so a resume cannot get a newer version of the file.

        The commit comes from the `X-Repo-Commit` header the hub sends for the file, asked of the URL's own host with
        the token the download will send. A URL that already names a commit is left as it is. If the host names no
        commit or refuses the request, the download goes ahead unpinned and reports any failure itself; a transient
        failure raises `ModelMetadataUnavailableError`.
        """
        if len(remote_files) != 1 or not isinstance(source, URLModelSource):
            return remote_files
        parts = _hf_file_url_parts(str(remote_files[0].url), trusted_endpoint)
        if parts is None or self._single_hf_file(source, trusted_endpoint) is None or is_commit_hash(parts.revision):
            return remote_files
        response = self._head_hf_file(parts.with_revision(parts.revision), self._single_hf_file_token(source))
        if response is None:
            return remote_files
        if response.status_code == 429 or response.status_code >= 500:
            raise ModelMetadataUnavailableError(f"Could not look up {parts.repo_id}: HTTP {response.status_code}")
        commit = response.headers.get("X-Repo-Commit")
        if not is_commit_hash(commit):
            self._logger.warning(f"Downloading {source} unpinned: its host named no commit for it.")
            return remote_files
        assert commit is not None
        return [remote_files[0].model_copy(update={"url": Url(parts.with_revision(commit))})]

    def _head_hf_file(self, url: str, token: Optional[str]) -> Optional[requests.Response]:
        """HEAD a Hugging Face file URL as downloads are sent (the download queue's proxy and address policy).

        Like huggingface_hub, follow redirects on the same host until a response names the commit: a renamed repo
        answers a relative 307, a small file one to its cache, and `http://` a 301 to `https://`. A redirect to another
        host (the CDN serving large files) is not followed, so the token never leaves the URL's host. None when the
        address policy refuses the URL; the download then refuses it too.
        """
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        for _ in range(_HF_HEAD_MAX_REDIRECTS + 1):
            try:
                response = self._download_queue.head(url, headers=headers, timeout=HF_METADATA_REQUEST_TIMEOUT_SECONDS)
            except UnsafeDownloadURLException:
                return None
            except requests.RequestException as e:
                raise ModelMetadataUnavailableError(f"Could not reach {url}") from e
            location = response.headers.get("Location")
            if "X-Repo-Commit" in response.headers or not response.is_redirect or not location:
                return response
            target = urljoin(url, location)
            source, destination = urlparse(url), urlparse(target)
            # The same host and scheme, or an upgrade to https on it; never a downgrade that would send the token
            # over plain http.
            same_host = (destination.scheme, destination.hostname, destination.port) == (
                source.scheme,
                source.hostname,
                source.port,
            ) or (destination.hostname == source.hostname and (source.scheme, destination.scheme) == ("http", "https"))
            if not same_host:
                return response
            url = target
        return response

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
            self._discard_unpinned_hf_staging(job)
            remote_files, metadata = self._remote_files_from_source(job.source, **self._revision_kwargs(job))
            if not remote_files:
                raise RuntimeError("No remote files are available to download")
            resume_metadata = self._recorded_resume_metadata(job)
            if job._source_revision is None and not resume_metadata:
                # Nothing staged: the download starts over, pinned to the commit the repository is at now.
                remote_files = self._pin_new_hf_file_download(job.source, remote_files, job._source_hf_endpoint)
                job._source_revision = self._download_revision(metadata, remote_files, job._source_hf_endpoint)
            subfolders = job.source.subfolders if isinstance(job.source, HFModelSource) else []
            job.status = InstallStatus.WAITING
            self._enqueue_remote_download(
                job=job,
                source=job.source,
                remote_files=remote_files,
                metadata=metadata,
                subfolder=job.source.subfolder
                if isinstance(job.source, HFModelSource) and len(subfolders) <= 1
                else None,
                subfolders=subfolders if len(subfolders) > 1 else None,
                resume_metadata=resume_metadata,
                report_missing_partials=True,
            )
        except BaseException:
            job.status = previous_status
            raise
        finally:
            if not operation_reserved:
                self._end_remote_download_operation(job)

    @staticmethod
    def _revision_kwargs(job: ModelInstallJob) -> dict[str, Any]:
        """`_remote_files_from_source` arguments that keep a job's download on its pinned revision and endpoint."""
        kwargs: dict[str, Any] = {"revision": job._source_revision}
        if job._source_hf_endpoint:
            kwargs["trusted_endpoint"] = job._source_hf_endpoint
        return kwargs

    def _begin_remote_download_operation(self, job: ModelInstallJob, *, require_paused: bool = False) -> bool:
        """Serialize resume/restart handoffs and reject replacements before old callbacks finish."""
        with self._remote_download_condition:
            if self._stop_event.is_set():
                raise InstallDownloadConflictError("The model install service is stopping; retry after it starts.")
            operation_active = job.id in self._remote_download_operations
            callback_pending = any(cached_job is job for cached_job in self._download_cache.values())
            download_active = operation_active or callback_pending
            if require_paused:
                if download_active:
                    raise (InstallDownloadConflictError if operation_active else _DownloadCallbackPendingError)(
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
        self._install_completed_event.set()  # wake wait_for_job(), which waits out a resume of a paused job

    def _remote_download_in_progress(self, job: ModelInstallJob) -> bool:
        with self._lock:
            return job.id in self._remote_download_operations or any(
                cached_job is job for cached_job in self._download_cache.values()
            )

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
                self._ledger.sweep_stale_claims()
                self._start_installer_thread()
                self._ledger.remove_dangling_staging(
                    (self._app_config.models_path / model.path).resolve()
                    for model in self.record_store.all_models()
                    if model.path
                )
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
                if self._requires_recovery(job):
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

            staging = self._staging_of(job)
            if preserve_download_status is not None:
                self._write_install_marker(job, status=preserve_download_status)
                if not download_callback_pending:
                    staging.release_claim()
            elif multifile_job is not None and downloads_complete:
                self._write_install_marker(job, status=InstallStatus.DOWNLOADS_DONE)
                if not download_callback_pending:
                    staging.release_claim()
            elif multifile_job is not None:
                self._logger.warning(f"Pausing job {job.id}")
                for part in multifile_job.download_parts:
                    self._download_queue.pause_job(part)
                self._write_install_marker(job, status=InstallStatus.PAUSED)
                if self._stop_event.is_set() and not download_callback_pending:
                    # The pause callback already ran, so no download worker can still touch staging.
                    staging.release_claim()
            elif staging.tmpdir is not None:
                self._logger.warning(f"Cancelling job {job.id}")
                self._discard_cancelled_staging(job)
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
        staging = InstallStaging(self._ledger)
        staging.protect_source(model_path)
        registered = False
        try:
            model_id = self._register(model_path, config)
            registered = True
            return model_id
        finally:
            # Registering a recovered source resolves the recovery data it held.
            staging.release(preserve_recovery=False, completed=registered)

    # TODO: Replace this with a proper fix for underlying problem of Windows holding open
    # the file when it needs to be moved.
    @staticmethod
    def _move_with_retries(src: Path, dst: Path, attempts: int = 5) -> None:
        """Workaround for Windows file-handle issues when moving files."""
        delay = _MOVE_RETRY_DELAY_SECONDS
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

    @classmethod
    def _restore_moved_path(cls, src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(f"Refusing to overwrite recreated source {dst}")
        try:
            rename_noreplace(src, dst)
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
                rename_noreplace(staged_path, dst)
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
        staging = InstallStaging(self._ledger)
        staging.protect_source(model_path)
        preserve_source = False
        installed = False
        try:
            info: AnyModelConfig = self._probe(Path(model_path), config)  # type: ignore

            def begin_transfer() -> bool:
                staging.begin_transfer()
                return True

            model_id = self._install_path_with_info(model_path, config, info, before_transfer=begin_transfer)
            installed = True
            return model_id
        except InstallRecoveryRequiredError:
            preserve_source = True
            raise
        finally:
            staging.release(preserve_recovery=preserve_source, completed=installed)

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
            dest_claim = self._ledger.claim(dest_dir)
        except FileExistsError as e:
            raise InstallCancellationConflictError(
                f"Cannot install model to {dest_dir}: another install or orphan cleanup is using it."
            ) from e
        with dest_claim:
            return self._install_path_with_info_claimed(model_path, config, info, before_transfer)

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
            self._ledger.mark_recovery_required(dest_dir)
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
            self._ledger.retain_recovery(dest_dir)
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
                    self._ledger.retain_recovery(dest_dir)
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
                    self._ledger.clear_recovery(dest_dir)
                except Exception as cleanup_error:
                    self._ledger.retain_recovery(dest_dir)
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
                    self._ledger.clear_recovery(dest_dir)
                except Exception as cleanup_error:
                    self._ledger.retain_recovery(dest_dir)
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

    def wait_for_job(self, job: ModelInstallJob, timeout: float = 0) -> ModelInstallJob:
        """Block until the indicated job is terminal or settled in pause, or the timeout limit is reached."""
        deadline = time.time() + timeout if timeout > 0 else None
        # A paused job needs a resume or retry from outside, so waiting on it would only run out the timeout. One whose
        # resume or retry is still being set up, or whose last download callback is still running, has not settled.
        while not (job.in_terminal_state or (job.paused and not self._remote_download_in_progress(job))):
            poll = _JOB_WAIT_POLL_SECONDS
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Timeout exceeded")
                poll = min(poll, remaining)
            if self._install_completed_event.wait(timeout=poll):  # in case we miss an event
                self._install_completed_event.clear()
        return job

    def resume_and_wait_for_job(self, job: ModelInstallJob, timeout: float = 0) -> ModelInstallJob:  # noqa D102
        deadline = time.time() + timeout if timeout > 0 else None
        while True:
            retry_after_wait = False
            if job.paused:
                try:
                    self.resume_job(job)
                except ModelRevisionUnavailableError:
                    pass  # resume_job() errored the job; its error says why and wait_for_job() returns it
                except _DownloadCallbackPendingError:
                    # Only an unfinished download callback is worth outlasting: wait_for_job() waits for it, and the
                    # resume is tried again if the job is still paused. A stopping service will not accept the resume.
                    retry_after_wait = not self._stop_event.is_set()
                except InstallDownloadConflictError:
                    # A concurrent resume or restart decides this job's outcome itself, which wait_for_job() waits
                    # for. Any other refusal (a stopping service, a partial file that cannot be deleted) is not
                    # resolved by trying again; the job is returned paused.
                    pass
            remaining = 0.0
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Timeout exceeded")
            self.wait_for_job(job, timeout=remaining)
            if not (retry_after_wait and job.paused):
                return job

    def wait_for_installs(self, timeout: int = 0) -> List[ModelInstallJob]:  # noqa D102
        """Wait for active download and install work; recoverably paused jobs are returned without being completed."""
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
            if self._requires_recovery(job):
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
        self._discard_cancelled_staging(job)

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
            was_paused = job.paused
            job.status = InstallStatus.PAUSED
        self._logger.warning(f"Pausing {job.source}")
        if dj := job._multifile_job:
            for part in dj.download_parts:
                self._download_queue.pause_job(part)
        self._write_install_marker(job, status=InstallStatus.PAUSED)
        if not was_paused:
            self._signal_job_paused(job)

    def resume_job(self, job: ModelInstallJob) -> None:
        """Resume a previously paused job."""
        if not self._begin_remote_download_operation(job, require_paused=True):
            return
        self._logger.info(f"Resuming {job.source}")
        try:
            self._resume_remote_download(job, operation_reserved=True)
        except ModelRevisionUnavailableError as e:
            self._fail_install_pinned_to_vanished_revision(job, e)
            raise
        finally:
            self._end_remote_download_operation(job)

    def _fail_install_pinned_to_vanished_revision(self, job: ModelInstallJob, error: Exception) -> None:
        """Error an install whose pinned repository revision is gone: its staged files can never be completed.

        The staged files are deleted, so the job keeps no download parts: there is nothing left to resume or restart,
        and a retry has to start a new install from the source, which pins the repository's current commit.
        """
        with self._lock:
            job._pinned_revision_vanished = True
            job.download_parts = set()
            job._resume_metadata = None
            self._set_error(job, error)
            self._staging_of(job).discard()

    def restart_failed(self, job: ModelInstallJob) -> None:
        """Restart failed or non-resumable downloads for a job, with every other file missing from staging."""
        if self._requires_recovery(job):
            raise InstallRecoveryRequiredError("Cannot restart an install that requires recovery.")
        if not isinstance(job.source, (HFModelSource, URLModelSource)):
            return
        if not job.download_parts:
            return
        if not any(part.resume_required or part.errored for part in job.download_parts):
            return
        self._restart_remote_download(job, job.source)

    def restart_file(self, job: ModelInstallJob, file_source: str) -> None:
        """Restart a specific file download for a job, with every other file missing from staging."""
        if self._requires_recovery(job):
            raise InstallRecoveryRequiredError("Cannot restart an install that requires recovery.")
        if not isinstance(job.source, (HFModelSource, URLModelSource)):
            return
        self._restart_remote_download(job, job.source, requested_file=file_source)

    def _restart_remote_download(
        self,
        job: ModelInstallJob,
        source: HFModelSource | URLModelSource,
        requested_file: Optional[str] = None,
    ) -> None:
        """Re-queue `requested_file` and every file of the source whose staged file is not on disk.

        Disk decides, not `job.download_parts`: a download that errored deleted the whole staging tree, completed files
        included; one that paused after a transient failure kept it; and a previous restart leaves `download_parts`
        holding only the files it re-queued.

        Only the files being restarted lose their partial download: the requested file, or, without one, every part
        the backend could not resume; and any part that errored. Every other missing file resumes from its partial
        as `_enqueue_remote_download` allows.
        """
        self._begin_remote_download_operation(job)
        previous_status = job.status
        try:
            if job._pinned_revision_vanished:
                # Its staging was given up; downloading files into it would install an incomplete model.
                return
            discarded_unpinned_hf_staging = self._discard_unpinned_hf_staging(job)
            remote_files, metadata = self._remote_files_from_source(source, **self._revision_kwargs(job))
            if requested_file is not None and all(str(rf.url) != requested_file for rf in remote_files):
                return
            if discarded_unpinned_hf_staging:
                # Nothing is staged any more: every file downloads again, pinned to the commit the repository is at now.
                remote_files = self._pin_new_hf_file_download(source, remote_files, job._source_hf_endpoint)
                job._source_revision = self._download_revision(metadata, remote_files, job._source_hf_endpoint)
                requested_file = None
            subfolders = source.subfolders if isinstance(source, HFModelSource) else []
            subfolder = source.subfolder if isinstance(source, HFModelSource) and len(subfolders) <= 1 else None
            subfolders_arg = subfolders if len(subfolders) > 1 else None
            destdir = self._staging_of(job).tmpdir
            assert destdir is not None, "a remote install downloads into its staging directory"
            staged_files = self._staged_remote_files(remote_files, subfolder, subfolders_arg)
            # A file is present only once the download renamed it from `.downloading`. A single-file URL source is
            # staged at "." until the response names its file, so it never counts as present.
            missing = [
                remote_file
                for remote_file, staged_file in zip(remote_files, staged_files, strict=True)
                if str(remote_file.url) == requested_file or not (destdir / staged_file.path).is_file()
            ]
            if not missing:
                raise RuntimeError("No remote files are available to download")
            restart_from_zero = {
                str(part.source)
                for part in job.download_parts
                if part.errored or (requested_file is None and part.resume_required)
            }
            if requested_file is not None:
                restart_from_zero.add(requested_file)
            job.status = InstallStatus.WAITING
            self._enqueue_remote_download(
                job=job,
                source=source,
                remote_files=missing,
                metadata=metadata,
                subfolder=subfolder,
                subfolders=subfolders_arg,
                resume_metadata=self._recorded_resume_metadata(job),
                restart_from_zero=restart_from_zero,
                # A paused install kept its staging, so a partial missing from it vanished; an errored one deleted it.
                report_missing_partials=previous_status == InstallStatus.PAUSED,
            )
        except BaseException as e:
            job.status = previous_status
            if isinstance(e, ModelRevisionUnavailableError):
                self._fail_install_pinned_to_vanished_revision(job, e)
            raise
        finally:
            self._end_remote_download_operation(job)

    def prune_jobs(self) -> List[int]:
        """Prune all completed and errored jobs."""
        # Filter and rebind under the condition. Unlocked, a registration made by a concurrent import_model()
        # between the two would be dropped, leaving a live install invisible to the duplicate check. Rebind rather
        # than mutating in place so that readers already iterating the old list are not silently truncated.
        with self._install_condition:
            pruned = []
            retained = []
            for job in self._install_jobs:
                (pruned if job.in_terminal_state else retained).append(job)
            self._install_jobs = retained
            return [job.id for job in pruned]

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
        self,
        source: ModelSource,
        revision: Optional[str] = None,
        trusted_endpoint: Optional[str] = None,
    ) -> Tuple[List[RemoteModelFile], Optional[AnyModelRepoMetadata]]:
        """List the files to download for a remote source, from repository `revision` when one is given.

        Without a revision, Hugging Face file URLs name the commit the repository is at now, never a branch, so the
        files of one install all come from one commit however often it is resumed or retried.
        """
        metadata = None
        if isinstance(source, HFModelSource):
            metadata = HuggingFaceMetadataFetch(self._session).from_id(
                source.repo_id, source.variant, revision, pin_to_commit=True
            )
            assert isinstance(metadata, ModelMetadataWithFiles)
            # Use subfolders property which handles '+' separated multiple subfolders
            subfolders = source.subfolders
            return (
                metadata.download_urls(
                    variant=source.variant or self._guess_variant(),
                    subfolder=source.subfolder if len(subfolders) <= 1 else None,
                    subfolders=subfolders if len(subfolders) > 1 else None,
                    session=self._session,
                    access_token_for=self._download_token_for(source, trusted_endpoint),
                ),
                metadata,
            )

        if isinstance(source, URLModelSource):
            try:
                fetcher = self.get_fetcher_from_url(str(source.url))
                kwargs: dict[str, Any] = {"session": self._session}
                metadata = fetcher(**kwargs).from_url(source.url, revision=revision, pin_to_commit=True)
                assert isinstance(metadata, ModelMetadataWithFiles)
                return (
                    metadata.download_urls(
                        session=self._session, access_token_for=self._download_token_for(source, trusted_endpoint)
                    ),
                    metadata,
                )
            except ValueError:
                pass

            file_url = self._normalize_huggingface_blob_url(source.url, trusted_endpoint)
            parts = _hf_file_url_parts(str(file_url), trusted_endpoint)
            if revision is not None and parts is not None and is_commit_hash(revision):
                file_url = self._pinned_hf_file_url(source, parts, revision)
            return [RemoteModelFile(url=file_url, path=Path("."), size=0)], None

        raise Exception(f"No files associated with {source}")

    def _pinned_hf_file_url(self, source: URLModelSource, parts: _HFFileURL, commit: str) -> Url:
        """The download URL of a Hugging Face file at the commit its install is pinned to.

        A commit the hub reports gone (404 with `X-Error-Code` `EntryNotFound` or `RevisionNotFound`) raises
        `ModelRevisionUnavailableError`. Nothing else is decided here - not a refusal, not a 404 without that code
        (from a mirror or proxy), not a failed request: the download reports those itself, and keeps its staged files.
        """
        url = parts.with_revision(commit)
        try:
            response = self._head_hf_file(url, self._single_hf_file_token(source))
        except ModelMetadataUnavailableError:
            return Url(url)
        if (
            response is not None
            and response.status_code == 404
            and response.headers.get("X-Error-Code") in {"EntryNotFound", "RevisionNotFound"}
        ):
            raise ModelRevisionUnavailableError(revision_unavailable_message(parts.repo_id, commit))
        return Url(url)

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

    def _install_next_item(self) -> None:
        self._logger.debug(f"Installer thread {threading.get_ident()} starting")
        while True:
            if self._stop_event.is_set():
                break
            self._logger.debug(f"Installer thread {threading.get_ident()} polling")
            try:
                job = self._install_queue.get(timeout=_INSTALL_QUEUE_POLL_SECONDS)
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
                            f"Failed to persist install recovery marker in {self._staging_of(job).tmpdir}: {marker_error}"
                        )

            finally:
                if should_process:
                    with self._lock:
                        if self._active_install_job is job:
                            self._active_install_job = None
                        job._install_phase = None
                    # Discarding keeps our staging claim until the tree is gone, so orphan cleanup cannot
                    # acquire the path in the gap before the directory is removed.
                    staging = self._staging_of(job)
                    if not job._recovery_required:
                        staging.discard()
                    staging.finish(recovery_required=job._recovery_required, completed=job.complete)
                self._install_completed_event.set()
                self._install_queue.task_done()
        self._logger.info(f"Installer thread {threading.get_ident()} exiting")

    def _begin_install_transfer(self, job: ModelInstallJob) -> bool:
        with self._lock:
            if job._cancel_requested:
                return False
            self._staging_of(job).begin_transfer()
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
        staging = self._staging_of(job)
        staging.claim()

        if isinstance(job.source, LocalModelSource):
            staging.protect_source(Path(job.source.path))

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

        staging.delete_marker()

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
                try:
                    self._write_install_marker(job, status=job.status)
                except Exception as marker_error:
                    self._logger.error(f"Failed to persist install recovery marker in {staging.tmpdir}: {marker_error}")
                raise
            except Exception:
                staging.end_transfer()
                raise
            staging.end_transfer()
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

        # The bool returned by this callback determines if the model is added to the list of models found by the search
        def on_model_found(model_path: Path) -> bool:
            resolved_path = model_path.resolve()
            if self._ledger.is_protected(resolved_path):
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
        self._ledger.clear_recovery(recovery_root)
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
        job = ModelInstallJob(
            id=self._next_id(),
            source=source,
            config_in=config or ModelRecordChanges(),
            local_path=Path(source.path),
            inplace=source.inplace or False,
        )
        job._staging = InstallStaging(self._ledger)
        return job

    def _import_from_hf(
        self,
        source: HFModelSource,
        config: Optional[ModelRecordChanges] = None,
    ) -> ModelInstallJob:
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
            remote_files=self._pin_new_hf_file_download(source, remote_files),
        )

    def _import_external_model(
        self,
        source: ExternalModelSource,
        config: Optional[ModelRecordChanges] = None,
    ) -> ModelInstallJob:
        job = ModelInstallJob(
            id=self._next_id(),
            source=source,
            config_in=config or ModelRecordChanges(),
            local_path=self._app_config.models_path,
            inplace=True,
        )
        job._staging = InstallStaging(self._ledger)
        return job

    def _import_remote_model(
        self,
        source: HFModelSource | URLModelSource,
        remote_files: List[RemoteModelFile],
        metadata: Optional[AnyModelRepoMetadata],
        config: Optional[ModelRecordChanges],
    ) -> ModelInstallJob:
        if len(remote_files) == 0:
            raise ValueError(f"{source}: No downloadable files found")
        source_hf_endpoint = self._hf_endpoint_of(source, remote_files)
        revision = self._download_revision(metadata, remote_files, source_hf_endpoint)
        # Hugging Face staging is reused only from the same commit: a branch moves, so files staged from it may be
        # older than the ones this install would download. A staging directory's recorded endpoint still identifies
        # it as Hugging Face staging after HF_ENDPOINT changes.
        staging = self._ledger.create_staging(
            str(source),
            revision,
            reusable=lambda staged_endpoint: (
                is_commit_hash(revision) or not self._is_hf_source(source, staged_endpoint or source_hf_endpoint)
            ),
        )
        assert staging.tmpdir is not None
        install_job = ModelInstallJob(
            id=self._next_id(),
            source=source,
            config_in=config or ModelRecordChanges(),
            source_metadata=metadata,
            local_path=staging.tmpdir,  # local path may change once the download has started due to content-disposition handling
            bytes=0,
            total_bytes=0,
        )
        install_job._staging = staging
        install_job._source_hf_endpoint = source_hf_endpoint
        install_job._source_revision = revision

        # Handle multiple subfolders for HFModelSource
        subfolders = source.subfolders if isinstance(source, HFModelSource) else []
        return self._enqueue_remote_download(
            job=install_job,
            source=source,
            remote_files=remote_files,
            metadata=metadata,
            subfolder=source.subfolder if isinstance(source, HFModelSource) and len(subfolders) <= 1 else None,
            subfolders=subfolders if len(subfolders) > 1 else None,
        )

    def _enqueue_remote_download(
        self,
        job: ModelInstallJob,
        source: HFModelSource | URLModelSource,
        remote_files: List[RemoteModelFile],
        metadata: Optional[AnyModelRepoMetadata],
        subfolder: Optional[Path] = None,
        subfolders: Optional[List[Path]] = None,
        resume_metadata: Optional[dict[str, dict[str, Any]]] = None,
        restart_from_zero: Optional[set[str]] = None,
        report_missing_partials: bool = False,
    ) -> ModelInstallJob:
        """Queue the download of `remote_files` into `destdir` for `job`.

        `resume_metadata` maps each URL to what a previous download recorded for it (see `part_resume_metadata`).
        A part resumes its partial file only with a recorded validator, sent as If-Range: a server whose file changed
        sends the whole new revision instead, and the part pauses as restart-required rather than splicing the two.
        A partial without a validator, or one whose resume the server already refused, is deleted and its part
        reported as restarted from scratch. The URLs in `restart_from_zero` start over at the caller's request.

        With `report_missing_partials`, a part that had started (its recorded download path is set) but has neither a
        partial nor a finished file is reported as restarted from scratch too. A resume sets it, and so does a restart of
        a paused install, whose staging folder was kept; a restart after an error does not, as the error deleted the
        staging folder.
        """
        previous_source_metadata = job.source_metadata
        previous_status = job.status
        previous_local_path = job.local_path
        previous_total_bytes = job.total_bytes
        previous_multifile_job = job._multifile_job
        previous_download_parts = job.download_parts
        staging = self._staging_of(job)
        destdir = staging.tmpdir
        assert destdir is not None, "a remote install downloads into its staging directory"
        claim_was_already_held = staging.claimed
        staging.claim()

        multifile_job: Optional[MultiFileDownloadJob] = None
        try:
            multifile_job = self._multifile_download(
                remote_files=remote_files,
                dest=destdir,
                subfolder=subfolder,
                subfolders=subfolders,
                # The source holds only a token sent with the install. The Hugging Face login is read on every
                # (re)queue so a retry after logging in uses it; it is never copied onto the source, which the
                # install marker persists and a resubmit sends back as an explicit token.
                access_token=self._download_access_token(source, job._source_hf_endpoint),
                submit_job=False,  # Important! Don't submit the job until we have set our _download_cache dict
            )
            resume_metadata = resume_metadata or {}
            restart_from_zero = restart_from_zero or set()
            for part in multifile_job.download_parts:
                meta = resume_metadata.get(str(part.source))
                if not meta:
                    continue
                part.canonical_url = meta.get("canonical_url") or part.canonical_url
                part.expected_total_bytes = meta.get("expected_total_bytes") or part.expected_total_bytes
                part.final_url = meta.get("final_url") or part.final_url
                if recorded_path := meta.get("download_path"):
                    if staging.contains(Path(recorded_path)):
                        part.download_path = Path(recorded_path)
                    else:
                        self._logger.warning(
                            f"Ignoring recorded download path {recorded_path} outside the staging folder {destdir}"
                        )
            with self._lock:
                if self._stop_event.is_set():
                    raise InstallDownloadConflictError(
                        "The model install service is stopping; the download was not submitted."
                    )
                if job.cancelled or job.paused:
                    raise InstallDownloadConflictError(
                        "The install changed state before the download could be submitted; retry the operation."
                    )
                for part in multifile_job.download_parts:
                    url = str(part.source)
                    if url in restart_from_zero:
                        self._discard_download(part)
                        continue
                    meta = resume_metadata.get(url) or {}
                    if part.resumable_partial() is None:
                        if report_missing_partials and meta.get("download_path") and self._download_vanished(part):
                            part.resume_from_scratch = True
                            part.resume_message = _MISSING_PARTIAL_MESSAGE
                        continue
                    part.etag = meta.get("etag")
                    part.last_modified = meta.get("last_modified")
                    part.resume_refused = bool(meta.get("resume_refused"))
                    if part.may_resume_partial():
                        continue
                    self._discard_download(part)
                    part.resume_from_scratch = True
                    part.resume_message = (
                        _REFUSED_PARTIAL_MESSAGE if part.resume_refused else _UNVALIDATED_PARTIAL_MESSAGE
                    )
                    # The partial is gone (a failed delete raised). A validator left on the part would report it as
                    # missing.
                    part.etag = part.last_modified = None
                    part.resume_refused = False
                job.source_metadata = metadata
                job.local_path = destdir
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
                job.total_bytes = previous_total_bytes
                job._multifile_job = previous_multifile_job
                job.download_parts = previous_download_parts
            if not claim_was_already_held:
                staging.release_claim()
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
                job.total_bytes = previous_total_bytes
                job._multifile_job = previous_multifile_job
                job.download_parts = previous_download_parts
            if not claim_was_already_held:
                staging.release_claim()
            raise
        return job

    @staticmethod
    def _download_vanished(part: DownloadJob) -> bool:
        """Whether neither the finished file nor the partial file of a part is on disk."""
        path = part.expected_download_path()
        return path is not None and not path.exists() and not in_progress_path(path).exists()

    def _discard_download(self, part: DownloadJob) -> None:
        """Delete the final and partial file of a part that must start over, so its download finds neither.

        Raises `InstallDownloadConflictError` when one cannot be deleted (on Windows, while another process holds it
        open). The download would otherwise resume a partial meant to be dropped, with a Range request that no
        validator guards, and could splice two revisions of the file.
        """
        path = part.expected_download_path()
        if path is None:
            return
        for stale_path in (path, in_progress_path(path)):
            if not stale_path.is_file():
                continue
            self._logger.info(f"Deleting partial file before restart: {stale_path}")
            try:
                stale_path.unlink()
            except OSError as e:
                raise InstallDownloadConflictError(
                    f"Could not delete {stale_path.name} to restart its download; retry once no other program is using "
                    f"it: {e}"
                ) from e

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
        return self._download_queue.multifile_download(
            parts=self._staged_remote_files(remote_files, subfolder, subfolders),
            dest=dest,
            access_token=access_token,
            submit_job=submit_job,
            on_start=self._download_started_callback,
            on_progress=self._download_progress_callback,
            on_complete=self._download_complete_callback,
            on_error=self._download_error_callback,
            on_cancelled=self._download_cancelled_callback,
        )

    @staticmethod
    def _staged_remote_files(
        remote_files: List[RemoteModelFile],
        subfolder: Optional[Path] = None,
        subfolders: Optional[List[Path]] = None,
    ) -> List[RemoteModelFile]:
        """Map each remote file, in order, to its path relative to the staging directory."""
        if not remote_files:
            return []
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
        return parts

    # ------------------------------------------------------------------
    # Callbacks are executed by the download queue in a separate thread
    # ------------------------------------------------------------------
    def _download_started_callback(self, download_job: MultiFileDownloadJob) -> None:
        with self._lock:
            if install_job := self._download_cache.get(download_job.id, None):
                if install_job.cancelled or install_job.paused:
                    return
                first_part = install_job.status != InstallStatus.DOWNLOADING
                install_job.status = InstallStatus.DOWNLOADING

                if install_job.local_path == self._staging_of(install_job).tmpdir:  # first time
                    assert download_job.download_path
                    install_job.local_path = download_job.download_path
                install_job.download_parts = download_job.download_parts
                install_job.bytes = sum(x.bytes for x in download_job.download_parts)
                total_parts = sum(x.total_bytes for x in download_job.download_parts)
                if total_parts > 0:
                    install_job.total_bytes = max(install_job.total_bytes or 0, total_parts)
                if first_part:
                    self._signal_job_download_started(install_job)
                else:
                    # Record the validators and file name of the part that just started, which a restore after a
                    # crash needs to resume its partial file.
                    self._write_install_marker(install_job, status=InstallStatus.DOWNLOADING)

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
                self._install_completed_event.set()  # wake wait_for_job(), which waits out this callback
                if install_job.cancelled:
                    self._discard_cancelled_staging(install_job)
                elif install_job.paused:
                    # A pause requested before this callback must not turn into an automatic install.
                    self._write_install_marker(install_job, status=InstallStatus.PAUSED)
                    if self._stop_event.is_set():
                        self._staging_of(install_job).release_claim()
                else:
                    self._signal_job_downloads_done(install_job)
                    if self._stop_event.is_set():
                        # Every part is complete, so the staging tree is safe to recover without this process's claim.
                        self._staging_of(install_job).release_claim()
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

    def _staging_has_recoverable_data(self, job: ModelInstallJob) -> bool:
        """Return True if the staging tree holds downloaded bytes a resume would keep: a completed file, or a partial
        file the resume would continue.

        This reads the disk rather than the current download's parts: a retry enqueues only the files that had not
        finished, so its parts do not know about the ones completed by an earlier attempt. A partial counts only with
        a recorded validator its server has not refused, since a resume deletes any other partial and starts over.
        Empty files are skipped because they cost nothing to fetch again.
        """
        tmpdir = self._staging_of(job).tmpdir
        if tmpdir is None:
            return False
        resumable_partials = {
            in_progress_path(Path(meta["download_path"])).resolve()
            for meta in self._recorded_resume_metadata(job).values()
            if meta.get("download_path")
            and (meta.get("etag") or meta.get("last_modified"))
            and not meta.get("resume_refused")
        }
        for root, _dirs, files in os.walk(tmpdir):
            for name in files:
                if name == INSTALL_MARKER_FILENAME or name.startswith(f"{INSTALL_MARKER_FILENAME}."):
                    continue
                path = Path(root, name)
                try:
                    if path.stat().st_size == 0:
                        continue
                except OSError:
                    continue
                if not name.endswith(".downloading") or path.resolve() in resumable_partials:
                    return True
        return False

    def _download_error_callback(self, download_job: MultiFileDownloadJob, excp: Optional[Exception] = None) -> None:
        with self._lock:
            if install_job := self._download_cache.pop(download_job.id, None):
                self._install_completed_event.set()  # wake wait_for_job(), which waits out this callback
                assert excp is not None
                staging = self._staging_of(install_job)
                if install_job.cancelled:
                    self._discard_cancelled_staging(install_job)
                elif install_job.paused:
                    # Preserve the explicit pause even if an already-terminal download reports its error late.
                    self._write_install_marker(install_job, status=InstallStatus.PAUSED)
                    if self._stop_event.is_set():
                        staging.release_claim()
                elif _is_transient_download_error(excp) and self._staging_has_recoverable_data(install_job):
                    # A transient part failure must not discard completed parts and resumable partials.
                    # Mirror the pause path: mark the install paused and persist the marker so the tmpdir
                    # survives the startup dangling-dir sweep and restart_failed()/resume_job() can recover
                    # the failed parts. The failed part keeps its error in download_parts.
                    install_job.status = InstallStatus.PAUSED
                    self._write_install_marker(install_job, status=InstallStatus.PAUSED)
                    self._signal_job_paused(install_job)
                    if self._stop_event.is_set():
                        staging.release_claim()
                else:
                    self._set_error(install_job, excp)
                    self._download_queue.cancel_job(download_job)
                    self._staging_of(install_job).discard()

                # Let other threads know that the number of downloads has changed
                self._downloads_changed_event.set()

    def _download_cancelled_callback(self, download_job: MultiFileDownloadJob) -> None:
        with self._lock:
            if install_job := self._download_cache.pop(download_job.id, None):
                self._install_completed_event.set()  # wake wait_for_job(), which waits out this callback
                self._downloads_changed_event.set()
                if install_job.cancelled:
                    self._discard_cancelled_staging(install_job)
                    return
                if install_job.paused or any(part.resume_required for part in download_job.download_parts):
                    was_paused = install_job.paused
                    install_job.status = InstallStatus.PAUSED
                    self._write_install_marker(install_job, status=InstallStatus.PAUSED)
                    if not was_paused:
                        self._signal_job_paused(install_job)
                    # A user-paused install remains protected from orphan cleanup until resumed or cancelled.
                    # Shutdown releases only after the download queue has quiesced this callback.
                    if self._stop_event.is_set():
                        self._staging_of(install_job).release_claim()
                    self._downloads_changed_event.set()
                    return
                staging = self._staging_of(install_job)
                # if install job has already registered an error, then do not replace its status with cancelled
                if not install_job.errored and not install_job.paused:
                    install_job.cancel()
                    self._discard_cancelled_staging(install_job)
                staging.release_claim()

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

    def _signal_job_paused(self, job: ModelInstallJob) -> None:
        if self._event_bus:
            self._event_bus.emit_model_install_paused(job)
        self._install_completed_event.set()  # wake wait_for_job(), which returns paused jobs

    def _signal_job_completed(self, job: ModelInstallJob) -> None:
        job.status = InstallStatus.COMPLETED
        assert job.config_out
        self._logger.info(f"Model install complete: {job.source}")
        self._logger.debug(f"{job.local_path} registered key {job.config_out.key}")
        self._staging_of(job).delete_marker()
        if self._event_bus:
            assert job.local_path is not None
            assert job.config_out is not None
            self._event_bus.emit_model_install_complete(job)

    def _signal_job_errored(self, job: ModelInstallJob) -> None:
        self._logger.error(f"Model install error: {job.source}\n{job.error_type}: {job.error}")
        self._staging_of(job).delete_marker()
        if self._event_bus:
            assert job.error_type is not None
            assert job.error is not None
            self._event_bus.emit_model_install_error(job)
        self._install_completed_event.set()  # wake wait_for_job()

    def _signal_job_cancelled(self, job: ModelInstallJob) -> None:
        self._logger.info(f"Model install canceled: {job.source}")
        self._staging_of(job).delete_marker()
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
    def _normalize_huggingface_blob_url(url: AnyHttpUrl, trusted_endpoint: Optional[str] = None) -> Url:
        """Convert Hugging Face file page URLs (on huggingface.co or the configured mirror) to download URLs."""
        parts = _hf_file_url_parts(str(url), trusted_endpoint)
        if parts is None or parts.kind != "blob":
            return Url(str(url))
        return Url(parts.with_revision(parts.revision))
