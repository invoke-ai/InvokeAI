# Copyright (c) 2023, Lincoln D. Stein
"""Implementation of multithreaded download queue for invokeai."""

import os
import re
import shutil
import threading
import time
import traceback
from pathlib import Path
from queue import PriorityQueue
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import requests
from huggingface_hub import HfApi, hf_hub_url
from pydantic import Field, parse_obj_as, validator
from pydantic.networks import AnyHttpUrl
from requests import HTTPError

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util import InvokeAILogger, Logger

from ..storage import DuplicateModelException
from .base import DownloadEventHandler, DownloadJobBase, DownloadJobStatus, DownloadQueueBase, UnknownJobIDException

# Maximum number of bytes to download during each call to requests.iter_content()
DOWNLOAD_CHUNK_SIZE = 100000

# marker that the queue is done and that thread should exit
STOP_JOB = DownloadJobBase(id=-99, priority=-99, source="dummy", destination="/")

# regular expression for picking up a URL
HTTP_RE = r"^https?://"


class DownloadJobPath(DownloadJobBase):
    """Download from a local Path."""

    source: Path = Field(description="Local filesystem Path where model can be found")


class DownloadJobRemoteSource(DownloadJobBase):
    """A DownloadJob from a remote source that provides progress info."""

    bytes: int = Field(default=0, description="Bytes downloaded so far")
    total_bytes: int = Field(default=0, description="Total bytes to download")
    access_token: Optional[str] = Field(description="access token needed to access this resource")


class DownloadJobURL(DownloadJobRemoteSource):
    """Job declaration for downloading individual URLs."""

    source: AnyHttpUrl = Field(description="URL to download")


class DownloadQueue(DownloadQueueBase):
    """Class for queued download of models."""

    _jobs: Dict[int, DownloadJobBase]
    _worker_pool: Set[threading.Thread]
    _queue: PriorityQueue
    _lock: threading.RLock
    _logger: Logger
    _event_handlers: List[DownloadEventHandler] = Field(default_factory=list)
    _next_job_id: int = 0
    _sequence: int = 0  # This is for debugging and used to tag jobs in dequeueing order
    _requests: requests.sessions.Session
    _quiet: bool = False

    def __init__(
        self,
        max_parallel_dl: int = 5,
        event_handlers: List[DownloadEventHandler] = [],
        requests_session: Optional[requests.sessions.Session] = None,
        config: Optional[InvokeAIAppConfig] = None,
        quiet: bool = False,
    ):
        """
        Initialize DownloadQueue.

        :param max_parallel_dl: Number of simultaneous downloads allowed [5].
        :param event_handler: Optional callable that will be called each time a job status changes.
        :param requests_session: Optional requests.sessions.Session object, for unit tests.
        """
        self._jobs = dict()
        self._next_job_id = 0
        self._queue = PriorityQueue()
        self._worker_pool = set()
        self._lock = threading.RLock()
        self._logger = InvokeAILogger.get_logger(config=config)
        self._event_handlers = event_handlers
        self._requests = requests_session or requests.Session()
        self._quiet = quiet

        self._start_workers(max_parallel_dl)

    def create_download_job(
        self,
        source: Union[str, Path, AnyHttpUrl],
        destdir: Path,
        start: bool = True,
        priority: int = 10,
        filename: Optional[Path] = None,
        variant: Optional[str] = None,
        access_token: Optional[str] = None,
        event_handlers: List[DownloadEventHandler] = [],
    ) -> DownloadJobBase:
        """Create a download job and return its ID."""
        kwargs: Dict[str, Optional[str]] = dict()

        cls = DownloadJobBase
        if Path(source).exists():
            cls = DownloadJobPath
        elif re.match(HTTP_RE, str(source)):
            cls = DownloadJobURL
            kwargs.update(access_token=access_token)
        else:
            raise NotImplementedError(f"Don't know what to do with this type of source: {source}")

        job = cls(
            source=source,
            destination=Path(destdir) / (filename or "."),
            event_handlers=event_handlers,
            priority=priority,
            **kwargs,
        )

        return self.submit_download_job(job, start)

    def submit_download_job(
        self,
        job: DownloadJobBase,
        start: bool = True,
    ):
        """Submit a job."""
        # add the queue's handlers
        for handler in self._event_handlers:
            job.add_event_handler(handler)
        try:
            self._lock.acquire()
            job.id = self._next_job_id
            self._jobs[job.id] = job
            self._next_job_id += 1
        finally:
            self._lock.release()
        if start:
            self.start_job(job)
        return job

    def release(self):
        """Signal our threads to exit when queue done."""
        for thread in self._worker_pool:
            if thread.is_alive():
                self._queue.put(STOP_JOB)

    def join(self):
        """Wait for all jobs to complete."""
        self._queue.join()

    def list_jobs(self) -> List[DownloadJobBase]:
        """List all the jobs."""
        return list(self._jobs.values())

    def change_priority(self, job: DownloadJobBase, delta: int):
        """Change the priority of a job. Smaller priorities run first."""
        try:
            self._lock.acquire()
            assert isinstance(self._jobs[job.id], DownloadJobBase)
            job.priority += delta
        except (AssertionError, KeyError) as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def prune_jobs(self):
        """Prune completed and errored queue items from the job list."""
        try:
            to_delete = set()
            self._lock.acquire()
            for job_id, job in self._jobs.items():
                if self._in_terminal_state(job):
                    to_delete.add(job_id)
            for job_id in to_delete:
                del self._jobs[job_id]
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def cancel_job(self, job: DownloadJobBase, preserve_partial: bool = False):
        """
        Cancel the indicated job.

        If it is running it will be stopped.
        job.status will be set to DownloadJobStatus.CANCELLED
        """
        try:
            self._lock.acquire()
            assert isinstance(self._jobs[job.id], DownloadJobBase)
            job.preserve_partial_downloads = preserve_partial
            self._update_job_status(job, DownloadJobStatus.CANCELLED)
            job.cleanup()
        except (AssertionError, KeyError) as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def id_to_job(self, id: int) -> DownloadJobBase:
        """Translate a job ID into a DownloadJobBase object."""
        try:
            return self._jobs[id]
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp

    def start_job(self, job: DownloadJobBase):
        """Enqueue (start) the indicated job."""
        try:
            assert isinstance(self._jobs[job.id], DownloadJobBase)
            self._update_job_status(job, DownloadJobStatus.ENQUEUED)
            self._queue.put(job)
        except (AssertionError, KeyError) as excp:
            raise UnknownJobIDException("Unrecognized job") from excp

    def pause_job(self, job: DownloadJobBase):
        """
        Pause (dequeue) the indicated job.

        The job can be restarted with start_job() and the download will pick up
        from where it left off.
        """
        try:
            self._lock.acquire()
            assert isinstance(self._jobs[job.id], DownloadJobBase)
            self._update_job_status(job, DownloadJobStatus.PAUSED)
            job.cleanup()
        except (AssertionError, KeyError) as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def start_all_jobs(self):
        """Start (enqueue) all jobs that are idle or paused."""
        try:
            self._lock.acquire()
            for job in self._jobs.values():
                if job.status in [DownloadJobStatus.IDLE, DownloadJobStatus.PAUSED]:
                    self.start_job(job)
        finally:
            self._lock.release()

    def pause_all_jobs(self):
        """Pause all running jobs."""
        try:
            self._lock.acquire()
            for job in self._jobs.values():
                if job.status == DownloadJobStatus.RUNNING:
                    self.pause_job(job)
        finally:
            self._lock.release()

    def cancel_all_jobs(self, preserve_partial: bool = False):
        """Cancel all running jobs."""
        try:
            self._lock.acquire()
            for job in self._jobs.values():
                if not self._in_terminal_state(job):
                    self.cancel_job(job, preserve_partial)
        finally:
            self._lock.release()

    def _in_terminal_state(self, job: DownloadJobBase):
        return job.status in [
            DownloadJobStatus.COMPLETED,
            DownloadJobStatus.ERROR,
            DownloadJobStatus.CANCELLED,
        ]

    def _start_workers(self, max_workers: int):
        """Start the requested number of worker threads."""
        for i in range(0, max_workers):
            worker = threading.Thread(target=self._download_next_item, daemon=True)
            worker.start()
            self._worker_pool.add(worker)

    def _download_next_item(self):
        """Worker thread gets next job on priority queue."""
        done = False
        while not done:
            job = self._queue.get()

            try:  # this is for debugging priority
                self._lock.acquire()
                job.job_sequence = self._sequence
                self._sequence += 1
            finally:
                self._lock.release()

            if job == STOP_JOB:  # marker that queue is done
                done = True

            if job.status == DownloadJobStatus.ENQUEUED:  # Don't do anything for non-enqueued jobs (shouldn't happen)
                if not self._quiet:
                    self._logger.info(f"{job.source}: Downloading to {job.destination}")
                do_download = self.select_downloader(job)
                do_download(job)

            if job.status == DownloadJobStatus.CANCELLED:
                self._cleanup_cancelled_job(job)

            self._queue.task_done()

    def select_downloader(self, job: DownloadJobBase) -> Callable[[DownloadJobBase], None]:
        """Based on the job type select the download method."""
        if isinstance(job, DownloadJobURL):
            return self._download_with_resume
        elif isinstance(job, DownloadJobPath):
            return self._download_path
        else:
            raise NotImplementedError(f"Don't know what to do with this job: {job}, type={type(job)}")

    def get_url_for_job(self, job: DownloadJobBase) -> AnyHttpUrl:
        return job.source

    def _download_with_resume(self, job: DownloadJobBase):
        """Do the actual download."""
        dest = None
        try:
            assert isinstance(job, DownloadJobRemoteSource)
            url = self.get_url_for_job(job)
            header = {"Authorization": f"Bearer {job.access_token}"} if job.access_token else {}
            open_mode = "wb"
            exist_size = 0

            resp = self._requests.get(url, headers=header, stream=True)
            content_length = int(resp.headers.get("content-length", 0))
            job.total_bytes = content_length

            if job.destination.is_dir():
                try:
                    file_name = ""
                    if match := re.search('filename="(.+)"', resp.headers["Content-Disposition"]):
                        file_name = match.group(1)
                    assert file_name != ""
                    self._validate_filename(
                        job.destination.as_posix(), file_name
                    )  # will raise a ValueError exception if file_name is suspicious
                except ValueError:
                    self._logger.warning(
                        f"Invalid filename '{file_name}' returned by source {url}, using last component of URL instead"
                    )
                    file_name = os.path.basename(url)
                except (KeyError, AssertionError):
                    file_name = os.path.basename(url)
                job.destination = job.destination / file_name
                dest = job.destination
            else:
                dest = job.destination
                dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists():
                job.bytes = dest.stat().st_size
                header["Range"] = f"bytes={job.bytes}-"
                open_mode = "ab"
                resp = self._requests.get(url, headers=header, stream=True)  # new request with range

            if exist_size > content_length:
                self._logger.warning("corrupt existing file found. re-downloading")
                os.remove(dest)
                exist_size = 0

            if resp.status_code == 416 or (content_length > 0 and exist_size == content_length):
                self._logger.warning(f"{dest}: complete file found. Skipping.")
                self._update_job_status(job, DownloadJobStatus.COMPLETED)
                return

            if resp.status_code == 206 or exist_size > 0:
                self._logger.warning(f"{dest}: partial file found. Resuming")
            elif resp.status_code != 200:
                raise HTTPError(resp.reason)
            else:
                self._logger.debug(f"{job.source}: Downloading {job.destination}")

            report_delta = job.total_bytes / 100  # report every 1% change
            last_report_bytes = 0

            self._update_job_status(job, DownloadJobStatus.RUNNING)
            with open(dest, open_mode) as file:
                for data in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if job.status != DownloadJobStatus.RUNNING:  # cancelled, paused or errored
                        return
                    job.bytes += file.write(data)
                    if job.bytes - last_report_bytes >= report_delta:
                        last_report_bytes = job.bytes
                        self._update_job_status(job)
            if job.status != DownloadJobStatus.RUNNING:  # cancelled, paused or errored
                return
            self._update_job_status(job, DownloadJobStatus.COMPLETED)
        except KeyboardInterrupt as excp:
            raise excp
        except DuplicateModelException as excp:
            self._logger.error(f"A model with the same hash as {dest} is already installed.")
            job.error = excp
            self._update_job_status(job, DownloadJobStatus.ERROR)
        except Exception as excp:
            self._logger.error(f"An error occurred while downloading/installing {job.source}: {str(excp)}")
            print(traceback.format_exc())
            job.error = excp
            self._update_job_status(job, DownloadJobStatus.ERROR)

    def _validate_filename(self, directory: str, filename: str):
        pc_name_max = os.pathconf(directory, "PC_NAME_MAX") if hasattr(os, "pathconf") else 260
        if "/" in filename:
            raise ValueError
        if filename.startswith(".."):
            raise ValueError
        if len(filename) > pc_name_max:
            raise ValueError
        if len(os.path.join(directory, filename)) > os.pathconf(directory, "PC_PATH_MAX"):
            raise ValueError

    def _update_job_status(self, job: DownloadJobBase, new_status: Optional[DownloadJobStatus] = None):
        """Optionally change the job status and send an event indicating a change of state."""
        if new_status:
            job.status = new_status
        self._logger.debug(f"Status update for download job {job.id}: {job}")
        if new_status == DownloadJobStatus.RUNNING and not job.job_started:
            job.job_started = time.time()
        elif new_status in [DownloadJobStatus.COMPLETED, DownloadJobStatus.ERROR]:
            job.job_ended = time.time()
        if job.event_handlers:
            for handler in job.event_handlers:
                try:
                    handler(job)
                except KeyboardInterrupt as excp:
                    raise excp
                except Exception as excp:
                    job.error = excp
                    self._update_job_status(job, DownloadJobStatus.ERROR)

    def _download_path(self, job: DownloadJobBase):
        """Call when the source is a Path or pathlike object."""
        source = Path(job.source).resolve()
        destination = Path(job.destination).resolve()
        try:
            self._update_job_status(job, DownloadJobStatus.RUNNING)
            if source != destination:
                shutil.move(source, destination)
            self._update_job_status(job, DownloadJobStatus.COMPLETED)
        except OSError as excp:
            job.error = excp
            self._update_job_status(job, DownloadJobStatus.ERROR)

    def _cleanup_cancelled_job(self, job: DownloadJobBase):
        if job.preserve_partial_downloads:
            return
        self._logger.warning("Cleaning up leftover files from cancelled download job {job.destination}")
        dest = Path(job.destination)
        if dest.is_file():
            dest.unlink()
        elif dest.is_dir():
            shutil.rmtree(dest.as_posix(), ignore_errors=True)
