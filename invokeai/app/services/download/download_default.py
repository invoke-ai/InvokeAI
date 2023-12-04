# Copyright (c) 2023, Lincoln D. Stein
"""Implementation of multithreaded download queue for invokeai."""

import os
import re
import threading
import traceback
from logging import Logger
from pathlib import Path
from queue import PriorityQueue
from typing import Dict, List, Optional, Set, Union

import requests
from pydantic.networks import AnyHttpUrl, Url
from requests import HTTPError
from tqdm import tqdm

from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.backend.util.logging import InvokeAILogger

from .download_base import (
    DownloadEventHandler,
    DownloadJob,
    DownloadJobCancelledException,
    DownloadJobStatus,
    DownloadQueueServiceBase,
    UnknownJobIDException,
)

# Maximum number of bytes to download during each call to requests.iter_content()
DOWNLOAD_CHUNK_SIZE = 100000

# marker that the queue is done and that thread should exit
STOP_JOB = DownloadJob(id=-99, priority=-99, source=Url("http://dummy.com/"), dest=Path("/"))


class DownloadQueueService(DownloadQueueServiceBase):
    """Class for queued download of models."""

    _jobs: Dict[int, DownloadJob]
    _worker_pool: Set[threading.Thread]
    _queue: PriorityQueue
    _lock: threading.Lock
    _logger: Logger
    _events: Optional[EventServiceBase] = None
    _next_job_id: int = 0
    _requests: requests.sessions.Session

    def __init__(
        self,
        max_parallel_dl: int = 5,
        event_bus: Optional[EventServiceBase] = None,
        requests_session: Optional[requests.sessions.Session] = None,
    ):
        """
        Initialize DownloadQueue.

        :param max_parallel_dl: Number of simultaneous downloads allowed [5].
        :param requests_session: Optional requests.sessions.Session object, for unit tests.
        """
        self._jobs = {}
        self._next_job_id = 0
        self._queue = PriorityQueue()
        self._worker_pool = set()
        self._lock = threading.Lock()
        self._logger = InvokeAILogger.get_logger("DownloadQueueService")
        self._event_bus = event_bus
        self._requests = requests_session or requests.Session()

        self._start_workers(max_parallel_dl)

    def download(
        self,
        source: Union[str, AnyHttpUrl],
        dest: Union[str, Path],
        priority: int = 10,
        access_token: Optional[str] = None,
        on_start: Optional[DownloadEventHandler] = None,
        on_progress: Optional[DownloadEventHandler] = None,
        on_complete: Optional[DownloadEventHandler] = None,
        on_error: Optional[DownloadEventHandler] = None,
    ) -> DownloadJob:
        """Create a download job and return its ID."""
        with self._lock:
            id = self._next_job_id
            self._next_job_id += 1
            job = DownloadJob(
                id=id,
                source=Url(source)
                if isinstance(source, str)
                else source,  # maybe a validator in DownloadJob would be better here
                dest=Path(dest),
                priority=priority,
                access_token=access_token,
            )
            job.set_callbacks(
                on_start=on_start,
                on_progress=on_progress,
                on_complete=on_complete,
                on_error=on_error,
            )
            self._jobs[id] = job
            self._queue.put(job)
        return job

    def release(self):
        """Signal our threads to exit when queue done."""
        for thread in self._worker_pool:
            if thread.is_alive():
                self._queue.put(STOP_JOB)

    def join(self):
        """Wait for all jobs to complete."""
        self._queue.join()

    def list_jobs(self) -> List[DownloadJob]:
        """List all the jobs."""
        return list(self._jobs.values())

    def prune_jobs(self):
        """Prune completed and errored queue items from the job list."""
        with self._lock:
            to_delete = set()
            try:
                for job_id, job in self._jobs.items():
                    if self._in_terminal_state(job):
                        to_delete.add(job_id)
                for job_id in to_delete:
                    del self._jobs[job_id]
            except KeyError as excp:
                raise UnknownJobIDException("Unrecognized job") from excp

    def id_to_job(self, id: int) -> DownloadJob:
        """Translate a job ID into a DownloadJob object."""
        try:
            return self._jobs[id]
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp

    def cancel_job(self, job: DownloadJob):
        """
        Cancel the indicated job.

        If it is running it will be stopped.
        job.status will be set to DownloadJobStatus.CANCELLED
        """
        with self._lock:
            job.cancel()

    def cancel_all_jobs(self, preserve_partial: bool = False):
        """Cancel all jobs (those not in enqueued, running or paused state)."""
        for job in self._jobs.values():
            if not self._in_terminal_state(job):
                self.cancel_job(job)

    def _in_terminal_state(self, job: DownloadJob):
        return job.status in [
            DownloadJobStatus.COMPLETED,
            DownloadJobStatus.ERROR,
        ]

    def _start_workers(self, max_workers: int):
        """Start the requested number of worker threads."""
        for i in range(0, max_workers):  # noqa B007
            worker = threading.Thread(target=self._download_next_item, daemon=True)
            worker.start()
            self._worker_pool.add(worker)

    def _download_next_item(self):
        """Worker thread gets next job on priority queue."""
        done = False
        while not done:
            job = self._queue.get()

            try:
                if job == STOP_JOB:  # marker that queue is done
                    done = True
                    continue

                job.job_started = get_iso_timestamp()
                self._do_download(job)
                self._signal_job_complete(job)

            except (DownloadJobCancelledException, OSError, HTTPError) as excp:
                job.error_type = excp.__class__.__name__ + f"({str(excp)})"
                job.error = traceback.format_exc()
                self._signal_job_error(job)
                if isinstance(excp, DownloadJobCancelledException):
                    self._cleanup_cancelled_job(job)

            finally:
                job.job_ended = get_iso_timestamp()
                self._queue.task_done()

    def _do_download(self, job: DownloadJob) -> None:
        """Do the actual download."""
        url = job.source
        header = {"Authorization": f"Bearer {job.access_token}"} if job.access_token else {}
        open_mode = "wb"

        # Make a streaming request. This will retrieve headers including
        # content-length and content-disposition, but not fetch any content itself
        resp = self._requests.get(str(url), headers=header, stream=True)
        if not resp.ok:
            raise HTTPError(resp.reason)
        content_length = int(resp.headers.get("content-length", 0))
        job.total_bytes = content_length

        if job.dest.is_dir():
            file_name = os.path.basename(str(url.path))  # default is to use the last bit of the URL

            if match := re.search('filename="(.+)"', resp.headers.get("Content-Disposition", "")):
                remote_name = match.group(1)
                if self._validate_filename(job.dest.as_posix(), remote_name):
                    file_name = remote_name

            job.download_path = job.dest / file_name

        else:
            job.dest.parent.mkdir(parents=True, exist_ok=True)
            job.download_path = job.dest

        assert job.download_path

        # potentially resume a partial previous download
        if job.download_path.exists():
            job.bytes = job.download_path.stat().st_size
            header["Range"] = f"bytes={job.bytes}-"
            open_mode = "ab"
            resp = self._requests.get(str(url), headers=header, stream=True)  # new range request

        # signal caller that the download is starting. At this point, key fields such as
        # download_path and total_bytes will be populated. We call it here because the might
        # discover that the local file is already complete and generate a COMPLETED status.
        self._signal_job_started(job)

        # "range not satisfiable" - local file is at least as large as the remote file
        if resp.status_code == 416 or (content_length > 0 and job.bytes >= content_length):
            self._logger.warning(f"{job.download_path}: complete file found. Skipping.")
            return

        # "partial content" - local file is smaller than remote file
        elif resp.status_code == 206 or job.bytes > 0:
            self._logger.warning(f"{job.download_path}: partial file found. Resuming")

        # some other error
        elif resp.status_code != 200:
            raise HTTPError(resp.reason)

        self._logger.debug(f"{job.source}: Downloading {job.download_path}")
        report_delta = job.total_bytes / 100  # report every 1% change
        last_report_bytes = 0

        # DOWNLOAD LOOP
        with open(job.download_path, open_mode) as file:
            for data in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if job.cancelled:
                    raise DownloadJobCancelledException("Job was cancelled at caller's request")

                job.bytes += file.write(data)
                if (job.bytes - last_report_bytes >= report_delta) or (job.bytes >= job.total_bytes):
                    last_report_bytes = job.bytes
                    self._signal_job_progress(job)

    def _validate_filename(self, directory: str, filename: str) -> bool:
        pc_name_max = os.pathconf(directory, "PC_NAME_MAX") if hasattr(os, "pathconf") else 260   # hardcoded for windows
        pc_path_max = os.pathconf(directory, "PC_PATH_MAX") if hasattr(os, "pathconf") else 32767 # hardcoded for windows with long names enabled
        if "/" in filename:
            return False
        if filename.startswith(".."):
            return False
        if len(filename) > pc_name_max:
            return False
        if len(os.path.join(directory, filename)) > pc_path_max:
            return False
        return True

    def _signal_job_started(self, job: DownloadJob) -> None:
        job.status = DownloadJobStatus.RUNNING
        if job.on_start:
            try:
                job.on_start(job)
            except Exception as e:
                self._logger.error(e)
        if self._event_bus:
            assert job.download_path
            self._event_bus.emit_download_started(str(job.source), job.download_path.as_posix())

    def _signal_job_progress(self, job: DownloadJob) -> None:
        if job.on_progress:
            try:
                job.on_progress(job)
            except Exception as e:
                self._logger.error(e)
        if self._event_bus:
            assert job.download_path
            self._event_bus.emit_download_progress(
                str(job.source),
                download_path=job.download_path.as_posix(),
                current_bytes=job.bytes,
                total_bytes=job.total_bytes,
            )

    def _signal_job_complete(self, job: DownloadJob) -> None:
        job.status = DownloadJobStatus.COMPLETED
        if job.on_complete:
            try:
                job.on_complete(job)
            except Exception as e:
                self._logger.error(e)
        if self._event_bus:
            assert job.download_path
            self._event_bus.emit_download_complete(
                str(job.source), download_path=job.download_path.as_posix(), total_bytes=job.total_bytes
            )

    def _signal_job_error(self, job: DownloadJob) -> None:
        job.status = DownloadJobStatus.ERROR
        if job.on_error:
            try:
                job.on_error(job)
            except Exception as e:
                self._logger.error(e)
        if self._event_bus:
            assert job.error_type
            assert job.error
            self._event_bus.emit_download_error(str(job.source), error_type=job.error_type, error=job.error)

    def _cleanup_cancelled_job(self, job: DownloadJob):
        self._logger.warning(f"Cleaning up leftover files from cancelled download job {job.download_path}")
        try:
            if partial_file := job.download_path:
                partial_file.unlink()
        except OSError as excp:
            self._logger.warning(excp)


# Example on_progress event handler to display a TQDM status bar
# Activate with:
#   download_service.download('http://foo.bar/baz', '/tmp', on_progress=TqdmProgress().job_update
class TqdmProgress(object):
    """TQDM-based progress bar object to use in on_progress handlers."""

    _bars: Dict[int, tqdm]  # the tqdm object
    _last: Dict[int, int]  # last bytes downloaded

    def __init__(self):  # noqa D107
        self._bars = {}
        self._last = {}

    def update(self, job: DownloadJob):  # noqa D102
        job_id = job.id
        # new job
        if job_id not in self._bars:
            assert job.download_path
            dest = Path(job.download_path).name
            self._bars[job_id] = tqdm(
                desc=dest,
                initial=0,
                total=job.total_bytes,
                unit="iB",
                unit_scale=True,
            )
            self._last[job_id] = 0
        self._bars[job_id].update(job.bytes - self._last[job_id])
        self._last[job_id] = job.bytes
