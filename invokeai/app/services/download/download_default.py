# Copyright (c) 2023, Lincoln D. Stein
"""Implementation of multithreaded download queue for invokeai."""

import os
import re
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, PriorityQueue
from typing import Any, Dict, List, Optional, Set

import requests
from pydantic.networks import AnyHttpUrl
from requests import HTTPError
from tqdm import tqdm

from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.backend.util.logging import InvokeAILogger

from .download_base import (
    DownloadEventHandler,
    DownloadExceptionHandler,
    DownloadJob,
    DownloadJobCancelledException,
    DownloadJobStatus,
    DownloadQueueServiceBase,
    ServiceInactiveException,
    UnknownJobIDException,
)

# Maximum number of bytes to download during each call to requests.iter_content()
DOWNLOAD_CHUNK_SIZE = 100000


class DownloadQueueService(DownloadQueueServiceBase):
    """Class for queued download of models."""

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
        self._jobs: Dict[int, DownloadJob] = {}
        self._next_job_id = 0
        self._queue: PriorityQueue[DownloadJob] = PriorityQueue()
        self._stop_event = threading.Event()
        self._job_completed_event = threading.Event()
        self._worker_pool: Set[threading.Thread] = set()
        self._lock = threading.Lock()
        self._logger = InvokeAILogger.get_logger("DownloadQueueService")
        self._event_bus = event_bus
        self._requests = requests_session or requests.Session()
        self._accept_download_requests = False
        self._max_parallel_dl = max_parallel_dl

    def start(self, *args: Any, **kwargs: Any) -> None:
        """Start the download worker threads."""
        with self._lock:
            if self._worker_pool:
                raise Exception("Attempt to start the download service twice")
            self._stop_event.clear()
            self._start_workers(self._max_parallel_dl)
            self._accept_download_requests = True

    def stop(self, *args: Any, **kwargs: Any) -> None:
        """Stop the download worker threads."""
        with self._lock:
            if not self._worker_pool:
                raise Exception("Attempt to stop the download service before it was started")
            self._accept_download_requests = False  # reject attempts to add new jobs to queue
            queued_jobs = [x for x in self.list_jobs() if x.status == DownloadJobStatus.WAITING]
            active_jobs = [x for x in self.list_jobs() if x.status == DownloadJobStatus.RUNNING]
            if queued_jobs:
                self._logger.warning(f"Cancelling {len(queued_jobs)} queued downloads")
            if active_jobs:
                self._logger.info(f"Waiting for {len(active_jobs)} active download jobs to complete")
            with self._queue.mutex:
                self._queue.queue.clear()
            self.cancel_all_jobs()
            self._stop_event.set()
            for thread in self._worker_pool:
                thread.join()
            self._worker_pool.clear()

    def submit_download_job(
        self,
        job: DownloadJob,
        on_start: Optional[DownloadEventHandler] = None,
        on_progress: Optional[DownloadEventHandler] = None,
        on_complete: Optional[DownloadEventHandler] = None,
        on_cancelled: Optional[DownloadEventHandler] = None,
        on_error: Optional[DownloadExceptionHandler] = None,
    ) -> None:
        """Enqueue a download job."""
        if not self._accept_download_requests:
            raise ServiceInactiveException(
                "The download service is not currently accepting requests. Please call start() to initialize the service."
            )
        with self._lock:
            job.id = self._next_job_id
            self._next_job_id += 1
            job.set_callbacks(
                on_start=on_start,
                on_progress=on_progress,
                on_complete=on_complete,
                on_cancelled=on_cancelled,
                on_error=on_error,
            )
            self._jobs[job.id] = job
            self._queue.put(job)

    def download(
        self,
        source: AnyHttpUrl,
        dest: Path,
        priority: int = 10,
        access_token: Optional[str] = None,
        on_start: Optional[DownloadEventHandler] = None,
        on_progress: Optional[DownloadEventHandler] = None,
        on_complete: Optional[DownloadEventHandler] = None,
        on_cancelled: Optional[DownloadEventHandler] = None,
        on_error: Optional[DownloadExceptionHandler] = None,
    ) -> DownloadJob:
        """Create and enqueue a download job and return it."""
        if not self._accept_download_requests:
            raise ServiceInactiveException(
                "The download service is not currently accepting requests. Please call start() to initialize the service."
            )
        job = DownloadJob(
            source=source,
            dest=dest,
            priority=priority,
            access_token=access_token,
        )
        self.submit_download_job(
            job,
            on_start=on_start,
            on_progress=on_progress,
            on_complete=on_complete,
            on_cancelled=on_cancelled,
            on_error=on_error,
        )
        return job

    def join(self) -> None:
        """Wait for all jobs to complete."""
        self._queue.join()

    def list_jobs(self) -> List[DownloadJob]:
        """List all the jobs."""
        return list(self._jobs.values())

    def prune_jobs(self) -> None:
        """Prune completed and errored queue items from the job list."""
        with self._lock:
            to_delete = set()
            for job_id, job in self._jobs.items():
                if job.in_terminal_state:
                    to_delete.add(job_id)
            for job_id in to_delete:
                del self._jobs[job_id]

    def id_to_job(self, id: int) -> DownloadJob:
        """Translate a job ID into a DownloadJob object."""
        try:
            return self._jobs[id]
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp

    def cancel_job(self, job: DownloadJob) -> None:
        """
        Cancel the indicated job.

        If it is running it will be stopped.
        job.status will be set to DownloadJobStatus.CANCELLED
        """
        with self._lock:
            job.cancel()

    def cancel_all_jobs(self) -> None:
        """Cancel all jobs (those not in enqueued, running or paused state)."""
        for job in self._jobs.values():
            if not job.in_terminal_state:
                self.cancel_job(job)

    def wait_for_job(self, job: DownloadJob, timeout: int = 0) -> DownloadJob:
        """Block until the indicated job has reached terminal state, or when timeout limit reached."""
        start = time.time()
        while not job.in_terminal_state:
            if self._job_completed_event.wait(timeout=0.25):  # in case we miss an event
                self._job_completed_event.clear()
            if timeout > 0 and time.time() - start > timeout:
                raise TimeoutError("Timeout exceeded")
        return job

    def _start_workers(self, max_workers: int) -> None:
        """Start the requested number of worker threads."""
        self._stop_event.clear()
        for i in range(0, max_workers):  # noqa B007
            worker = threading.Thread(target=self._download_next_item, daemon=True)
            self._logger.debug(f"Download queue worker thread {worker.name} starting.")
            worker.start()
            self._worker_pool.add(worker)

    def _download_next_item(self) -> None:
        """Worker thread gets next job on priority queue."""
        done = False
        while not done:
            if self._stop_event.is_set():
                done = True
                continue
            try:
                job = self._queue.get(timeout=1)
            except Empty:
                continue
            try:
                job.job_started = get_iso_timestamp()
                self._do_download(job)
                self._signal_job_complete(job)
            except (OSError, HTTPError) as excp:
                job.error_type = excp.__class__.__name__ + f"({str(excp)})"
                job.error = traceback.format_exc()
                self._signal_job_error(job, excp)
            except DownloadJobCancelledException:
                self._signal_job_cancelled(job)
                self._cleanup_cancelled_job(job)

            finally:
                job.job_ended = get_iso_timestamp()
                self._job_completed_event.set()  # signal a change to terminal state
                self._queue.task_done()
        self._logger.debug(f"Download queue worker thread {threading.current_thread().name} exiting.")

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

        job.content_type = resp.headers.get("Content-Type")
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

        # Don't clobber an existing file. See commit 82c2c85202f88c6d24ff84710f297cfc6ae174af
        # for code that instead resumes an interrupted download.
        if job.download_path.exists():
            raise OSError(f"[Errno 17] File {job.download_path} exists")

        # append ".downloading" to the path
        in_progress_path = self._in_progress_path(job.download_path)

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
        with open(in_progress_path, open_mode) as file:
            for data in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if job.cancelled:
                    raise DownloadJobCancelledException("Job was cancelled at caller's request")

                job.bytes += file.write(data)
                if (job.bytes - last_report_bytes >= report_delta) or (job.bytes >= job.total_bytes):
                    last_report_bytes = job.bytes
                    self._signal_job_progress(job)

        # if we get here we are done and can rename the file to the original dest
        self._logger.debug(f"{job.source}: saved to {job.download_path} (bytes={job.bytes})")
        in_progress_path.rename(job.download_path)

    def _validate_filename(self, directory: str, filename: str) -> bool:
        pc_name_max = os.pathconf(directory, "PC_NAME_MAX") if hasattr(os, "pathconf") else 260  # hardcoded for windows
        pc_path_max = (
            os.pathconf(directory, "PC_PATH_MAX") if hasattr(os, "pathconf") else 32767
        )  # hardcoded for windows with long names enabled
        if "/" in filename:
            return False
        if filename.startswith(".."):
            return False
        if len(filename) > pc_name_max:
            return False
        if len(os.path.join(directory, filename)) > pc_path_max:
            return False
        return True

    def _in_progress_path(self, path: Path) -> Path:
        return path.with_name(path.name + ".downloading")

    def _signal_job_started(self, job: DownloadJob) -> None:
        job.status = DownloadJobStatus.RUNNING
        if job.on_start:
            try:
                job.on_start(job)
            except Exception as e:
                self._logger.error(
                    f"An error occurred while processing the on_start callback: {traceback.format_exception(e)}"
                )
        if self._event_bus:
            assert job.download_path
            self._event_bus.emit_download_started(str(job.source), job.download_path.as_posix())

    def _signal_job_progress(self, job: DownloadJob) -> None:
        if job.on_progress:
            try:
                job.on_progress(job)
            except Exception as e:
                self._logger.error(
                    f"An error occurred while processing the on_progress callback: {traceback.format_exception(e)}"
                )
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
                self._logger.error(
                    f"An error occurred while processing the on_complete callback: {traceback.format_exception(e)}"
                )
        if self._event_bus:
            assert job.download_path
            self._event_bus.emit_download_complete(
                str(job.source), download_path=job.download_path.as_posix(), total_bytes=job.total_bytes
            )

    def _signal_job_cancelled(self, job: DownloadJob) -> None:
        if job.status not in [DownloadJobStatus.RUNNING, DownloadJobStatus.WAITING]:
            return
        job.status = DownloadJobStatus.CANCELLED
        if job.on_cancelled:
            try:
                job.on_cancelled(job)
            except Exception as e:
                self._logger.error(
                    f"An error occurred while processing the on_cancelled callback: {traceback.format_exception(e)}"
                )
        if self._event_bus:
            self._event_bus.emit_download_cancelled(str(job.source))

    def _signal_job_error(self, job: DownloadJob, excp: Optional[Exception] = None) -> None:
        job.status = DownloadJobStatus.ERROR
        self._logger.error(f"{str(job.source)}: {traceback.format_exception(excp)}")
        if job.on_error:
            try:
                job.on_error(job, excp)
            except Exception as e:
                self._logger.error(
                    f"An error occurred while processing the on_error callback: {traceback.format_exception(e)}"
                )
        if self._event_bus:
            assert job.error_type
            assert job.error
            self._event_bus.emit_download_error(str(job.source), error_type=job.error_type, error=job.error)

    def _cleanup_cancelled_job(self, job: DownloadJob) -> None:
        self._logger.debug(f"Cleaning up leftover files from cancelled download job {job.download_path}")
        try:
            if job.download_path:
                partial_file = self._in_progress_path(job.download_path)
                partial_file.unlink()
        except OSError as excp:
            self._logger.warning(excp)


# Example on_progress event handler to display a TQDM status bar
# Activate with:
#   download_service.download(DownloadJob('http://foo.bar/baz', '/tmp', on_progress=TqdmProgress().update))
class TqdmProgress(object):
    """TQDM-based progress bar object to use in on_progress handlers."""

    _bars: Dict[int, tqdm]  # type: ignore
    _last: Dict[int, int]  # last bytes downloaded

    def __init__(self) -> None:  # noqa D107
        self._bars = {}
        self._last = {}

    def update(self, job: DownloadJob) -> None:  # noqa D102
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
