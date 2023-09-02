# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Abstract base class for a multithreaded model download queue.
"""

import os
import re
import requests
import threading

from abc import ABC, abstractmethod
from enum import Enum
from functools import total_ordering
from queue import PriorityQueue
from pathlib import Path
from threading import Thread
from typing import Set, List, Optional, Dict, Callable

from pydantic import BaseModel, Field, validator, PrivateAttr
from pydantic.networks import AnyHttpUrl

from invokeai.backend.util.logging import InvokeAILogger


class EventServicesBase:   # forward declaration
    pass

class DownloadJobStatus(str, Enum):
    """State of a download job."""

    IDLE = "idle"            # not enqueued, will not run
    ENQUEUED = "enqueued"    # enqueued but not yet active
    RUNNING = "running"      # actively downloading
    PAUSED = "paused"        # previously started, now paused
    COMPLETED = "completed"  # finished running
    ERROR = "error"          # terminated with an error message


class UnknownJobIDException(Exception):
    """Raised when an invalid Job ID is requested."""


class CancelledJobException(Exception):
    """Raised when a job is cancelled."""


DownloadEventHandler = Callable[["DownloadJobBase"], None]


@total_ordering
class DownloadJob(BaseModel):
    """Class to monitor and control a model download request."""

    priority: int = Field(default=10, description="Queue priority; lower values are higher priority")
    id: int = Field(description="Numeric ID of this job")
    url: AnyHttpUrl = Field(description="URL to download")
    destination: Path = Field(description="Destination of URL on local disk")
    access_token: Optional[str] = Field(description="access token needed to access this resource")
    status: DownloadJobStatus = Field(default=DownloadJobStatus.IDLE, description="Status of the download")
    bytes: int = Field(default=0, description="Bytes downloaded so far")
    total_bytes: int = Field(default=0, description="Total bytes to download")
    event_handler: Optional[DownloadEventHandler] = Field(description="Callable will be called whenever job status changes")
    error: Exception = Field(default=None, description="Exception that caused an error")

    class Config():
        """Config object for this pydantic class."""

        arbitrary_types_allowed = True
        validate_assignment = True

    # @validator('destination')
    # def path_doesnt_exist(cls, v):
    #     """Don't allow a destination to clobber an existing file."""
    #     if v.exists():
    #         raise ValueError(f"{v} already exists")
    #     return v

    def __lt__(self, other: "DownloadJob") -> bool:
        """
        Return True if self.priority < other.priority.

        :param other: The DownloadJob that this will be compared against.
        """
        if not hasattr(other, "id"):
            return NotImplemented
        return self.id < other.id


class DownloadQueueBase(ABC):
    """Abstract base class for managing model downloads."""

    @abstractmethod
    def create_download_job(
            self,
            url: str,
            destdir: Path,
            filename: Optional[Path] = None,
            start: bool = True,
            access_token: Optional[str] = None,
            event_handler: Optional[DownloadEventHandler] = None,
    ) -> int:
        """
        Create a download job.

        :param url: URL to download.
        :param destdir: Directory to download into.
        :param filename: Optional name of file, if not provided
        will use the content-disposition field to assign the name.
        :param start: Immediately start job [True]
        :param event_handler: Optional callable that will be called whenever job status changes.
        :returns job id: The numeric ID of the DownloadJob object for this task.
        """
        pass

    @abstractmethod
    def list_jobs(self) -> List[DownloadJob]:
        """
        List active DownloadJobs.

        :returns List[DownloadJob]: List of download jobs whose state is not "completed."
        """
        pass

    @abstractmethod
    def id_to_job(self, id: int) -> DownloadJob:
        """
        Return the DownloadJob corresponding to the string ID.

        :param id: ID of the DownloadJob.

        Exceptions:
        * UnknownJobIDException
        """
        pass

    @abstractmethod
    def start_all_jobs(self):
        """Enqueue all stopped jobs."""
        pass

    @abstractmethod
    def pause_all_jobs(self):
        """Pause and dequeue all active jobs."""
        pass

    @abstractmethod
    def cancel_all_jobs(self):
        """Cancel all active and enquedjobs."""
        pass

    @abstractmethod
    def start_job(self, id: int):
        """Start the job putting it into ENQUEUED state."""
        pass

    @abstractmethod
    def pause_job(self, id: int):
        """Pause the job, putting it into PAUSED state."""
        pass

    @abstractmethod
    def cancel_job(self, id: int):
        """Cancel the job, clearing partial downloads and putting it into ERROR state."""
        pass

    @abstractmethod
    def change_priority(self, id: int, delta: int):
        """
        Change the job's priority.

        :param id: ID of the job
        :param delta: Value to increment or decrement priority.

        Lower values are higher priority.  The default starting value is 10.
        Thus to make this a really high priority job:
           job.change_priority(-10).
        """
        pass

    @abstractmethod
    def join(self):
        """Wait until all jobs are off the queue."""
        pass


class DownloadQueue(DownloadQueueBase):
    """Class for queued download of models."""

    _jobs: Dict[int, DownloadJob]
    _worker_pool: Set[Thread]
    _queue: PriorityQueue
    _lock: threading.Lock
    _logger: InvokeAILogger
    _event_bus: Optional[EventServicesBase]
    _event_handler: Optional[DownloadEventHandler]
    _next_job_id: int = 0

    def __init__(self,
                 max_parallel_dl: int = 5,
                 events: Optional["EventServicesBase"] = None,
                 event_handler: Optional[DownloadEventHandler] = None,
                 ):
        """
        Initialize DownloadQueue.

        :param max_parallel_dl: Number of simultaneous downloads allowed [5].
        :param events: Optional EventServices bus for reporting events.
        :param event_handler: Optional callable that will be called each time a job status changes.
        """
        self._jobs = dict()
        self._next_job_id = 0
        self._queue = PriorityQueue()
        self._worker_pool = set()
        self._lock = threading.RLock()
        self._logger = InvokeAILogger.getLogger()
        self._event_bus = events
        self._event_handler = event_handler

        self._start_workers(max_parallel_dl)

    def create_download_job(
            self,
            url: str,
            destdir: Path,
            filename: Optional[Path] = None,
            start: bool = True,
            access_token: Optional[str] = None,
            event_handler: Optional[DownloadEventHandler] = None,
    ) -> int:
        try:
            self._lock.acquire()
            id = self._next_job_id
            self._jobs[id] = DownloadJob(
                id=id,
                url=url,
                destination=Path(destdir) / (filename or "."),
                access_token=access_token,
                event_handler=(event_handler or self._event_handler),
            )
            self._next_job_id += 1
            job = self._jobs[id]
        finally:
            self._lock.release()
        if start:
            self.start_job(id)
        return job.id

    def join(self):
        self._queue.join()

    def list_jobs(self) -> List[DownloadJob]:
        return self._jobs.values()


    def change_priority(self, id: int, delta: int):
        try:
            self._lock.acquire()
            job = self._jobs[id]
            job.priority += delta
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def cancel_job(self, job: DownloadJob):
        try:
            self._lock.acquire()
            job.status = DownloadJobStatus.ERROR
            job.error = CancelledJobException(f"Job {job.id} cancelled at caller's request")
            self._update_job_status
            del self._jobs[job.id]
        finally:
            self._lock.release()

    def id_to_job(self, id: int) -> DownloadJob:
        try:
            return self._jobs[id]
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp

    def start_job(self, id: int):
        try:
            job = self._jobs[id]
            self._update_job_status(job, DownloadJobStatus.ENQUEUED)
            self._queue.put(job)
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp

    def pause_job(self, id: int):
        try:
            self._lock.acquire()
            job = self._jobs[id]
            self._update_job_status(job, DownloadJobStatus.PAUSED)
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def start_all_jobs(self):
        try:
            self._lock.acquire()
            for id in self._jobs:
                self.start_job(id)
        finally:
            self._lock.release()

    def pause_all_jobs(self, id: int):
        try:
            self._lock.acquire()
            for id in self._jobs:
                self.pause_job(id)
        finally:
            self._lock.release()

    def cancel_all_jobs(self, id: int):
        try:
            self._lock.acquire()
            for id in self._jobs:
                self.cancel_job(id)
        finally:
            self._lock.release()

    def _start_workers(self, max_workers: int):
        for i in range(0, max_workers):
            worker = Thread(target=self._download_next_item, daemon=True)
            worker.start()
            self._worker_pool.add(worker)

    def _download_next_item(self):
        """Worker thread gets next job on priority queue."""
        while True:
            job = self._queue.get()
            if job.status == DownloadJobStatus.ENQUEUED:    # Don't do anything for cancelled or errored jobs
                self._download_with_resume(job)
            self._queue.task_done()

    def _download_with_resume(self, job: DownloadJob):
        """Do the actual download."""
        header = {"Authorization": f"Bearer {job.access_token}"} if job.access_token else {}
        open_mode = "wb"
        exist_size = 0

        resp = requests.get(job.url, header, stream=True)
        content_length = int(resp.headers.get("content-length", 0))
        job.total_bytes = content_length

        if job.destination.is_dir():
            try:
                file_name = re.search('filename="(.+)"', resp.headers.get("Content-Disposition")).group(1)
            except AttributeError:
                file_name = os.path.basename(job.url)
            dest = job.destination / file_name
        else:
            dest = job.destination
            dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            job.bytes = dest.stat().st_size
            header["Range"] = f"bytes={job.bytes}-"
            open_mode = "ab"
            resp = requests.get(job.url, headers=header, stream=True)  # new request with range

        if exist_size > content_length:
            self._logger.warning("corrupt existing file found. re-downloading")
            os.remove(dest)
            exist_size = 0

        if resp.status_code == 416 or (content_length > 0 and exist_size == content_length):
            self._logger.warning(f"{dest}: complete file found. Skipping.")
            self._update_job_status(job, DownloadJobStatus.COMPLETED)
            return

        if resp.status_code == 206 or exist_size > 0:
            self._logger.warning(f"{dest}: partial file found. Resuming...")
        elif resp.status_code != 200:
            self._logger.error(f"An error occurred during downloading {dest}: {resp.reason}")
        else:
            self._logger.info(f"{dest}: Downloading...")

        self._update_job_status(job, DownloadJobStatus.RUNNING)
        report_delta = job.total_bytes / 100  # report every 1% change
        last_report_bytes = 0
        try:
            with open(dest, open_mode) as file:
                for data in resp.iter_content(chunk_size=1024):
                    if job.status != DownloadJobStatus.RUNNING:   # cancelled, paused or errored
                        return
                    job.bytes += file.write(data)
                    if job.bytes - last_report_bytes >= report_delta:
                        last_report_bytes = job.bytes
                        self._update_job_status(job)

            self._update_job_status(job, DownloadJobStatus.COMPLETED)
            del self._jobs[job.id]
        except Exception as excp:
            self._logger.error(f"An error occurred while downloading {dest}: {str(excp)}")
            job.error = excp
            self._update_job_status(job, DownloadJobStatus.ERROR)

    def _update_job_status(self,
                           job: DownloadJob,
                           new_status: Optional[DownloadJobStatus] = None
                           ):
        """Optionally change the job status and send an event indicating a change of state."""
        if new_status:
            job.status = new_status
        if bus := self._event_bus:
            bus.emit_model_download_event(job)
        self._logger.debug(f"Status update for download job {job.id}: {job}")
        if job.event_handler:
            job.event_handler(job)
