# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Abstract base class for a multithreaded model download queue.
"""

from abc import ABC, abstractmethod
from enum import Enum
from functools import total_ordering
from queue import PriorityQueue
from pathlib import Path
from threading import Thread
from typing import Set, List, Optional, Dict

from pydantic import BaseModel, Field, validator
from pydantic.networks import AnyHttpUrl


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


@total_ordering
class DownloadJobBase(ABC, BaseModel):
    """Class to monitor and control a model download request."""

    priority: int = Field(default=10, description="Queue priority; lower values are higher priority")
    id: int = Field(description="Numeric ID of this job")
    url: AnyHttpUrl = Field(description="URL to download")
    destination: Path = Field(description="Destination of URL on local disk")
    status: DownloadJobStatus = Field(default=DownloadJobStatus.IDLE, description="Status of the download")
    bytes: int = Field(default=0, description="Bytes downloaded so far")
    total_bytes: int = Field(default=0, description="Total bytes to download")
    error: Exception = Field(default=None, description="Exception that caused an error")

    class Config():
        """Config object for this pydantic class."""

        arbitrary_types_allowed = True
        validate_assignment = True

    @validator('destination')
    def path_doesnt_exist(cls, v):
        """Don't allow a destination to clobber an existing file."""
        if v.exists():
            raise ValueError(f"{v} already exists")
        return v

    @abstractmethod
    def start(self):
        """Start the job putting it into ENQUEUED state."""
        pass

    @abstractmethod
    def pause(self):
        """Pause the job, putting it into PAUSED state."""
        pass

    @abstractmethod
    def cancel(self):
        """Cancel the job, clearing partial downloads and putting it into ERROR state."""
        pass

    @abstractmethod
    def change_priority(self, delta: int=+1):
        """
        Change the job's priority.

        :param delta: Value to increment or decrement priority.

        Lower values are higher priority.  The default starting value is 10.
        Thus to make this a really high priority job:
           job.change_priority(-10).
        """
        pass

    def __lt__(self, other: "DownloadJobBase") -> bool:
        """
        Return True if self.priority < other.priority.

        :param other: The DownloadJobBase that this will be compared against.
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
    ) -> DownloadJobBase:
        """
        Create a download job.

        :param url: URL to download.
        :param destdir: Directory to download into.
        :param filename: Optional name of file, if not provided
        will use the content-disposition field to assign the name.
        :returns DownloadJob: The DownloadJob object for this task.
        """
        pass

    @abstractmethod
    def list_jobs(self) -> List[DownloadJobBase]:
        """
        List active DownloadJobBases.

        :returns List[DownloadJobBase]: List of download jobs whose state is not "completed."
        """
        pass

    @abstractmethod
    def delete_job(self, job: DownloadJobBase):
        """
        Cancel a download and delete its job.

        :param job: DownloadJobBase to delete
        """
        pass

    @abstractmethod
    def id_to_job(self, id: int) -> DownloadJobBase:
        """
        Return the DownloadJobBase corresponding to the string ID.

        :param id: ID of the DownloadJobBase.

        Exceptions:
        * UnknownJobIDException
        """
        pass

    @abstractmethod
    def start_all_jobs(self):
        """Enqueue all stopped jobs."""
        pass

    @abstractmethod
    def stop_all_jobs(self):
        """Pause and dequeue all active jobs."""
        pass


class DownloadJob(DownloadJobBase):
    """Implementation of DownloadJobBase"""

    _queue: DownloadQueueBase = PrivateAttr()

    def __init__(self, queue: DownloadQueueBase, **kwargs):
        """
        Create a new DownloadJob.

        :param queue: Reference to the DownloadQueueBase object that created us.
        """
        super().__init__(**kwargs)
        self._queue = queue
                 
class DownloadQueue(DownloadQueueBase):
    """Class for queued download of models."""

    _jobs: Dict[int, DownloadJobBase]
    _worker_pool: Set[Thread]
    _queue: PriorityQueue
    _next_job: int = 0

    def __init__(self, max_parallel_dl: int = 5):
        """
        Initialize DownloadQueue.

        :param max_parallel_dl: Number of simultaneous downloads allowed.
        """
        self._jobs = dict()
        self._next_job = 0
        self._queue = PriorityQueue()
        self._worker_pool = set()
        for i in range(0, max_parallel_dl):
            worker = Thread(target=self._download_next_item, daemon=True)
            worker.start()
            self._worker_pool.add(worker)

        def create_download_job(
            self,
            url: str,
            destdir: Path,
            filename: Optional[Path] = None,
        ) -> DownloadJob:
            
