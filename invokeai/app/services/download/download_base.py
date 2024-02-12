# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""Model download service."""

from abc import ABC, abstractmethod
from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field, PrivateAttr
from pydantic.networks import AnyHttpUrl


class DownloadJobStatus(str, Enum):
    """State of a download job."""

    WAITING = "waiting"  # not enqueued, will not run
    RUNNING = "running"  # actively downloading
    COMPLETED = "completed"  # finished running
    CANCELLED = "cancelled"  # user cancelled
    ERROR = "error"  # terminated with an error message


class DownloadJobCancelledException(Exception):
    """This exception is raised when a download job is cancelled."""


class UnknownJobIDException(Exception):
    """This exception is raised when an invalid job id is referened."""


class ServiceInactiveException(Exception):
    """This exception is raised when user attempts to initiate a download before the service is started."""


DownloadEventHandler = Callable[["DownloadJob"], None]
DownloadExceptionHandler = Callable[["DownloadJob", Optional[Exception]], None]


@total_ordering
class DownloadJob(BaseModel):
    """Class to monitor and control a model download request."""

    # required variables to be passed in on creation
    source: AnyHttpUrl = Field(description="Where to download from. Specific types specified in child classes.")
    dest: Path = Field(description="Destination of downloaded model on local disk; a directory or file path")
    access_token: Optional[str] = Field(default=None, description="authorization token for protected resources")
    # automatically assigned on creation
    id: int = Field(description="Numeric ID of this job", default=-1)  # default id is a sentinel
    priority: int = Field(default=10, description="Queue priority; lower values are higher priority")

    # set internally during download process
    status: DownloadJobStatus = Field(default=DownloadJobStatus.WAITING, description="Status of the download")
    download_path: Optional[Path] = Field(default=None, description="Final location of downloaded file")
    job_started: Optional[str] = Field(default=None, description="Timestamp for when the download job started")
    job_ended: Optional[str] = Field(
        default=None, description="Timestamp for when the download job ende1d (completed or errored)"
    )
    content_type: Optional[str] = Field(default=None, description="Content type of downloaded file")
    bytes: int = Field(default=0, description="Bytes downloaded so far")
    total_bytes: int = Field(default=0, description="Total file size (bytes)")

    # set when an error occurs
    error_type: Optional[str] = Field(default=None, description="Name of exception that caused an error")
    error: Optional[str] = Field(default=None, description="Traceback of the exception that caused an error")

    # internal flag
    _cancelled: bool = PrivateAttr(default=False)

    # optional event handlers passed in on creation
    _on_start: Optional[DownloadEventHandler] = PrivateAttr(default=None)
    _on_progress: Optional[DownloadEventHandler] = PrivateAttr(default=None)
    _on_complete: Optional[DownloadEventHandler] = PrivateAttr(default=None)
    _on_cancelled: Optional[DownloadEventHandler] = PrivateAttr(default=None)
    _on_error: Optional[DownloadExceptionHandler] = PrivateAttr(default=None)

    def __hash__(self) -> int:
        """Return hash of the string representation of this object, for indexing."""
        return hash(str(self))

    def __le__(self, other: "DownloadJob") -> bool:
        """Return True if this job's priority is less than another's."""
        return self.priority <= other.priority

    def cancel(self) -> None:
        """Call to cancel the job."""
        self._cancelled = True

    # cancelled and the callbacks are private attributes in order to prevent
    # them from being serialized and/or used in the Json Schema
    @property
    def cancelled(self) -> bool:
        """Call to cancel the job."""
        return self._cancelled

    @property
    def complete(self) -> bool:
        """Return true if job completed without errors."""
        return self.status == DownloadJobStatus.COMPLETED

    @property
    def running(self) -> bool:
        """Return true if the job is running."""
        return self.status == DownloadJobStatus.RUNNING

    @property
    def errored(self) -> bool:
        """Return true if the job is errored."""
        return self.status == DownloadJobStatus.ERROR

    @property
    def in_terminal_state(self) -> bool:
        """Return true if job has finished, one way or another."""
        return self.status not in [DownloadJobStatus.WAITING, DownloadJobStatus.RUNNING]

    @property
    def on_start(self) -> Optional[DownloadEventHandler]:
        """Return the on_start event handler."""
        return self._on_start

    @property
    def on_progress(self) -> Optional[DownloadEventHandler]:
        """Return the on_progress event handler."""
        return self._on_progress

    @property
    def on_complete(self) -> Optional[DownloadEventHandler]:
        """Return the on_complete event handler."""
        return self._on_complete

    @property
    def on_error(self) -> Optional[DownloadExceptionHandler]:
        """Return the on_error event handler."""
        return self._on_error

    @property
    def on_cancelled(self) -> Optional[DownloadEventHandler]:
        """Return the on_cancelled event handler."""
        return self._on_cancelled

    def set_callbacks(
        self,
        on_start: Optional[DownloadEventHandler] = None,
        on_progress: Optional[DownloadEventHandler] = None,
        on_complete: Optional[DownloadEventHandler] = None,
        on_cancelled: Optional[DownloadEventHandler] = None,
        on_error: Optional[DownloadExceptionHandler] = None,
    ) -> None:
        """Set the callbacks for download events."""
        self._on_start = on_start
        self._on_progress = on_progress
        self._on_complete = on_complete
        self._on_error = on_error
        self._on_cancelled = on_cancelled


class DownloadQueueServiceBase(ABC):
    """Multithreaded queue for downloading models via URL."""

    @abstractmethod
    def start(self, *args: Any, **kwargs: Any) -> None:
        """Start the download worker threads."""

    @abstractmethod
    def stop(self, *args: Any, **kwargs: Any) -> None:
        """Stop the download worker threads."""

    @abstractmethod
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
        """
        Create and enqueue download job.

        :param source: Source of the download as a URL.
        :param dest: Path to download to. See below.
        :param on_start, on_progress, on_complete, on_error: Callbacks for the indicated
         events.
        :returns: A DownloadJob object for monitoring the state of the download.

        The `dest` argument is a Path object. Its behavior is:

        1. If the path exists and is a directory, then the URL contents will be downloaded
            into that directory using the filename indicated in the response's `Content-Disposition` field.
            If no content-disposition is present, then the last component of the URL will be used (similar to
            wget's behavior).
        2. If the path does not exist, then it is taken as the name of a new file to create with the downloaded
           content.
        3. If the path exists and is an existing file, then the downloader will try to resume the download from
           the end of the existing file.

        """
        pass

    @abstractmethod
    def submit_download_job(
        self,
        job: DownloadJob,
        on_start: Optional[DownloadEventHandler] = None,
        on_progress: Optional[DownloadEventHandler] = None,
        on_complete: Optional[DownloadEventHandler] = None,
        on_cancelled: Optional[DownloadEventHandler] = None,
        on_error: Optional[DownloadExceptionHandler] = None,
    ) -> None:
        """
        Enqueue a download job.

        :param job: The DownloadJob
        :param on_start, on_progress, on_complete, on_error: Callbacks for the indicated
         events.
        """
        pass

    @abstractmethod
    def list_jobs(self) -> List[DownloadJob]:
        """
        List active download jobs.

        :returns List[DownloadJob]: List of download jobs whose state is not "completed."
        """
        pass

    @abstractmethod
    def id_to_job(self, id: int) -> DownloadJob:
        """
        Return the DownloadJob corresponding to the integer ID.

        :param id: ID of the DownloadJob.

        Exceptions:
        * UnknownJobIDException
        """
        pass

    @abstractmethod
    def cancel_all_jobs(self) -> None:
        """Cancel all active and enquedjobs."""
        pass

    @abstractmethod
    def prune_jobs(self) -> None:
        """Prune completed and errored queue items from the job list."""
        pass

    @abstractmethod
    def cancel_job(self, job: DownloadJob) -> None:
        """Cancel the job, clearing partial downloads and putting it into ERROR state."""
        pass

    @abstractmethod
    def join(self) -> None:
        """Wait until all jobs are off the queue."""
        pass

    @abstractmethod
    def wait_for_job(self, job: DownloadJob, timeout: int = 0) -> DownloadJob:
        """Wait until the indicated download job has reached a terminal state.

        This will block until the indicated install job has completed,
        been cancelled, or errored out.

        :param job: The job to wait on.
        :param timeout: Wait up to indicated number of seconds. Raise a TimeoutError if
        the job hasn't completed within the indicated time.
        """
        pass
