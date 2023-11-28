# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""Model download service."""

from abc import ABC, abstractmethod
from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import Any, List, Optional, Callable

from pydantic import BaseModel, Field
from pydantic.networks import AnyHttpUrl
from invokeai.app.services.events.events_base import EventServiceBase

class DownloadJobStatus(str, Enum):
    """State of a download job."""

    WAITING = "waiting"  # not enqueued, will not run
    RUNNING = "running"  # actively downloading
    COMPLETED = "completed"  # finished running
    ERROR = "error"  # terminated with an error message

class DownloadJobCancelledException(Exception):
    """This exception is raised when a download job is cancelled."""

DownloadEventHandler = Callable[["DownloadJob", ], None]

@total_ordering
class DownloadJob(BaseModel):
    """Class to monitor and control a model download request."""

    id          : int = Field(description="Numeric ID of this job", default=-1)  # default id is a sentinel
    priority    : int = Field(default=10, description="Queue priority; lower values are higher priority")
    source      : AnyHttpUrl = Field(description="Where to download from. Specific types specified in child classes.")
    dest        : Path = Field(description="Destination of downloaded model on local disk")
    status      : DownloadJobStatus = Field(default=DownloadJobStatus.WAITING, description="Status of the download")
    job_started : int = Field(default=0, description="Timestamp for when the download job started")
    job_ended   : int = Field(default=0, description="Timestamp for when the download job ende1d (completed or errored)")
    on_start    : Optional[DownloadEventHandler] = Field(default=None, description="handler for download start event")
    on_progress : Optional[DownloadEventHandler] = Field(default=None, description="handler for download progress events")
    on_complete : Optional[DownloadEventHandler] = Field(default=None, description="handler for download completion events")
    on_error    : Optional[DownloadEventHandler] = Field(default=None, description="handler for download error events")
    error_type  : Optional[str] = Field(default=None, description="Name of exception that caused an error")
    error       : Optional[str] = Field(default=None, description="Traceback of the exception that caused an error")

    def __le__(self, other: "DownloadJob") -> bool:
        """Return True if this job's priority is less than another's."""
        return self.priority <= other.priority



class DownloadQueueServiceBase(ABC):
    """Multithreaded queue for downloading models via URL."""

    @abstractmethod
    def download_url(
            self,
            source: AnyHttpUrl,
            dest: Path,
            access_token: Optional[str] = None,
            on_start: Optional[DownloadEventHandler] = None,
            on_progress: Optional[DownloadEventHandler] = None,
            on_complete: Optional[DownloadEventHandler] = None,
            on_error: Optional[DownloadEventHandler] = None
    ) -> DownloadJob:
        """
        Create a download job.

        :param source: Source of the download as a URL.
        :param destdir: Path to download to. See below. 
        :param on_start, on_progress, on_complete, on_error: Callbacks for the indicated
         events.
        :returns: A DownloadJob object for monitoring the state of the download.

        The `dest` argument is a Path object. Its behavior is:

        1. If the path exists and is a directory, then the URL contents will be downloaded 
            into the directory using the filename indicated in the response's `Content-Disposition` field. 
        2. If the path does not exist, then it is taken as the name of a new file to create with the downloaded
           content.
        3. If the path exists and is an existing file, then this generates an OSError 
           with a `[Errno 17] File exists` message.

        """
        pass

    @abstractmethod
    def list_jobs(self) -> List[DownloadJob]:
        """
        List active DownloadJobBases.

        :returns List[DownloadJobBase]: List of download jobs whose state is not "completed."
        """
        pass

    @abstractmethod
    def id_to_job(self, id: int) -> DownloadJob:
        """
        Return the DownloadJobBase corresponding to the integer ID.

        :param id: ID of the DownloadJobBase.

        Exceptions:
        * UnknownJobIDException
        """
        pass

    @abstractmethod
    def cancel_all_jobs(self):
        """Cancel all active and enquedjobs."""
        pass

    @abstractmethod
    def prune_jobs(self):
        """Prune completed and errored queue items from the job list."""
        pass

    @abstractmethod
    def cancel_job(self, job: DownloadJob):
        """Cancel the job, clearing partial downloads and putting it into ERROR state."""
        pass

    @abstractmethod
    def join(self):
        """Wait until all jobs are off the queue."""
        pass
