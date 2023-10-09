# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""Abstract base class for a multithreaded model download queue."""

from abc import ABC, abstractmethod
from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import requests
from pydantic import BaseModel, Field
from pydantic.networks import AnyHttpUrl

from invokeai.app.services.config import InvokeAIAppConfig


class DownloadJobStatus(str, Enum):
    """State of a download job."""

    IDLE = "idle"  # not enqueued, will not run
    ENQUEUED = "enqueued"  # enqueued but not yet active
    RUNNING = "running"  # actively downloading
    PAUSED = "paused"  # previously started, now paused
    COMPLETED = "completed"  # finished running
    ERROR = "error"  # terminated with an error message
    CANCELLED = "cancelled"  # terminated by caller


class UnknownJobIDException(Exception):
    """Raised when an invalid Job is referenced."""


DownloadEventHandler = Callable[["DownloadJobBase"], None]


@total_ordering
class DownloadJobBase(BaseModel):
    """Class to monitor and control a model download request."""

    priority: int = Field(default=10, description="Queue priority; lower values are higher priority")
    id: int = Field(description="Numeric ID of this job", default=-1)  # default id is a sentinel
    source: Any = Field(description="Where to download from. Specific types specified in child classes.")
    destination: Path = Field(description="Destination of downloaded model on local disk")
    access_token: Optional[str] = Field(description="access token needed to access this resource")
    status: DownloadJobStatus = Field(default=DownloadJobStatus.IDLE, description="Status of the download")
    event_handlers: Optional[List[DownloadEventHandler]] = Field(
        description="Callables that will be called whenever job status changes",
        default_factory=list,
    )
    job_started: Optional[float] = Field(description="Timestamp for when the download job started")
    job_ended: Optional[float] = Field(description="Timestamp for when the download job ended (completed or errored)")
    job_sequence: Optional[int] = Field(
        description="Counter that records order in which this job was dequeued (used in unit testing)"
    )
    preserve_partial_downloads: bool = Field(
        description="if true, then preserve partial downloads when cancelled or errored", default=False
    )
    error: Optional[Exception] = Field(default=None, description="Exception that caused an error")

    def add_event_handler(self, handler: DownloadEventHandler):
        """Add an event handler to the end of the handlers list."""
        if self.event_handlers is not None:
            self.event_handlers.append(handler)

    def clear_event_handlers(self):
        """Clear all event handlers."""
        self.event_handlers = list()

    def cleanup(self, preserve_partial_downloads: bool = False):
        """Possibly do some action when work is finished."""
        pass

    class Config:
        """Config object for this pydantic class."""

        arbitrary_types_allowed = True
        validate_assignment = True

    def __lt__(self, other: "DownloadJobBase") -> bool:
        """
        Return True if self.priority < other.priority.

        :param other: The DownloadJobBase that this will be compared against.
        """
        if not hasattr(other, "priority"):
            return NotImplemented
        return self.priority < other.priority


class DownloadQueueBase(ABC):
    """Abstract base class for managing model downloads."""

    @abstractmethod
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
        :param config: InvokeAIAppConfig object, used to configure the logger and other options.
        :param quiet: If true, don't log the start of download jobs. Useful for subrequests.
        """
        pass

    @abstractmethod
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
        """
        Create and submit a download job.

        :param source: Source of the download - URL, repo_id or Path
        :param destdir: Directory to download into.
        :param priority: Initial priority for this job [10]
        :param filename: Optional name of file, if not provided
        will use the content-disposition field to assign the name.
        :param start: Immediately start job [True]
        :param variant: Variant to download, such as "fp16" (repo_ids only).
        :param event_handlers: Optional callables that will be called whenever job status changes.
        :returns the job: job.id will be a non-negative value after execution

        Known variants currently are:
        1. onnx
        2. openvino
        3. fp16
        4. None (usually returns fp32 model)
        """
        pass

    def submit_download_job(
        self,
        job: DownloadJobBase,
        start: bool = True,
    ):
        """
        Submit a download job.

        :param job: A DownloadJobBase
        :param start: Immediately start job [True]

        After execution, `job.id` will be set to a non-negative value.
        """
        pass

    @abstractmethod
    def release(self):
        """
        Release resources used by queue.

        If threaded downloads are
        used, then this will stop the threads.
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
    def id_to_job(self, id: int) -> DownloadJobBase:
        """
        Return the DownloadJobBase corresponding to the string ID.

        :param id: ID of the DownloadJobBase.

        Exceptions:
        * UnknownJobException

        Note that once a job is completed, id_to_job() may no longer
        recognize the job. Call id_to_job() before the job completes
        if you wish to keep the job object around after it has
        completed work.
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
    def prune_jobs(self):
        """Prune completed and errored queue items from the job list."""
        pass

    @abstractmethod
    def cancel_all_jobs(self, preserve_partial: bool = False):
        """
        Cancel all jobs (those in enqueued, running and paused states).

        :param preserve_partial: Keep partially downloaded files [False].
        """
        pass

    @abstractmethod
    def start_job(self, job: DownloadJobBase):
        """Start the job putting it into ENQUEUED state."""
        pass

    @abstractmethod
    def pause_job(self, job: DownloadJobBase):
        """Pause the job, putting it into PAUSED state."""
        pass

    @abstractmethod
    def cancel_job(self, job: DownloadJobBase, preserve_partial: bool = False):
        """
        Cancel the job, clearing partial downloads and putting it into CANCELLED state.

        :param preserve_partial: Keep partial downloads [False]
        """
        pass

    @abstractmethod
    def change_priority(self, job: DownloadJobBase, delta: int):
        """
        Change the job's priority.

        :param job: Job to change
        :param delta: Value to increment or decrement priority.

        Lower values are higher priority.  The default starting value is 10.
        Thus to make this a really high priority job:
           job.change_priority(-10).
        """
        pass

    @abstractmethod
    def join(self):
        """
        Wait until all jobs are off the queue.

        Note that once a job is completed, id_to_job() will
        no longer recognize the job.
        """
        pass

    @abstractmethod
    def select_downloader(self, job: DownloadJobBase) -> Callable[[DownloadJobBase], None]:
        """Based on the job type select the download method."""
        pass

    @abstractmethod
    def get_url_for_job(self, job: DownloadJobBase) -> AnyHttpUrl:
        """
        Given a job, translate its source field into a downloadable URL.

        Intended to be subclassed to cover various source types.
        """
        pass
