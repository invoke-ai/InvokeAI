# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Model download service.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from pydantic.networks import AnyHttpUrl

from invokeai.backend.model_manager.download import DownloadEventHandler, DownloadJobBase, DownloadQueue

from .events import EventServiceBase


class DownloadQueueServiceBase(ABC):
    """Multithreaded queue for downloading models via URL or repo_id."""

    @abstractmethod
    def create_download_job(
        self,
        source: Union[str, Path, AnyHttpUrl],
        destdir: Path,
        filename: Optional[Path] = None,
        start: bool = True,
        access_token: Optional[str] = None,
        event_handlers: Optional[List[DownloadEventHandler]] = None,
    ) -> DownloadJobBase:
        """
        Create a download job.

        :param source: Source of the download - URL, repo_id or local Path
        :param destdir: Directory to download into.
        :param filename: Optional name of file, if not provided
        will use the content-disposition field to assign the name.
        :param start: Immediately start job [True]
        :param event_handler: Callable that receives a DownloadJobBase and acts on it.
        :returns job id: The numeric ID of the DownloadJobBase object for this task.
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
    def start_job(self, job: DownloadJobBase):
        """Start the job putting it into ENQUEUED state."""
        pass

    @abstractmethod
    def pause_job(self, job: DownloadJobBase):
        """Pause the job, putting it into PAUSED state."""
        pass

    @abstractmethod
    def cancel_job(self, job: DownloadJobBase):
        """Cancel the job, clearing partial downloads and putting it into ERROR state."""
        pass

    @abstractmethod
    def change_priority(self, job: DownloadJobBase, delta: int):
        """
        Change the job's priority.

        :param job: Job to  apply change to
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


class DownloadQueueService(DownloadQueueServiceBase):
    """Multithreaded queue for downloading models via URL or repo_id."""

    _event_bus: EventServiceBase
    _queue: DownloadQueue

    def __init__(self, event_bus: EventServiceBase, **kwargs):
        """
        Initialize new DownloadQueueService object.

        :param event_bus: EventServiceBase object for reporting progress.
        :param **kwargs: Any of the arguments taken by invokeai.backend.model_manager.download.DownloadQueue.
        e.g. `max_parallel_dl`.
        """
        self._event_bus = event_bus
        self._queue = DownloadQueue(**kwargs)

    def create_download_job(
        self,
        source: Union[str, Path, AnyHttpUrl],
        destdir: Path,
        filename: Optional[Path] = None,
        start: bool = True,
        access_token: Optional[str] = None,
        event_handlers: Optional[List[DownloadEventHandler]] = None,
    ) -> DownloadJobBase:  # noqa D102
        event_handlers = event_handlers or []
        if self._event_bus:
            event_handlers = [*event_handlers, self._event_bus.emit_model_event]
        return self._queue.create_download_job(
            source=source,
            destdir=destdir,
            filename=filename,
            start=start,
            access_token=access_token,
            event_handlers=event_handlers,
        )

    def list_jobs(self) -> List[DownloadJobBase]:  # noqa D102
        return self._queue.list_jobs()

    def id_to_job(self, id: int) -> DownloadJobBase:  # noqa D102
        return self._queue.id_to_job(id)

    def start_all_jobs(self):  # noqa D102
        return self._queue.start_all_jobs()

    def pause_all_jobs(self):  # noqa D102
        return self._queue.pause_all_jobs()

    def cancel_all_jobs(self):  # noqa D102
        return self._queue.cancel_all_jobs()

    def start_job(self, job: DownloadJobBase):  # noqa D102
        return self._queue.start_job(job)

    def pause_job(self, job: DownloadJobBase):  # noqa D102
        return self._queue.pause_job(job)

    def cancel_job(self, job: DownloadJobBase):  # noqa D102
        return self._queue.cancel_job(job)

    def change_priority(self, job: DownloadJobBase, delta: int):  # noqa D102
        return self._queue.change_priority(job, delta)

    def join(self):  # noqa D102
        return self._queue.join()
