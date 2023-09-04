# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Model download service.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from .events import EventServicesBase
from invokeai.backend.model_manager.download import DownloadQueue, DownloadJobBase, DownloadEventHandler

class DownloadQueueServiceBase(ABC):
    """Multithreaded queue for downloading models via URL or repo_id."""

    @abstractmethod
    def create_download_job(
            self,
            source: str,
            destdir: Path,
            filename: Optional[Path] = None,
            start: bool = True,
            access_token: Optional[str] = None,
    ) -> int:
        """
        Create a download job.

        :param source: Source of the download - URL or repo_id
        :param destdir: Directory to download into.
        :param filename: Optional name of file, if not provided
        will use the content-disposition field to assign the name.
        :param start: Immediately start job [True]
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


class DownloadQueueService(DownloadQueueServiceBase):
    """Multithreaded queue for downloading models via URL or repo_id."""

    _event_bus: EventServicesBase
    _queue: DownloadQueue

    def __init__(self, event_bus: EventServicesBase, **kwargs):
        """
        Initialize new DownloadQueueService object.

        :param event_bus: EventServicesBase object for reporting progress.
        :param **kwargs: Any of the arguments taken by invokeai.backend.model_manager.download.DownloadQueue.
        e.g. `max_parallel_dl`.
        """
        self._event_bus = event_bus
        self._queue = DownloadQueue(event_handler=self._forward_event)

    def _forward_event(self, job: DownloadJobBase):
        if self._event_bus:
            self._event_bus.emit_model_download_event(job)

    def _wrap_handler(self, event_handler: DownloadEventHandler) -> DownloadEventHandler:
        def __wrapper(job: DownloadJobBase):
            self._forward_events(job)
            event_handler(job)
        return __wrapper

    def create_download_job(
        self,
        source: str,
        destdir: Path,
        filename: Optional[Path] = None,
        start: bool = True,
        access_token: Optional[str] = None,
        event_handler: Optional[DownloadEventHandler] = None,
    ) -> int:
        if event_handler:
            event_handler = self._wrap_handler(event_handler)
        return self._queue.create_download_job(
            source, destdir, filename, start, access_token,
            event_handler=event_handler
        )

    def list_jobs(self) -> List[DownloadJobBase]:
        return self._queue.list_jobs()

    def id_to_job(self, id: int) -> DownloadJobBase:
        return self._queue.id_to_job(id)

    def start_all_jobs(self):
        return self._queue.start_all_jobs()

    def pause_all_jobs(self):
        return self._queue.pause_all_jobs()

    def cancel_all_jobs(self):
        return self._queue.cancel_all_jobs()

    def start_job(self, id: int):
        return self._queue.start_job(id)

    def pause_job(self, id: int):
        return self._queue.pause_job(id)

    def cancel_job(self, id: int):
        return self._queue.cancel_job(id)

    def change_priority(self, id: int, delta: int):
        return self._queue.change_priority(id, delta)

    def join(self):
        return self._queue.join()
