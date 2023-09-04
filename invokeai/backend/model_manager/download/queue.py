# Copyright (c) 2023, Lincoln D. Stein
"""Implementation of multithreaded download queue for invokeai."""

import re
import os
import requests
import threading

from pathlib import Path
from typing import Dict, Optional, Set, List

from pydantic import Field, validator, ValidationError
from pydantic.networks import AnyHttpUrl
from queue import PriorityQueue

from invokeai.backend.util.logging import InvokeAILogger
from .base import (
    DownloadQueueBase,
    DownloadJobStatus,
    DownloadEventHandler,
    UnknownJobIDException,
    CancelledJobException,
    DownloadJobBase
)

# marker that the queue is done and that thread should exit
STOP_JOB = DownloadJobBase(
    id=-99,
    priority=-99,
    source='dummy',
    destination='/')


class DownloadJobURL(DownloadJobBase):
    """Job declaration for downloading individual URLs."""

    source: AnyHttpUrl = Field(description="URL to download")


class DownloadJobRepoID(DownloadJobBase):
    """Download repo ids."""

    variant: Optional[str] = Field(description="Variant, such as 'fp16', to download")

    @validator('source')
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if not re.match(r'^\w+/\w+$', v):
            raise ValidationError(f'{v} invalid repo_id')
        return v


class DownloadQueue(DownloadQueueBase):
    """Class for queued download of models."""

    _jobs: Dict[int, DownloadJobBase]
    _worker_pool: Set[threading.Thread]
    _queue: PriorityQueue
    _lock: threading.Lock
    _logger: InvokeAILogger
    _event_handler: Optional[DownloadEventHandler]
    _next_job_id: int = 0

    def __init__(self,
                 max_parallel_dl: int = 5,
                 event_handler: Optional[DownloadEventHandler] = None,
                 ):
        """
        Initialize DownloadQueue.

        :param max_parallel_dl: Number of simultaneous downloads allowed [5].
        :param event_handler: Optional callable that will be called each time a job status changes.
        """
        print('IN __INIT__')
        self._jobs = dict()
        self._next_job_id = 0
        self._queue = PriorityQueue()
        self._worker_pool = set()
        self._lock = threading.RLock()
        self._logger = InvokeAILogger.getLogger()
        self._event_handler = event_handler

        self._start_workers(max_parallel_dl)

    def create_download_job(
            self,
            source: str,
            destdir: Path,
            filename: Optional[Path] = None,
            start: bool = True,
            variant: Optional[str] = None,
            access_token: Optional[str] = None,
            event_handler: Optional[DownloadEventHandler] = None,
    ) -> int:
        if re.match(r'^\w+/\w+$', source):
            cls = DownloadJobRepoID
            kwargs = dict(variant=variant)
        else:
            cls = DownloadJobURL
            kwargs = dict()
        try:
            self._lock.acquire()
            id = self._next_job_id
            self._jobs[id] = cls(
                id=id,
                source=source,
                destination=Path(destdir) / (filename or "."),
                access_token=access_token,
                event_handler=(event_handler or self._event_handler),
                **kwargs,
            )
            self._next_job_id += 1
            job = self._jobs[id]
        finally:
            self._lock.release()
        if start:
            self.start_job(id)
        return job.id

    def release(self):
        """Signal our threads to exit when queue done."""
        for thread in self._worker_pool:
            if thread.is_alive():
                self._queue.put(STOP_JOB)

    def join(self):
        self._queue.join()

    def list_jobs(self) -> List[DownloadJobBase]:
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

    def cancel_job(self, job: DownloadJobBase):
        try:
            self._lock.acquire()
            job.status = DownloadJobStatus.ERROR
            job.error = CancelledJobException(f"Job {job.id} cancelled at caller's request")
            self._update_job_status
            del self._jobs[job.id]
        finally:
            self._lock.release()

    def id_to_job(self, id: int) -> DownloadJobBase:
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
            worker = threading.Thread(target=self._download_next_item, daemon=True)
            worker.start()
            self._worker_pool.add(worker)

    def _download_next_item(self):
        """Worker thread gets next job on priority queue."""
        while True:
            job = self._queue.get()
            if job == STOP_JOB:  # marker that queue is done
                print(f'DEBUG: thread {threading.current_thread().native_id} exiting')
                break
            if job.status == DownloadJobStatus.ENQUEUED:    # Don't do anything for cancelled or errored jobs
                if isinstance(job, DownloadJobURL):
                    self._download_with_resume(job)
                elif isinstance(job, DownloadJobRepoID):
                    raise self._download_repoid(job)
                else:
                    raise NotImplementedError(f"Don't know what to do with this job: {job}")
            self._queue.task_done()

    def _download_with_resume(self, job: DownloadJobBase):
        """Do the actual download."""
        header = {"Authorization": f"Bearer {job.access_token}"} if job.access_token else {}
        open_mode = "wb"
        exist_size = 0

        resp = requests.get(job.source, header, stream=True)
        content_length = int(resp.headers.get("content-length", 0))
        job.total_bytes = content_length

        if job.destination.is_dir():
            try:
                file_name = re.search('filename="(.+)"', resp.headers.get("Content-Disposition")).group(1)
            except AttributeError:
                file_name = os.path.basename(job.source)
            job.destination = job.destination / file_name
            dest = job.destination
        else:
            dest = job.destination
            dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            job.bytes = dest.stat().st_size
            header["Range"] = f"bytes={job.bytes}-"
            open_mode = "ab"
            resp = requests.get(job.source, headers=header, stream=True)  # new request with range

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
                           job: DownloadJobBase,
                           new_status: Optional[DownloadJobStatus] = None
                           ):
        """Optionally change the job status and send an event indicating a change of state."""
        if new_status:
            job.status = new_status
        self._logger.debug(f"Status update for download job {job.id}: {job}")
        if job.event_handler:
            job.event_handler(job)

    def _download_repoid(self, job: DownloadJobBase):
        """Download a job that holds a huggingface repoid."""
        repo_id = job.source
        variant = job.variant
        
