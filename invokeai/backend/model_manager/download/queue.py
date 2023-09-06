# Copyright (c) 2023, Lincoln D. Stein
"""Implementation of multithreaded download queue for invokeai."""

import re
import os
import requests
import threading
import time

from pathlib import Path
from typing import Dict, Optional, Set, List, Tuple

from pydantic import Field, validator, ValidationError
from pydantic.networks import AnyHttpUrl
from queue import PriorityQueue

from huggingface_hub import HfApi, hf_hub_url

from invokeai.backend.util.logging import InvokeAILogger
from .base import (
    DownloadQueueBase,
    DownloadJobStatus,
    DownloadEventHandler,
    UnknownJobIDException,
    CancelledJobException,
    DownloadJobBase,
)

# marker that the queue is done and that thread should exit
STOP_JOB = DownloadJobBase(id=-99, priority=-99, source="dummy", destination="/")


class DownloadJobURL(DownloadJobBase):
    """Job declaration for downloading individual URLs."""

    source: AnyHttpUrl = Field(description="URL to download")


class DownloadJobRepoID(DownloadJobBase):
    """Download repo ids."""

    variant: Optional[str] = Field(description="Variant, such as 'fp16', to download")

    @validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if not re.match(r"^[\w-]+/[\w-]+$", v):
            raise ValidationError(f"{v} invalid repo_id")
        return v


class DownloadQueue(DownloadQueueBase):
    """Class for queued download of models."""

    _jobs: Dict[int, DownloadJobBase]
    _worker_pool: Set[threading.Thread]
    _queue: PriorityQueue
    _lock: threading.Lock
    _logger: InvokeAILogger
    _event_handlers: Optional[List[DownloadEventHandler]]
    _next_job_id: int = 0
    _sequence: int = 0  # This is for debugging and used to tag jobs in dequeueing order
    _requests: requests.sessions.Session

    def __init__(
        self,
        max_parallel_dl: int = 5,
        event_handlers: Optional[List[DownloadEventHandler]] = None,
        requests_session: Optional[requests.sessions.Session] = None,
    ):
        """
        Initialize DownloadQueue.

        :param max_parallel_dl: Number of simultaneous downloads allowed [5].
        :param event_handler: Optional callable that will be called each time a job status changes.
        :param requests_session: Optional requests.sessions.Session object, for unit tests.
        """
        self._jobs = dict()
        self._next_job_id = 0
        self._queue = PriorityQueue()
        self._worker_pool = set()
        self._lock = threading.RLock()
        self._logger = InvokeAILogger.getLogger()
        self._event_handlers = event_handlers
        self._requests = requests_session or requests.Session()

        self._start_workers(max_parallel_dl)

    def create_download_job(
        self,
        source: str,
        destdir: Path,
        filename: Optional[Path] = None,
        start: bool = True,
        variant: Optional[str] = None,
        access_token: Optional[str] = None,
        event_handlers: Optional[List[DownloadEventHandler]] = None,
    ) -> int:
        """Create a download job and return its ID."""
        if re.match(r"^[\w-]+/[\w-]+$", source):
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
                event_handlers=(event_handlers or self._event_handlers),
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
        """Wait for all jobs to complete."""
        self._queue.join()

    def list_jobs(self) -> List[DownloadJobBase]:
        """List all the jobs."""
        return self._jobs.values()

    def change_priority(self, id: int, delta: int):
        """Change the priority of a job. Smaller priorities run first."""
        try:
            self._lock.acquire()
            job = self._jobs[id]
            job.priority += delta
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def cancel_job(self, id: str):
        """
        Cancel the indicated job.

        If it is running it will be stopped.
        job.error will be set to CancelledJobException.
        """
        try:
            self._lock.acquire()
            job = self._jobs[id]
            job.status = DownloadJobStatus.ERROR
            job.error = CancelledJobException(f"Job {job.id} cancelled at caller's request")
            self._update_job_status
            del self._jobs[job.id]
        finally:
            self._lock.release()

    def id_to_job(self, id: int) -> DownloadJobBase:
        """Translate a job ID into a DownloadJobBase object."""
        try:
            return self._jobs[id]
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp

    def start_job(self, id: int):
        """Enqueue (start) the indicated job."""
        try:
            job = self._jobs[id]
            self._update_job_status(job, DownloadJobStatus.ENQUEUED)
            self._queue.put(job)
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp

    def pause_job(self, id: int):
        """
        Pause (dequeue) the indicated job.

        In theory the job can be restarted and the download will pick up
        from where it left off.
        """
        try:
            self._lock.acquire()
            job = self._jobs[id]
            self._update_job_status(job, DownloadJobStatus.PAUSED)
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def start_all_jobs(self):
        """Start (enqueue) all jobs that are idle or paused."""
        try:
            self._lock.acquire()
            for id, job in self._jobs.items():
                if job.status in [DownloadJobStatus.IDLE or DownloadJobStatus.PAUSED]:
                    self.start_job(id)
        finally:
            self._lock.release()

    def pause_all_jobs(self, id: int):
        """Pause all running jobs."""
        try:
            self._lock.acquire()
            for id, job in self._jobs.items():
                if job.stats == DownloadJobStatus.RUNNING:
                    self.pause_job(id)
        finally:
            self._lock.release()

    def cancel_all_jobs(self):
        """Cancel all running jobs."""
        try:
            self._lock.acquire()
            for id, job in self._jobs.items():
                if job.status in [DownloadJobStatus.RUNNING, DownloadJobStatus.PAUSED, DownloadJobStatus.ENQUEUED]:
                    self.cancel_job(id)
        finally:
            self._lock.release()

    def _start_workers(self, max_workers: int):
        """Start the requested number of worker threads."""
        for i in range(0, max_workers):
            worker = threading.Thread(target=self._download_next_item, daemon=True)
            worker.start()
            self._worker_pool.add(worker)

    def _download_next_item(self):
        """Worker thread gets next job on priority queue."""
        while True:
            job = self._queue.get()

            try:
                self._lock.acquire()
                job.job_sequence = self._sequence
                self._sequence += 1
            finally:
                self._lock.release()

            if job == STOP_JOB:  # marker that queue is done
                break
            if job.status == DownloadJobStatus.ENQUEUED:  # Don't do anything for cancelled or errored jobs
                if isinstance(job, DownloadJobURL):
                    self._download_with_resume(job)
                elif isinstance(job, DownloadJobRepoID):
                    self._download_repoid(job)
                else:
                    raise NotImplementedError(f"Don't know what to do with this job: {job}")
            self._queue.task_done()

    def _download_with_resume(self, job: DownloadJobBase):
        """Do the actual download."""
        header = {"Authorization": f"Bearer {job.access_token}"} if job.access_token else {}
        open_mode = "wb"
        exist_size = 0

        resp = self._requests.get(job.source, headers=header, stream=True)
        content_length = int(resp.headers.get("content-length", 0))
        job.total_bytes = content_length

        if job.destination.is_dir():
            try:
                file_name = re.search('filename="(.+)"', resp.headers.get("Content-Disposition", "bug-noname")).group(1)
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
            resp = self._requests.get(job.source, headers=header, stream=True)  # new request with range

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
                    if job.status != DownloadJobStatus.RUNNING:  # cancelled, paused or errored
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

    def _update_job_status(self, job: DownloadJobBase, new_status: Optional[DownloadJobStatus] = None):
        """Optionally change the job status and send an event indicating a change of state."""
        if new_status:
            job.status = new_status
        self._logger.debug(f"Status update for download job {job.id}: {job}")
        if new_status == DownloadJobStatus.RUNNING and not job.job_started:
            job.job_started = time.time()
        elif new_status in [DownloadJobStatus.COMPLETED, DownloadJobStatus.ERROR]:
            job.job_ended = time.time()
        if job.event_handlers:
            for handler in job.event_handlers:
                handler(job)

    def _download_repoid(self, job: DownloadJobBase):
        """Download a job that holds a huggingface repoid."""

        def subdownload_event(subjob: DownloadJobBase):
            if subjob.status == DownloadJobStatus.RUNNING:
                bytes_downloaded[subjob.id] = subjob.bytes
                job.bytes = sum(bytes_downloaded.values())
                self._update_job_status(job, DownloadJobStatus.RUNNING)

            elif subjob.status == DownloadJobStatus.ERROR:
                job.error = subjob.error
                subqueue.cancel_all_jobs()
                self._update_job_status(job, DownloadJobStatus.ERROR)

        subqueue = self.__class__(event_handlers=[subdownload_event])
        try:
            repo_id = job.source
            variant = job.variant
            urls_to_download = self._get_repo_urls(repo_id, variant)
            job.destination = job.destination / Path(repo_id).name
            bytes_downloaded = dict()

            for url, subdir, file, size in urls_to_download:
                job.total_bytes += size
                subqueue.create_download_job(
                    source=url,
                    destdir=job.destination / subdir,
                    filename=file,
                    variant=variant,
                    access_token=job.access_token,
                )
        except Exception as excp:
            job.status = DownloadJobStatus.ERROR
            job.error = excp
            self._logger.error(job.error)
        finally:
            subqueue.join()
            if not job.status == DownloadJobStatus.ERROR:
                self._update_job_status(job, DownloadJobStatus.COMPLETED)
            subqueue.release()  # get rid of the subqueue

    # def _get_download_size(self, url: AnyHttpUrl) -> int:
    #     resp = self._requests.get(url, stream=True)
    #     resp.raise_for_status()
    #     return int(resp.headers.get("content-length", 0))

    def _get_repo_urls(self, repo_id: str, variant: Optional[str] = None) -> List[Tuple[AnyHttpUrl, Path, Path]]:
        """Given a repo_id and an optional variant, return list of URLs to download to get the model."""
        model_info = HfApi().model_info(repo_id=repo_id, files_metadata=True)
        sibs = model_info.siblings
        paths = [x.rfilename for x in sibs]
        sizes = {x.rfilename: x.size for x in sibs}
        if "model_index.json" in paths:
            url = hf_hub_url(repo_id, filename="model_index.json")
            resp = self._requests.get(url)
            resp.raise_for_status()  # will raise an HTTPError on non-200 status
            submodels = resp.json()
            paths = [x for x in paths if Path(x).parent.as_posix() in submodels]
            paths.insert(0, "model_index.json")
        return [
            (hf_hub_url(repo_id, filename=x.as_posix()), x.parent or Path("."), x.name, sizes[x.as_posix()])
            for x in self._select_variants(paths, variant)
        ]

    def _select_variants(self, paths: List[str], variant: Optional[str] = None) -> Set[Path]:
        """Select the proper variant files from a list of HuggingFace repo_id paths."""
        result = set()
        basenames = dict()
        for p in paths:
            path = Path(p)
            if path.suffix in [".bin", ".safetensors", ".pt"]:
                parent = path.parent
                suffixes = path.suffixes
                if len(suffixes) == 2:
                    file_variant, suffix = suffixes
                    basename = parent / Path(path.stem).stem
                else:
                    file_variant = None
                    suffix = suffixes[0]
                    basename = parent / path.stem

                if previous := basenames.get(basename):
                    if previous.suffix != ".safetensors" and suffix == ".safetensors":
                        basenames[basename] = path
                    if file_variant == f".{variant}":
                        basenames[basename] = path
                    elif not variant and not file_variant:
                        basenames[basename] = path
                else:
                    basenames[basename] = path
            else:
                result.add(path)
        for v in basenames.values():
            result.add(v)
        return result
