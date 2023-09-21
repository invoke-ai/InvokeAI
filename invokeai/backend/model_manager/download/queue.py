# Copyright (c) 2023, Lincoln D. Stein
"""Implementation of multithreaded download queue for invokeai."""

import os
import re
import shutil
import threading
import time
import traceback
from json import JSONDecodeError
from pathlib import Path
from queue import PriorityQueue
from typing import Dict, List, Optional, Set, Tuple, Union

import requests
from huggingface_hub import HfApi, hf_hub_url
from pydantic import Field, ValidationError, validator
from pydantic.networks import AnyHttpUrl
from requests import HTTPError

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util.logging import InvokeAILogger

from ..storage import DuplicateModelException
from . import HTTP_RE, REPO_ID_RE
from .base import (
    DownloadEventHandler,
    DownloadJobBase,
    DownloadJobStatus,
    DownloadQueueBase,
    ModelSourceMetadata,
    UnknownJobIDException,
)

# Maximum number of bytes to download during each call to requests.iter_content()
DOWNLOAD_CHUNK_SIZE = 100000

# marker that the queue is done and that thread should exit
STOP_JOB = DownloadJobBase(id=-99, priority=-99, source="dummy", destination="/")

# endpoint for civitai get-model API
CIVITAI_MODEL_DOWNLOAD = "https://civitai.com/api/download/models/"
CIVITAI_MODEL_PAGE = "https://civitai.com/models/"
CIVITAI_MODELS_ENDPOINT = "https://civitai.com/api/v1/models/"
CIVITAI_VERSIONS_ENDPOINT = "https://civitai.com/api/v1/model-versions/"


class DownloadJobURL(DownloadJobBase):
    """Job declaration for downloading individual URLs."""

    source: AnyHttpUrl = Field(description="URL to download")


class DownloadJobRepoID(DownloadJobBase):
    """Download repo ids."""

    variant: Optional[str] = Field(description="Variant, such as 'fp16', to download")

    @validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        if not re.match(REPO_ID_RE, v):
            raise ValidationError(f"{v} invalid repo_id")
        return v


class DownloadJobPath(DownloadJobBase):
    """Handle file paths."""

    source: Path = Field(description="Path to a file or directory to install")


class DownloadQueue(DownloadQueueBase):
    """Class for queued download of models."""

    _jobs: Dict[int, DownloadJobBase]
    _worker_pool: Set[threading.Thread]
    _queue: PriorityQueue
    _lock: threading.Lock
    _logger: InvokeAILogger
    _event_handlers: List[DownloadEventHandler] = Field(default_factory=list)
    _next_job_id: int = 0
    _sequence: int = 0  # This is for debugging and used to tag jobs in dequeueing order
    _requests: requests.sessions.Session
    _quiet: bool = False

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
        """
        self._jobs = dict()
        self._next_job_id = 0
        self._queue = PriorityQueue()
        self._worker_pool = set()
        self._lock = threading.RLock()
        self._logger = InvokeAILogger.getLogger(config=config)
        self._event_handlers = event_handlers
        self._requests = requests_session or requests.Session()
        self._quiet = quiet

        self._start_workers(max_parallel_dl)

    def create_download_job(
        self,
        source: Union[str, Path, AnyHttpUrl],
        destdir: Path,
        filename: Optional[Path] = None,
        start: bool = True,
        variant: Optional[str] = None,
        access_token: Optional[str] = None,
        event_handlers: Optional[List[DownloadEventHandler]] = None,
    ) -> DownloadJobBase:
        """
        Create a download job and return its ID.
        """
        kwargs = dict()

        if Path(source).exists():
            cls = DownloadJobPath
        elif re.match(REPO_ID_RE, str(source)):
            cls = DownloadJobRepoID
            kwargs = dict(variant=variant)
        elif re.match(HTTP_RE, str(source)):
            cls = DownloadJobURL
        else:
            raise NotImplementedError(f"Don't know what to do with this type of source: {source}")

        job = cls(
            source=source,
            destination=Path(destdir) / (filename or "."),
            access_token=access_token,
            event_handlers=(event_handlers or self._event_handlers),
            **kwargs,
        )

        return self.submit_download_job(job, start)

    def submit_download_job(
        self,
        job: DownloadJobBase,
        start: bool = True,
    ):
        """Submit a job."""
        # add the queue's handlers
        for handler in self._event_handlers:
            job.add_event_handler(handler)
        try:
            self._lock.acquire()
            job.id = self._next_job_id
            self._jobs[job.id] = job
            self._next_job_id += 1
        finally:
            self._lock.release()
        if start:
            self.start_job(job)
        return job

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

    def change_priority(self, job: DownloadJobBase, delta: int):
        """Change the priority of a job. Smaller priorities run first."""
        try:
            self._lock.acquire()
            assert isinstance(self._jobs[job.id], DownloadJobBase)
            job.priority += delta
        except (AssertionError, KeyError) as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def cancel_job(self, job: DownloadJobBase, preserve_partial: bool = False):
        """
        Cancel the indicated job.

        If it is running it will be stopped.
        job.status will be set to DownloadJobStatus.CANCELLED
        """
        try:
            self._lock.acquire()
            assert isinstance(self._jobs[job.id], DownloadJobBase)
            job.preserve_partial_downloads = preserve_partial
            self._update_job_status(job, DownloadJobStatus.CANCELLED)
            if job.subqueue:
                job.subqueue.cancel_all_jobs(preserve_partial=preserve_partial)
        except (AssertionError, KeyError) as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def id_to_job(self, id: int) -> DownloadJobBase:
        """Translate a job ID into a DownloadJobBase object."""
        try:
            return self._jobs[id]
        except KeyError as excp:
            raise UnknownJobIDException("Unrecognized job") from excp

    def start_job(self, job: DownloadJobBase):
        """Enqueue (start) the indicated job."""
        try:
            assert isinstance(self._jobs[job.id], DownloadJobBase)
            self._update_job_status(job, DownloadJobStatus.ENQUEUED)
            self._queue.put(job)
        except (AssertionError, KeyError) as excp:
            raise UnknownJobIDException("Unrecognized job") from excp

    def pause_job(self, job: DownloadJobBase):
        """
        Pause (dequeue) the indicated job.

        The job can be restarted with start_job() and the download will pick up
        from where it left off.
        """
        try:
            self._lock.acquire()
            assert isinstance(self._jobs[job.id], DownloadJobBase)
            self._update_job_status(job, DownloadJobStatus.PAUSED)
            if job.subqueue:
                job.subqueue.cancel_all_jobs(preserve_partial=True)
                job.subqueue.release()
        except (AssertionError, KeyError) as excp:
            raise UnknownJobIDException("Unrecognized job") from excp
        finally:
            self._lock.release()

    def start_all_jobs(self):
        """Start (enqueue) all jobs that are idle or paused."""
        try:
            self._lock.acquire()
            for job in self._jobs.values():
                if job.status in [DownloadJobStatus.IDLE, DownloadJobStatus.PAUSED]:
                    self.start_job(job)
        finally:
            self._lock.release()

    def pause_all_jobs(self):
        """Pause all running jobs."""
        try:
            self._lock.acquire()
            for job in self._jobs.values():
                if job.status == DownloadJobStatus.RUNNING:
                    self.pause_job(job)
        finally:
            self._lock.release()

    def cancel_all_jobs(self, preserve_partial: bool = False):
        """Cancel all running jobs."""
        try:
            self._lock.acquire()
            for job in self._jobs.values():
                if not self._in_terminal_state(job):
                    self.cancel_job(job, preserve_partial)
        finally:
            self._lock.release()

    def _in_terminal_state(self, job: DownloadJobBase):
        return job.status in [
            DownloadJobStatus.COMPLETED,
            DownloadJobStatus.ERROR,
            DownloadJobStatus.CANCELLED,
        ]

    def _start_workers(self, max_workers: int):
        """Start the requested number of worker threads."""
        for i in range(0, max_workers):
            worker = threading.Thread(target=self._download_next_item, daemon=True)
            worker.start()
            self._worker_pool.add(worker)

    def _download_next_item(self):
        """Worker thread gets next job on priority queue."""
        done = False
        while not done:
            job = self._queue.get()

            try:  # this is for debugging priority
                self._lock.acquire()
                job.job_sequence = self._sequence
                self._sequence += 1
            finally:
                self._lock.release()

            if job == STOP_JOB:  # marker that queue is done
                done = True

            if job.status == DownloadJobStatus.ENQUEUED:  # Don't do anything for non-enqueued jobs (shouldn't happen)
                # There should be a better way to dispatch on the job type
                if not self._quiet:
                    self._logger.info(f"{job.source}: Downloading to {job.destination}")
                if isinstance(job, DownloadJobURL):
                    self._download_with_resume(job)
                elif isinstance(job, DownloadJobRepoID):
                    self._download_repoid(job)
                elif isinstance(job, DownloadJobPath):
                    self._download_path(job)
                else:
                    raise NotImplementedError(f"Don't know what to do with this job: {job}")

            if job.status == DownloadJobStatus.CANCELLED:
                self._cleanup_cancelled_job(job)

            if self._in_terminal_state(job):
                del self._jobs[job.id]

            self._queue.task_done()

    def _get_metadata_and_url(self, job: DownloadJobBase) -> AnyHttpUrl:
        """
        Fetch metadata from certain well-known URLs.

        The metadata will be stashed in job.metadata, if found
        Return the download URL.
        """
        metadata = job.metadata
        url = job.source
        metadata_url = url
        try:
            # a Civitai download URL
            if match := re.match(CIVITAI_MODEL_DOWNLOAD + r"(\d+)", metadata_url):
                version = match.group(1)
                resp = self._requests.get(CIVITAI_VERSIONS_ENDPOINT + version).json()
                print(f"DEBUG: resp={resp}")
                metadata.thumbnail_url = metadata.thumbnail_url or resp["images"][0]["url"]
                metadata.description = metadata.description or (
                    f"Trigger terms: {(', ').join(resp['trainedWords'])}"
                    if resp["trainedWords"]
                    else resp["description"]
                )
                metadata_url = CIVITAI_MODEL_PAGE + str(resp["modelId"])

            # a Civitai model page
            if match := re.match(CIVITAI_MODEL_PAGE + r"(\d+)", metadata_url):
                model = match.group(1)
                resp = self._requests.get(CIVITAI_MODELS_ENDPOINT + str(model)).json()

                # note that we munge the URL here to get the download URL of the first model
                url = resp["modelVersions"][0]["downloadUrl"]

                metadata.author = metadata.author or resp["creator"]["username"]
                metadata.tags = metadata.tags or resp["tags"]
                metadata.thumbnail_url = metadata.thumbnail_url or resp["modelVersions"][0]["images"][0]["url"]
                metadata.license = (
                    metadata.license
                    or f"allowCommercialUse={resp['allowCommercialUse']}; allowDerivatives={resp['allowDerivatives']}; allowNoCredit={resp['allowNoCredit']}"
                )
        except (HTTPError, KeyError, TypeError, JSONDecodeError) as excp:
            self._logger.warn(excp)

        # return the download url
        return url

    def _download_with_resume(self, job: DownloadJobBase):
        """Do the actual download."""
        try:
            url = self._get_metadata_and_url(job)

            header = {"Authorization": f"Bearer {job.access_token}"} if job.access_token else {}
            open_mode = "wb"
            exist_size = 0

            resp = self._requests.get(url, headers=header, stream=True)
            content_length = int(resp.headers.get("content-length", 0))
            job.total_bytes = content_length

            if job.destination.is_dir():
                try:
                    file_name = re.search('filename="(.+)"', resp.headers["Content-Disposition"]).group(1)
                    self._validate_filename(
                        job.destination, file_name
                    )  # will raise a ValueError exception if file_name is suspicious
                except ValueError:
                    self._logger.warning(
                        f"Invalid filename '{file_name}' returned by source {url}, using last component of URL instead"
                    )
                    file_name = os.path.basename(url)
                except KeyError:
                    file_name = os.path.basename(url)
                job.destination = job.destination / file_name
                dest = job.destination
            else:
                dest = job.destination
                dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists():
                job.bytes = dest.stat().st_size
                header["Range"] = f"bytes={job.bytes}-"
                open_mode = "ab"
                resp = self._requests.get(url, headers=header, stream=True)  # new request with range

            if exist_size > content_length:
                self._logger.warning("corrupt existing file found. re-downloading")
                os.remove(dest)
                exist_size = 0

            if resp.status_code == 416 or (content_length > 0 and exist_size == content_length):
                self._logger.warning(f"{dest}: complete file found. Skipping.")
                self._update_job_status(job, DownloadJobStatus.COMPLETED)
                return

            if resp.status_code == 206 or exist_size > 0:
                self._logger.warning(f"{dest}: partial file found. Resuming")
            elif resp.status_code != 200:
                raise HTTPError(resp.reason)
            else:
                self._logger.debug(f"{job.source}: Downloading {job.destination}")

            report_delta = job.total_bytes / 100  # report every 1% change
            last_report_bytes = 0

            self._update_job_status(job, DownloadJobStatus.RUNNING)
            with open(dest, open_mode) as file:
                for data in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if job.status != DownloadJobStatus.RUNNING:  # cancelled, paused or errored
                        return
                    job.bytes += file.write(data)
                    if job.bytes - last_report_bytes >= report_delta:
                        last_report_bytes = job.bytes
                        self._update_job_status(job)

            self._update_job_status(job, DownloadJobStatus.COMPLETED)
        except KeyboardInterrupt as excp:
            raise excp
        except DuplicateModelException as excp:
            self._logger.error(f"A model with the same hash as {dest} is already installed.")
            job.error = excp
            self._update_job_status(job, DownloadJobStatus.ERROR)
        except Exception as excp:
            self._logger.error(f"An error occurred while downloading/installing {dest}: {str(excp)}")
            print(traceback.format_exc())
            job.error = excp
            self._update_job_status(job, DownloadJobStatus.ERROR)

    def _validate_filename(self, directory: str, filename: str):
        if "/" in filename:
            raise ValueError
        if filename.startswith(".."):
            raise ValueError
        if len(filename) > os.pathconf(directory, "PC_NAME_MAX"):
            raise ValueError
        if len(os.path.join(directory, filename)) > os.pathconf(directory, "PC_PATH_MAX"):
            raise ValueError

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
                try:
                    handler(job)
                except KeyboardInterrupt as excp:
                    raise excp
                except Exception as excp:
                    job.error = excp
                    self._update_job_status(job, DownloadJobStatus.ERROR)

    def _download_repoid(self, job: DownloadJobBase):
        """Download a job that holds a huggingface repoid."""

        def subdownload_event(subjob: DownloadJobBase):
            if subjob.status == DownloadJobStatus.RUNNING:
                bytes_downloaded[subjob.id] = subjob.bytes
                job.bytes = sum(bytes_downloaded.values())
                self._update_job_status(job, DownloadJobStatus.RUNNING)
                return

            if subjob.status == DownloadJobStatus.ERROR:
                job.error = subjob.error
                subjob.subqueue.cancel_all_jobs()
                self._update_job_status(job, DownloadJobStatus.ERROR)
                return

            if subjob.status == DownloadJobStatus.COMPLETED:
                bytes_downloaded[subjob.id] = subjob.bytes
                job.bytes = sum(bytes_downloaded.values())
                self._update_job_status(job, DownloadJobStatus.RUNNING)
                return

        job.subqueue = self.__class__(
            event_handlers=[subdownload_event],
            requests_session=self._requests,
            quiet=True,
        )
        try:
            repo_id = job.source
            variant = job.variant
            if not job.metadata:
                job.metadata = ModelSourceMetadata()
            urls_to_download = self._get_repo_info(repo_id, variant=variant, metadata=job.metadata)
            if job.destination.name != Path(repo_id).name:
                job.destination = job.destination / Path(repo_id).name
            bytes_downloaded = dict()
            job.total_bytes = 0

            for url, subdir, file, size in urls_to_download:
                job.total_bytes += size
                job.subqueue.create_download_job(
                    source=url,
                    destdir=job.destination / subdir,
                    filename=file,
                    variant=variant,
                    access_token=job.access_token,
                )
        except KeyboardInterrupt as excp:
            raise excp
        except Exception as excp:
            job.error = excp
            self._update_job_status(job, DownloadJobStatus.ERROR)
            self._logger.error(job.error)
        finally:
            job.subqueue.join()
            if job.status == DownloadJobStatus.RUNNING:
                self._update_job_status(job, DownloadJobStatus.COMPLETED)
            job.subqueue.release()  # get rid of the subqueue

    def _get_repo_info(
        self,
        repo_id: str,
        metadata: ModelSourceMetadata,
        variant: Optional[str] = None,
    ) -> Tuple[List[Tuple[AnyHttpUrl, Path, Path]], ModelSourceMetadata]:
        """
        Given a repo_id and an optional variant, return list of URLs to download to get the model.
        The metadata field will be updated with model metadata from HuggingFace.

        Known variants currently are:
        1. onnx
        2. openvino
        3. fp16
        4. None (usually returns fp32 model)
        """
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
        urls = [
            (hf_hub_url(repo_id, filename=x.as_posix()), x.parent or Path("."), x.name, sizes[x.as_posix()])
            for x in self._select_variants(paths, variant)
        ]
        if hasattr(model_info, "cardData"):
            metadata.license = metadata.license or model_info.cardData.get("license")
        metadata.tags = metadata.tags or model_info.tags
        metadata.author = metadata.author or model_info.author
        return urls

    def _select_variants(self, paths: List[str], variant: Optional[str] = None) -> Set[Path]:
        """Select the proper variant files from a list of HuggingFace repo_id paths."""
        result = set()
        basenames = dict()
        for p in paths:
            path = Path(p)

            if path.suffix == ".onnx":
                if variant == "onnx":
                    result.add(path)

            elif path.name.startswith("openvino_model"):
                if variant == "openvino":
                    result.add(path)

            elif path.suffix in [".json", ".txt"]:
                result.add(path)

            elif path.suffix in [".bin", ".safetensors", ".pt"] and variant in ["fp16", None]:
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
                continue

        for v in basenames.values():
            result.add(v)

        return result

    def _download_path(self, job: DownloadJobBase):
        """Call when the source is a Path or pathlike object."""
        source = Path(job.source).resolve()
        destination = Path(job.destination).resolve()
        try:
            if source != destination:
                shutil.move(source, destination)
            self._update_job_status(job, DownloadJobStatus.COMPLETED)
        except OSError as excp:
            job.error = excp
            self._update_job_status(job, DownloadJobStatus.ERROR)

    def _cleanup_cancelled_job(self, job: DownloadJobBase):
        if job.preserve_partial_downloads:
            return
        self._logger.warning("Cleaning up leftover files from cancelled download job {job.destination}")
        dest = Path(job.destination)
        if dest.is_file():
            dest.unlink()
        elif dest.is_dir():
            shutil.rmtree(dest.as_posix(), ignore_errors=True)
