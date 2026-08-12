# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for the download queue."""

import asyncio
from pathlib import Path as FsPath
from pathlib import PurePosixPath, PureWindowsPath
from typing import List, Optional

from fastapi import Body, Path, Response
from fastapi.routing import APIRouter
from pydantic.networks import AnyHttpUrl
from starlette.exceptions import HTTPException

from invokeai.app.api.auth_dependencies import AdminUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.download import (
    DownloadJob,
    UnknownJobIDException,
)
from invokeai.app.util.ssrf import UnsafeDownloadURLException, validate_download_url

download_queue_router = APIRouter(prefix="/v1/download_queue", tags=["download_queue"])


def _download_queue_path(cache_dir: FsPath) -> FsPath:
    """Return the untrusted API-download directory beside the model cache."""
    return cache_dir.parent / f"{cache_dir.name}.downloads"


def _api_job(job: DownloadJob) -> DownloadJob:
    """Keep the API destination relative while the queued job retains its absolute path."""
    queue_root = _download_queue_path(ApiDependencies.invoker.services.configuration.download_cache_path).resolve()
    if job.dest.is_relative_to(queue_root):
        return job.model_copy(update={"dest": job.dest.relative_to(queue_root)})
    return job


def _validate_dest(dest: str) -> FsPath:
    """Resolve `dest` to an absolute path inside the API download directory.

    `dest` is a relative POSIX- or Windows-style path. It is anchored to
    a separate directory beside `download_cache_path` and the result is checked for containment, so a caller can
    never choose where on the filesystem the download lands. Anchoring matters on its
    own: a bare relative path handed to the download service resolves against the
    server process's working directory, which in a source or container install holds
    the application's own code.

    The API download directory is intentionally separate from the model cache. Model
    loading trusts files under `download_cache_path`, while this route accepts arbitrary
    administrator-selected URLs and filenames.

    Raises 400 on suspicious input so the download service never sees it.
    """
    if not dest or not dest.strip():
        raise HTTPException(status_code=400, detail="Download destination must not be empty.")

    posix = PurePosixPath(dest)
    windows = PureWindowsPath(dest)
    if posix.is_absolute() or windows.is_absolute():
        raise HTTPException(status_code=400, detail="Download destination must be a relative path.")

    if ".." in posix.parts or ".." in windows.parts:
        raise HTTPException(status_code=400, detail="Download destination must not contain '..' segments.")

    if "\x00" in dest:
        raise HTTPException(status_code=400, detail="Download destination must not contain null bytes.")

    cache_dir = ApiDependencies.invoker.services.configuration.download_cache_path
    queue_dir = _download_queue_path(cache_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    queue_root = queue_dir.resolve()
    # `resolve()` follows symlinks, so a link planted inside the queue directory cannot be
    # used to step outside it either.
    resolved = (queue_root / FsPath(dest)).resolve()
    if not resolved.is_relative_to(queue_root):
        raise HTTPException(status_code=400, detail="Download destination must be inside the download queue directory.")

    return resolved


@download_queue_router.get(
    "/",
    operation_id="list_downloads",
)
async def list_downloads(current_admin: AdminUserOrDefault) -> List[DownloadJob]:
    """Get a list of active and inactive jobs."""
    queue = ApiDependencies.invoker.services.download_queue
    return [_api_job(job) for job in queue.list_jobs()]


@download_queue_router.patch(
    "/",
    operation_id="prune_downloads",
    responses={
        204: {"description": "All completed jobs have been pruned"},
        400: {"description": "Bad request"},
    },
)
async def prune_downloads(current_admin: AdminUserOrDefault) -> Response:
    """Prune completed and errored jobs."""
    queue = ApiDependencies.invoker.services.download_queue
    queue.prune_jobs()
    return Response(status_code=204)


@download_queue_router.post(
    "/i/",
    operation_id="download",
)
async def download(
    current_admin: AdminUserOrDefault,
    source: AnyHttpUrl = Body(description="download source"),
    dest: str = Body(description="download destination, relative to the separate download queue directory"),
    priority: int = Body(default=10, description="queue priority"),
    access_token: Optional[str] = Body(default=None, description="token for authorization to download"),
) -> DownloadJob:
    """Download the source URL to the file or directory indicted in dest."""
    validated_dest = await asyncio.to_thread(_validate_dest, dest)
    config = ApiDependencies.invoker.services.configuration
    try:
        await asyncio.to_thread(
            validate_download_url, str(source), allow_private_urls=config.allow_private_download_urls
        )
    except UnsafeDownloadURLException:
        raise HTTPException(status_code=400, detail="Download URL resolves to a non-public address.")
    queue = ApiDependencies.invoker.services.download_queue
    return _api_job(queue.download(source, validated_dest, priority, access_token))


@download_queue_router.get(
    "/i/{id}",
    operation_id="get_download_job",
    responses={
        200: {"description": "Success"},
        404: {"description": "The requested download JobID could not be found"},
    },
)
async def get_download_job(
    current_admin: AdminUserOrDefault,
    id: int = Path(description="ID of the download job to fetch."),
) -> DownloadJob:
    """Get a download job using its ID."""
    try:
        job = ApiDependencies.invoker.services.download_queue.id_to_job(id)
        return _api_job(job)
    except UnknownJobIDException as e:
        raise HTTPException(status_code=404, detail=str(e))


@download_queue_router.delete(
    "/i/{id}",
    operation_id="cancel_download_job",
    responses={
        204: {"description": "Job has been cancelled"},
        404: {"description": "The requested download JobID could not be found"},
    },
)
async def cancel_download_job(
    current_admin: AdminUserOrDefault,
    id: int = Path(description="ID of the download job to cancel."),
) -> Response:
    """Cancel a download job using its ID."""
    try:
        queue = ApiDependencies.invoker.services.download_queue
        job = queue.id_to_job(id)
        queue.cancel_job(job)
        return Response(status_code=204)
    except UnknownJobIDException as e:
        raise HTTPException(status_code=404, detail=str(e))


@download_queue_router.delete(
    "/i",
    operation_id="cancel_all_download_jobs",
    responses={
        204: {"description": "Download jobs have been cancelled"},
    },
)
async def cancel_all_download_jobs(current_admin: AdminUserOrDefault) -> Response:
    """Cancel all download jobs."""
    ApiDependencies.invoker.services.download_queue.cancel_all_jobs()
    return Response(status_code=204)
