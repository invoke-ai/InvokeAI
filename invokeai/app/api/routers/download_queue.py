# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for the download queue."""

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


def _validate_dest(dest: str) -> FsPath:
    """Resolve `dest` to an absolute path inside the configured download cache directory.

    `dest` is a relative POSIX- or Windows-style path. It is anchored to
    `download_cache_path` and the result is checked for containment, so a caller can
    never choose where on the filesystem the download lands. Anchoring matters on its
    own: a bare relative path handed to the download service resolves against the
    server process's working directory, which in a source or container install holds
    the application's own code.

    Containment is not the same as harmlessness: entries under the download cache are
    handed to `torch.load`/`torch.jit.load` by `download_and_cache_model`, so a write in
    here can still poison a cached model. That is why this route is admin-only — an
    administrator can already install arbitrary models — and why the containment check
    must not be treated as the whole defence.

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
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_root = cache_dir.resolve()
    # `resolve()` follows symlinks, so a link planted inside the cache cannot be used to
    # step outside it either. `is_relative_to` is true for the cache directory itself,
    # which is the ordinary "download into the cache" case.
    resolved = (cache_root / FsPath(dest)).resolve()
    if not resolved.is_relative_to(cache_root):
        raise HTTPException(status_code=400, detail="Download destination must be inside the download cache directory.")

    return resolved


@download_queue_router.get(
    "/",
    operation_id="list_downloads",
)
async def list_downloads(current_admin: AdminUserOrDefault) -> List[DownloadJob]:
    """Get a list of active and inactive jobs."""
    queue = ApiDependencies.invoker.services.download_queue
    return queue.list_jobs()


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
    dest: str = Body(description="download destination, relative to the download cache directory"),
    priority: int = Body(default=10, description="queue priority"),
    access_token: Optional[str] = Body(default=None, description="token for authorization to download"),
) -> DownloadJob:
    """Download the source URL to the file or directory indicted in dest."""
    validated_dest = _validate_dest(dest)
    config = ApiDependencies.invoker.services.configuration
    try:
        validate_download_url(str(source), allow_private_urls=config.allow_private_download_urls)
    except UnsafeDownloadURLException as e:
        raise HTTPException(status_code=400, detail=str(e))
    queue = ApiDependencies.invoker.services.download_queue
    return queue.download(source, validated_dest, priority, access_token)


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
        return job
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
