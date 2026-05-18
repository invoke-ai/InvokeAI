# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for the download queue."""

from pathlib import Path as FsPath
from pathlib import PurePosixPath, PureWindowsPath
from typing import List, Optional

from fastapi import Body, Path, Response
from fastapi.routing import APIRouter
from pydantic.networks import AnyHttpUrl
from starlette.exceptions import HTTPException

from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.download import (
    DownloadJob,
    UnknownJobIDException,
)

download_queue_router = APIRouter(prefix="/v1/download_queue", tags=["download_queue"])


def _validate_dest(dest: str) -> str:
    """Reject absolute paths and parent-traversal segments.

    Accepts a relative POSIX- or Windows-style path. Returns the original string
    for the caller to wrap in `Path(...)`. Raises 400 on suspicious input so the
    download service never sees it.
    """
    if not dest or not dest.strip():
        raise HTTPException(status_code=400, detail="Download destination must not be empty.")

    posix = PurePosixPath(dest)
    windows = PureWindowsPath(dest)
    if posix.is_absolute() or windows.is_absolute():
        raise HTTPException(status_code=400, detail="Download destination must be a relative path.")

    parts = set(posix.parts) | set(windows.parts)
    if ".." in parts:
        raise HTTPException(status_code=400, detail="Download destination must not contain '..' segments.")

    return dest


@download_queue_router.get(
    "/",
    operation_id="list_downloads",
)
async def list_downloads(current_user: CurrentUserOrDefault) -> List[DownloadJob]:
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
async def prune_downloads(current_user: AdminUserOrDefault) -> Response:
    """Prune completed and errored jobs."""
    queue = ApiDependencies.invoker.services.download_queue
    queue.prune_jobs()
    return Response(status_code=204)


@download_queue_router.post(
    "/i/",
    operation_id="download",
)
async def download(
    current_user: CurrentUserOrDefault,
    source: AnyHttpUrl = Body(description="download source"),
    dest: str = Body(description="download destination"),
    priority: int = Body(default=10, description="queue priority"),
    access_token: Optional[str] = Body(default=None, description="token for authorization to download"),
) -> DownloadJob:
    """Download the source URL to the file or directory indicted in dest."""
    validated_dest = _validate_dest(dest)
    queue = ApiDependencies.invoker.services.download_queue
    return queue.download(source, FsPath(validated_dest), priority, access_token)


@download_queue_router.get(
    "/i/{id}",
    operation_id="get_download_job",
    responses={
        200: {"description": "Success"},
        404: {"description": "The requested download JobID could not be found"},
    },
)
async def get_download_job(
    current_user: CurrentUserOrDefault,
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
    current_user: CurrentUserOrDefault,
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
async def cancel_all_download_jobs(current_user: AdminUserOrDefault) -> Response:
    """Cancel all download jobs."""
    ApiDependencies.invoker.services.download_queue.cancel_all_jobs()
    return Response(status_code=204)
