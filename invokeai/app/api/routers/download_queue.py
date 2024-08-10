# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for the download queue."""

from typing import List, Optional

from fastapi import Body, Path, Response
from fastapi.routing import APIRouter
from pydantic.networks import AnyHttpUrl
from starlette.exceptions import HTTPException

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.download import (
    DownloadJob,
    UnknownJobIDException,
)

download_queue_router = APIRouter(prefix="/v1/download_queue", tags=["download_queue"])


@download_queue_router.get(
    "/",
    operation_id="list_downloads",
)
async def list_downloads() -> List[DownloadJob]:
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
async def prune_downloads() -> Response:
    """Prune completed and errored jobs."""
    queue = ApiDependencies.invoker.services.download_queue
    queue.prune_jobs()
    return Response(status_code=204)


@download_queue_router.post(
    "/i/",
    operation_id="download",
)
async def download(
    source: AnyHttpUrl = Body(description="download source"),
    dest: str = Body(description="download destination"),
    priority: int = Body(default=10, description="queue priority"),
    access_token: Optional[str] = Body(default=None, description="token for authorization to download"),
) -> DownloadJob:
    """Download the source URL to the file or directory indicted in dest."""
    queue = ApiDependencies.invoker.services.download_queue
    return queue.download(source, Path(dest), priority, access_token)


@download_queue_router.get(
    "/i/{id}",
    operation_id="get_download_job",
    responses={
        200: {"description": "Success"},
        404: {"description": "The requested download JobID could not be found"},
    },
)
async def get_download_job(
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
async def cancel_all_download_jobs() -> Response:
    """Cancel all download jobs."""
    ApiDependencies.invoker.services.download_queue.cancel_all_jobs()
    return Response(status_code=204)
