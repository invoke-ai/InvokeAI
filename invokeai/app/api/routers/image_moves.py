from fastapi import HTTPException, status
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.auth_dependencies import AdminUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.image_moves.image_moves_default import (
    ImageMoveBackgroundOperation,
    ImageMoveBackgroundStatus,
    ImageMoveJob,
    ImageMoveJobAlreadyRunning,
    ImageMoveQueueActive,
    MoveJobState,
)

image_moves_router = APIRouter(prefix="/v1/image_moves", tags=["image_moves"])


class ImageMoveJobResponse(BaseModel):
    id: int = Field(description="The image move job id.")
    state: MoveJobState = Field(description="The image move job state.")
    error_message: str | None = Field(default=None, description="The last error recorded for the job, if any.")


class ImageMoveStatusResponse(BaseModel):
    is_running: bool = Field(description="Whether an image move background operation is currently running.")
    operation: ImageMoveBackgroundOperation | None = Field(
        default=None, description="The active background operation, if any."
    )
    active_job_id: int | None = Field(default=None, description="The active journal job id, if any.")
    latest_job: ImageMoveJobResponse | None = Field(default=None, description="The latest journal job, if any.")
    last_error: str | None = Field(default=None, description="The last background worker error, if any.")
    needs_move_count: int = Field(description="The number of images that do not match the current subfolder strategy.")


def _get_image_move_service():
    image_moves = getattr(ApiDependencies.invoker.services, "image_moves", None)
    if image_moves is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Image move service unavailable")
    return image_moves


def _job_to_response(job: ImageMoveJob | None) -> ImageMoveJobResponse | None:
    if job is None:
        return None
    return ImageMoveJobResponse(id=job.id, state=job.state, error_message=job.error_message)


def _status_to_response(service_status: ImageMoveBackgroundStatus | dict) -> ImageMoveStatusResponse:
    if isinstance(service_status, dict):
        return ImageMoveStatusResponse(**service_status)
    return ImageMoveStatusResponse(
        is_running=service_status.is_running,
        operation=service_status.operation,
        active_job_id=service_status.active_job_id,
        latest_job=_job_to_response(service_status.latest_job),
        last_error=service_status.last_error,
        needs_move_count=service_status.needs_move_count,
    )


@image_moves_router.post(
    "/start",
    operation_id="start_image_move",
    response_model=ImageMoveStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_image_move(_: AdminUserOrDefault) -> ImageMoveStatusResponse:
    try:
        return _status_to_response(_get_image_move_service().start_background_move_all())
    except (ImageMoveJobAlreadyRunning, ImageMoveQueueActive) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e


@image_moves_router.post(
    "/recover",
    operation_id="start_image_move_recovery",
    response_model=ImageMoveStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_image_move_recovery(_: AdminUserOrDefault) -> ImageMoveStatusResponse:
    try:
        return _status_to_response(_get_image_move_service().start_background_recovery())
    except ImageMoveJobAlreadyRunning as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e


@image_moves_router.get(
    "/status",
    operation_id="get_image_move_status",
    response_model=ImageMoveStatusResponse,
)
async def get_image_move_status(_: AdminUserOrDefault) -> ImageMoveStatusResponse:
    return _status_to_response(_get_image_move_service().get_background_status())
