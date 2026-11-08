import logging
from typing import NoReturn

from fastapi import HTTPException, status
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import AdminUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.gallery_maintenance.gallery_maintenance_common import (
    GalleryMaintenanceConflict,
    GalleryMaintenanceExecuteRequest,
    GalleryMaintenanceOperation,
    GalleryMaintenancePreview,
    GalleryMaintenancePreviewChanged,
    GalleryMaintenancePreviewRequest,
    GalleryMaintenanceResult,
)
from invokeai.app.services.image_moves.image_moves_default import ImageMoveJobAlreadyRunning, ImageMoveQueueActive

logger = logging.getLogger(__name__)

gallery_maintenance_router = APIRouter(prefix="/v1/app/gallery/maintenance", tags=["gallery_maintenance"])


def _get_service():
    service = getattr(ApiDependencies.invoker.services, "gallery_maintenance", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Gallery maintenance unavailable")
    return service


def _raise_safe_error(error: Exception) -> NoReturn:
    logger.exception("Gallery maintenance operation failed")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Gallery maintenance failed"
    ) from error


@gallery_maintenance_router.post(
    "/preview",
    operation_id="preview_gallery_maintenance",
    response_model=GalleryMaintenancePreview,
    status_code=status.HTTP_200_OK,
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
        409: {"description": "Gallery maintenance is already active"},
        500: {"description": "Gallery maintenance preview failed"},
        503: {"description": "Gallery maintenance unavailable"},
    },
)
def preview_gallery_maintenance(
    request: GalleryMaintenancePreviewRequest, _: AdminUserOrDefault
) -> GalleryMaintenancePreview:
    service = _get_service()
    try:
        return service.preview(request.operation)
    except (GalleryMaintenanceConflict, ImageMoveJobAlreadyRunning, ImageMoveQueueActive) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Image storage is busy") from error
    except Exception as error:
        _raise_safe_error(error)


def _execute_gallery_maintenance(
    operation: GalleryMaintenanceOperation, request: GalleryMaintenanceExecuteRequest
) -> GalleryMaintenanceResult:
    service = _get_service()
    try:
        return service.execute(operation, request.fingerprint)
    except GalleryMaintenancePreviewChanged as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Gallery contents changed after preview. Preview again before continuing.",
        ) from error
    except (GalleryMaintenanceConflict, ImageMoveJobAlreadyRunning, ImageMoveQueueActive) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Image storage is busy") from error
    except Exception as error:
        _raise_safe_error(error)


@gallery_maintenance_router.post(
    "/remove-missing",
    operation_id="remove_missing_images",
    response_model=GalleryMaintenanceResult,
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
        409: {"description": "Preview is stale or image storage is busy"},
        500: {"description": "Gallery maintenance failed"},
        503: {"description": "Gallery maintenance unavailable"},
    },
)
def remove_missing_images(request: GalleryMaintenanceExecuteRequest, _: AdminUserOrDefault) -> GalleryMaintenanceResult:
    return _execute_gallery_maintenance(GalleryMaintenanceOperation.REMOVE_MISSING, request)


@gallery_maintenance_router.post(
    "/archive-untracked",
    operation_id="archive_untracked_images",
    response_model=GalleryMaintenanceResult,
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
        409: {"description": "Preview is stale or image storage is busy"},
        500: {"description": "Gallery maintenance failed"},
        503: {"description": "Gallery maintenance unavailable"},
    },
)
def archive_untracked_images(
    request: GalleryMaintenanceExecuteRequest, _: AdminUserOrDefault
) -> GalleryMaintenanceResult:
    return _execute_gallery_maintenance(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED, request)


@gallery_maintenance_router.post(
    "/regenerate-thumbnails",
    operation_id="regenerate_missing_thumbnails",
    response_model=GalleryMaintenanceResult,
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Admin privileges required"},
        409: {"description": "Preview is stale or image storage is busy"},
        500: {"description": "Gallery maintenance failed"},
        503: {"description": "Gallery maintenance unavailable"},
    },
)
def regenerate_missing_thumbnails(
    request: GalleryMaintenanceExecuteRequest, _: AdminUserOrDefault
) -> GalleryMaintenanceResult:
    return _execute_gallery_maintenance(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS, request)
