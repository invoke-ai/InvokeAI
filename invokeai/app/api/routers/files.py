from fastapi import HTTPException, Path, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.files.files_base import FileServiceBase
from invokeai.app.services.files.files_common import (
    FileAccessDeniedException,
    FileDTO,
    FileMetadataException,
    FileNotFoundException,
    FileTooLargeException,
    UnsupportedFileTypeException,
)

files_router = APIRouter(prefix="/v1/files", tags=["files"])


def _get_file_service() -> FileServiceBase:
    file_service = ApiDependencies.invoker.services.files
    if file_service is None:
        raise HTTPException(status_code=503, detail="Managed file service is not available")
    return file_service


@files_router.post(
    "/upload",
    operation_id="upload_file",
    responses={
        201: {"description": "The file was uploaded successfully"},
        413: {"description": "The file is too large"},
        415: {"description": "The file type is not supported"},
    },
    status_code=201,
    response_model=FileDTO,
)
async def upload_file(
    current_user: CurrentUserOrDefault,
    file: UploadFile,
    response: Response,
) -> FileDTO:
    """Uploads a managed file for node inputs."""
    try:
        file_dto = await run_in_threadpool(
            _get_file_service().save,
            file_name=file.filename or "",
            content_type=file.content_type,
            file=file.file,
            user_id=current_user.user_id,
        )
        response.status_code = 201
        return file_dto
    except UnsupportedFileTypeException as e:
        raise HTTPException(status_code=415, detail=str(e))
    except FileTooLargeException as e:
        raise HTTPException(status_code=413, detail=str(e))
    except FileMetadataException as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


@files_router.get(
    "/i/{file_id}",
    operation_id="get_file_dto",
    response_model=FileDTO,
)
async def get_file_dto(
    current_user: CurrentUserOrDefault,
    file_id: str = Path(description="The managed file ID."),
) -> FileDTO:
    """Gets metadata for a managed file."""
    try:
        return await run_in_threadpool(_get_file_service().get_dto, file_id, user_id=current_user.user_id)
    except FileAccessDeniedException:
        raise HTTPException(status_code=403, detail="Not authorized to access this file")
    except FileNotFoundException:
        raise HTTPException(status_code=404, detail="File not found")
    except FileMetadataException as e:
        raise HTTPException(status_code=500, detail=str(e))


@files_router.delete(
    "/i/{file_id}",
    operation_id="delete_file",
    status_code=204,
)
async def delete_file(
    current_user: CurrentUserOrDefault,
    file_id: str = Path(description="The managed file ID."),
) -> None:
    """Deletes a managed file."""
    try:
        await run_in_threadpool(_get_file_service().delete, file_id, user_id=current_user.user_id)
    except FileAccessDeniedException:
        raise HTTPException(status_code=403, detail="Not authorized to delete this file")
    except FileNotFoundException:
        raise HTTPException(status_code=404, detail="File not found")
    except FileMetadataException as e:
        raise HTTPException(status_code=500, detail=str(e))
