import traceback
from pathlib import Path
from typing import Optional

from fastapi import Body, File, Form, HTTPException, Query, Response, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.canvas_project_records.canvas_project_records_common import CanvasProjectRecordChanges
from invokeai.app.services.canvas_projects.canvas_projects_common import (
    AddCanvasProjectsToBoardResult,
    CanvasProjectDTO,
    DeleteCanvasProjectsResult,
    RemoveCanvasProjectsFromBoardResult,
    StarredCanvasProjectsResult,
    UnstarredCanvasProjectsResult,
)
from invokeai.app.services.image_records.image_records_common import ResourceOrigin
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection

canvas_projects_router = APIRouter(prefix="/v1/canvas_projects", tags=["canvas_projects"])
board_canvas_projects_router = APIRouter(prefix="/v1/board_canvas_projects", tags=["board_canvas_projects"])

# Reasonable size cap for `.invk` uploads. Canvas projects bundle layer image bytes, which
# can grow large for complex compositions — but they should not be measured in gigabytes.
MAX_UPLOAD_SIZE = 256 * 1024 * 1024  # 256 MB
UPLOAD_CHUNK_SIZE = 1024 * 1024


def _assert_project_owner(project_name: str, current_user: CurrentUserOrDefault) -> None:
    from invokeai.app.services.board_records.board_records_common import BoardVisibility

    if current_user.is_admin:
        return
    owner = ApiDependencies.invoker.services.canvas_project_records.get_user_id(project_name)
    if owner is not None and owner == current_user.user_id:
        return

    board_id = ApiDependencies.invoker.services.board_canvas_project_records.get_board_for_project(project_name)
    if board_id is not None:
        try:
            board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
            if board.user_id == current_user.user_id:
                return
            if board.board_visibility == BoardVisibility.Public:
                return
        except Exception:
            pass

    raise HTTPException(status_code=403, detail="Not authorized to modify this canvas project")


def _assert_project_direct_owner(project_name: str, current_user: CurrentUserOrDefault) -> None:
    if current_user.is_admin:
        return
    owner = ApiDependencies.invoker.services.canvas_project_records.get_user_id(project_name)
    if owner is not None and owner == current_user.user_id:
        return
    raise HTTPException(status_code=403, detail="Not authorized to move this canvas project")


def _assert_board_write_access(board_id: str, current_user: CurrentUserOrDefault) -> None:
    from invokeai.app.services.board_records.board_records_common import BoardVisibility

    try:
        board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Board not found")
    if current_user.is_admin:
        return
    if board.user_id == current_user.user_id:
        return
    if board.board_visibility == BoardVisibility.Public:
        return
    raise HTTPException(status_code=403, detail="Not authorized to modify this board")


def _assert_project_read_access(project_name: str, current_user: CurrentUserOrDefault) -> None:
    from invokeai.app.services.board_records.board_records_common import BoardVisibility

    if current_user.is_admin:
        return
    owner = ApiDependencies.invoker.services.canvas_project_records.get_user_id(project_name)
    if owner is not None and owner == current_user.user_id:
        return

    board_id = ApiDependencies.invoker.services.board_canvas_project_records.get_board_for_project(project_name)
    if board_id is not None:
        try:
            board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
            if board.board_visibility in (BoardVisibility.Shared, BoardVisibility.Public):
                return
        except Exception:
            pass

    raise HTTPException(status_code=403, detail="Not authorized to access this canvas project")


@canvas_projects_router.post(
    "/upload",
    operation_id="upload_canvas_project",
    responses={
        201: {"description": "The canvas project was uploaded successfully"},
        413: {"description": "Canvas project file too large"},
        415: {"description": "Not a supported canvas project file"},
    },
    status_code=201,
    response_model=CanvasProjectDTO,
)
async def upload_canvas_project(
    current_user: CurrentUserOrDefault,
    response: Response,
    file: UploadFile = File(description="The canvas project ZIP (.invk) file"),
    name: str = Form(description="The user-facing project name"),
    app_version: str = Form(description="The InvokeAI app version captured at save time"),
    width: int = Form(description="The bbox width at save time"),
    height: int = Form(description="The bbox height at save time"),
    image_count: int = Form(description="The number of embedded image files"),
    thumbnail: Optional[UploadFile] = File(default=None, description="Optional preview WebP thumbnail"),
    board_id: Optional[str] = Form(default=None, description="Optional board to attach the project to"),
    is_intermediate: bool = Form(default=False, description="Whether this is an intermediate project"),
) -> CanvasProjectDTO:
    """Uploads a canvas project ZIP for the current user, optionally placing it on a board."""
    if board_id is not None:
        _assert_board_write_access(board_id, current_user)

    if not (file.filename or "").lower().endswith(".invk"):
        raise HTTPException(status_code=415, detail="Not a supported canvas project file (.invk expected)")

    try:
        total = 0
        chunks: list[bytes] = []
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Canvas project upload exceeds maximum size ({MAX_UPLOAD_SIZE} bytes)",
                )
            chunks.append(chunk)
        zip_bytes = b"".join(chunks)

        thumbnail_bytes: Optional[bytes] = None
        if thumbnail is not None:
            thumbnail_bytes = await thumbnail.read()
            if len(thumbnail_bytes) == 0:
                thumbnail_bytes = None

        try:
            project_dto = ApiDependencies.invoker.services.canvas_projects.create(
                zip_bytes=zip_bytes,
                name=name,
                app_version=app_version,
                width=width,
                height=height,
                image_count=image_count,
                thumbnail_bytes=thumbnail_bytes,
                project_origin=ResourceOrigin.EXTERNAL,
                board_id=board_id,
                is_intermediate=is_intermediate,
                user_id=current_user.user_id,
            )
            response.status_code = 201
            response.headers["Location"] = project_dto.project_url
            return project_dto
        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Failed to create canvas project")
    except HTTPException:
        raise
    except Exception:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to read canvas project upload")


@canvas_projects_router.delete(
    "/i/{project_name}",
    operation_id="delete_canvas_project",
    response_model=DeleteCanvasProjectsResult,
)
async def delete_canvas_project(
    current_user: CurrentUserOrDefault,
    project_name: str = PathParam(description="The name of the canvas project to delete"),
) -> DeleteCanvasProjectsResult:
    _assert_project_owner(project_name, current_user)

    deleted_projects: set[str] = set()
    affected_boards: set[str] = set()
    try:
        project_dto = ApiDependencies.invoker.services.canvas_projects.get_dto(project_name)
        board_id = project_dto.board_id or "none"
        ApiDependencies.invoker.services.canvas_projects.delete(project_name)
        deleted_projects.add(project_name)
        affected_boards.add(board_id)
    except Exception:
        pass

    return DeleteCanvasProjectsResult(
        deleted_projects=list(deleted_projects),
        affected_boards=list(affected_boards),
    )


@canvas_projects_router.post(
    "/delete",
    operation_id="delete_canvas_projects_from_list",
    response_model=DeleteCanvasProjectsResult,
)
async def delete_canvas_projects_from_list(
    current_user: CurrentUserOrDefault,
    project_names: list[str] = Body(description="The list of canvas project names to delete", embed=True),
) -> DeleteCanvasProjectsResult:
    deleted_projects: set[str] = set()
    affected_boards: set[str] = set()
    for project_name in project_names:
        try:
            _assert_project_owner(project_name, current_user)
            project_dto = ApiDependencies.invoker.services.canvas_projects.get_dto(project_name)
            board_id = project_dto.board_id or "none"
            ApiDependencies.invoker.services.canvas_projects.delete(project_name)
            deleted_projects.add(project_name)
            affected_boards.add(board_id)
        except HTTPException:
            raise
        except Exception:
            pass
    return DeleteCanvasProjectsResult(
        deleted_projects=list(deleted_projects),
        affected_boards=list(affected_boards),
    )


@canvas_projects_router.patch(
    "/i/{project_name}",
    operation_id="update_canvas_project",
    response_model=CanvasProjectDTO,
)
async def update_canvas_project(
    current_user: CurrentUserOrDefault,
    project_name: str = PathParam(description="The name of the canvas project to update"),
    project_changes: CanvasProjectRecordChanges = Body(description="The changes to apply"),
) -> CanvasProjectDTO:
    _assert_project_owner(project_name, current_user)
    try:
        return ApiDependencies.invoker.services.canvas_projects.update(project_name, project_changes)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to update canvas project")


@canvas_projects_router.put(
    "/i/{project_name}/file",
    operation_id="replace_canvas_project_file",
    responses={
        200: {"description": "The canvas project file was replaced successfully"},
        413: {"description": "Canvas project file too large"},
        415: {"description": "Not a supported canvas project file"},
    },
    response_model=CanvasProjectDTO,
)
async def replace_canvas_project_file(
    current_user: CurrentUserOrDefault,
    project_name: str = PathParam(description="The name of the canvas project to replace"),
    file: UploadFile = File(description="The new canvas project ZIP (.invk) file"),
    name: Optional[str] = Form(default=None, description="Optional new user-facing project name"),
    app_version: str = Form(description="The InvokeAI app version captured at save time"),
    width: int = Form(description="The bbox width at save time"),
    height: int = Form(description="The bbox height at save time"),
    image_count: int = Form(description="The number of embedded image files"),
    thumbnail: Optional[UploadFile] = File(default=None, description="Optional new WebP thumbnail"),
) -> CanvasProjectDTO:
    """Replaces the on-disk ZIP and thumbnail for an existing canvas project. Keeps project_name,
    board assignment, starred state, ownership. Updates width/height/image_count/app_version and
    `has_thumbnail` (when a thumbnail is supplied). Optionally renames via the `name` form field."""
    _assert_project_owner(project_name, current_user)

    if not (file.filename or "").lower().endswith(".invk"):
        raise HTTPException(status_code=415, detail="Not a supported canvas project file (.invk expected)")

    try:
        total = 0
        chunks: list[bytes] = []
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Canvas project upload exceeds maximum size ({MAX_UPLOAD_SIZE} bytes)",
                )
            chunks.append(chunk)
        zip_bytes = b"".join(chunks)

        thumbnail_bytes: Optional[bytes] = None
        if thumbnail is not None:
            thumbnail_bytes = await thumbnail.read()
            if len(thumbnail_bytes) == 0:
                thumbnail_bytes = None

        try:
            project_dto = ApiDependencies.invoker.services.canvas_projects.replace_file(
                project_name=project_name,
                zip_bytes=zip_bytes,
                width=width,
                height=height,
                image_count=image_count,
                app_version=app_version,
                thumbnail_bytes=thumbnail_bytes,
            )
            # If the caller also wants to rename, apply that as a separate record change.
            if name is not None and name != project_dto.name:
                project_dto = ApiDependencies.invoker.services.canvas_projects.update(
                    project_name, CanvasProjectRecordChanges(name=name)
                )
            return project_dto
        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Failed to replace canvas project file")
    except HTTPException:
        raise
    except Exception:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to read canvas project upload")


@canvas_projects_router.post(
    "/star",
    operation_id="star_canvas_projects_in_list",
    response_model=StarredCanvasProjectsResult,
)
async def star_canvas_projects_in_list(
    current_user: CurrentUserOrDefault,
    project_names: list[str] = Body(description="The list of canvas project names to star", embed=True),
) -> StarredCanvasProjectsResult:
    starred_projects: set[str] = set()
    affected_boards: set[str] = set()
    for project_name in project_names:
        try:
            _assert_project_owner(project_name, current_user)
            ApiDependencies.invoker.services.canvas_projects.update(
                project_name, CanvasProjectRecordChanges(starred=True)
            )
            starred_projects.add(project_name)
            board_id = ApiDependencies.invoker.services.board_canvas_project_records.get_board_for_project(
                project_name
            )
            affected_boards.add(board_id or "none")
        except HTTPException:
            raise
        except Exception:
            pass
    return StarredCanvasProjectsResult(
        starred_projects=list(starred_projects),
        affected_boards=list(affected_boards),
    )


@canvas_projects_router.post(
    "/unstar",
    operation_id="unstar_canvas_projects_in_list",
    response_model=UnstarredCanvasProjectsResult,
)
async def unstar_canvas_projects_in_list(
    current_user: CurrentUserOrDefault,
    project_names: list[str] = Body(description="The list of canvas project names to unstar", embed=True),
) -> UnstarredCanvasProjectsResult:
    unstarred_projects: set[str] = set()
    affected_boards: set[str] = set()
    for project_name in project_names:
        try:
            _assert_project_owner(project_name, current_user)
            ApiDependencies.invoker.services.canvas_projects.update(
                project_name, CanvasProjectRecordChanges(starred=False)
            )
            unstarred_projects.add(project_name)
            board_id = ApiDependencies.invoker.services.board_canvas_project_records.get_board_for_project(
                project_name
            )
            affected_boards.add(board_id or "none")
        except HTTPException:
            raise
        except Exception:
            pass
    return UnstarredCanvasProjectsResult(
        unstarred_projects=list(unstarred_projects),
        affected_boards=list(affected_boards),
    )


@canvas_projects_router.get(
    "/i/{project_name}",
    operation_id="get_canvas_project_dto",
    response_model=CanvasProjectDTO,
)
async def get_canvas_project_dto(
    current_user: CurrentUserOrDefault,
    project_name: str = PathParam(description="The name of canvas project to get"),
) -> CanvasProjectDTO:
    _assert_project_read_access(project_name, current_user)
    try:
        return ApiDependencies.invoker.services.canvas_projects.get_dto(project_name)
    except Exception:
        raise HTTPException(status_code=404)


@canvas_projects_router.get(
    "/i/{project_name}/full",
    operation_id="get_canvas_project_full",
    response_class=Response,
    responses={
        200: {"description": "Return the canvas project ZIP", "content": {"application/zip": {}}},
        404: {"description": "Canvas project not found"},
    },
)
async def get_canvas_project_full(
    project_name: str = PathParam(description="The name of canvas project file to get"),
) -> Response:
    """Serves the canvas project ZIP (.invk).

    Like the image/video equivalents, this endpoint is intentionally unauthenticated so the
    browser can fetch it via standard download flow. Project names are UUIDs, providing
    security through unguessability.
    """
    try:
        path_str = ApiDependencies.invoker.services.canvas_projects.get_path(project_name)
    except Exception:
        raise HTTPException(status_code=404)

    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404)

    return FileResponse(
        path=path,
        media_type="application/zip",
        filename=f"{project_name}.invk",
        headers={"Content-Disposition": f'attachment; filename="{project_name}.invk"'},
    )


@canvas_projects_router.get(
    "/i/{project_name}/thumbnail",
    operation_id="get_canvas_project_thumbnail",
    response_class=Response,
    responses={
        200: {"description": "Return the canvas project thumbnail", "content": {"image/webp": {}}},
        404: {"description": "Canvas project thumbnail not found"},
    },
)
async def get_canvas_project_thumbnail(
    project_name: str = PathParam(description="The name of canvas project whose thumbnail to get"),
) -> Response:
    """Serves the canvas project preview thumbnail (WebP). Unauthenticated for the same reason
    as `get_canvas_project_full` — project names are UUIDs."""
    try:
        path_str = ApiDependencies.invoker.services.canvas_projects.get_path(project_name, thumbnail=True)
    except Exception:
        raise HTTPException(status_code=404)

    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404)

    return FileResponse(path=path, media_type="image/webp")


@canvas_projects_router.get(
    "/",
    operation_id="list_canvas_project_dtos",
    response_model=OffsetPaginatedResults[CanvasProjectDTO],
)
async def list_canvas_project_dtos(
    current_user: CurrentUserOrDefault,
    project_origin: Optional[ResourceOrigin] = Query(default=None, description="Filter by project origin"),
    is_intermediate: Optional[bool] = Query(default=None, description="Filter by is_intermediate flag"),
    board_id: Optional[str] = Query(default=None, description="Filter by board_id ('none' for unassigned)"),
    offset: int = Query(default=0, description="The page offset"),
    limit: int = Query(default=10, description="The number of canvas projects per page"),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="Sort direction"),
    starred_first: bool = Query(default=True, description="Whether starred projects come first"),
    search_term: Optional[str] = Query(default=None, description="A free-text search term"),
) -> OffsetPaginatedResults[CanvasProjectDTO]:
    """Lists canvas project DTOs with pagination and filtering."""
    return ApiDependencies.invoker.services.canvas_projects.get_many(
        offset=offset,
        limit=limit,
        starred_first=starred_first,
        order_dir=order_dir,
        project_origin=project_origin,
        is_intermediate=is_intermediate,
        board_id=board_id,
        search_term=search_term,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
    )


@board_canvas_projects_router.post(
    "/",
    operation_id="add_canvas_project_to_board",
    response_model=AddCanvasProjectsToBoardResult,
)
async def add_canvas_project_to_board(
    current_user: CurrentUserOrDefault,
    board_id: str = Body(description="The id of the board to add the project to"),
    project_name: str = Body(description="The name of the canvas project to add"),
) -> AddCanvasProjectsToBoardResult:
    _assert_project_direct_owner(project_name, current_user)
    _assert_board_write_access(board_id, current_user)

    affected_boards: set[str] = set()
    added_projects: list[str] = []
    try:
        existing_board = ApiDependencies.invoker.services.board_canvas_project_records.get_board_for_project(
            project_name
        )
        if existing_board is not None:
            affected_boards.add(existing_board)
        ApiDependencies.invoker.services.board_canvas_project_records.add_project_to_board(
            board_id=board_id, project_name=project_name
        )
        affected_boards.add(board_id)
        added_projects.append(project_name)
    except Exception:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to add canvas project to board")

    return AddCanvasProjectsToBoardResult(
        added_projects=added_projects,
        affected_boards=list(affected_boards),
    )


@board_canvas_projects_router.delete(
    "/",
    operation_id="remove_canvas_project_from_board",
    response_model=RemoveCanvasProjectsFromBoardResult,
)
async def remove_canvas_project_from_board(
    current_user: CurrentUserOrDefault,
    project_name: str = Body(description="The name of the canvas project to remove from its board", embed=True),
) -> RemoveCanvasProjectsFromBoardResult:
    _assert_project_direct_owner(project_name, current_user)

    affected_boards: set[str] = set()
    removed_projects: list[str] = []
    try:
        existing_board = ApiDependencies.invoker.services.board_canvas_project_records.get_board_for_project(
            project_name
        )
        if existing_board is not None:
            affected_boards.add(existing_board)
        ApiDependencies.invoker.services.board_canvas_project_records.remove_project_from_board(project_name)
        affected_boards.add("none")
        removed_projects.append(project_name)
    except Exception:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to remove canvas project from board")

    return RemoveCanvasProjectsFromBoardResult(
        removed_projects=removed_projects,
        affected_boards=list(affected_boards),
    )
