import re
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from fastapi import Body, HTTPException, Query, Request, Response, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from invokeai.app.api.auth_dependencies import CurrentMediaUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.images import WorkflowAndGraphResponse, _assert_board_read_access
from invokeai.app.invocations.fields import MetadataField, MetadataFieldValidator
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.video_records.video_records_common import VideoNamesResult, VideoRecordChanges
from invokeai.app.services.videos.videos_common import (
    AddVideosToBoardResult,
    DeleteVideosResult,
    RemoveVideosFromBoardResult,
    StarredVideosResult,
    UnstarredVideosResult,
    VideoDTO,
    VideoUrlsDTO,
)
from invokeai.app.util.video_thumbnails import probe_video

videos_router = APIRouter(prefix="/v1/videos", tags=["videos"])

# Videos are immutable; set a high max-age (1 year)
VIDEO_MAX_AGE = 31536000

# MP4 only — the names service emits `{uuid}.mp4` unconditionally and we don't transcode on
# upload. Accepting .mov/.webm/.mkv here previously caused those containers to be stored
# under a .mp4 name and served with the .mp4 MIME type, which silently broke playback in
# browsers when the container did not match.
ACCEPTED_VIDEO_MIME_PREFIXES = ("video/mp4",)
ACCEPTED_VIDEO_EXTENSIONS = (".mp4",)

# Per-chunk size for HTTP Range responses (1 MB)
RANGE_CHUNK_SIZE = 1024 * 1024

# Upload streaming chunk size (1 MB) and a coarse per-upload size cap. The cap is generous
# because Wan-generated MP4s for long sequences can run into the hundreds of megabytes;
# the goal is to prevent a single client from exhausting RAM, not to be a content policy.
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_UPLOAD_SIZE = 1024 * 1024 * 1024  # 1 GB
# Pre-parse ingress cap enforced by VideoUploadLimitASGIMiddleware, applied to the whole
# request body *before* the multipart parser spools it to temp storage. Slightly larger
# than MAX_UPLOAD_SIZE to allow for multipart framing and the metadata form field.
MAX_UPLOAD_REQUEST_SIZE = MAX_UPLOAD_SIZE + 10 * 1024 * 1024
# Global bound on concurrent video uploads — each in-flight upload can hold up to two
# copies of the file in temp storage (the multipart spool + the route's own tmp file).
MAX_CONCURRENT_VIDEO_UPLOADS = 4


def _get_video_cache_control() -> str:
    if ApiDependencies.invoker.services.configuration.multiuser:
        return "private, no-store"
    return f"max-age={VIDEO_MAX_AGE}"


def _assert_video_owner(video_name: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user does not own the video and is not an admin."""
    from invokeai.app.services.board_records.board_records_common import BoardVisibility

    if current_user.is_admin:
        return
    owner = ApiDependencies.invoker.services.video_records.get_user_id(video_name)
    if owner is not None and owner == current_user.user_id:
        return

    board_id = ApiDependencies.invoker.services.board_video_records.get_board_for_video(video_name)
    if board_id is not None:
        try:
            board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
            if board.user_id == current_user.user_id:
                return
            if board.board_visibility == BoardVisibility.Public:
                return
        except Exception:
            pass

    raise HTTPException(status_code=403, detail="Not authorized to modify this video")


def _assert_video_direct_owner(video_name: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user is not the direct owner of the video.

    Intentionally stricter than _assert_video_owner: board-ownership and public-board
    fallbacks are NOT honored. Mirrors _assert_image_direct_owner in board_images.py —
    board-move operations need to verify the *original* owner, otherwise a user could
    move someone else's video onto their own board via the board-owner branch.
    """
    if current_user.is_admin:
        return
    owner = ApiDependencies.invoker.services.video_records.get_user_id(video_name)
    if owner is not None and owner == current_user.user_id:
        return
    raise HTTPException(status_code=403, detail="Not authorized to move this video")


def _assert_board_write_access(board_id: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user may not mutate the given board.

    Mirrors _assert_board_write_access in board_images.py: admins and the board owner
    may write; public boards accept contributions from any user.
    """
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


def _assert_video_read_access(video_name: str, current_user: CurrentUserOrDefault) -> None:
    """Raise 403 if the current user may not view the video."""
    from invokeai.app.services.board_records.board_records_common import BoardVisibility

    if current_user.is_admin:
        return
    owner = ApiDependencies.invoker.services.video_records.get_user_id(video_name)
    if owner is not None and owner == current_user.user_id:
        return

    board_id = ApiDependencies.invoker.services.board_video_records.get_board_for_video(video_name)
    if board_id is not None:
        try:
            board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
            if board.board_visibility in (BoardVisibility.Shared, BoardVisibility.Public):
                return
        except Exception:
            pass

    raise HTTPException(status_code=403, detail="Not authorized to access this video")


def _is_accepted_video_upload(file: UploadFile) -> bool:
    if file.content_type and file.content_type.startswith(ACCEPTED_VIDEO_MIME_PREFIXES):
        return True
    if file.filename:
        return file.filename.lower().endswith(ACCEPTED_VIDEO_EXTENSIONS)
    return False


def _is_mp4_file(path: Path) -> bool:
    try:
        with open(path, "rb") as video_file:
            search_limit = min(path.stat().st_size, 64 * 1024)
            position = 0
            while position + 8 <= search_limit:
                video_file.seek(position)
                header = video_file.read(8)
                box_size = int.from_bytes(header[:4], byteorder="big")
                box_type = header[4:8]
                header_size = 8
                if box_size == 1:
                    extended_size = video_file.read(8)
                    if len(extended_size) != 8:
                        return False
                    box_size = int.from_bytes(extended_size, byteorder="big")
                    header_size = 16
                if box_size < header_size:
                    return False
                if box_type == b"ftyp":
                    major_brand = video_file.read(4)
                    return len(major_brand) == 4 and major_brand != b"qt  "
                position += box_size
    except OSError:
        return False
    return False


@videos_router.post(
    "/upload",
    operation_id="upload_video",
    responses={
        201: {"description": "The video was uploaded successfully"},
        415: {"description": "Video upload failed"},
    },
    status_code=201,
    response_model=VideoDTO,
)
async def upload_video(
    current_user: CurrentUserOrDefault,
    file: UploadFile,
    request: Request,
    response: Response,
    video_category: ImageCategory = Query(description="The category of the video"),
    is_intermediate: bool = Query(description="Whether this is an intermediate video"),
    board_id: Optional[str] = Query(default=None, description="The board to add this video to, if any"),
    session_id: Optional[str] = Query(default=None, description="The session ID associated with this upload, if any"),
    metadata: Optional[str] = Body(
        default=None,
        description="The metadata to associate with the video, must be a stringified JSON dict",
        embed=True,
    ),
) -> VideoDTO:
    """Uploads a video for the current user."""
    if metadata is not None:
        try:
            MetadataFieldValidator.validate_json(metadata)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail="Metadata must be a JSON object") from e

    # Check board access for uploads to a specific board.
    if board_id is not None:
        from invokeai.app.services.board_records.board_records_common import BoardVisibility

        try:
            board = ApiDependencies.invoker.services.boards.get_dto(board_id=board_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Board not found")
        if (
            not current_user.is_admin
            and board.user_id != current_user.user_id
            and board.board_visibility != BoardVisibility.Public
        ):
            raise HTTPException(status_code=403, detail="Not authorized to upload to this board")

    if not _is_accepted_video_upload(file):
        raise HTTPException(status_code=415, detail="Not a supported video file")

    # Stream the upload to a tmp file so we can probe and then hand its path to the service.
    # Reading the full body into memory first risked exhausting RAM on multi-GB uploads;
    # chunk-stream instead and enforce a hard size cap. Filesystem writes, container
    # validation, ffmpeg probing, and thumbnail extraction are all blocking — run them in
    # the thread pool so a slow (or hostile) file can't stall the event loop and every
    # other API request with it.
    tmp = tempfile.NamedTemporaryFile(prefix="invokeai_upload_", suffix=".mp4", delete=False)
    tmp_path = Path(tmp.name)
    try:
        total = 0
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE:
                tmp.close()
                raise HTTPException(
                    status_code=413,
                    detail=f"Video upload exceeds maximum size ({MAX_UPLOAD_SIZE} bytes)",
                )
            await run_in_threadpool(tmp.write, chunk)
        tmp.close()

        if not await run_in_threadpool(_is_mp4_file, tmp_path):
            raise HTTPException(status_code=415, detail="Not an MP4 video file")

        try:
            width, height, duration, fps = await run_in_threadpool(probe_video, tmp_path)
        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=415, detail="Failed to read video")

        try:
            video_dto = await run_in_threadpool(
                lambda: ApiDependencies.invoker.services.videos.create(
                    source_path=tmp_path,
                    width=width,
                    height=height,
                    duration=duration,
                    fps=fps,
                    video_origin=ResourceOrigin.EXTERNAL,
                    video_category=video_category,
                    session_id=session_id,
                    board_id=board_id,
                    metadata=metadata,
                    workflow=None,
                    graph=None,
                    is_intermediate=is_intermediate,
                    user_id=current_user.user_id,
                )
            )

            response.status_code = 201
            response.headers["Location"] = video_dto.video_url
            return video_dto
        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Failed to create video")
    finally:
        # If create() succeeded the file was moved; this unlink is a no-op then.
        try:
            tmp.close()
        except Exception:
            pass
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@videos_router.delete("/i/{video_name}", operation_id="delete_video", response_model=DeleteVideosResult)
async def delete_video(
    current_user: CurrentUserOrDefault,
    video_name: str = PathParam(description="The name of the video to delete"),
) -> DeleteVideosResult:
    _assert_video_owner(video_name, current_user)

    # Let service-level failures surface as 500s rather than swallowing them and returning a
    # success-shaped response. A previous version of this handler caught everything and
    # returned an empty ``deleted_videos`` list with HTTP 200; the frontend treated that as
    # success, dropped the item from its cache, and the video stayed on disk — a silent
    # data-consistency failure that only became visible on the next page reload.
    try:
        video_dto = ApiDependencies.invoker.services.videos.get_dto(video_name)
    except Exception:
        raise HTTPException(status_code=404, detail="Video not found")

    board_id = video_dto.board_id or "none"
    try:
        ApiDependencies.invoker.services.videos.delete(video_name)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete video")

    return DeleteVideosResult(
        deleted_videos=[video_name],
        affected_boards=[board_id],
    )


@videos_router.post("/delete", operation_id="delete_videos_from_list", response_model=DeleteVideosResult)
async def delete_videos_from_list(
    current_user: CurrentUserOrDefault,
    video_names: list[str] = Body(description="The list of names of videos to delete", embed=True),
) -> DeleteVideosResult:
    # Skip — but do not re-raise — auth failures so a foreign name mid-batch doesn't
    # discard the response payload for items the caller had already legitimately deleted.
    # Without this, the client cache never learns about the partial successes and the
    # already-deleted records reappear in the UI until the next full refresh.
    deleted_videos: set[str] = set()
    affected_boards: set[str] = set()
    for video_name in video_names:
        try:
            _assert_video_owner(video_name, current_user)
            video_dto = ApiDependencies.invoker.services.videos.get_dto(video_name)
            board_id = video_dto.board_id or "none"
            ApiDependencies.invoker.services.videos.delete(video_name)
            deleted_videos.add(video_name)
            affected_boards.add(board_id)
        except HTTPException:
            continue
        except Exception:
            pass
    return DeleteVideosResult(
        deleted_videos=list(deleted_videos),
        affected_boards=list(affected_boards),
    )


@videos_router.delete("/uncategorized", operation_id="delete_uncategorized_videos", response_model=DeleteVideosResult)
async def delete_uncategorized_videos(
    current_user: CurrentUserOrDefault,
) -> DeleteVideosResult:
    """Deletes all uncategorized videos owned by the current user (or all if admin).

    Mirrors ``delete_uncategorized_images`` so the "Delete All Uncategorized
    Images/Videos" board action covers both media kinds.
    """
    names_result = ApiDependencies.invoker.services.videos.get_video_names(
        board_id="none",
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
    )
    deleted_videos: set[str] = set()
    affected_boards: set[str] = set()
    for video_name in names_result.video_names:
        try:
            _assert_video_owner(video_name, current_user)
            ApiDependencies.invoker.services.videos.delete(video_name)
            deleted_videos.add(video_name)
            affected_boards.add("none")
        except HTTPException:
            # Skip videos not owned by the current user
            continue
        except Exception:
            pass
    return DeleteVideosResult(
        deleted_videos=list(deleted_videos),
        affected_boards=list(affected_boards),
    )


@videos_router.patch("/i/{video_name}", operation_id="update_video", response_model=VideoDTO)
async def update_video(
    current_user: CurrentUserOrDefault,
    video_name: str = PathParam(description="The name of the video to update"),
    video_changes: VideoRecordChanges = Body(description="The changes to apply to the video"),
) -> VideoDTO:
    _assert_video_owner(video_name, current_user)
    try:
        return ApiDependencies.invoker.services.videos.update(video_name, video_changes)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to update video")


@videos_router.get("/i/{video_name}", operation_id="get_video_dto", response_model=VideoDTO)
async def get_video_dto(
    current_user: CurrentUserOrDefault,
    video_name: str = PathParam(description="The name of video to get"),
) -> VideoDTO:
    _assert_video_read_access(video_name, current_user)
    try:
        return ApiDependencies.invoker.services.videos.get_dto(video_name)
    except Exception:
        raise HTTPException(status_code=404)


@videos_router.get(
    "/i/{video_name}/metadata", operation_id="get_video_metadata", response_model=Optional[MetadataField]
)
async def get_video_metadata(
    current_user: CurrentUserOrDefault,
    video_name: str = PathParam(description="The name of video to get"),
) -> Optional[MetadataField]:
    _assert_video_read_access(video_name, current_user)
    try:
        return ApiDependencies.invoker.services.videos.get_metadata(video_name)
    except Exception:
        raise HTTPException(status_code=404)


@videos_router.get(
    "/i/{video_name}/workflow", operation_id="get_video_workflow", response_model=WorkflowAndGraphResponse
)
async def get_video_workflow(
    current_user: CurrentUserOrDefault,
    video_name: str = PathParam(description="The name of video whose workflow to get"),
) -> WorkflowAndGraphResponse:
    """Gets the workflow and graph saved with a generated video (mirrors the image route)."""
    _assert_video_read_access(video_name, current_user)
    try:
        workflow = ApiDependencies.invoker.services.videos.get_workflow(video_name)
        graph = ApiDependencies.invoker.services.videos.get_graph(video_name)
        return WorkflowAndGraphResponse(workflow=workflow, graph=graph)
    except Exception:
        raise HTTPException(status_code=404)


def _parse_range_header(range_header: str, file_size: int) -> Optional[tuple[int, int]]:
    """Parses an HTTP Range header of the form `bytes=START-END`. Returns inclusive (start, end)
    byte offsets, or None if the header is malformed or unsatisfiable."""
    match = re.match(r"^bytes=(\d*)-(\d*)$", range_header.strip())
    if match is None:
        return None
    start_str, end_str = match.group(1), match.group(2)
    if start_str == "" and end_str == "":
        return None
    if start_str == "":
        # suffix range: last N bytes
        try:
            suffix_len = int(end_str)
        except ValueError:
            return None
        if suffix_len == 0:
            return None
        start = max(file_size - suffix_len, 0)
        end = file_size - 1
    else:
        try:
            start = int(start_str)
        except ValueError:
            return None
        if end_str == "":
            end = file_size - 1
        else:
            try:
                end = int(end_str)
            except ValueError:
                return None
        if start > end or start >= file_size:
            return None
        end = min(end, file_size - 1)
    return start, end


@videos_router.get(
    "/i/{video_name}/full",
    operation_id="get_video_full",
    response_class=Response,
    responses={
        200: {"description": "Return the full video file", "content": {"video/mp4": {}}},
        206: {"description": "Return a byte-range of the video file", "content": {"video/mp4": {}}},
        404: {"description": "Video not found"},
    },
)
@videos_router.head(
    "/i/{video_name}/full",
    operation_id="get_video_full_head",
    response_class=Response,
    responses={
        200: {"description": "Return the full video file", "content": {"video/mp4": {}}},
        404: {"description": "Video not found"},
    },
)
async def get_video_full(
    request: Request,
    current_user: CurrentMediaUserOrDefault,
    video_name: str = PathParam(description="The name of video file to get"),
) -> Response:
    """Serves the video file with HTTP Range support so HTML5 <video> seek/scrub works.

    Browser media requests authenticate with the path-scoped HttpOnly cookie set at login.
    """
    _assert_video_read_access(video_name, current_user)
    try:
        path_str = ApiDependencies.invoker.services.videos.get_path(video_name, thumbnail=False)
    except Exception:
        raise HTTPException(status_code=404)

    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404)

    file_size = path.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")

    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": _get_video_cache_control(),
        "Content-Disposition": f'inline; filename="{video_name}"',
    }

    # HEAD: respond with metadata only.
    if request.method == "HEAD":
        return Response(
            status_code=200,
            media_type="video/mp4",
            headers={**common_headers, "Content-Length": str(file_size)},
        )

    if range_header is None:
        # Stream the file via sendfile() rather than reading it into RAM — multi-GB
        # MP4 downloads (clients without Range, CLI tools, CDN edge fetches) would
        # otherwise allocate a multi-GB Python bytes object per request.
        return FileResponse(
            path,
            media_type="video/mp4",
            headers=common_headers,
        )

    parsed = _parse_range_header(range_header, file_size)
    if parsed is None:
        # Unsatisfiable range.
        return Response(
            status_code=416,
            headers={**common_headers, "Content-Range": f"bytes */{file_size}"},
        )
    start, end = parsed
    length = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        # Read at most one chunk; clients ask for more via subsequent ranges.
        read_length = min(length, RANGE_CHUNK_SIZE)
        chunk = f.read(read_length)
        actual_end = start + len(chunk) - 1
    return Response(
        chunk,
        status_code=206,
        media_type="video/mp4",
        headers={
            **common_headers,
            "Content-Range": f"bytes {start}-{actual_end}/{file_size}",
            "Content-Length": str(len(chunk)),
        },
    )


@videos_router.get(
    "/i/{video_name}/thumbnail",
    operation_id="get_video_thumbnail",
    response_class=Response,
    responses={
        200: {"description": "Return the video thumbnail", "content": {"image/webp": {}}},
        404: {"description": "Video not found"},
    },
)
async def get_video_thumbnail(
    current_user: CurrentMediaUserOrDefault,
    video_name: str = PathParam(description="The name of thumbnail file to get"),
) -> Response:
    """Returns the first-frame WebP thumbnail of an authorized video."""
    _assert_video_read_access(video_name, current_user)
    try:
        path = ApiDependencies.invoker.services.videos.get_path(video_name, thumbnail=True)
    except Exception:
        raise HTTPException(status_code=404)
    # FileResponse stats the file lazily *after* the route returns, which means a missing
    # thumbnail surfaces as a server-side error rather than the documented 404. Check up
    # front so callers get the right status. Video saves are allowed without a thumbnail
    # (see video_files_disk.save), so this is a reachable path.
    if not Path(path).is_file():
        raise HTTPException(status_code=404)
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": _get_video_cache_control()},
    )


@videos_router.get("/i/{video_name}/urls", operation_id="get_video_urls", response_model=VideoUrlsDTO)
async def get_video_urls(
    current_user: CurrentUserOrDefault,
    video_name: str = PathParam(description="The name of the video whose URL to get"),
) -> VideoUrlsDTO:
    _assert_video_read_access(video_name, current_user)
    try:
        video_url = ApiDependencies.invoker.services.videos.get_url(video_name)
        thumbnail_url = ApiDependencies.invoker.services.videos.get_url(video_name, thumbnail=True)
        return VideoUrlsDTO(video_name=video_name, video_url=video_url, thumbnail_url=thumbnail_url)
    except Exception:
        raise HTTPException(status_code=404)


@videos_router.get("/", operation_id="list_video_dtos", response_model=OffsetPaginatedResults[VideoDTO])
async def list_video_dtos(
    current_user: CurrentUserOrDefault,
    video_origin: Optional[ResourceOrigin] = Query(default=None, description="The origin of videos to list."),
    categories: Optional[list[ImageCategory]] = Query(default=None, description="The categories of video to include."),
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate videos."),
    board_id: Optional[str] = Query(
        default=None,
        description="The board id to filter by. Use 'none' to find videos without a board.",
    ),
    offset: int = Query(default=0, description="The page offset"),
    limit: int = Query(default=10, description="The number of videos per page"),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred videos first"),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
) -> OffsetPaginatedResults[VideoDTO]:
    """Gets a list of video DTOs for the current user."""
    # Validate that the caller can read from this board. "none" is handled by the SQL layer.
    if board_id is not None and board_id != "none":
        _assert_board_read_access(board_id, current_user)

    return ApiDependencies.invoker.services.videos.get_many(
        offset,
        limit,
        starred_first,
        order_dir,
        video_origin,
        categories,
        is_intermediate,
        board_id,
        search_term,
        current_user.user_id,
        current_user.is_admin,
    )


@videos_router.get("/names", operation_id="get_video_names")
async def get_video_names(
    current_user: CurrentUserOrDefault,
    video_origin: Optional[ResourceOrigin] = Query(default=None, description="The origin of videos to list."),
    categories: Optional[list[ImageCategory]] = Query(default=None, description="The categories of video to include."),
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate videos."),
    board_id: Optional[str] = Query(
        default=None,
        description="The board id to filter by. Use 'none' to find videos without a board.",
    ),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred videos first"),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
) -> VideoNamesResult:
    """Gets ordered list of video names with metadata for optimistic updates."""
    # Validate that the caller can read from this board. "none" is handled by the SQL layer.
    if board_id is not None and board_id != "none":
        _assert_board_read_access(board_id, current_user)

    try:
        return ApiDependencies.invoker.services.videos.get_video_names(
            starred_first=starred_first,
            order_dir=order_dir,
            video_origin=video_origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=current_user.user_id,
            is_admin=current_user.is_admin,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get video names")


@videos_router.post("/star", operation_id="star_videos_in_list", response_model=StarredVideosResult)
async def star_videos_in_list(
    current_user: CurrentUserOrDefault,
    video_names: list[str] = Body(description="The list of names of videos to star", embed=True),
) -> StarredVideosResult:
    # Skip — but do not re-raise — auth failures so a foreign name mid-batch doesn't
    # discard the response payload for items that were already starred. Mirrors
    # delete_videos_from_list: re-raising turned partial successes into an error-shaped
    # response, so the client never invalidated caches for the videos that did change.
    starred_videos: set[str] = set()
    affected_boards: set[str] = set()
    for video_name in video_names:
        try:
            _assert_video_owner(video_name, current_user)
            updated = ApiDependencies.invoker.services.videos.update(
                video_name, changes=VideoRecordChanges(starred=True)
            )
            starred_videos.add(video_name)
            affected_boards.add(updated.board_id or "none")
        except HTTPException:
            continue
        except Exception:
            pass
    return StarredVideosResult(starred_videos=list(starred_videos), affected_boards=list(affected_boards))


@videos_router.post("/unstar", operation_id="unstar_videos_in_list", response_model=UnstarredVideosResult)
async def unstar_videos_in_list(
    current_user: CurrentUserOrDefault,
    video_names: list[str] = Body(description="The list of names of videos to unstar", embed=True),
) -> UnstarredVideosResult:
    # See star_videos_in_list: skip foreign names instead of re-raising mid-batch.
    unstarred_videos: set[str] = set()
    affected_boards: set[str] = set()
    for video_name in video_names:
        try:
            _assert_video_owner(video_name, current_user)
            updated = ApiDependencies.invoker.services.videos.update(
                video_name, changes=VideoRecordChanges(starred=False)
            )
            unstarred_videos.add(video_name)
            affected_boards.add(updated.board_id or "none")
        except HTTPException:
            continue
        except Exception:
            pass
    return UnstarredVideosResult(unstarred_videos=list(unstarred_videos), affected_boards=list(affected_boards))


class VideoBoardArg(BaseModel):
    board_id: str = Field(description="The id of the board to add or remove the video from")
    video_name: str = Field(description="The name of the video to add to / remove from the board")


@videos_router.post(
    "/board",
    operation_id="add_video_to_board",
    response_model=AddVideosToBoardResult,
)
async def add_video_to_board(
    current_user: CurrentUserOrDefault,
    arg: VideoBoardArg = Body(),
) -> AddVideosToBoardResult:
    _assert_board_write_access(arg.board_id, current_user)
    _assert_video_direct_owner(arg.video_name, current_user)
    try:
        # Capture the source board BEFORE mutating so the frontend can invalidate both
        # the old and new board caches. Mirrors add_image_to_board.
        old_board_id = (
            ApiDependencies.invoker.services.board_video_records.get_board_for_video(arg.video_name) or "none"
        )
        ApiDependencies.invoker.services.board_video_records.add_video_to_board(
            board_id=arg.board_id, video_name=arg.video_name
        )
        return AddVideosToBoardResult(
            added_videos=[arg.video_name],
            affected_boards=list({arg.board_id, old_board_id}),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to add video to board")


@videos_router.delete(
    "/board",
    operation_id="remove_video_from_board",
    response_model=RemoveVideosFromBoardResult,
)
async def remove_video_from_board(
    current_user: CurrentUserOrDefault,
    video_name: str = Body(description="The name of the video to remove from its board", embed=True),
) -> RemoveVideosFromBoardResult:
    # A video association can be removed by EITHER the direct video owner OR a user with
    # write access to the destination board (admin, board owner, or any contributor when the
    # board is Public). This mirrors remove_image_from_board and prevents a video from being
    # stranded when a non-owner uploads into a Public board that is later made Shared/Private:
    # without the board-write fallback, neither the uploader nor the board owner could
    # detach the video. See PR #9163 review.
    old_board_id = ApiDependencies.invoker.services.board_video_records.get_board_for_video(video_name)
    try:
        _assert_video_direct_owner(video_name, current_user)
    except HTTPException:
        if old_board_id is None:
            raise
        _assert_board_write_access(old_board_id, current_user)
    try:
        ApiDependencies.invoker.services.board_video_records.remove_video_from_board(video_name=video_name)
        return RemoveVideosFromBoardResult(
            removed_videos=[video_name],
            affected_boards=list({old_board_id or "none", "none"}),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to remove video from board")
