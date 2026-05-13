import re
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from fastapi import Body, HTTPException, Query, Request, Response, UploadFile
from fastapi import Path as PathParam
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.video_records.video_records_common import VideoNamesResult, VideoRecordChanges
from invokeai.app.services.videos.videos_common import (
    DeleteVideosResult,
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
    tmp = tempfile.NamedTemporaryFile(prefix="invokeai_upload_", suffix=".mp4", delete=False)
    tmp_path = Path(tmp.name)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        try:
            width, height, duration, fps = probe_video(tmp_path)
        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=415, detail="Failed to read video")

        try:
            video_dto = ApiDependencies.invoker.services.videos.create(
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

            response.status_code = 201
            response.headers["Location"] = video_dto.video_url
            return video_dto
        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Failed to create video")
    finally:
        # If create() succeeded the file was moved; this unlink is a no-op then.
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

    deleted_videos: set[str] = set()
    affected_boards: set[str] = set()
    try:
        video_dto = ApiDependencies.invoker.services.videos.get_dto(video_name)
        board_id = video_dto.board_id or "none"
        ApiDependencies.invoker.services.videos.delete(video_name)
        deleted_videos.add(video_name)
        affected_boards.add(board_id)
    except Exception:
        pass

    return DeleteVideosResult(
        deleted_videos=list(deleted_videos),
        affected_boards=list(affected_boards),
    )


@videos_router.post("/delete", operation_id="delete_videos_from_list", response_model=DeleteVideosResult)
async def delete_videos_from_list(
    current_user: CurrentUserOrDefault,
    video_names: list[str] = Body(description="The list of names of videos to delete", embed=True),
) -> DeleteVideosResult:
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
            raise
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
    video_name: str = PathParam(description="The name of video file to get"),
) -> Response:
    """Serves the video file with HTTP Range support so HTML5 <video> seek/scrub works.

    Like the image equivalent, this endpoint is intentionally unauthenticated because browsers
    load videos via <video src> tags which cannot send Bearer tokens. Video names are UUIDs,
    providing security through unguessability.
    """
    try:
        path_str = ApiDependencies.invoker.services.videos.get_path(video_name)
    except Exception:
        raise HTTPException(status_code=404)

    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404)

    file_size = path.stat().st_size
    range_header = request.headers.get("range") or request.headers.get("Range")

    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": f"max-age={VIDEO_MAX_AGE}",
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
        with open(path, "rb") as f:
            content = f.read()
        return Response(
            content,
            media_type="video/mp4",
            headers={**common_headers, "Content-Length": str(file_size)},
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
    video_name: str = PathParam(description="The name of thumbnail file to get"),
) -> Response:
    """Returns the first-frame WebP thumbnail of a video. Unauthenticated; UUIDs provide unguessability."""
    try:
        path = ApiDependencies.invoker.services.videos.get_path(video_name, thumbnail=True)
        with open(path, "rb") as f:
            content = f.read()
        response = Response(content, media_type="image/webp")
        response.headers["Cache-Control"] = f"max-age={VIDEO_MAX_AGE}"
        return response
    except Exception:
        raise HTTPException(status_code=404)


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
            raise
        except Exception:
            pass
    return StarredVideosResult(starred_videos=list(starred_videos), affected_boards=list(affected_boards))


@videos_router.post("/unstar", operation_id="unstar_videos_in_list", response_model=UnstarredVideosResult)
async def unstar_videos_in_list(
    current_user: CurrentUserOrDefault,
    video_names: list[str] = Body(description="The list of names of videos to unstar", embed=True),
) -> UnstarredVideosResult:
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
            raise
        except Exception:
            pass
    return UnstarredVideosResult(unstarred_videos=list(unstarred_videos), affected_boards=list(affected_boards))


class VideoBoardArg(BaseModel):
    board_id: str = Field(description="The id of the board to add or remove the video from")
    video_name: str = Field(description="The name of the video to add to / remove from the board")


@videos_router.post(
    "/board",
    operation_id="add_video_to_board",
    response_model=VideoDTO,
)
async def add_video_to_board(
    current_user: CurrentUserOrDefault,
    arg: VideoBoardArg = Body(),
) -> VideoDTO:
    _assert_video_owner(arg.video_name, current_user)
    try:
        ApiDependencies.invoker.services.board_video_records.add_video_to_board(
            board_id=arg.board_id, video_name=arg.video_name
        )
        return ApiDependencies.invoker.services.videos.get_dto(arg.video_name)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to add video to board")


@videos_router.delete(
    "/board",
    operation_id="remove_video_from_board",
    response_model=VideoDTO,
)
async def remove_video_from_board(
    current_user: CurrentUserOrDefault,
    video_name: str = Body(description="The name of the video to remove from its board", embed=True),
) -> VideoDTO:
    _assert_video_owner(video_name, current_user)
    try:
        ApiDependencies.invoker.services.board_video_records.remove_video_from_board(video_name=video_name)
        return ApiDependencies.invoker.services.videos.get_dto(video_name)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to remove video from board")
