import json
import os
import re
import tempfile
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Optional

from fastapi import Body, HTTPException, Query, Request, Response
from fastapi import Path as PathParam
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from PIL import Image as PILImage
from pydantic import BaseModel, Field, StringConstraints, ValidationError
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from invokeai.app.api.auth_dependencies import CurrentMediaUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.extract_metadata import extract_metadata_from_video
from invokeai.app.api.routers._access import (
    assert_board_read_access as _assert_board_read_access,
)
from invokeai.app.api.routers._access import (
    assert_board_write_access as _assert_board_write_access,
)
from invokeai.app.api.routers._access import assert_project_owned
from invokeai.app.api.routers._access import (
    assert_video_owner as _assert_video_owner,
)
from invokeai.app.api.routers._access import (
    assert_video_read_access as _assert_video_read_access,
)
from invokeai.app.api.routers._access import (
    board_share_recipients as _board_share_recipients,
)
from invokeai.app.api.routers._limits import MAX_COPY_BATCH_SIZE
from invokeai.app.api.routers.images import WorkflowAndGraphResponse
from invokeai.app.invocations.fields import MetadataField, MetadataFieldValidator
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin, is_gallery_category
from invokeai.app.services.shared.pagination import MAX_PAGE_SIZE, OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.video_records.video_records_common import (
    VideoNamesResult,
    VideoRecordChanges,
    VideoRecordNotFoundException,
)
from invokeai.app.services.videos.videos_common import (
    AddVideosToBoardResult,
    DeleteVideosResult,
    RemoveVideosFromBoardResult,
    StarredVideosResult,
    UnstarredVideosResult,
    VideoDTO,
    VideoUrlsDTO,
)
from invokeai.app.util.mp4_metadata import read_ftyp_major_brand
from invokeai.app.util.video_ingest import VideoIngestError, ingest_media_to_mp4, probe_media_streams
from invokeai.app.util.video_thumbnails import (
    VideoDecodeTimeoutError,
    extract_representative_video_frame,
    probe_video_with_codec,
)

videos_router = APIRouter(prefix="/v1/videos", tags=["videos"])

# Videos are immutable; set a high max-age (1 year)
VIDEO_MAX_AGE = 31536000

# The names service emits `{uuid}.mp4` unconditionally and files are served as video/mp4,
# so everything *stored* must genuinely be H.264 MP4 — storing foreign containers under a
# .mp4 name silently broke browser playback, which is why uploads were historically
# restricted to already-compliant files. Uploads are now normalized at ingest instead
# (see invokeai.app.util.video_ingest): H.264 in a foreign container is losslessly
# remuxed, other codecs (iPhone HEVC, ProRes, VP9) are transcoded, and audio-only files
# are wrapped into waveform videos so audio clips flow through the video pipeline
# (gallery, trim, audio-only reference conditioning) without a first-class audio type.
ACCEPTED_VIDEO_MIME_PREFIXES = ("video/",)
# The extension lists are the fallback for uploads whose type the browser could not
# determine (they arrive as application/octet-stream); the MIME prefixes above accept the
# ordinary case. Everything here is demuxable and decodable by the bundled ffmpeg — ASF
# (Windows Media) included — and lands as H.264/AAC MP4 through the ingest converter.
ACCEPTED_VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
    ".mkv",
    ".avi",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".wmv",
    ".asf",
)
ACCEPTED_AUDIO_MIME_PREFIXES = ("audio/",)
ACCEPTED_AUDIO_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".aiff",
    ".aif",
    ".wma",
)

# Per-chunk size for HTTP Range responses (1 MB)
RANGE_CHUNK_SIZE = 1024 * 1024

# Coarse per-upload size cap, enforced against the file part as it streams in. Generous
# because Wan-generated MP4s for long sequences can run into the hundreds of megabytes;
# the goal is to prevent a single client from exhausting RAM/disk, not to be a content policy.
MAX_UPLOAD_SIZE = 1024 * 1024 * 1024  # 1 GB
# Pre-parse ingress cap enforced by VideoUploadLimitASGIMiddleware, applied to the whole
# request body *before* the upload route parses it. Slightly larger than MAX_UPLOAD_SIZE
# to allow for multipart framing and the metadata form field.
MAX_UPLOAD_REQUEST_SIZE = MAX_UPLOAD_SIZE + 10 * 1024 * 1024
# The `metadata` form field is a stringified JSON dict; it is buffered in memory while the
# body streams, so it gets its own (generous) cap.
MAX_UPLOAD_METADATA_SIZE = 1024 * 1024
# Global bound on concurrent video uploads — each in-flight upload holds one full-size copy
# of the file in temp storage until probe/thumbnail/create finish with it, and a second (also
# capped at MAX_UPLOAD_SIZE) while the ingest converter is writing its output or, inside
# create(), while the metadata remux writes its replacement next to the stored file.
MAX_CONCURRENT_VIDEO_UPLOADS = 2
# Per-user bound (multiuser mode only): keeps one tenant's slow uploads from holding
# every global slot and starving the other users into 429s.
MAX_CONCURRENT_VIDEO_UPLOADS_PER_USER = 1
MAX_VIDEO_BATCH_SIZE = 1000
VideoName = Annotated[str, StringConstraints(max_length=255)]


class VideoNamesBatch(BaseModel):
    video_names: list[VideoName] = Field(
        max_length=MAX_VIDEO_BATCH_SIZE,
        description="The list of video names to process",
    )


def _get_video_cache_control() -> str:
    if ApiDependencies.invoker.services.configuration.multiuser:
        return "private, no-store"
    return f"max-age={VIDEO_MAX_AGE}"


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


def _classify_upload(filename: Optional[str], content_type: Optional[str]) -> Optional[str]:
    """Returns 'video' or 'audio' for an accepted upload, None for a rejected one.

    The classification is advisory (it picks the 415 message and nothing else): the
    ingest path probes actual stream content, so a mislabeled file still converts
    correctly or fails with a clear error.
    """
    if content_type:
        if content_type.startswith(ACCEPTED_VIDEO_MIME_PREFIXES):
            return "video"
        if content_type.startswith(ACCEPTED_AUDIO_MIME_PREFIXES):
            return "audio"
    if filename:
        name = filename.lower()
        if name.endswith(ACCEPTED_VIDEO_EXTENSIONS):
            return "video"
        if name.endswith(ACCEPTED_AUDIO_EXTENSIONS):
            return "audio"
    return None


def _with_media_origin(metadata: Optional[str], origin: str) -> str:
    """Merges a `media_origin` marker into the upload's (already-validated) metadata JSON."""
    parsed = json.loads(metadata) if metadata else {}
    parsed.setdefault("media_origin", origin)
    return json.dumps(parsed)


class _VideoUploadStreamParser:
    """Parses the multipart upload body, writing the file part straight to `destination`.

    Declaring `file: UploadFile` on the route makes Starlette parse the body into its own
    spooled temp file first, so the route's copy to a named temp file was a SECOND
    full-size copy: every in-flight upload occupied up to 2 x MAX_UPLOAD_SIZE of temp
    storage (x MAX_CONCURRENT_VIDEO_UPLOADS) for the whole probe/thumbnail/create phase.
    The spool's path cannot be reused instead — once rolled over it is an unlinked
    anonymous file, and both ffmpeg and videos.create need a real path.

    Parsing the stream ourselves keeps exactly one copy of the upload on disk (the ingest
    step, when a file needs it, adds its own output file on top). It also lets the file-type
    and size checks fire while the body is still arriving, rather than after the whole thing
    has been written somewhere.

    Callbacks run inside `MultipartParser.write`, which the route calls in a worker thread
    — the disk writes must not happen on the event loop.
    """

    def __init__(self, destination: BinaryIO) -> None:
        self._destination = destination
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._headers: dict[bytes, bytes] = {}
        self._part_name: Optional[bytes] = None
        # A bytearray, not a list of per-callback chunks: the parser hands the metadata
        # field over in whatever pieces the client sent it, and one `bytes` object per
        # piece would let a client that dribbles the field in 2-byte pieces hold ~22x the
        # size cap in memory. `len(self._metadata)` is the retained size, so the cap
        # below bounds memory, not just payload.
        self._metadata = bytearray()
        self.filename: Optional[str] = None
        self.content_type: Optional[str] = None
        self.upload_kind: Optional[str] = None
        self.metadata: Optional[str] = None
        self.file_size = 0
        self.saw_file_part = False
        # `MultipartParser.finalize()` is a no-op that does NOT check the parser reached its
        # end state, so a body that stops after the file bytes — truncated upload, aborted
        # client, or the parser's own silent `max_size` truncation — would otherwise look
        # exactly like a complete one and get probed and persisted. `on_end` fires only when
        # the closing `--boundary--` is parsed, so it is the proof of a complete body.
        self.saw_end = False

    @property
    def callbacks(self) -> dict[str, object]:
        return {
            "on_part_begin": self._on_part_begin,
            "on_part_data": self._on_part_data,
            "on_part_end": self._on_part_end,
            "on_header_field": self._on_header_field,
            "on_header_value": self._on_header_value,
            "on_header_end": self._on_header_end,
            "on_headers_finished": self._on_headers_finished,
            "on_end": self._on_end,
        }

    def _on_part_begin(self) -> None:
        self._headers = {}
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._part_name = None
        self._metadata = bytearray()

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field.extend(data[start:end])

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value.extend(data[start:end])

    def _on_header_end(self) -> None:
        self._headers[bytes(self._header_field).lower()] = bytes(self._header_value)
        self._header_field = bytearray()
        self._header_value = bytearray()

    def _on_headers_finished(self) -> None:
        _, options = parse_options_header(self._headers.get(b"content-disposition", b""))
        self._part_name = options.get(b"name")
        if self._part_name != b"file":
            return
        if self.saw_file_part:
            raise HTTPException(status_code=422, detail="Expected exactly one video file")
        self.saw_file_part = True
        filename = options.get(b"filename")
        self.filename = filename.decode("utf-8", errors="replace") if filename is not None else None
        content_type = self._headers.get(b"content-type")
        self.content_type = content_type.decode("latin-1") if content_type is not None else None
        # Reject the wrong kind of file before any of its bytes reach the disk.
        self.upload_kind = _classify_upload(self.filename, self.content_type)
        if self.upload_kind is None:
            raise HTTPException(status_code=415, detail="Not a supported video or audio file")

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self._part_name == b"file":
            self.file_size += len(chunk)
            if self.file_size > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Video upload exceeds maximum size ({MAX_UPLOAD_SIZE} bytes)",
                )
            self._destination.write(chunk)
        elif self._part_name == b"metadata":
            if len(self._metadata) + len(chunk) > MAX_UPLOAD_METADATA_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"Video metadata exceeds maximum size ({MAX_UPLOAD_METADATA_SIZE} bytes)",
                )
            self._metadata.extend(chunk)
        # Any other field is dropped rather than buffered: an unknown part must not be a
        # way to make the server hold arbitrary bytes in memory.

    def _on_part_end(self) -> None:
        if self._part_name == b"metadata":
            try:
                self.metadata = self._metadata.decode("utf-8")
            except UnicodeDecodeError as error:
                raise HTTPException(status_code=422, detail="Metadata must be UTF-8 encoded") from error
        self._part_name = None
        self._metadata = bytearray()

    def _on_end(self) -> None:
        self.saw_end = True


async def _stream_video_upload(request: Request, destination: BinaryIO) -> _VideoUploadStreamParser:
    """Streams the request body through the multipart parser into `destination`."""
    media_type, options = parse_options_header(request.headers.get("content-type", ""))
    boundary = options.get(b"boundary")
    # Content-Type is case-insensitive (RFC 7231) and parse_options_header preserves case.
    if media_type.lower() != b"multipart/form-data" or boundary is None:
        raise HTTPException(status_code=422, detail="Expected a multipart/form-data video upload")

    parser_state = _VideoUploadStreamParser(destination)
    # max_size is the pre-parse ingress cap the middleware already enforces; repeating it
    # here bounds the parser itself for any path that reaches it directly. The parser
    # silently *truncates* past it rather than erroring, so on such a path the body would
    # look short rather than rejected — the saw_end check below is what catches that.
    parser = MultipartParser(boundary, parser_state.callbacks, max_size=MAX_UPLOAD_REQUEST_SIZE)
    try:
        async for chunk in request.stream():
            # Parsing writes to disk, so it belongs in the thread pool alongside the rest of
            # the blocking upload work.
            await run_in_threadpool(parser.write, chunk)
        await run_in_threadpool(parser.finalize)
    except MultipartParseError as error:
        # A malformed body is the client's fault, not a server error.
        raise HTTPException(status_code=422, detail="Malformed multipart body") from error
    except ClientDisconnect as error:
        # The client went away, or VideoUploadLimitASGIMiddleware cut it off for exceeding the
        # ingress cap, the idle timeout or the duration cap. Left to propagate this surfaces as
        # a 500 raised above the middleware, which both misreports a client-side abort as a
        # server fault and bypasses the middleware's connection-close handling.
        raise HTTPException(status_code=400, detail="Upload ended before the body was complete") from error

    if not parser_state.saw_end:
        # No closing boundary: the body was truncated. Proceeding would probe and persist a
        # partial file whenever the truncated bytes happen to survive the MP4 checks.
        raise HTTPException(status_code=422, detail="Incomplete multipart body")
    if not parser_state.saw_file_part:
        raise HTTPException(status_code=422, detail="Expected a video file in the upload")
    return parser_state


def _is_mp4_file(path: Path) -> bool:
    try:
        major_brand = read_ftyp_major_brand(path)
    except OSError:
        return False
    return major_brand is not None and major_brand != b"qt  "


def _probe_decodable_video(path: Path) -> tuple[tuple[int, int, float, Optional[float]], Optional[PILImage.Image]]:
    """Probes metadata and proves the video has a decodable frame.

    Returns the metadata plus the decoded frame so the save path can reuse it as the
    thumbnail source instead of spawning another decode worker. The frame comes from the
    thumbnail seek ladder (see extract_representative_video_frame): the first informative
    frame found starting ~1s in, else the best-scoring one, with frame 0 as the last rung.
    Acceptance is thereby slightly WIDER than a frame-0 check: a file whose frame 0 is
    corrupt but whose later frame decodes is accepted rather than 415'd. A decode timeout
    is contention on a loaded server, not evidence the video is bad — probe_video already
    succeeded — so it yields (metadata, None) and the upload proceeds, with save-time
    thumbnail extraction as the backstop.
    """
    width, height, duration, fps, codec = probe_video_with_codec(path)
    if codec is None or codec.lower() not in {"h264", "avc", "avc1", "libx264"}:
        raise ValueError("Video must use a browser-compatible H.264/AVC codec")
    metadata = (width, height, duration, fps)
    try:
        first_frame = extract_representative_video_frame(path, duration, fps, raise_on_timeout=True)
    except VideoDecodeTimeoutError:
        return metadata, None
    if first_frame is None:
        raise ValueError("Video has no decodable frame")
    return metadata, first_frame


# The body of a video upload is parsed by hand (see _stream_video_upload) so the file lands in
# exactly one temp file, which means FastAPI cannot infer the request schema from the route
# signature. This spells out the same multipart body the `file` + `metadata` parameters used to
# generate, so the documented contract is unchanged. Every route that ingests through
# `ingest_uploaded_video` documents its body with it.
VIDEO_UPLOAD_OPENAPI_EXTRA: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "properties": {
                        # Key order and shape mirror what FastAPI generates for the sibling
                        # upload routes (see Body_upload_image), so the documented contract
                        # stays byte-identical to what `file` + `metadata` produced.
                        "file": {
                            "type": "string",
                            "contentMediaType": "application/octet-stream",
                            "title": "File",
                        },
                        "metadata": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "title": "Metadata",
                            "description": "The metadata to associate with the video, must be a stringified JSON dict",
                        },
                    },
                    "type": "object",
                    "required": ["file"],
                    "title": "Body_upload_video",
                }
            }
        },
    }
}


async def ingest_uploaded_video(
    request: Request,
    current_user: CurrentUserOrDefault,
    *,
    video_category: ImageCategory,
    is_intermediate: bool,
    board_id: Optional[str],
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> VideoDTO:
    """Stream a multipart video upload from `request` into the gallery and announce it.

    Callers must be routes that `VideoUploadLimitASGIMiddleware` matches: the ingress size cap
    and the concurrency slots are enforced there, before this runs.
    """
    # Check board access for uploads to a specific board.
    board = await run_in_threadpool(_assert_board_write_access, board_id, current_user)
    await run_in_threadpool(assert_project_owned, project_id, current_user)

    # Stream the upload straight into a tmp file so we can probe it and then hand its path
    # to the service. Reading the full body into memory first risked exhausting RAM on
    # multi-GB uploads; the parser streams it instead and enforces a hard size cap as the
    # bytes arrive. Filesystem writes, container validation, ffmpeg probing, and thumbnail
    # extraction are all blocking — run them in the thread pool so a slow (or hostile) file
    # can't stall the event loop and every other API request with it.
    tmp = tempfile.NamedTemporaryFile(prefix="invokeai_upload_", suffix=".mp4", delete=False)
    tmp_path = Path(tmp.name)
    try:
        upload = await _stream_video_upload(request, tmp)
        tmp.close()

        upload_kind = upload.upload_kind
        if upload.metadata is not None:
            try:
                MetadataFieldValidator.validate_json(upload.metadata)
            except ValidationError as e:
                raise HTTPException(status_code=422, detail="Metadata must be a JSON object") from e

        # An MP4 that InvokeAI produced carries its metadata, workflow and graph as keyed metadata, the way a
        # PNG carries text chunks. Read them from the file as uploaded: the ingest remux below does not
        # preserve keyed metadata. As for images, client-supplied metadata wins over the embedded copy, while
        # workflow and graph are never client-overridable.
        extracted = await run_in_threadpool(
            extract_metadata_from_video,
            tmp_path,
            upload.metadata,
            None,
            None,
            ApiDependencies.invoker.services.logger,
        )
        metadata = extracted.invokeai_metadata
        if metadata is not None and len(metadata.encode("utf-8")) > MAX_UPLOAD_METADATA_SIZE:
            # The form field is capped while it streams; an embedded record gets the same bound
            # before it reaches the database. Workflow and graph legitimately run larger.
            ApiDependencies.invoker.services.logger.info("Ignoring oversized metadata embedded in uploaded video")
            metadata = None

        # Already-compliant H.264 MP4s skip conversion (their only rewrite is the metadata
        # remux inside create(), and only when there is a record to embed);
        # everything else — foreign containers, foreign codecs, audio-only files — is
        # normalized by the ingest converter. The conversion runs inside this upload's
        # concurrency slot: a long HEVC transcode holds one of MAX_CONCURRENT_VIDEO_UPLOADS
        # slots for its duration, which is the intended backpressure.
        needs_ingest = True
        if upload_kind == "video" and await run_in_threadpool(_is_mp4_file, tmp_path):
            try:
                container_probe = await run_in_threadpool(probe_media_streams, tmp_path)
            except VideoIngestError as e:
                raise HTTPException(status_code=415, detail=str(e))
            # Both codecs must already be browser-safe to skip ingest: an mp4-family
            # container can legally carry h264 video with AMR/opus/mp3/ac3 audio (e.g.
            # older Android .3gp camera files), which browsers render as silent video.
            # The remux branch stream-copies the h264 and normalizes only the audio.
            needs_ingest = container_probe.video_codec != "h264" or container_probe.audio_codec not in (None, "aac")

        if needs_ingest:
            converted = tempfile.NamedTemporaryFile(prefix="invokeai_ingest_", suffix=".mp4", delete=False)
            converted_path = Path(converted.name)
            converted.close()
            try:
                ingest_action = await run_in_threadpool(
                    ingest_media_to_mp4, tmp_path, converted_path, max_output_bytes=MAX_UPLOAD_SIZE
                )
                await run_in_threadpool(os.replace, converted_path, tmp_path)
            except VideoIngestError as e:
                ApiDependencies.invoker.services.logger.info(f"Video upload ingest failed: {e}")
                raise HTTPException(status_code=415, detail=str(e))
            finally:
                converted_path.unlink(missing_ok=True)
            if ingest_action == "audio_wrap":
                # Mark wrapped audio uploads so clients can treat them as audio clips
                # (e.g. defaulting an audio-only reference conditioning mode).
                metadata = _with_media_origin(metadata, "audio_upload")

        try:
            (width, height, duration, fps), first_frame = await run_in_threadpool(_probe_decodable_video, tmp_path)
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
                    first_frame=first_frame,
                    video_origin=ResourceOrigin.EXTERNAL,
                    video_category=video_category,
                    session_id=session_id,
                    board_id=board_id,
                    metadata=metadata,
                    workflow=extracted.invokeai_workflow,
                    graph=extracted.invokeai_graph,
                    is_intermediate=is_intermediate,
                    user_id=current_user.user_id,
                    project_id=project_id,
                )
            )
        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Failed to create video")

        if not is_intermediate and is_gallery_category(video_category):
            shared_user_ids = await run_in_threadpool(_board_share_recipients, board)
            ApiDependencies.invoker.services.events.emit_video_uploaded(
                video_dto, user_id=current_user.user_id, board=board, shared_user_ids=shared_user_ids
            )

        return video_dto
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


@videos_router.post(
    "/upload",
    operation_id="upload_video",
    responses={
        201: {"description": "The video was uploaded successfully"},
        415: {"description": "Video upload failed"},
    },
    status_code=201,
    response_model=VideoDTO,
    openapi_extra=VIDEO_UPLOAD_OPENAPI_EXTRA,
)
async def upload_video(
    current_user: CurrentUserOrDefault,
    request: Request,
    response: Response,
    video_category: ImageCategory = Query(description="The category of the video"),
    is_intermediate: bool = Query(description="Whether this is an intermediate video"),
    board_id: Optional[str] = Query(default=None, description="The board to add this video to, if any"),
    session_id: Optional[str] = Query(default=None, description="The session ID associated with this upload, if any"),
    project_id: Optional[str] = Query(
        default=None,
        min_length=1,
        max_length=255,
        description="The caller's project this upload originates in, if any; recorded for intermediates cleanup",
    ),
) -> VideoDTO:
    """Uploads a video for the current user."""
    video_dto = await ingest_uploaded_video(
        request,
        current_user,
        video_category=video_category,
        is_intermediate=is_intermediate,
        board_id=board_id,
        session_id=session_id,
        project_id=project_id,
    )
    response.status_code = 201
    response.headers["Location"] = video_dto.video_url
    return video_dto


# Declared sync (`def`, not `async def`) so FastAPI runs it in the threadpool: every call
# below is blocking SQLite/filesystem work, which would stall the event loop — and with it
# every other request and socket event — for the duration of the delete. The batch
# siblings below are sync for the same reason.
@videos_router.delete("/i/{video_name}", operation_id="delete_video", response_model=DeleteVideosResult)
def delete_video(
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
        failed_videos=[],
        affected_boards=[board_id],
    )


@videos_router.post("/delete", operation_id="delete_videos_from_list", response_model=DeleteVideosResult)
def delete_videos_from_list(
    current_user: CurrentUserOrDefault,
    batch: VideoNamesBatch,
) -> DeleteVideosResult:
    # Skip — but do not re-raise — auth failures so a foreign name mid-batch doesn't
    # discard the response payload for items the caller had already legitimately deleted.
    # Without this, the client cache never learns about the partial successes and the
    # already-deleted records reappear in the UI until the next full refresh.
    #
    # HTTPException here means an intentional skip (a foreign name, or a 404 because a
    # concurrent session deleted it first) — not a failed deletion. Reporting those in
    # failed_videos would toast a spurious "could not be deleted" warning; the images
    # endpoints skip them silently, and mixed selections must behave consistently.
    deleted_videos: set[str] = set()
    failed_videos: set[str] = set()
    affected_boards: set[str] = set()
    # Dedup while preserving order: a name repeated in the request would otherwise be
    # processed twice, and the second pass's not-found error would land the same name
    # in both deleted_videos and failed_videos.
    for video_name in dict.fromkeys(batch.video_names):
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
            failed_videos.add(video_name)
    return DeleteVideosResult(
        deleted_videos=list(deleted_videos),
        failed_videos=list(failed_videos),
        affected_boards=list(affected_boards),
    )


@videos_router.delete("/uncategorized", operation_id="delete_uncategorized_videos", response_model=DeleteVideosResult)
def delete_uncategorized_videos(
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
    failed_videos: set[str] = set()
    affected_boards: set[str] = set()
    for video_name in names_result.video_names:
        try:
            _assert_video_owner(video_name, current_user)
            ApiDependencies.invoker.services.videos.delete(video_name)
            deleted_videos.add(video_name)
            affected_boards.add("none")
        except HTTPException:
            # Skip videos not owned by the current user — an intentional skip, not a
            # failed deletion, so it must not be reported (and toasted) as one.
            continue
        except Exception:
            failed_videos.add(video_name)
    return DeleteVideosResult(
        deleted_videos=list(deleted_videos),
        failed_videos=list(failed_videos),
        affected_boards=list(affected_boards),
    )


# Sync for the same reason as delete_video: the update is a blocking SQLite write.
@videos_router.patch("/i/{video_name}", operation_id="update_video", response_model=VideoDTO)
def update_video(
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
def get_video_dto(
    current_user: CurrentUserOrDefault,
    video_name: str = PathParam(description="The name of video to get"),
) -> VideoDTO:
    _assert_video_read_access(video_name, current_user)
    try:
        return ApiDependencies.invoker.services.videos.get_dto(video_name)
    except VideoRecordNotFoundException:
        # See get_image_dto: this is the 404 a workflow's video field drops its reference on,
        # so only a genuinely missing record may produce it.
        raise HTTPException(status_code=404)


@videos_router.get(
    "/i/{video_name}/metadata", operation_id="get_video_metadata", response_model=Optional[MetadataField]
)
def get_video_metadata(
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
def get_video_workflow(
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
    if file_size <= 0:
        # No byte range is satisfiable against an empty file (a suffix range would
        # otherwise "satisfy" with the invalid pair (0, -1)).
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
def get_video_full(
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

    # Open once and serve every branch from the fd. Deletion stages files away via an
    # atomic rename, so any later path-based stat/open — including FileResponse's lazy
    # open after the route returns — races with a concurrent delete and surfaces as an
    # uncontrolled 500. An open fd is immune: the data stays readable until the handle
    # closes, even after the path is gone.
    video_file: Optional[BinaryIO] = None
    try:
        video_file = open(path_str, "rb")
    except OSError:
        raise HTTPException(status_code=404)

    try:
        file_size = os.fstat(video_file.fileno()).st_size
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
            # Stream from the already-open fd in bounded chunks rather than reading the
            # file into RAM — multi-GB MP4 downloads (clients without Range, CLI tools,
            # CDN edge fetches) would otherwise allocate a multi-GB bytes object. This
            # trades FileResponse's sendfile() for delete-race immunity.
            handle = video_file
            video_file = None  # ownership moves to the streaming iterator

            def iter_video() -> Iterator[bytes]:
                try:
                    while chunk := handle.read(RANGE_CHUNK_SIZE):
                        yield chunk
                finally:
                    handle.close()

            return StreamingResponse(
                iter_video(),
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
        video_file.seek(start)
        # Read at most one chunk; clients ask for more via subsequent ranges.
        read_length = min(length, RANGE_CHUNK_SIZE)
        chunk = video_file.read(read_length)
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
    finally:
        if video_file is not None:
            video_file.close()


@videos_router.get(
    "/i/{video_name}/thumbnail",
    operation_id="get_video_thumbnail",
    response_class=Response,
    responses={
        200: {"description": "Return the video thumbnail", "content": {"image/webp": {}}},
        404: {"description": "Video not found"},
    },
)
def get_video_thumbnail(
    current_user: CurrentMediaUserOrDefault,
    video_name: str = PathParam(description="The name of thumbnail file to get"),
) -> Response:
    """Returns the WebP thumbnail of an authorized video."""
    _assert_video_read_access(video_name, current_user)
    try:
        path = ApiDependencies.invoker.services.videos.get_path(video_name, thumbnail=True)
    except Exception:
        raise HTTPException(status_code=404)
    try:
        with open(path, "rb") as thumbnail_file:
            thumbnail = thumbnail_file.read()
    except OSError:
        raise HTTPException(status_code=404)

    return Response(
        thumbnail,
        media_type="image/webp",
        headers={"Cache-Control": _get_video_cache_control()},
    )


@videos_router.get("/i/{video_name}/urls", operation_id="get_video_urls", response_model=VideoUrlsDTO)
def get_video_urls(
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
def list_video_dtos(
    current_user: CurrentUserOrDefault,
    video_origin: Optional[ResourceOrigin] = Query(default=None, description="The origin of videos to list."),
    categories: Optional[list[ImageCategory]] = Query(default=None, description="The categories of video to include."),
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate videos."),
    board_id: Optional[str] = Query(
        default=None,
        description="The board id to filter by. Use 'none' to find videos without a board.",
    ),
    # Bounds matter: these flow verbatim into SQL, and a negative LIMIT means
    # *unlimited* in SQLite — one request would materialize every video row.
    offset: int = Query(default=0, ge=0, description="The page offset"),
    limit: int = Query(default=10, ge=0, le=MAX_PAGE_SIZE, description="The number of videos per page"),
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


@videos_router.get("/names", operation_id="get_video_names", deprecated=True)
def get_video_names(
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
    """Gets ordered list of video names with metadata for optimistic updates.

    Deprecated: use `GET /v1/gallery/item_names`, which returns images and videos interleaved
    in one ordered list. This video-only endpoint predates the polymorphic gallery.
    """
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


class CopiedVideo(BaseModel):
    source_video_name: str = Field(description="The video that was copied")
    video_name: str = Field(description="The name assigned to the copy")


class CopyVideosResult(BaseModel):
    copied: list[CopiedVideo] = Field(description="The copies that were made, in request order")
    failed: list[str] = Field(description="The source video names that could not be copied")


@videos_router.post("/copy", operation_id="copy_videos_to_board", response_model=CopyVideosResult)
def copy_videos_to_board(
    current_user: CurrentUserOrDefault,
    video_names: list[str] = Body(description="The names of the videos to copy", max_length=MAX_COPY_BATCH_SIZE),
    board_id: Optional[str] = Body(default=None, description="The board to put the copies on, if any"),
) -> CopyVideosResult:
    """Copies videos, optionally onto a board, and returns the new names.

    The image twin of this route explains why copies rather than shared references: `board_videos`
    keys on `video_name`, so one video sits on exactly one board.

    Blocking work (the service copies the file off disk) runs on FastAPI's threadpool by virtue of
    this being a sync `def`, so a large batch cannot stall the event loop.

    `move_source=False` is load-bearing, not defensive: `create` consumes the path it is given,
    because every other caller hands it a temp file. Here the path is the *source's own* file.

    Read access is enough to copy, which means a video on a board shared with you can be copied
    into something you own, and the copy outlives the share. That is deliberate — it is what makes
    a shared board usable as a source — but it is a real widening of what "read-only" means, so it
    is stated rather than left to be discovered.
    """
    _assert_board_write_access(board_id, current_user)

    copied: list[CopiedVideo] = []
    failed: list[str] = []

    for video_name in video_names:
        try:
            _assert_video_read_access(video_name, current_user)
            video_dto = ApiDependencies.invoker.services.videos.copy(
                source_video_name=video_name,
                board_id=board_id,
                user_id=current_user.user_id,
            )
            copied.append(CopiedVideo(source_video_name=video_name, video_name=video_dto.video_name))
        except Exception:
            ApiDependencies.invoker.services.logger.error(f"Failed to copy video {video_name}", exc_info=True)
            failed.append(video_name)

    return CopyVideosResult(copied=copied, failed=failed)


@videos_router.post("/star", operation_id="star_videos_in_list", response_model=StarredVideosResult)
def star_videos_in_list(
    current_user: CurrentUserOrDefault,
    batch: VideoNamesBatch,
) -> StarredVideosResult:
    # Skip — but do not re-raise — auth failures so a foreign name mid-batch doesn't
    # discard the response payload for items that were already starred. Mirrors
    # delete_videos_from_list: re-raising turned partial successes into an error-shaped
    # response, so the client never invalidated caches for the videos that did change.
    starred_videos: set[str] = set()
    failed_videos: set[str] = set()
    affected_boards: set[str] = set()
    for video_name in dict.fromkeys(batch.video_names):
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
            failed_videos.add(video_name)
    return StarredVideosResult(
        starred_videos=list(starred_videos),
        failed_videos=list(failed_videos),
        affected_boards=list(affected_boards),
    )


@videos_router.post("/unstar", operation_id="unstar_videos_in_list", response_model=UnstarredVideosResult)
def unstar_videos_in_list(
    current_user: CurrentUserOrDefault,
    batch: VideoNamesBatch,
) -> UnstarredVideosResult:
    # See star_videos_in_list: skip foreign names instead of re-raising mid-batch.
    unstarred_videos: set[str] = set()
    failed_videos: set[str] = set()
    affected_boards: set[str] = set()
    for video_name in dict.fromkeys(batch.video_names):
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
            failed_videos.add(video_name)
    return UnstarredVideosResult(
        unstarred_videos=list(unstarred_videos),
        failed_videos=list(failed_videos),
        affected_boards=list(affected_boards),
    )


class VideoBoardArg(BaseModel):
    board_id: str = Field(description="The id of the board to add or remove the video from")
    video_name: str = Field(description="The name of the video to add to / remove from the board")


@videos_router.post(
    "/board",
    operation_id="add_video_to_board",
    response_model=AddVideosToBoardResult,
)
def add_video_to_board(
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
def remove_video_from_board(
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
