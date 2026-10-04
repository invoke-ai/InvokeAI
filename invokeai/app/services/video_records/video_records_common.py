import datetime
import re
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, StrictBool, StrictStr

from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ResourceOrigin,
)
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class VideoRecordNotFoundException(Exception):
    """Raised when a video record is not found."""

    def __init__(self, message="Video record not found"):
        super().__init__(message)


class VideoRecordSaveException(Exception):
    """Raised when a video record cannot be saved."""

    def __init__(self, message="Video record not saved"):
        super().__init__(message)


class VideoRecordDeleteException(Exception):
    """Raised when a video record cannot be deleted."""

    def __init__(self, message="Video record not deleted"):
        super().__init__(message)


# The `media_origin` marker, projected out of the `metadata` JSON blob. Kept as a bare
# expression so the polymorphic gallery query can alias it into its own UNION half.
#
# The `json_valid` guard is not decoration: `json_extract` RAISES on unparseable text, and
# this expression now runs on every row of every video listing. An unguarded call would let
# a single malformed blob fail the whole page rather than one video -- and the column is
# plain TEXT with no CHECK constraint, so nothing but convention keeps one out. Every
# in-tree writer goes through `MetadataField`, so this is insurance, not a live bug.
MEDIA_ORIGIN_JSON_EXPR = (
    "CASE WHEN json_valid(videos.metadata) THEN json_extract(videos.metadata, '$.media_origin') END"
)
MEDIA_ORIGIN_SQL_EXPR = f"{MEDIA_ORIGIN_JSON_EXPR} AS media_origin"

# The longest marker worth carrying. Only `audio_upload` has meaning today, but the field is
# an open vocabulary, so this is a sanity bound rather than an allowlist. It matters because
# upload metadata is client-supplied and unbounded, and this one key now rides EVERY row of
# EVERY listing: without a cap, one upload with a 200 KB marker is echoed back on every
# gallery page that includes it.
MEDIA_ORIGIN_MAX_LENGTH = 64

# A marker is an identifier-shaped token (`audio_upload` is the only one so far). Matching
# the SHAPE rather than an allowlist keeps the vocabulary open, while rejecting the two
# things `json_extract` hands back as strings without them being markers: a JSON object or
# array arrives as its serialized text (`'{"a":"b"}'`, `'[1,2]'`), which `isinstance(str)`
# alone would happily propagate.
MEDIA_ORIGIN_PATTERN = re.compile(r"\A[A-Za-z0-9_-]+\Z")


def coerce_media_origin(value: Any) -> Optional[str]:
    """The `media_origin` marker as a string, or None for anything that is not one.

    Upload metadata is validated only as a JSON *object* (`MetadataField` is a
    `RootModel[dict[str, Any]]`), so a client may store any JSON value under this key, and
    `json_extract` hands back the matching SQLite type — an int for a JSON number, 0/1 for a
    JSON boolean. Feeding one of those to the `Optional[str]` field would raise a
    `ValidationError` while *deserializing the row*, which does not merely mislabel the
    video: it makes the record, its DTO, and every gallery listing that contains it fail to
    build. One odd upload would take out the gallery. An unrecognized marker means the same
    thing as an absent one, so it is dropped here rather than propagated -- an empty string,
    one past :data:`MEDIA_ORIGIN_MAX_LENGTH`, and anything not shaped like a marker per
    :data:`MEDIA_ORIGIN_PATTERN`.
    """
    if not isinstance(value, str) or len(value) > MEDIA_ORIGIN_MAX_LENGTH:
        return None
    return value if MEDIA_ORIGIN_PATTERN.match(value) else None


VIDEO_DTO_COLS = ", ".join(
    [
        "videos." + c
        for c in [
            "video_name",
            "video_origin",
            "video_category",
            "width",
            "height",
            "duration",
            "fps",
            "session_id",
            "node_id",
            "has_workflow",
            "is_intermediate",
            "created_at",
            "updated_at",
            "deleted_at",
            "starred",
            "video_subfolder",
            "project_id",
            "file_size_bytes",
        ]
    ]
    # `media_origin` is not a column: it is the one key of the `metadata` JSON blob the
    # frontend needs on every row (it marks an upload the ingest converter wrapped from an
    # audio file). Extracting just that key keeps listings from carrying whole metadata
    # blobs. `json_extract` yields NULL for a NULL or non-object blob; what it yields for a
    # non-string value is `coerce_media_origin`'s problem, not the query's.
    + [MEDIA_ORIGIN_SQL_EXPR]
)


class VideoRecord(BaseModelExcludeNull):
    """Deserialized video record: the columns, plus the one `media_origin` key projected out of metadata."""

    video_name: str = Field(description="The unique name of the video.")
    video_origin: ResourceOrigin = Field(description="The origin of the video.")
    video_category: ImageCategory = Field(description="The category of the video (reuses ImageCategory).")
    width: int = Field(description="The pixel width of the video.")
    height: int = Field(description="The pixel height of the video.")
    duration: float = Field(description="The duration of the video in seconds.")
    fps: Optional[float] = Field(default=None, description="The frames-per-second of the video, if known.")
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the video.")
    updated_at: Union[datetime.datetime, str] = Field(description="The updated timestamp of the video.")
    deleted_at: Optional[Union[datetime.datetime, str]] = Field(
        default=None, description="The deleted timestamp of the video."
    )
    is_intermediate: bool = Field(description="Whether this is an intermediate video.")
    session_id: Optional[str] = Field(default=None, description="The session ID that produced this video, if any.")
    node_id: Optional[str] = Field(default=None, description="The node ID that produced this video, if any.")
    starred: bool = Field(description="Whether this video is starred.")
    has_workflow: bool = Field(description="Whether this video has a workflow associated.")
    video_subfolder: str = Field(default="", description="The subfolder where the video is stored on disk.")
    project_id: Optional[str] = Field(
        default=None, description="The project this video originated in, if it was made for one."
    )
    file_size_bytes: Optional[int] = Field(
        default=None, description="Bytes the video, its thumbnail and sidecar occupy on disk; null until measured."
    )
    media_origin: Optional[str] = Field(
        default=None,
        description="How this video entered the gallery, if it was marked: 'audio_upload' for an uploaded audio "
        "file the server wrapped into a waveform video.",
    )


class VideoRecordChanges(BaseModelExcludeNull, extra="allow"):
    """Allowed mutations on a video record."""

    video_category: Optional[ImageCategory] = Field(default=None, description="The video's new category.")
    session_id: Optional[StrictStr] = Field(default=None, description="The video's new session ID.")
    is_intermediate: Optional[StrictBool] = Field(default=None, description="The video's new `is_intermediate` flag.")
    starred: Optional[StrictBool] = Field(default=None, description="The video's new `starred` state.")


def deserialize_video_record(video_dict: dict) -> VideoRecord:
    """Deserializes a video record from a sqlite row dict."""
    video_name = video_dict.get("video_name", "unknown")
    video_origin = ResourceOrigin(video_dict.get("video_origin", ResourceOrigin.INTERNAL.value))
    video_category = ImageCategory(video_dict.get("video_category", ImageCategory.GENERAL.value))
    width = video_dict.get("width", 0)
    height = video_dict.get("height", 0)
    duration = video_dict.get("duration", 0.0)
    fps_raw = video_dict.get("fps", None)
    fps = float(fps_raw) if fps_raw is not None else None
    session_id = video_dict.get("session_id", None)
    node_id = video_dict.get("node_id", None)
    created_at = video_dict.get("created_at", get_iso_timestamp())
    updated_at = video_dict.get("updated_at", get_iso_timestamp())
    deleted_at = video_dict.get("deleted_at", None)
    is_intermediate = video_dict.get("is_intermediate", False)
    starred = video_dict.get("starred", False)
    has_workflow = video_dict.get("has_workflow", False)
    video_subfolder = video_dict.get("video_subfolder", "")
    media_origin = coerce_media_origin(video_dict.get("media_origin", None))
    project_id = video_dict.get("project_id", None)
    file_size_bytes = video_dict.get("file_size_bytes", None)

    return VideoRecord(
        video_name=video_name,
        video_origin=video_origin,
        video_category=video_category,
        width=width,
        height=height,
        duration=float(duration),
        fps=fps,
        session_id=session_id,
        node_id=node_id,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=deleted_at,
        is_intermediate=is_intermediate,
        starred=starred,
        has_workflow=has_workflow,
        video_subfolder=video_subfolder,
        media_origin=media_origin,
        project_id=project_id,
        file_size_bytes=file_size_bytes,
    )


class VideoNamesResult(BaseModel):
    """Response containing ordered video names with metadata for optimistic updates."""

    video_names: list[str] = Field(description="Ordered list of video names")
    starred_count: int = Field(description="Number of starred videos (when starred_first=True)")
    total_count: int = Field(description="Total number of videos matching the query")
