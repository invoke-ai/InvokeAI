"""Polymorphic gallery types: images and videos appearing in a single time-sorted stream."""

import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.util.metaenum import MetaEnum
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class GalleryItemKind(str, Enum, metaclass=MetaEnum):
    """Discriminator for polymorphic gallery items."""

    IMAGE = "image"
    VIDEO = "video"


class GalleryItemRef(BaseModel):
    """A thin reference to a gallery item — used for ordered name lists."""

    kind: GalleryItemKind = Field(description="Whether the item is an image or video.")
    name: str = Field(description="The unique name of the image or video.")


class GalleryItemLocation(BaseModel):
    """A gallery item's position in a filtered, ordered listing."""

    kind: GalleryItemKind = Field(description="Whether the item is an image or video.")
    name: str = Field(description="The unique name of the image or video.")
    index: int = Field(description="The item's zero-based index in the listing.")
    total: int = Field(description="Number of items matching the listing filters.")


class GalleryItem(BaseModelExcludeNull):
    """A gallery item — either an image or a video, with shared fields and a discriminator.

    Frontend code should dispatch on `kind` to render image- vs video-specific UI.
    """

    kind: GalleryItemKind = Field(description="Whether the item is an image or video.")
    name: str = Field(description="The unique name of the image or video.")
    full_url: str = Field(description="URL to the full-resolution image PNG or the full-quality video MP4.")
    thumbnail_url: str = Field(description="URL to the static (WebP) thumbnail.")
    width: int = Field(description="The width of the item in pixels.")
    height: int = Field(description="The height of the item in pixels.")
    category: ImageCategory = Field(description="The category of the item (images and videos share the same enum).")
    starred: bool = Field(description="Whether the item is starred.")
    is_intermediate: bool = Field(description="Whether the item is an intermediate output.")
    board_id: Optional[str] = Field(default=None, description="Owning board id, if any.")
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the item.")
    # Video-only fields. None for images.
    duration: Optional[float] = Field(default=None, description="Video duration in seconds. None for images.")
    fps: Optional[float] = Field(default=None, description="Video frames per second. None for images.")
    media_origin: Optional[str] = Field(
        default=None,
        description="How a video entered the gallery, if it was marked: 'audio_upload' for an uploaded audio file "
        "the server wrapped into a waveform video. None for images and for unmarked videos.",
    )


class GalleryItemNamesResult(BaseModel):
    """Ordered list of gallery item references plus counts for optimistic UI.

    Deprecated in favour of :class:`GalleryItemNames`. Wrapping every name in an object to
    carry a `kind` discriminator costs ~800ms of model construction on a 200k-item library,
    for a field callers derive from the filename extension anyway.
    """

    items: list[GalleryItemRef] = Field(description="Ordered list of (kind, name) references.")
    starred_count: int = Field(description="Number of starred items (when starred_first=True).")
    total_count: int = Field(description="Total number of items matching the query.")


class GalleryItemNames(BaseModel):
    """Ordered flat list of gallery item names plus counts for optimistic UI.

    Names are polymorphic — images and videos are interleaved by `created_at`. The kind of a
    given name is its file extension (`.mp4` is a video), which is how every consumer already
    discriminates. Mirrors the shape of the image-only `ImageNamesResult`.
    """

    item_names: list[str] = Field(description="Ordered list of image and video names.")
    starred_count: int = Field(description="Number of starred items (when starred_first=True).")
    total_count: int = Field(description="Total number of items matching the query.")


# A usage check names at most this many documents (`documents_total` counts them all), and at most this many media
# names per document or queue entry (their counts are exact).
MAX_USAGE_DOCUMENTS = 100
MAX_USAGE_NAMES = 1000

GalleryMediaUsageDocumentKind = Literal["project", "workflow", "legacy_editor", "quarantined_project"]


class GalleryMediaUsageDocument(BaseModel):
    """One of the caller's saved documents that names checked media."""

    kind: GalleryMediaUsageDocumentKind = Field(
        description="'project': a project's content and the caller's own workspace in it. 'workflow': a library "
        "workflow. 'legacy_editor': the legacy editor's saved state, all of it as one entry. 'quarantined_project': "
        "a project kept for repair."
    )
    id: str = Field(description="The project or workflow id; 'legacy_editor' for the legacy editor's state.")
    name: Optional[str] = Field(description="The document's name; null for the legacy editor or a missing document.")
    image_names: list[str] = Field(
        description=f"The checked images the document names, sorted; at most {MAX_USAGE_NAMES}."
    )
    video_names: list[str] = Field(
        description=f"The checked videos the document names, sorted; at most {MAX_USAGE_NAMES}."
    )
    image_count: int = Field(description="How many checked images the document names, past the list's cap too.")
    video_count: int = Field(description="How many checked videos the document names, past the list's cap too.")
    revision: Optional[int] = Field(
        description="For a project: its content revision when the names were read, as the project read reports it "
        "(`revision`). Null for other kinds, and for a project the caller does not own."
    )
    workspace_revision: Optional[int] = Field(
        description="For a project: the caller's workspace revision when the names were read, as the project read "
        "reports it (`workspace.revision`); null while the caller has no workspace in it. Null for other kinds."
    )


class GalleryMediaUsageQueue(BaseModel):
    """The caller's active queue work that names checked media."""

    item_count: int = Field(
        description="The caller's active queue items (pending, in progress or waiting, and the completed children of "
        "an active root) whose sessions name checked media."
    )
    image_names: list[str] = Field(
        description=f"The checked images those items name, sorted; at most {MAX_USAGE_NAMES}."
    )
    video_names: list[str] = Field(
        description=f"The checked videos those items name, sorted; at most {MAX_USAGE_NAMES}."
    )
    image_count: int = Field(description="How many checked images those items name, past the list's cap too.")
    video_count: int = Field(description="How many checked videos those items name, past the list's cap too.")


class GalleryMediaUsageOthers(BaseModel):
    """Other accounts' documents and queue work that name checked media, without identifying them."""

    document_count: int = Field(description="Other accounts' documents naming checked media.")
    queue_item_count: int = Field(description="Other accounts' active queue items naming checked media.")
    image_count: int = Field(description="Distinct checked images other accounts' documents or queue work name.")
    video_count: int = Field(description="Distinct checked videos other accounts' documents or queue work name.")


class GalleryMediaUsage(BaseModel):
    """What still uses media a deletion would remove. Saved documents count conservatively: anything they name,
    history included."""

    checked_image_count: int = Field(
        description="Images checked: the requested ones the caller may delete, or the images the board's deletion "
        "would delete."
    )
    checked_video_count: int = Field(
        description="Videos checked: the requested ones the caller may delete, or the videos the board's deletion "
        "would delete."
    )
    in_use_image_count: int = Field(description="Distinct checked images that anything below names.")
    in_use_video_count: int = Field(description="Distinct checked videos that anything below names.")
    documents: list[GalleryMediaUsageDocument] = Field(
        description=f"The caller's own documents naming checked media, by kind, then name, then id; at most "
        f"{MAX_USAGE_DOCUMENTS}."
    )
    documents_total: int = Field(description="How many of the caller's documents name checked media.")
    queue: GalleryMediaUsageQueue = Field(description="The caller's own active queue work naming checked media.")
    other_accounts: GalleryMediaUsageOthers = Field(
        description="Other accounts' documents and queue work naming checked media, as counts only."
    )


@dataclass(frozen=True)
class BoardMediaSummary:
    cover_image_name: Optional[str] = None
    cover_video_name: Optional[str] = None
    image_count: int = 0
    video_count: int = 0
    asset_count: int = 0
    asset_video_count: int = 0
