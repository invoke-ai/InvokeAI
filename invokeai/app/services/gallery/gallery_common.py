"""Polymorphic gallery types: images and videos appearing in a single time-sorted stream."""

import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

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


@dataclass(frozen=True)
class BoardMediaSummary:
    cover_image_name: Optional[str] = None
    cover_video_name: Optional[str] = None
    image_count: int = 0
    video_count: int = 0
    asset_count: int = 0
