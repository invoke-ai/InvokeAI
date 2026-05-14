"""Polymorphic gallery types: images, videos and canvas projects in a single time-sorted stream."""

import datetime
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
    CANVAS_PROJECT = "canvas_project"


class GalleryItemRef(BaseModel):
    """A thin reference to a gallery item — used for ordered name lists."""

    kind: GalleryItemKind = Field(description="Whether the item is an image, video or canvas project.")
    name: str = Field(description="The unique name of the image, video or canvas project.")


class GalleryItem(BaseModelExcludeNull):
    """A gallery item — image, video or canvas project, with shared fields and a discriminator.

    Frontend code should dispatch on `kind` to render kind-specific UI.
    """

    kind: GalleryItemKind = Field(description="Whether the item is an image, video or canvas project.")
    name: str = Field(description="The unique name of the image, video or canvas project.")
    full_url: str = Field(description="URL to the full-resolution image PNG, video MP4 or canvas project .invk ZIP.")
    thumbnail_url: str = Field(description="URL to the static (WebP) thumbnail.")
    width: int = Field(description="The width of the item in pixels.")
    height: int = Field(description="The height of the item in pixels.")
    category: ImageCategory = Field(description="The category of the item (canvas projects always GENERAL).")
    starred: bool = Field(description="Whether the item is starred.")
    is_intermediate: bool = Field(description="Whether the item is an intermediate output.")
    board_id: Optional[str] = Field(default=None, description="Owning board id, if any.")
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the item.")
    # Video-only fields. None for images and canvas projects.
    duration: Optional[float] = Field(default=None, description="Video duration in seconds. None for non-videos.")
    fps: Optional[float] = Field(default=None, description="Video frames per second. None for non-videos.")
    # Canvas-project-only field. None for images and videos.
    image_count: Optional[int] = Field(
        default=None, description="Number of embedded images in a canvas project. None for non-projects."
    )


class GalleryItemNamesResult(BaseModel):
    """Ordered list of gallery item references plus counts for optimistic UI."""

    items: list[GalleryItemRef] = Field(description="Ordered list of (kind, name) references.")
    starred_count: int = Field(description="Number of starred items (when starred_first=True).")
    total_count: int = Field(description="Total number of items matching the query.")
