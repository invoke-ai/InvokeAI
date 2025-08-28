# TODO: Should these excpetions subclass existing python exceptions?
import datetime
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, StrictBool, StrictStr

from invokeai.app.util.metaenum import MetaEnum
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class ResourceOrigin(str, Enum, metaclass=MetaEnum):
    """The origin of a resource (eg image).

    - INTERNAL: The resource was created by the application.
    - EXTERNAL: The resource was not created by the application.
    This may be a user-initiated upload, or an internal application upload (eg Canvas init image).
    """

    INTERNAL = "internal"
    """The resource was created by the application."""
    EXTERNAL = "external"
    """The resource was not created by the application.
    This may be a user-initiated upload, or an internal application upload (eg Canvas init image).
    """


class InvalidOriginException(ValueError):
    """Raised when a provided value is not a valid ResourceOrigin.

    Subclasses `ValueError`.
    """

    def __init__(self, message="Invalid resource origin."):
        super().__init__(message)


class ImageCategory(str, Enum, metaclass=MetaEnum):
    """The category of an image.

    - GENERAL: The image is an output, init image, or otherwise an image without a specialized purpose.
    - MASK: The image is a mask image.
    - CONTROL: The image is a ControlNet control image.
    - USER: The image is a user-provide image.
    - OTHER: The image is some other type of image with a specialized purpose. To be used by external nodes.
    """

    GENERAL = "general"
    """GENERAL: The image is an output, init image, or otherwise an image without a specialized purpose."""
    MASK = "mask"
    """MASK: The image is a mask image."""
    CONTROL = "control"
    """CONTROL: The image is a ControlNet control image."""
    USER = "user"
    """USER: The image is a user-provide image."""
    OTHER = "other"
    """OTHER: The image is some other type of image with a specialized purpose. To be used by external nodes."""


class InvalidImageCategoryException(ValueError):
    """Raised when a provided value is not a valid ImageCategory.

    Subclasses `ValueError`.
    """

    def __init__(self, message="Invalid image category."):
        super().__init__(message)


class ImageRecordNotFoundException(Exception):
    """Raised when an image record is not found."""

    def __init__(self, message="Image record not found"):
        super().__init__(message)


class ImageRecordSaveException(Exception):
    """Raised when an image record cannot be saved."""

    def __init__(self, message="Image record not saved"):
        super().__init__(message)


class ImageRecordDeleteException(Exception):
    """Raised when an image record cannot be deleted."""

    def __init__(self, message="Image record not deleted"):
        super().__init__(message)


IMAGE_DTO_COLS = ", ".join(
    [
        "images." + c
        for c in [
            "image_name",
            "image_origin",
            "image_category",
            "width",
            "height",
            "session_id",
            "node_id",
            "has_workflow",
            "is_intermediate",
            "created_at",
            "updated_at",
            "deleted_at",
            "starred",
        ]
    ]
)


class ImageRecord(BaseModelExcludeNull):
    """Deserialized image record without metadata."""

    image_name: str = Field(description="The unique name of the image.")
    """The unique name of the image."""
    image_origin: ResourceOrigin = Field(description="The type of the image.")
    """The origin of the image."""
    image_category: ImageCategory = Field(description="The category of the image.")
    """The category of the image."""
    width: int = Field(description="The width of the image in px.")
    """The actual width of the image in px. This may be different from the width in metadata."""
    height: int = Field(description="The height of the image in px.")
    """The actual height of the image in px. This may be different from the height in metadata."""
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the image.")
    """The created timestamp of the image."""
    updated_at: Union[datetime.datetime, str] = Field(description="The updated timestamp of the image.")
    """The updated timestamp of the image."""
    deleted_at: Optional[Union[datetime.datetime, str]] = Field(
        default=None, description="The deleted timestamp of the image."
    )
    """The deleted timestamp of the image."""
    is_intermediate: bool = Field(description="Whether this is an intermediate image.")
    """Whether this is an intermediate image."""
    session_id: Optional[str] = Field(
        default=None,
        description="The session ID that generated this image, if it is a generated image.",
    )
    """The session ID that generated this image, if it is a generated image."""
    node_id: Optional[str] = Field(
        default=None,
        description="The node ID that generated this image, if it is a generated image.",
    )
    """The node ID that generated this image, if it is a generated image."""
    starred: bool = Field(description="Whether this image is starred.")
    """Whether this image is starred."""
    has_workflow: bool = Field(description="Whether this image has a workflow.")


class ImageRecordChanges(BaseModelExcludeNull, extra="allow"):
    """A set of changes to apply to an image record.

    Only limited changes are valid:
      - `image_category`: change the category of an image
      - `session_id`: change the session associated with an image
      - `is_intermediate`: change the image's `is_intermediate` flag
      - `starred`: change whether the image is starred
    """

    image_category: Optional[ImageCategory] = Field(default=None, description="The image's new category.")
    """The image's new category."""
    session_id: Optional[StrictStr] = Field(
        default=None,
        description="The image's new session ID.",
    )
    """The image's new session ID."""
    is_intermediate: Optional[StrictBool] = Field(default=None, description="The image's new `is_intermediate` flag.")
    """The image's new `is_intermediate` flag."""
    starred: Optional[StrictBool] = Field(default=None, description="The image's new `starred` state")
    """The image's new `starred` state."""


def deserialize_image_record(image_dict: dict) -> ImageRecord:
    """Deserializes an image record."""

    # Retrieve all the values, setting "reasonable" defaults if they are not present.

    # TODO: do we really need to handle default values here? ideally the data is the correct shape...
    image_name = image_dict.get("image_name", "unknown")
    image_origin = ResourceOrigin(image_dict.get("image_origin", ResourceOrigin.INTERNAL.value))
    image_category = ImageCategory(image_dict.get("image_category", ImageCategory.GENERAL.value))
    width = image_dict.get("width", 0)
    height = image_dict.get("height", 0)
    session_id = image_dict.get("session_id", None)
    node_id = image_dict.get("node_id", None)
    created_at = image_dict.get("created_at", get_iso_timestamp())
    updated_at = image_dict.get("updated_at", get_iso_timestamp())
    deleted_at = image_dict.get("deleted_at", get_iso_timestamp())
    is_intermediate = image_dict.get("is_intermediate", False)
    starred = image_dict.get("starred", False)
    has_workflow = image_dict.get("has_workflow", False)

    return ImageRecord(
        image_name=image_name,
        image_origin=image_origin,
        image_category=image_category,
        width=width,
        height=height,
        session_id=session_id,
        node_id=node_id,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=deleted_at,
        is_intermediate=is_intermediate,
        starred=starred,
        has_workflow=has_workflow,
    )


class ImageCollectionCounts(BaseModel):
    starred_count: int = Field(description="The number of starred images in the collection.")
    unstarred_count: int = Field(description="The number of unstarred images in the collection.")


class ImageNamesResult(BaseModel):
    """Response containing ordered image names with metadata for optimistic updates."""

    image_names: list[str] = Field(description="Ordered list of image names")
    starred_count: int = Field(description="Number of starred images (when starred_first=True)")
    total_count: int = Field(description="Total number of images matching the query")
