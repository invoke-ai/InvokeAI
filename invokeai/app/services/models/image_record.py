import datetime
from typing import Optional, Union

from pydantic import BaseModel, Extra, Field, StrictBool, StrictStr

from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.app.util.misc import get_iso_timestamp


class ImageRecord(BaseModel):
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
    deleted_at: Union[datetime.datetime, str, None] = Field(description="The deleted timestamp of the image.")
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


class ImageRecordChanges(BaseModel, extra=Extra.forbid):
    """A set of changes to apply to an image record.

    Only limited changes are valid:
      - `image_category`: change the category of an image
      - `session_id`: change the session associated with an image
      - `is_intermediate`: change the image's `is_intermediate` flag
    """

    image_category: Optional[ImageCategory] = Field(description="The image's new category.")
    """The image's new category."""
    session_id: Optional[StrictStr] = Field(
        default=None,
        description="The image's new session ID.",
    )
    """The image's new session ID."""
    is_intermediate: Optional[StrictBool] = Field(default=None, description="The image's new `is_intermediate` flag.")
    """The image's new `is_intermediate` flag."""


class ImageUrlsDTO(BaseModel):
    """The URLs for an image and its thumbnail."""

    image_name: str = Field(description="The unique name of the image.")
    """The unique name of the image."""
    image_url: str = Field(description="The URL of the image.")
    """The URL of the image."""
    thumbnail_url: str = Field(description="The URL of the image's thumbnail.")
    """The URL of the image's thumbnail."""


class ImageDTO(ImageRecord, ImageUrlsDTO):
    """Deserialized image record, enriched for the frontend."""

    board_id: Optional[str] = Field(description="The id of the board the image belongs to, if one exists.")
    """The id of the board the image belongs to, if one exists."""
    pass


def image_record_to_dto(
    image_record: ImageRecord, image_url: str, thumbnail_url: str, board_id: Optional[str]
) -> ImageDTO:
    """Converts an image record to an image DTO."""
    return ImageDTO(
        **image_record.dict(),
        image_url=image_url,
        thumbnail_url=thumbnail_url,
        board_id=board_id,
    )


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
    )
