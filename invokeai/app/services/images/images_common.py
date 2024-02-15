from typing import Optional

from PIL.Image import Image as PILImageType
from pydantic import Field, BaseModel

from invokeai.app.services.image_records.image_records_common import ImageRecord
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull
from invokeai.app.invocations.baseinvocation import MetadataField
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID

class ImageUploadData(BaseModel):
    image: PILImageType
    image_name: Optional[str] = None
    image_origin: ResourceOrigin
    image_category: ImageCategory
    image_url: Optional[str] = None
    session_id: Optional[str] = None
    board_id: Optional[str] = None
    is_intermediate: Optional[bool] = False
    metadata: Optional[MetadataField] = None
    workflow: Optional[WorkflowWithoutID] = None
    width: Optional[int] = None
    height: Optional[int] = None
    starred: Optional[bool] = False
    node_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

class ImageUrlsDTO(BaseModelExcludeNull):
    """The URLs for an image and its thumbnail."""

    image_name: str = Field(description="The unique name of the image.")
    """The unique name of the image."""
    image_url: str = Field(description="The URL of the image.")
    """The URL of the image."""
    thumbnail_url: str = Field(description="The URL of the image's thumbnail.")
    """The URL of the image's thumbnail."""


class ImageDTO(ImageRecord, ImageUrlsDTO):
    """Deserialized image record, enriched for the frontend."""

    board_id: Optional[str] = Field(
        default=None, description="The id of the board the image belongs to, if one exists."
    )
    metadata: Optional[MetadataField] = Field(
        default=None, description="The metadata of the image."
    )
    workflow: Optional[WorkflowWithoutID] = Field(
        default=None, description="The workflow of the image."
    )
    """The id of the board the image belongs to, if one exists."""


def image_record_to_dto(
    image_record: ImageRecord,
    image_url: str,
    thumbnail_url: str,
    board_id: Optional[str],
) -> ImageDTO:
    """Converts an image record to an image DTO."""
    return ImageDTO(
        **image_record.model_dump(),
        image_url=image_url,
        thumbnail_url=thumbnail_url,
        board_id=board_id,
    )
