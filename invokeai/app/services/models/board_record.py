from typing import Optional, Union
from datetime import datetime
from pydantic import BaseModel, Extra, Field, StrictBool, StrictStr
from invokeai.app.util.misc import get_iso_timestamp


class BoardRecord(BaseModel):
    """Deserialized board record."""

    board_id: str = Field(description="The unique ID of the board.")
    """The unique ID of the board."""
    board_name: str = Field(description="The name of the board.")
    """The name of the board."""
    created_at: Union[datetime, str] = Field(description="The created timestamp of the board.")
    """The created timestamp of the image."""
    updated_at: Union[datetime, str] = Field(description="The updated timestamp of the board.")
    """The updated timestamp of the image."""
    deleted_at: Union[datetime, str, None] = Field(description="The deleted timestamp of the board.")
    """The updated timestamp of the image."""
    cover_image_name: Optional[str] = Field(description="The name of the cover image of the board.")
    """The name of the cover image of the board."""


class BoardDTO(BoardRecord):
    """Deserialized board record with cover image URL and image count."""

    cover_image_name: Optional[str] = Field(description="The name of the board's cover image.")
    """The URL of the thumbnail of the most recent image in the board."""
    image_count: int = Field(description="The number of images in the board.")
    """The number of images in the board."""


def deserialize_board_record(board_dict: dict) -> BoardRecord:
    """Deserializes a board record."""

    # Retrieve all the values, setting "reasonable" defaults if they are not present.

    board_id = board_dict.get("board_id", "unknown")
    board_name = board_dict.get("board_name", "unknown")
    cover_image_name = board_dict.get("cover_image_name", "unknown")
    created_at = board_dict.get("created_at", get_iso_timestamp())
    updated_at = board_dict.get("updated_at", get_iso_timestamp())
    deleted_at = board_dict.get("deleted_at", get_iso_timestamp())

    return BoardRecord(
        board_id=board_id,
        board_name=board_name,
        cover_image_name=cover_image_name,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=deleted_at,
    )
