from datetime import datetime
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field

from invokeai.app.util.metaenum import MetaEnum
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class BoardRecord(BaseModelExcludeNull):
    """Deserialized board record."""

    board_id: str = Field(description="The unique ID of the board.")
    """The unique ID of the board."""
    board_name: str = Field(description="The name of the board.")
    """The name of the board."""
    created_at: Union[datetime, str] = Field(description="The created timestamp of the board.")
    """The created timestamp of the image."""
    updated_at: Union[datetime, str] = Field(description="The updated timestamp of the board.")
    """The updated timestamp of the image."""
    deleted_at: Optional[Union[datetime, str]] = Field(default=None, description="The deleted timestamp of the board.")
    """The updated timestamp of the image."""
    cover_image_name: Optional[str] = Field(default=None, description="The name of the cover image of the board.")
    """The name of the cover image of the board."""
    archived: bool = Field(description="Whether or not the board is archived.")
    """Whether or not the board is archived."""
    is_private: Optional[bool] = Field(default=None, description="Whether the board is private.")
    """Whether the board is private."""


def deserialize_board_record(board_dict: dict) -> BoardRecord:
    """Deserializes a board record."""

    # Retrieve all the values, setting "reasonable" defaults if they are not present.

    board_id = board_dict.get("board_id", "unknown")
    board_name = board_dict.get("board_name", "unknown")
    cover_image_name = board_dict.get("cover_image_name", "unknown")
    created_at = board_dict.get("created_at", get_iso_timestamp())
    updated_at = board_dict.get("updated_at", get_iso_timestamp())
    deleted_at = board_dict.get("deleted_at", get_iso_timestamp())
    archived = board_dict.get("archived", False)
    is_private = board_dict.get("is_private", False)

    return BoardRecord(
        board_id=board_id,
        board_name=board_name,
        cover_image_name=cover_image_name,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=deleted_at,
        archived=archived,
        is_private=is_private,
    )


class BoardChanges(BaseModel, extra="forbid"):
    board_name: Optional[str] = Field(default=None, description="The board's new name.", max_length=300)
    cover_image_name: Optional[str] = Field(default=None, description="The name of the board's new cover image.")
    archived: Optional[bool] = Field(default=None, description="Whether or not the board is archived")


class BoardRecordOrderBy(str, Enum, metaclass=MetaEnum):
    """The order by options for board records"""

    CreatedAt = "created_at"
    Name = "board_name"


class BoardRecordNotFoundException(Exception):
    """Raised when an board record is not found."""

    def __init__(self, message="Board record not found"):
        super().__init__(message)


class BoardRecordSaveException(Exception):
    """Raised when an board record cannot be saved."""

    def __init__(self, message="Board record not saved"):
        super().__init__(message)


class BoardRecordDeleteException(Exception):
    """Raised when an board record cannot be deleted."""

    def __init__(self, message="Board record not deleted"):
        super().__init__(message)
