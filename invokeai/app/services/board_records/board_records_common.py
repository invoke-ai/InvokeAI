from datetime import datetime
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field

from invokeai.app.util.metaenum import MetaEnum
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class BoardVisibility(str, Enum, metaclass=MetaEnum):
    """The visibility options for a board."""

    Private = "private"
    """Only the board owner (and admins) can see and modify this board."""
    Shared = "shared"
    """All users can view this board, but only the owner (and admins) can modify it."""
    Public = "public"
    """All users can view this board; only the owner (and admins) can modify its structure."""


class BoardRecord(BaseModelExcludeNull):
    """Deserialized board record."""

    board_id: str = Field(description="The unique ID of the board.")
    """The unique ID of the board."""
    board_name: str = Field(description="The name of the board.")
    """The name of the board."""
    user_id: str = Field(description="The user ID of the board owner.")
    """The user ID of the board owner."""
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
    board_visibility: BoardVisibility = Field(
        default=BoardVisibility.Private, description="The visibility of the board."
    )
    """The visibility of the board (private, shared, or public)."""
    project_id: Optional[str] = Field(
        default=None, description="The id of the owner's project this board belongs to; absent for a Library board."
    )
    """A board lives in exactly one of its owner's projects, or in the Library when this is `None`.
    Project boards are private and unshared. Clients must treat the field as absent-or-null: the
    record's own `model_dump` drops `None` values, but a route's response serialization need not."""


def deserialize_board_record(board_dict: dict) -> BoardRecord:
    """Deserializes a board record."""

    # Retrieve all the values, setting "reasonable" defaults if they are not present.

    board_id = board_dict.get("board_id", "unknown")
    board_name = board_dict.get("board_name", "unknown")
    # Default to 'system' for backwards compatibility with boards created before multiuser support
    user_id = board_dict.get("user_id", "system")
    cover_image_name = board_dict.get("cover_image_name", "unknown")
    created_at = board_dict.get("created_at", get_iso_timestamp())
    updated_at = board_dict.get("updated_at", get_iso_timestamp())
    deleted_at = board_dict.get("deleted_at", get_iso_timestamp())
    archived = board_dict.get("archived", False)
    project_id = board_dict.get("project_id")
    board_visibility_raw = board_dict.get("board_visibility", BoardVisibility.Private.value)
    try:
        board_visibility = BoardVisibility(board_visibility_raw)
    except ValueError:
        board_visibility = BoardVisibility.Private

    return BoardRecord(
        board_id=board_id,
        board_name=board_name,
        user_id=user_id,
        cover_image_name=cover_image_name,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=deleted_at,
        archived=archived,
        board_visibility=board_visibility,
        project_id=project_id,
    )


BOARD_NAME_MAX_LENGTH = 300
"""The longest board name the API accepts.

Lives here because `BoardChanges` is what enforces it. A project's inbox takes the project's name,
and project names are unbounded, so both the claim path and the migration truncate to this — they
must agree with the generic route or they would write names it then refuses to touch.
"""


class BoardChanges(BaseModel, extra="forbid"):
    board_name: Optional[str] = Field(
        default=None, description="The board's new name.", max_length=BOARD_NAME_MAX_LENGTH
    )
    cover_image_name: Optional[str] = Field(
        default=None, max_length=255, description="The name of the board's new cover image."
    )
    archived: Optional[bool] = Field(default=None, description="Whether or not the board is archived")
    board_visibility: Optional[BoardVisibility] = Field(default=None, description="The visibility of the board.")
    project_id: Optional[str] = Field(
        default=None,
        description=(
            "Move the board into one of the owner's projects, or to the Library with an explicit null."
            " Omit the field to leave the board where it is."
        ),
    )

    @property
    def moves_board(self) -> bool:
        """Whether the request names a destination at all; `None` is the Library, absent is "stay"."""
        return "project_id" in self.model_fields_set


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


class BoardRecordInboxException(BoardRecordSaveException):
    """Raised when a generic board write would rename, archive, publish, move or delete a project's inbox."""

    def __init__(self, message="Board is a project's inbox"):
        super().__init__(message)


class BoardRecordProjectNotFoundException(BoardRecordSaveException):
    """Raised when a board is created in or moved to a project its owner does not have.

    Someone else's project reads as missing rather than forbidden: the caller has no business
    learning that a project id it does not own exists.
    """

    def __init__(self, message="Project not found"):
        super().__init__(message)


class BoardRecordProjectUnavailableException(BoardRecordSaveException):
    """Raised when a board cannot be in a project: it is shared or public, or would become so.

    Boards in a project are private and unshared, so a visibility change and a move are refused
    together whenever the result would be a non-private project board.
    """

    def __init__(self, message="Boards in a project must be private"):
        super().__init__(message)
