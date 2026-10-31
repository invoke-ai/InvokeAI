from datetime import datetime
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator

from invokeai.app.services.board_access.board_access_common import GeneralAccess
from invokeai.app.util.metaenum import MetaEnum
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class BoardVisibility(str, Enum, metaclass=MetaEnum):
    """Deprecated: a board's general access under its pre-7.0 name. Kept until 8.0 for older clients."""

    Private = "private"
    """General access `restricted`."""
    Shared = "shared"
    """General access `viewer`."""
    Public = "public"
    """General access `contributor`."""


GENERAL_ACCESS_BY_VISIBILITY = {
    BoardVisibility.Private: GeneralAccess.Restricted,
    BoardVisibility.Shared: GeneralAccess.Viewer,
    BoardVisibility.Public: GeneralAccess.Contributor,
}
VISIBILITY_BY_GENERAL_ACCESS = {access: visibility for visibility, access in GENERAL_ACCESS_BY_VISIBILITY.items()}


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
    general_access: GeneralAccess = Field(
        default=GeneralAccess.Restricted,
        description="The role every other account gets on the board: none (restricted), viewer or contributor.",
    )
    project_id: Optional[str] = Field(
        default=None, description="The id of the owner's project this board belongs to; absent for a Library board."
    )
    """A board lives in exactly one of its owner's projects, or in the Library when this is `None`.
    Clients must treat the field as absent-or-null: the record's own `model_dump` drops `None` values,
    but a route's response serialization need not."""


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
    try:
        general_access = GeneralAccess(board_dict.get("general_access", GeneralAccess.Restricted.value))
    except ValueError:
        general_access = GeneralAccess.Restricted

    return BoardRecord(
        board_id=board_id,
        board_name=board_name,
        user_id=user_id,
        cover_image_name=cover_image_name,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=deleted_at,
        archived=archived,
        general_access=general_access,
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
    general_access: Optional[GeneralAccess] = Field(
        default=None, description="The role every other account gets on the board. Only its owner may change it."
    )
    board_visibility: Optional[BoardVisibility] = Field(
        default=None,
        description="Deprecated, to be removed in 8.0: general access under its old name. Send `general_access`.",
        json_schema_extra={"deprecated": True},
    )
    project_id: Optional[str] = Field(
        default=None,
        description=(
            "Move the board into one of the owner's projects, or to the Library with an explicit null."
            " Omit the field to leave the board where it is."
        ),
    )

    @model_validator(mode="after")
    def _visibility_names_general_access(self) -> "BoardChanges":
        """`board_visibility` is `general_access` under its old name: only one of them may be sent."""
        if self.board_visibility is not None:
            if self.general_access is not None:
                raise ValueError("Send general_access or board_visibility, not both")
            self.general_access = GENERAL_ACCESS_BY_VISIBILITY[self.board_visibility]
            self.board_visibility = None
        return self

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
