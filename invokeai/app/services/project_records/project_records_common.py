"""Common types and errors for the project records service."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProjectRecordNotFoundError(Exception):
    """Raised when a project record is not found for the requesting user."""

    def __init__(self, project_id: str) -> None:
        super().__init__(f"Project {project_id} not found")


class ProjectRecordExistsError(Exception):
    """Raised when creating a project with an id the user already has."""

    def __init__(self, project_id: str) -> None:
        super().__init__(f"Project {project_id} already exists")


class ProjectRecordConflictError(Exception):
    """Raised when a save carries a stale revision (another client saved first)."""

    def __init__(self, project_id: str, expected_revision: int, current_revision: int) -> None:
        self.current_revision = current_revision
        super().__init__(
            f"Project {project_id} is at revision {current_revision}; the save expected revision {expected_revision}"
        )


class ProjectBoardNotFoundError(Exception):
    """Raised when a project is created against a board the user does not have.

    A board owned by someone else lands here rather than in a "forbidden" error: the caller has no
    business learning that a board id it does not own exists.
    """

    def __init__(self, board_id: str) -> None:
        super().__init__(f"Board {board_id} not found")


class ProjectBoardUnavailableError(Exception):
    """Raised when a board exists and is the user's, but cannot become a project's board.

    A project board is private, unshared, and owned by exactly one project; a board that is public,
    shared, explicitly shared with someone, or already claimed fails all three ways at once.
    """

    def __init__(self, board_id: str) -> None:
        super().__init__(f"Board {board_id} is not available to be claimed by a project")


class ProjectSummaryDTO(BaseModel):
    """Lightweight project listing entry; the document payload is omitted."""

    project_id: str = Field(description="The project's client-generated identifier")
    board_id: str = Field(description="The project's private board; only project APIs may rename or delete it")
    name: str = Field(description="The project's display name")
    revision: int = Field(description="Monotonic revision, incremented on every save")
    created_at: str = Field(description="When the project was created")
    updated_at: str = Field(description="When the project was last saved")


class ProjectRecordDTO(ProjectSummaryDTO):
    """Full project record including the client-owned document."""

    data: dict[str, Any] = Field(description="The opaque client-owned project document")


class ProjectBoardItemDTO(BaseModel):
    """One visible item on a project's board."""

    category: Literal["general", "control", "mask", "user"] = Field(description="The item's gallery category")
    kind: Literal["image", "video"] = Field(description="Which namespace the name belongs to")
    name: str = Field(description="The server-assigned image or video name")
    starred: bool = Field(description="Whether the item is starred")


class ProjectBoardSnapshotDTO(BaseModel):
    """Everything a project's board holds that the gallery would show.

    This is the enumeration an export needs in order to carry a project's whole workspace rather
    than only the media its document happens to reference. Intermediates and the canvas's private
    `other` category are excluded, because neither is something the gallery shows on the board.

    Deliberately unversioned: `.invk`'s `board.json` carries its own version, so the archive format
    is free to change without the wire format following it, and vice versa.
    """

    items: list[ProjectBoardItemDTO] = Field(description="The board's visible items, ordered by kind then name")
