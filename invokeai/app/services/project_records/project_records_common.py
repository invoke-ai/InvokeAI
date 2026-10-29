"""Common types and errors for the project records service."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION = 2
"""Compatibility floor assigned to projects created before schema negotiation existed.

The project service deliberately treats the document as opaque. Clients declare the newest canvas
schema they understand, while this stored floor says how new a client must be to edit the project.
Keeping the default at v2 lets older clients continue to use existing projects; once a project is
upgraded, the floor only moves forward.
"""


DEFAULT_PROJECT_DOCUMENT_SCHEMA_VERSION = 3
"""The document schema floor of projects saved before the floor existed, and the newest schema a client that does
not say is assumed to understand.

Schema 4 split the document into the project content, in `projects.data`, and each member's workspace, in
`project_workspaces`. A client that does not name the newest schema it understands predates the split, so it is
refused a split project rather than handed half of one.
"""

SPLIT_PROJECT_DOCUMENT_SCHEMA_VERSION = 4
"""The first document schema in which a project is content and workspaces. A workspace is stored only beside content
at this floor or above: beside a whole document, a build from before the split could still save the whole project
over it, and nothing would read the workspace."""


class ProjectRecordNotFoundError(Exception):
    """Raised when a project record is not found for the requesting user."""

    def __init__(self, project_id: str) -> None:
        super().__init__(f"Project {project_id} not found")


class ProjectRecordExistsError(Exception):
    """Raised when creating a project with an id a project already has: ids are unique across accounts."""

    def __init__(self, project_id: str) -> None:
        super().__init__(f"Project {project_id} already exists")


class ProjectRecordConflictError(Exception):
    """Raised when a save carries a stale revision (another client saved first)."""

    def __init__(self, project_id: str, expected_revision: int, current_revision: int) -> None:
        self.current_revision = current_revision
        super().__init__(
            f"Project {project_id} is at revision {current_revision}; the save expected revision {expected_revision}"
        )


PROJECT_DOCUMENT_MAX_BYTES = 32 * 1024 * 1024
"""Maximum compact UTF-8 JSON size accepted for a project document."""


class ProjectDocumentTooLargeError(Exception):
    """Raised before an oversized project document is written."""

    def __init__(self, actual_bytes: int, max_bytes: int = PROJECT_DOCUMENT_MAX_BYTES) -> None:
        self.actual_bytes = actual_bytes
        self.max_bytes = max_bytes
        super().__init__(f"Project document is {actual_bytes} bytes; the limit is {max_bytes} bytes")


class ProjectDocumentInvalidError(Exception):
    """Raised when a project document cannot be represented as standards-compliant UTF-8 JSON."""

    def __init__(self) -> None:
        super().__init__("Project document must contain valid UTF-8 JSON values")


class ProjectCanvasSchemaUnsupportedError(Exception):
    """Raised before an incapable client may read or write a project document."""

    def __init__(self, project_id: str, minimum_version: int, client_maximum_version: int) -> None:
        self.project_id = project_id
        self.minimum_version = minimum_version
        self.client_maximum_version = client_maximum_version
        super().__init__(
            f"Project {project_id} requires canvas schema {minimum_version}, but this client supports up to"
            f" schema {client_maximum_version}. Update the client before opening or editing this project."
        )


class ProjectDocumentSchemaUnsupportedError(Exception):
    """Raised before a client may read or write a project whose document schema it does not understand."""

    def __init__(self, project_id: str, minimum_version: int, client_maximum_version: int) -> None:
        self.project_id = project_id
        self.minimum_version = minimum_version
        self.client_maximum_version = client_maximum_version
        super().__init__(
            f"Project {project_id} requires document schema {minimum_version}, but this client supports up to"
            f" schema {client_maximum_version}. Update the client before opening or editing this project."
        )


class ProjectDocumentSchemaDowngradeError(Exception):
    """Raised when a save tries to lower a project's document schema floor."""

    def __init__(self, project_id: str, current_version: int, requested_version: int) -> None:
        self.project_id = project_id
        self.current_version = current_version
        self.requested_version = requested_version
        super().__init__(
            f"Project {project_id} requires document schema {current_version}; its floor cannot be lowered to"
            f" schema {requested_version}"
        )


class ProjectWorkspaceConflictError(Exception):
    """Raised when a workspace save is based on a revision other than the stored one (another tab or device saved
    first), or expects a workspace where there is none or none where there is one."""

    def __init__(self, project_id: str, expected_revision: Optional[int], current_revision: Optional[int]) -> None:
        self.current_revision = current_revision
        super().__init__(
            f"The workspace in project {project_id} is at revision {current_revision}; the save expected revision"
            f" {expected_revision}"
        )


class ProjectWorkspaceBeforeSplitError(Exception):
    """Raised when a save carries a workspace while leaving the project's document floor below the split."""

    def __init__(self, project_id: str, document_floor: int) -> None:
        self.document_floor = document_floor
        super().__init__(
            f"Project {project_id} would stay at document schema {document_floor}; a workspace is saved only with"
            f" content at schema {SPLIT_PROJECT_DOCUMENT_SCHEMA_VERSION} or newer"
        )


class ProjectSplitNeedsWorkspaceError(Exception):
    """Raised when a save raises a project's document floor across the split without the caller's workspace."""

    def __init__(self, project_id: str, document_floor: int) -> None:
        self.document_floor = document_floor
        super().__init__(
            f"Project {project_id} is held whole at document schema {document_floor}; the save that splits it into"
            f" content at schema {SPLIT_PROJECT_DOCUMENT_SCHEMA_VERSION} or newer must carry the caller's workspace"
        )


class ProjectCanvasSchemaDowngradeError(Exception):
    """Raised when a save tries to lower a project's compatibility floor."""

    def __init__(self, project_id: str, current_version: int, requested_version: int) -> None:
        self.project_id = project_id
        self.current_version = current_version
        self.requested_version = requested_version
        super().__init__(
            f"Project {project_id} requires canvas schema {current_version}; its compatibility floor cannot be"
            f" lowered to schema {requested_version}"
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


PROJECT_BOARD_SNAPSHOT_MAX_ITEMS = 20_000
"""The most items, over all of a project's boards, the snapshot route will enumerate.

The snapshot is unpaginated on purpose — the caller that needs it, exporting a project, has to hold
the whole list anyway. That is a reason not to page it, not a reason to have no ceiling: the answer
is built entirely in memory and any client with a project id can ask for it. Matched to the
archive's own `INVK_MAX_ENTRIES`, so a board this route refuses is one the export could not have
packed either.
"""


class ProjectBoardTooLargeError(Exception):
    """Raised when a project's boards hold more than the snapshot route will enumerate."""

    def __init__(self, project_id: str, limit: int) -> None:
        super().__init__(f"Project {project_id} has more than {limit} items on its boards")


class ProjectSummaryDTO(BaseModel):
    """Lightweight project listing entry; the document payload is omitted."""

    project_id: str = Field(description="The project's client-generated identifier")
    board_id: str = Field(description="The project's private board; only project APIs may rename or delete it")
    name: str = Field(description="The project's display name")
    revision: int = Field(description="Monotonic revision, incremented on every save")
    minimum_canvas_schema_version: int = Field(
        ge=1, description="Oldest canvas schema a client must support before it may read or write this project"
    )
    minimum_document_schema_version: int = Field(
        ge=1,
        description="Oldest project document schema a client must support before it may read or write this project",
    )
    created_at: str = Field(description="When the project was created")
    updated_at: str = Field(description="When the project content was last saved")
    workspace_updated_at: Optional[str] = Field(
        default=None, description="When the caller last saved their workspace in the project; null if they have none"
    )


class ProjectRecordDTO(ProjectSummaryDTO):
    """Full project record including the client-owned document."""

    data: dict[str, Any] = Field(description="The opaque client-owned project content (document schema 4 on)")


class ProjectWorkspaceDTO(BaseModel):
    """One account's workspace in a project: how they work on it (layout, widgets, generation settings)."""

    data: dict[str, Any] = Field(description="The opaque client-owned workspace document")
    revision: int = Field(description="Monotonic revision, incremented on every save of this workspace")
    updated_at: str = Field(description="When the workspace was last saved")


class ProjectWithWorkspaceDTO(ProjectRecordDTO):
    """A project's content together with the caller's workspace in it."""

    workspace: Optional[ProjectWorkspaceDTO] = Field(
        description=(
            "The caller's workspace. On a read, null until they save one, when the client builds a default; on a"
            " save, the workspace the save carried and stored, or null when it carried none"
        )
    )


class ProjectUpdateWorkspaceRequest(BaseModel):
    """The caller's workspace, saved in the same transaction as the project content it accompanies.

    Only one client needs it: one that splits a whole project (document schema 3 or older) into content and
    workspace. The raised floor, the content and the member's first workspace row then land together or not at all.
    The save must leave the document floor at the split schema or above, or it is refused whole.
    Landed by half -- the content stored, the workspace not -- the project would join every later load with a
    default workspace, and the member's layout, settings, history and staging would be gone for good.
    """

    data: dict[str, Any] = Field(description="The opaque client-owned workspace document")
    expected_revision: Optional[int] = Field(
        description="The workspace revision this save is based on, or null when the caller has no workspace in the"
        " project yet; a mismatch refuses the whole save, content included"
    )


class ProjectBoardItemDTO(BaseModel):
    """One visible item on a project's board."""

    category: Literal["general", "control", "mask", "user"] = Field(description="The item's gallery category")
    kind: Literal["image", "video"] = Field(description="Which namespace the name belongs to")
    name: str = Field(description="The server-assigned image or video name")
    starred: bool = Field(description="Whether the item is starred")


class ProjectBoardSnapshotBoardDTO(BaseModel):
    """One of the project's boards and the visible items on it."""

    board_id: str = Field(description="The board's id on this install")
    name: str = Field(description="The board's name; the inbox carries the project's")
    is_inbox: bool = Field(description="Whether this is the project's inbox, which every project has exactly one of")
    archived: bool = Field(description="Whether the board is archived")
    items: list[ProjectBoardItemDTO] = Field(description="The board's visible items, ordered by kind then name")


class ProjectBoardSnapshotDTO(BaseModel):
    """Everything a project's boards hold that the gallery would show.

    This is the enumeration an export needs in order to carry a project's whole workspace rather
    than only the media its document happens to reference. Intermediates and the canvas's private
    `other` category are excluded, because neither is something the gallery shows on a board.

    Deliberately unversioned: `.invk`'s `board.json` carries its own version, so the archive format
    is free to change without the wire format following it, and vice versa.
    """

    boards: list[ProjectBoardSnapshotBoardDTO] = Field(
        description="The project's boards, inbox first, then the rest in creation order"
    )
