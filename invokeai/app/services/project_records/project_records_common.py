"""Common types and errors for the project records service."""

from typing import Any

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


class ProjectSummaryDTO(BaseModel):
    """Lightweight project listing entry; the document payload is omitted."""

    project_id: str = Field(description="The project's client-generated identifier")
    name: str = Field(description="The project's display name")
    revision: int = Field(description="Monotonic revision, incremented on every save")
    created_at: str = Field(description="When the project was created")
    updated_at: str = Field(description="When the project was last saved")


class ProjectRecordDTO(ProjectSummaryDTO):
    """Full project record including the client-owned document."""

    data: dict[str, Any] = Field(description="The opaque client-owned project document")
