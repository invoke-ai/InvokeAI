import datetime
from typing import Optional, Union

from pydantic import BaseModel, Field, StrictBool, StrictStr

from invokeai.app.services.image_records.image_records_common import ResourceOrigin
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class CanvasProjectRecordNotFoundException(Exception):
    """Raised when a canvas project record is not found."""

    def __init__(self, message="Canvas project record not found"):
        super().__init__(message)


class CanvasProjectRecordSaveException(Exception):
    """Raised when a canvas project record cannot be saved."""

    def __init__(self, message="Canvas project record not saved"):
        super().__init__(message)


class CanvasProjectRecordDeleteException(Exception):
    """Raised when a canvas project record cannot be deleted."""

    def __init__(self, message="Canvas project record not deleted"):
        super().__init__(message)


CANVAS_PROJECT_DTO_COLS = ", ".join(
    [
        "canvas_projects." + c
        for c in [
            "project_name",
            "project_origin",
            "name",
            "app_version",
            "width",
            "height",
            "image_count",
            "has_thumbnail",
            "starred",
            "is_intermediate",
            "user_id",
            "project_subfolder",
            "created_at",
            "updated_at",
            "deleted_at",
        ]
    ]
)


class CanvasProjectRecord(BaseModelExcludeNull):
    """Deserialized canvas project record."""

    project_name: str = Field(description="The unique name (ID) of the canvas project.")
    project_origin: ResourceOrigin = Field(description="The origin of the canvas project.")
    name: str = Field(description="The user-facing display name of the project.")
    app_version: str = Field(description="The InvokeAI app version this project was saved under.")
    width: int = Field(description="The bbox width of the canvas at save time.")
    height: int = Field(description="The bbox height of the canvas at save time.")
    image_count: int = Field(description="The number of images embedded in the project ZIP.")
    has_thumbnail: bool = Field(description="Whether the project has a preview thumbnail on disk.")
    starred: bool = Field(description="Whether this project is starred.")
    is_intermediate: bool = Field(description="Whether this is an intermediate project (almost always False).")
    user_id: str = Field(description="The id of the user that owns this project.")
    project_subfolder: str = Field(default="", description="The subfolder where the project is stored on disk.")
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the project.")
    updated_at: Union[datetime.datetime, str] = Field(description="The updated timestamp of the project.")
    deleted_at: Optional[Union[datetime.datetime, str]] = Field(
        default=None, description="The deleted timestamp of the project."
    )


class CanvasProjectRecordChanges(BaseModelExcludeNull, extra="allow"):
    """Allowed mutations on a canvas project record."""

    name: Optional[StrictStr] = Field(default=None, description="The project's new display name.")
    starred: Optional[StrictBool] = Field(default=None, description="The project's new starred state.")
    is_intermediate: Optional[StrictBool] = Field(
        default=None, description="The project's new is_intermediate flag."
    )


def deserialize_canvas_project_record(project_dict: dict) -> CanvasProjectRecord:
    """Deserializes a canvas project record from a sqlite row dict."""
    return CanvasProjectRecord(
        project_name=project_dict.get("project_name", "unknown"),
        project_origin=ResourceOrigin(project_dict.get("project_origin", ResourceOrigin.INTERNAL.value)),
        name=project_dict.get("name", ""),
        app_version=project_dict.get("app_version", "unknown"),
        width=project_dict.get("width", 0),
        height=project_dict.get("height", 0),
        image_count=project_dict.get("image_count", 0),
        has_thumbnail=bool(project_dict.get("has_thumbnail", False)),
        starred=bool(project_dict.get("starred", False)),
        is_intermediate=bool(project_dict.get("is_intermediate", False)),
        user_id=project_dict.get("user_id", "system"),
        project_subfolder=project_dict.get("project_subfolder", ""),
        created_at=project_dict.get("created_at", get_iso_timestamp()),
        updated_at=project_dict.get("updated_at", get_iso_timestamp()),
        deleted_at=project_dict.get("deleted_at", None),
    )


class CanvasProjectNamesResult(BaseModel):
    """Response containing ordered canvas project names with metadata for optimistic updates."""

    project_names: list[str] = Field(description="Ordered list of canvas project names.")
    starred_count: int = Field(description="Number of starred projects (when starred_first=True).")
    total_count: int = Field(description="Total number of projects matching the query.")
