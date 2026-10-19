"""Typed contracts for application-owned gallery maintenance."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GalleryMaintenanceOperation(str, Enum):
    REMOVE_MISSING = "remove_missing"
    ARCHIVE_UNTRACKED = "archive_untracked"
    REGENERATE_THUMBNAILS = "regenerate_thumbnails"


class GalleryMaintenancePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: GalleryMaintenanceOperation


class GalleryMaintenanceExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class GalleryMaintenancePreview(BaseModel):
    operation: GalleryMaintenanceOperation
    fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    examined_count: int = Field(ge=0)
    affected_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list, max_length=20)
    archive_path: str | None = None


class GalleryMaintenanceResult(BaseModel):
    operation: GalleryMaintenanceOperation
    status: Literal["completed", "partial", "no_op", "failed"]
    examined_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    records_removed: int = Field(ge=0)
    images_archived: int = Field(ge=0)
    thumbnails_archived: int = Field(ge=0)
    thumbnails_regenerated: int = Field(ge=0)
    archive_path: str | None = None
    backup_path: str | None = None
    errors: list[str] = Field(default_factory=list, max_length=20)


class GalleryMaintenanceError(Exception):
    """A maintenance scan or operation could not safely complete."""


class GalleryMaintenancePreviewChanged(GalleryMaintenanceError):
    """The current inventory no longer matches the preview the administrator confirmed."""


class GalleryMaintenanceConflict(GalleryMaintenanceError):
    """Another operation or active queue work prevents maintenance from starting."""
