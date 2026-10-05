"""Domain types for the managed and configured font libraries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FontScope(StrEnum):
    """The account scope of an uploaded font."""

    PRIVATE = "private"
    SHARED = "shared"
    ALL = "all"


class FontSource(StrEnum):
    """How the font file is supplied to InvokeAI."""

    UPLOADED = "uploaded"
    DIRECTORY = "directory"


@dataclass(frozen=True)
class FontAxis:
    """One OpenType variation axis."""

    tag: str
    label: str
    minimum: float
    default: float
    maximum: float
    hidden: bool


@dataclass(frozen=True)
class FontInstance:
    """A named OpenType variation instance."""

    name: str
    coordinates: dict[str, float]


@dataclass(frozen=True)
class FontRecord:
    """A single font file face indexed by the service."""

    id: str
    family: str
    label: str
    style: str
    weight: int
    content_hash: str
    scope: FontScope
    source: FontSource
    filename: str
    byte_size: int
    axes: tuple[FontAxis, ...]
    instances: tuple[FontInstance, ...]
    # Internal ownership key. API serializers intentionally omit this field.
    owner_id: str | None = None
    storage_path: str | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class FontValidationResult:
    """Validated metadata for a file that has not been persisted yet."""

    filename: str
    family: str
    label: str
    style: str
    weight: int
    content_hash: str
    byte_size: int
    axes: tuple[FontAxis, ...]
    instances: tuple[FontInstance, ...]


@dataclass(frozen=True)
class FontUploadResult:
    """The result of an upload, including whether a hash dedupe was used."""

    font: FontRecord
    created: bool
