"""Domain types and errors for the managed and configured font libraries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

SUPPORTED_FONT_EXTENSIONS = frozenset({".ttf", ".otf", ".woff", ".woff2"})


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


class FontServiceError(RuntimeError):
    """Base class for errors raised by the font service."""


class FontNotFoundError(FontServiceError):
    """The font does not exist or is not visible to the requesting account."""


class FontForbiddenError(FontServiceError):
    """The caller is not permitted to perform the requested operation."""


class FontDeleteForbiddenError(FontForbiddenError):
    """A read-only or otherwise protected font cannot be deleted."""


class FontValidationError(FontServiceError):
    """The file is too large, unsupported, malformed, or missing font metadata."""


class FontQuotaExceededError(FontServiceError):
    """An upload would exceed the configured library quota."""


class FontChangedError(FontServiceError):
    """A font ID no longer points at the requested content hash."""


class FontInstanceError(FontServiceError):
    """A variable font instance could not be produced."""


class FontStorageError(FontServiceError):
    """A managed font file could not be read or safely stored."""
