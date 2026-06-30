from datetime import datetime

from pydantic import BaseModel, Field

DEFAULT_FILE_UPLOAD_MAX_BYTES = 50 * 1024 * 1024

SUPPORTED_FILE_EXTENSIONS = frozenset(
    {
        ".csv",
        ".json",
        ".md",
        ".markdown",
        ".pdf",
        ".txt",
        ".yaml",
        ".yml",
    }
)

SUPPORTED_FILE_MIME_TYPES = frozenset(
    {
        "application/json",
        "application/pdf",
        "application/x-yaml",
        "application/yaml",
        "text/csv",
        "text/markdown",
        "text/plain",
        "text/x-markdown",
        "text/x-yaml",
        "text/yaml",
    }
)


class FileDTO(BaseModel):
    file_id: str = Field(description="The managed file ID.")
    file_name: str = Field(description="The original file name.")
    content_type: str = Field(description="The uploaded file content type.")
    size_bytes: int = Field(description="The size of the file in bytes.", ge=0)
    created_at: datetime = Field(description="When the file was uploaded.")


class FileStorageException(Exception):
    """Base exception for managed file storage errors."""


class FileNotFoundException(FileStorageException):
    """Raised when a managed file cannot be found."""


class UnsupportedFileTypeException(FileStorageException):
    """Raised when a managed file upload is not allowed."""


class FileTooLargeException(FileStorageException):
    """Raised when a managed file upload exceeds the size limit."""


class FileAccessDeniedException(FileStorageException):
    """Raised when a user cannot access a managed file."""


class FileMetadataException(FileStorageException):
    """Raised when managed file metadata cannot be read or validated."""
