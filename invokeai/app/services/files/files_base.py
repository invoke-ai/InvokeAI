from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any, BinaryIO

from invokeai.app.services.files.files_common import FileDTO


class FileServiceBase(ABC):
    @abstractmethod
    def save(
        self,
        file_name: str,
        content_type: str | None,
        file: BinaryIO,
        user_id: str | None,
    ) -> FileDTO:
        """Saves a managed file and returns its DTO."""

    @abstractmethod
    def get_dto(self, file_id: str, user_id: str | None = None) -> FileDTO:
        """Gets a managed file DTO."""

    @abstractmethod
    def get_path(self, file_id: str, user_id: str | None = None) -> Path:
        """Gets the path to a managed file."""

    @abstractmethod
    def open(self, file_id: str, mode: str = "rb", user_id: str | None = None) -> IO[Any]:
        """Opens a managed file."""

    @abstractmethod
    def delete(self, file_id: str, user_id: str | None = None) -> None:
        """Deletes a managed file."""
