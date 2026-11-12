import json
import threading
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import IO, Any, BinaryIO

from pydantic import Field, TypeAdapter, ValidationError

from invokeai.app.services.files.files_base import FileServiceBase
from invokeai.app.services.files.files_common import (
    DEFAULT_FILE_LIBRARY_MAX_BYTES,
    DEFAULT_FILE_UPLOAD_MAX_BYTES,
    SUPPORTED_FILE_EXTENSIONS,
    SUPPORTED_FILE_MIME_TYPES,
    FileAccessDeniedException,
    FileDTO,
    FileMetadataException,
    FileNotFoundException,
    FileQuotaExceededException,
    FileTooLargeException,
    UnsupportedFileTypeException,
)
from invokeai.app.util.misc import uuid_string
from invokeai.backend.util.logging import InvokeAILogger


class FileRecord(FileDTO):
    stored_file_name: str = Field(description="The file name used on disk.")
    user_id: str | None = Field(default=None, description="The user who uploaded this file.")


FileRecordAdapter = TypeAdapter(FileRecord)


class DiskFileService(FileServiceBase):
    def __init__(
        self,
        storage_path: Path,
        max_file_size: int = DEFAULT_FILE_UPLOAD_MAX_BYTES,
        max_storage_size: int = DEFAULT_FILE_LIBRARY_MAX_BYTES,
    ) -> None:
        self._storage_path = storage_path
        self._max_file_size = max_file_size
        self._max_storage_size = max_storage_size
        self._logger = InvokeAILogger.get_logger(name=self.__class__.__name__)
        self._validate_storage_folder()
        self._storage_lock = threading.Lock()
        # Recount on every construction so restarts (and files removed outside the service) are reflected.
        self._used_bytes = self._calculate_used_bytes()

    def save(
        self,
        file_name: str,
        content_type: str | None,
        file: BinaryIO,
        user_id: str | None,
    ) -> FileDTO:
        safe_file_name = self._sanitize_file_name(file_name)
        extension = self._get_supported_extension(safe_file_name)
        normalized_content_type = self._normalize_content_type(content_type)
        self._validate_content_type(normalized_content_type)

        file_id = uuid_string()
        stored_file_name = f"{file_id}{extension}"
        file_path = self._get_file_path(stored_file_name)
        metadata_path = self._get_metadata_path(file_id)
        metadata_tmp_path = metadata_path.with_name(f"{metadata_path.name}.tmp")

        size_bytes = 0
        reserved_bytes = 0
        try:
            with open(file_path, "wb") as out_file:
                while True:
                    chunk = file.read(1024 * 1024)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > self._max_file_size:
                        raise FileTooLargeException(f"File exceeds the maximum size of {self._max_file_size} bytes.")
                    self._reserve_storage(len(chunk))
                    reserved_bytes += len(chunk)
                    out_file.write(chunk)

            record = FileRecord(
                file_id=file_id,
                file_name=safe_file_name,
                stored_file_name=stored_file_name,
                content_type=normalized_content_type,
                size_bytes=size_bytes,
                created_at=datetime.now(timezone.utc),
                user_id=user_id,
            )
            # Write the metadata beside its final path and swap it in, so a crash mid-write cannot leave an
            # unreadable record for a file that was saved completely.
            metadata_tmp_path.write_text(record.model_dump_json(), encoding="utf-8")
            metadata_tmp_path.replace(metadata_path)
            return self._record_to_dto(record)
        except Exception:
            file_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            metadata_tmp_path.unlink(missing_ok=True)
            self._release_storage(reserved_bytes)
            raise

    def get_dto(self, file_id: str, user_id: str | None = None) -> FileDTO:
        record = self._get_record(file_id, user_id=user_id)
        self._get_existing_file_path(record)
        return self._record_to_dto(record)

    def get_path(self, file_id: str, user_id: str | None = None) -> Path:
        record = self._get_record(file_id, user_id=user_id)
        return self._get_existing_file_path(record)

    def open(self, file_id: str, mode: str = "rb", user_id: str | None = None) -> IO[Any]:
        if mode not in {"rb", "r"}:
            raise ValueError("Managed files may only be opened for reading.")
        path = self.get_path(file_id, user_id=user_id)
        if mode == "r":
            return open(path, mode, encoding="utf-8")
        return open(path, mode)

    def delete(self, file_id: str, user_id: str | None = None) -> None:
        record = self._get_record(file_id, user_id=user_id)
        file_path = self._get_file_path(record.stored_file_name)
        try:
            size_bytes = file_path.stat().st_size
        except FileNotFoundError:
            size_bytes = 0
        file_path.unlink(missing_ok=True)
        self._get_metadata_path(file_id).unlink(missing_ok=True)
        self._release_storage(size_bytes)

    def _get_record(self, file_id: str, user_id: str | None = None) -> FileRecord:
        metadata_path = self._get_metadata_path(file_id)
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            record = FileRecordAdapter.validate_python(data)
        except FileNotFoundError as e:
            raise FileNotFoundException(f"File not found: {file_id}") from e
        except (JSONDecodeError, ValidationError) as e:
            self._logger.warning(f"Invalid managed file metadata for file: {file_id}")
            raise FileMetadataException(f"Invalid file metadata: {file_id}") from e
        except OSError as e:
            self._logger.warning(f"Unable to read managed file metadata for file: {file_id}: {e}")
            raise FileMetadataException(f"Unable to read file metadata: {file_id}") from e

        if record.user_id != user_id:
            raise FileAccessDeniedException(f"Not authorized to access file: {file_id}")
        return record

    def _get_existing_file_path(self, record: FileRecord) -> Path:
        path = self._get_file_path(record.stored_file_name)
        if not path.exists():
            raise FileNotFoundException(f"File not found: {record.file_id}")
        return path

    def _get_file_path(self, stored_file_name: str) -> Path:
        path = self._storage_path / stored_file_name
        resolved_base = self._storage_path.resolve()
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(resolved_base):
            raise ValueError("File path outside storage folder, potential directory traversal detected.")
        return resolved_path

    def _get_metadata_path(self, file_id: str) -> Path:
        if Path(file_id).name != file_id:
            raise ValueError("Invalid file ID, potential directory traversal detected.")
        return self._get_file_path(f"{file_id}.meta.json")

    def _validate_storage_folder(self) -> None:
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def _calculate_used_bytes(self) -> int:
        used_bytes = 0
        for entry in self._storage_path.iterdir():
            # Metadata is part of the record, not the stored document; counting only data files keeps the limit
            # meaningful and matches what `save` reserves.
            if entry.is_file() and not entry.name.endswith((".meta.json", ".tmp")):
                try:
                    used_bytes += entry.stat().st_size
                except FileNotFoundError:
                    continue
        return used_bytes

    def _reserve_storage(self, size_bytes: int) -> None:
        with self._storage_lock:
            if self._used_bytes + size_bytes > self._max_storage_size:
                raise FileQuotaExceededException(
                    f"Managed file storage is full (limit {self._max_storage_size} bytes)."
                )
            self._used_bytes += size_bytes

    def _release_storage(self, size_bytes: int) -> None:
        if size_bytes == 0:
            return
        with self._storage_lock:
            self._used_bytes = max(0, self._used_bytes - size_bytes)

    @staticmethod
    def _record_to_dto(record: FileRecord) -> FileDTO:
        return FileDTO(**record.model_dump(exclude={"stored_file_name", "user_id"}))

    @staticmethod
    def _sanitize_file_name(file_name: str) -> str:
        safe_file_name = Path(file_name.replace("\x00", "")).name.strip()
        if not safe_file_name:
            raise UnsupportedFileTypeException("Missing file name.")
        return safe_file_name

    @staticmethod
    def _get_supported_extension(file_name: str) -> str:
        extension = Path(file_name).suffix.lower()
        if extension not in SUPPORTED_FILE_EXTENSIONS:
            raise UnsupportedFileTypeException(f"Unsupported file extension: {extension}")
        return extension

    @staticmethod
    def _normalize_content_type(content_type: str | None) -> str:
        if not content_type:
            return "application/octet-stream"
        return content_type.split(";", 1)[0].strip().lower()

    @staticmethod
    def _validate_content_type(content_type: str) -> None:
        # Some browsers or reverse proxies send application/octet-stream for document files, and some report
        # vendor aliases such as application/vnd.ms-excel for CSV. The extension allowlist remains the primary
        # check; the MIME allowlist only rejects clearly unrelated types.
        if content_type in {"application/octet-stream", ""}:
            return
        if content_type not in SUPPORTED_FILE_MIME_TYPES:
            raise UnsupportedFileTypeException(f"Unsupported content type: {content_type}")
