from io import BytesIO
from pathlib import Path

import pytest

from invokeai.app.services.files.files_common import (
    FileAccessDeniedException,
    FileMetadataException,
    FileNotFoundException,
    FileTooLargeException,
    UnsupportedFileTypeException,
)
from invokeai.app.services.files.files_disk import DiskFileService


def test_save_open_get_path_and_delete_round_trip(tmp_path: Path) -> None:
    service = DiskFileService(tmp_path)

    dto = service.save(
        file_name="report.pdf",
        content_type="application/pdf",
        file=BytesIO(b"%PDF-1.7 test"),
        user_id="user-1",
    )

    assert dto.file_name == "report.pdf"
    assert dto.content_type == "application/pdf"
    assert dto.size_bytes == len(b"%PDF-1.7 test")

    path = service.get_path(dto.file_id, user_id="user-1")
    assert path.is_relative_to(tmp_path.resolve())
    assert path.name == f"{dto.file_id}.pdf"
    assert path.read_bytes() == b"%PDF-1.7 test"

    with service.open(dto.file_id, user_id="user-1") as opened_file:
        assert opened_file.read() == b"%PDF-1.7 test"

    service.delete(dto.file_id, user_id="user-1")

    with pytest.raises(FileNotFoundException):
        service.get_dto(dto.file_id, user_id="user-1")


def test_json_upload_does_not_collide_with_metadata(tmp_path: Path) -> None:
    service = DiskFileService(tmp_path)
    content = b'{"hello": "world"}'

    dto = service.save(
        file_name="data.json",
        content_type="application/json",
        file=BytesIO(content),
        user_id="user-1",
    )

    file_path = service.get_path(dto.file_id, user_id="user-1")
    metadata_path = tmp_path / f"{dto.file_id}.meta.json"

    assert file_path.name == f"{dto.file_id}.json"
    assert file_path.read_bytes() == content
    assert metadata_path.exists()
    assert metadata_path != file_path


def test_open_text_mode_reads_utf8(tmp_path: Path) -> None:
    service = DiskFileService(tmp_path)
    content = "cafe 日本語"
    dto = service.save(
        file_name="notes.md",
        content_type="text/markdown",
        file=BytesIO(content.encode("utf-8")),
        user_id=None,
    )

    with service.open(dto.file_id, mode="r", user_id=None) as opened_file:
        assert opened_file.read() == content


def test_get_dto_fails_when_file_is_missing(tmp_path: Path) -> None:
    service = DiskFileService(tmp_path)
    dto = service.save(file_name="notes.txt", content_type="text/plain", file=BytesIO(b"notes"), user_id=None)

    service.get_path(dto.file_id).unlink()

    with pytest.raises(FileNotFoundException):
        service.get_dto(dto.file_id)


def test_get_dto_fails_when_metadata_is_invalid(tmp_path: Path) -> None:
    service = DiskFileService(tmp_path)
    dto = service.save(file_name="notes.txt", content_type="text/plain", file=BytesIO(b"notes"), user_id=None)

    (tmp_path / f"{dto.file_id}.meta.json").write_text("{", encoding="utf-8")

    with pytest.raises(FileMetadataException):
        service.get_dto(dto.file_id)


def test_sanitizes_file_name_to_basename(tmp_path: Path) -> None:
    service = DiskFileService(tmp_path)

    dto = service.save(
        file_name="../nested/data.csv",
        content_type="text/csv",
        file=BytesIO(b"a,b\n1,2\n"),
        user_id=None,
    )

    assert dto.file_name == "data.csv"
    assert service.get_path(dto.file_id).is_relative_to(tmp_path.resolve())


@pytest.mark.parametrize("file_name", ["image.png", "archive.zip", "no-extension"])
def test_rejects_unsupported_extension(tmp_path: Path, file_name: str) -> None:
    service = DiskFileService(tmp_path)

    with pytest.raises(UnsupportedFileTypeException):
        service.save(file_name=file_name, content_type="application/octet-stream", file=BytesIO(b"data"), user_id=None)


@pytest.mark.parametrize("content_type", ["image/png", "application/zip"])
def test_rejects_unsupported_content_type(tmp_path: Path, content_type: str) -> None:
    service = DiskFileService(tmp_path)

    with pytest.raises(UnsupportedFileTypeException):
        service.save(file_name="data.json", content_type=content_type, file=BytesIO(b"{}"), user_id=None)


@pytest.mark.parametrize("content_type", [None, "", "application/octet-stream"])
def test_allows_octet_stream_for_allowed_extensions(tmp_path: Path, content_type: str | None) -> None:
    service = DiskFileService(tmp_path)

    dto = service.save(file_name="notes.md", content_type=content_type, file=BytesIO(b"# Notes"), user_id=None)

    assert dto.content_type == "application/octet-stream"


def test_rejects_files_over_size_limit_and_cleans_up(tmp_path: Path) -> None:
    service = DiskFileService(tmp_path, max_file_size=4)

    with pytest.raises(FileTooLargeException):
        service.save(file_name="large.txt", content_type="text/plain", file=BytesIO(b"12345"), user_id=None)

    assert list(tmp_path.iterdir()) == []


def test_user_scoped_access(tmp_path: Path) -> None:
    service = DiskFileService(tmp_path)
    dto = service.save(file_name="private.txt", content_type="text/plain", file=BytesIO(b"secret"), user_id="user-1")

    service.get_dto(dto.file_id, user_id="user-1")

    with pytest.raises(FileAccessDeniedException):
        service.get_dto(dto.file_id, user_id=None)

    with pytest.raises(FileAccessDeniedException):
        service.get_dto(dto.file_id, user_id="user-2")


def test_rejects_write_modes(tmp_path: Path) -> None:
    service = DiskFileService(tmp_path)
    dto = service.save(file_name="notes.txt", content_type="text/plain", file=BytesIO(b"notes"), user_id=None)

    with pytest.raises(ValueError, match="only be opened for reading"):
        service.open(dto.file_id, mode="wb")
