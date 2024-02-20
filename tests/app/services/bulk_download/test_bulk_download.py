import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile

import pytest

from invokeai.app.services.board_records.board_records_common import BoardRecord, BoardRecordNotFoundException
from invokeai.app.services.bulk_download.bulk_download_common import BulkDownloadTargetException
from invokeai.app.services.bulk_download.bulk_download_default import BulkDownloadService
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecordNotFoundException,
    ResourceOrigin,
)
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from tests.test_nodes import TestEventService


@pytest.fixture
def mock_image_dto() -> ImageDTO:
    """Create a mock ImageDTO."""
    return ImageDTO(
        image_name="mock_image.png",
        board_id="12345",
        image_url="None",
        width=100,
        height=100,
        thumbnail_url="None",
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        created_at="None",
        updated_at="None",
        starred=False,
        has_workflow=False,
        is_intermediate=False,
    )


@pytest.fixture(autouse=True)
def mock_temporary_directory(monkeypatch: Any, tmp_path: Path):
    """Mock the TemporaryDirectory class so that it uses the tmp_path fixture."""

    class MockTemporaryDirectory(TemporaryDirectory):
        def __init__(self):
            super().__init__(dir=tmp_path)
            self.name = tmp_path

    def mock_TemporaryDirectory(*args, **kwargs):
        return MockTemporaryDirectory()

    monkeypatch.setattr(
        "invokeai.app.services.bulk_download.bulk_download_default.TemporaryDirectory", mock_TemporaryDirectory
    )


def test_get_path_when_file_exists(tmp_path: Path) -> None:
    """Test get_path when the file exists."""

    bulk_download_service = BulkDownloadService()

    # Create a directory at tmp_path/bulk_downloads
    test_bulk_downloads_dir: Path = tmp_path / "bulk_downloads"
    test_bulk_downloads_dir.mkdir(parents=True, exist_ok=True)

    # Create a file at tmp_path/bulk_downloads/test.zip
    test_file_path: Path = test_bulk_downloads_dir / "test.zip"
    test_file_path.touch()

    assert bulk_download_service.get_path("test.zip") == str(test_file_path)


def test_get_path_when_file_does_not_exist(tmp_path: Path) -> None:
    """Test get_path when the file does not exist."""

    bulk_download_service = BulkDownloadService()
    with pytest.raises(BulkDownloadTargetException):
        bulk_download_service.get_path("test")


def test_bulk_downloads_dir_created_at_start(tmp_path: Path) -> None:
    """Test that the bulk_downloads directory is created at start."""

    BulkDownloadService()
    assert (tmp_path / "bulk_downloads").exists()


def test_handler_image_names(tmp_path: Path, monkeypatch: Any, mock_image_dto: ImageDTO, mock_invoker: Invoker):
    """Test that the handler creates the zip file correctly when given a list of image names."""

    expected_zip_path, expected_image_path, mock_image_contents = prepare_handler_test(
        tmp_path, monkeypatch, mock_image_dto, mock_invoker
    )

    bulk_download_service = BulkDownloadService()
    bulk_download_service.start(mock_invoker)
    bulk_download_service.handler([mock_image_dto.image_name], None, None)

    assert_handler_success(
        expected_zip_path, expected_image_path, mock_image_contents, tmp_path, mock_invoker.services.events
    )


def test_generate_id(monkeypatch: Any):
    """Test that the generate_id method generates a unique id."""

    bulk_download_service = BulkDownloadService()

    monkeypatch.setattr("invokeai.app.services.bulk_download.bulk_download_default.uuid_string", lambda: "test")

    assert bulk_download_service.generate_item_id(None) == "test"


def test_generate_id_with_board_id(monkeypatch: Any, mock_invoker: Invoker):
    """Test that the generate_id method generates a unique id with a board id."""

    bulk_download_service = BulkDownloadService()
    bulk_download_service.start(mock_invoker)

    def mock_board_get(*args, **kwargs):
        return BoardRecord(board_id="12345", board_name="test_board_name", created_at="None", updated_at="None")

    monkeypatch.setattr(mock_invoker.services.board_records, "get", mock_board_get)

    monkeypatch.setattr("invokeai.app.services.bulk_download.bulk_download_default.uuid_string", lambda: "test")

    assert bulk_download_service.generate_item_id("12345") == "test_board_name_test"


def test_generate_id_with_default_board_id(monkeypatch: Any):
    """Test that the generate_id method generates a unique id with a board id."""

    bulk_download_service = BulkDownloadService()

    monkeypatch.setattr("invokeai.app.services.bulk_download.bulk_download_default.uuid_string", lambda: "test")

    assert bulk_download_service.generate_item_id("none") == "Uncategorized_test"


def test_handler_board_id(tmp_path: Path, monkeypatch: Any, mock_image_dto: ImageDTO, mock_invoker: Invoker):
    """Test that the handler creates the zip file correctly when given a board id."""

    expected_zip_path, expected_image_path, mock_image_contents = prepare_handler_test(
        tmp_path, monkeypatch, mock_image_dto, mock_invoker
    )

    def mock_board_get(*args, **kwargs):
        return BoardRecord(board_id="12345", board_name="test_board_name", created_at="None", updated_at="None")

    monkeypatch.setattr(mock_invoker.services.board_records, "get", mock_board_get)

    def mock_get_many(*args, **kwargs):
        return OffsetPaginatedResults(limit=-1, total=1, offset=0, items=[mock_image_dto])

    monkeypatch.setattr(mock_invoker.services.images, "get_many", mock_get_many)

    bulk_download_service = BulkDownloadService()
    bulk_download_service.start(mock_invoker)
    bulk_download_service.handler([], "test", None)

    assert_handler_success(
        expected_zip_path, expected_image_path, mock_image_contents, tmp_path, mock_invoker.services.events
    )


def test_handler_board_id_default(tmp_path: Path, monkeypatch: Any, mock_image_dto: ImageDTO, mock_invoker: Invoker):
    """Test that the handler creates the zip file correctly when given a board id."""

    _, expected_image_path, mock_image_contents = prepare_handler_test(
        tmp_path, monkeypatch, mock_image_dto, mock_invoker
    )

    def mock_get_many(*args, **kwargs):
        return OffsetPaginatedResults(limit=-1, total=1, offset=0, items=[mock_image_dto])

    monkeypatch.setattr(mock_invoker.services.images, "get_many", mock_get_many)

    bulk_download_service = BulkDownloadService()
    bulk_download_service.start(mock_invoker)
    bulk_download_service.handler([], "none", None)

    expected_zip_path: Path = tmp_path / "bulk_downloads" / "test.zip"

    assert_handler_success(
        expected_zip_path, expected_image_path, mock_image_contents, tmp_path, mock_invoker.services.events
    )


def test_handler_bulk_download_item_id_given(
    tmp_path: Path, monkeypatch: Any, mock_image_dto: ImageDTO, mock_invoker: Invoker
):
    """Test that the handler creates the zip file correctly when given a pregenerated bulk download item id."""

    _, expected_image_path, mock_image_contents = prepare_handler_test(
        tmp_path, monkeypatch, mock_image_dto, mock_invoker
    )

    def mock_get_many(*args, **kwargs):
        return OffsetPaginatedResults(limit=-1, total=1, offset=0, items=[mock_image_dto])

    monkeypatch.setattr(mock_invoker.services.images, "get_many", mock_get_many)

    bulk_download_service = BulkDownloadService()
    bulk_download_service.start(mock_invoker)
    bulk_download_service.handler([mock_image_dto.image_name], None, "test_id")

    expected_zip_path: Path = tmp_path / "bulk_downloads" / "test_id.zip"

    assert_handler_success(
        expected_zip_path, expected_image_path, mock_image_contents, tmp_path, mock_invoker.services.events
    )


def prepare_handler_test(tmp_path: Path, monkeypatch: Any, mock_image_dto: ImageDTO, mock_invoker: Invoker):
    """Prepare the test for the handler tests."""

    def mock_uuid_string():
        return "test"

    # You have to patch the function within the module it's being imported into. This is strange, but it works.
    # See http://www.gregreda.com/2021/06/28/mocking-imported-module-function-python/
    monkeypatch.setattr("invokeai.app.services.bulk_download.bulk_download_default.uuid_string", mock_uuid_string)

    expected_zip_path: Path = tmp_path / "bulk_downloads" / "test.zip"
    expected_image_path: Path = (
        tmp_path / "bulk_downloads" / mock_image_dto.image_category.value / mock_image_dto.image_name
    )

    # Mock the get_dto method so that when the image dto needs to be retrieved it is returned
    def mock_get_dto(*args, **kwargs):
        return mock_image_dto

    monkeypatch.setattr(mock_invoker.services.images, "get_dto", mock_get_dto)

    # This is used when preparing all images for a given board
    def mock_get_all_board_image_names_for_board(*args, **kwargs):
        return [mock_image_dto.image_name]

    monkeypatch.setattr(
        mock_invoker.services.board_image_records,
        "get_all_board_image_names_for_board",
        mock_get_all_board_image_names_for_board,
    )

    # Create a mock image file so that the contents of the zip file are not empty
    mock_image_path: Path = tmp_path / mock_image_dto.image_name
    mock_image_contents: str = "Totally an image"
    mock_image_path.write_text(mock_image_contents)

    def mock_get_path(*args, **kwargs):
        return str(mock_image_path)

    monkeypatch.setattr(mock_invoker.services.images, "get_path", mock_get_path)

    return expected_zip_path, expected_image_path, mock_image_contents


def assert_handler_success(
    expected_zip_path: Path,
    expected_image_path: Path,
    mock_image_contents: str,
    tmp_path: Path,
    event_bus: TestEventService,
):
    """Assert that the handler was successful."""
    # Check that the zip file was created
    assert expected_zip_path.exists()
    assert expected_zip_path.is_file()
    assert expected_zip_path.stat().st_size > 0

    # Check that the zip contents are expected
    with ZipFile(expected_zip_path, "r") as zip_file:
        zip_file.extractall(tmp_path / "bulk_downloads")
        assert expected_image_path.exists()
        assert expected_image_path.is_file()
        assert expected_image_path.stat().st_size > 0
        assert expected_image_path.read_text() == mock_image_contents

    # Check that the correct events were emitted
    assert len(event_bus.events) == 2
    assert event_bus.events[0].event_name == "bulk_download_started"
    assert event_bus.events[1].event_name == "bulk_download_completed"
    assert event_bus.events[1].payload["bulk_download_item_name"] == os.path.basename(expected_zip_path)


def test_handler_on_image_not_found(tmp_path: Path, monkeypatch: Any, mock_image_dto: ImageDTO, mock_invoker: Invoker):
    """Test that the handler emits an error event when the image is not found."""
    exception: Exception = ImageRecordNotFoundException("Image not found")

    def mock_get_dto(*args, **kwargs):
        raise exception

    monkeypatch.setattr(mock_invoker.services.images, "get_dto", mock_get_dto)

    execute_handler_test_on_error(tmp_path, monkeypatch, mock_image_dto, mock_invoker, exception)


def test_handler_on_board_not_found(tmp_path: Path, monkeypatch: Any, mock_image_dto: ImageDTO, mock_invoker: Invoker):
    """Test that the handler emits an error event when the image is not found."""

    exception: Exception = BoardRecordNotFoundException("Image not found")

    def mock_get_board_name(*args, **kwargs):
        raise exception

    monkeypatch.setattr(mock_invoker.services.images, "get_dto", mock_get_board_name)

    execute_handler_test_on_error(tmp_path, monkeypatch, mock_image_dto, mock_invoker, exception)


def test_handler_on_generic_exception(
    tmp_path: Path, monkeypatch: Any, mock_image_dto: ImageDTO, mock_invoker: Invoker
):
    """Test that the handler emits an error event when the image is not found."""

    exception: Exception = Exception("Generic exception")

    def mock_get_board_name(*args, **kwargs):
        raise exception

    monkeypatch.setattr(mock_invoker.services.images, "get_dto", mock_get_board_name)

    with pytest.raises(Exception):  # noqa: B017
        execute_handler_test_on_error(tmp_path, monkeypatch, mock_image_dto, mock_invoker, exception)

    event_bus: TestEventService = mock_invoker.services.events

    assert len(event_bus.events) == 2
    assert event_bus.events[0].event_name == "bulk_download_started"
    assert event_bus.events[1].event_name == "bulk_download_failed"
    assert event_bus.events[1].payload["error"] == exception.__str__()


def execute_handler_test_on_error(
    tmp_path: Path, monkeypatch: Any, mock_image_dto: ImageDTO, mock_invoker: Invoker, error: Exception
):
    bulk_download_service = BulkDownloadService()
    bulk_download_service.start(mock_invoker)
    bulk_download_service.handler([mock_image_dto.image_name], None, None)

    event_bus: TestEventService = mock_invoker.services.events

    assert len(event_bus.events) == 2
    assert event_bus.events[0].event_name == "bulk_download_started"
    assert event_bus.events[1].event_name == "bulk_download_failed"
    assert event_bus.events[1].payload["error"] == error.__str__()


def test_delete(tmp_path: Path):
    """Test that the delete method removes the bulk download file."""

    bulk_download_service = BulkDownloadService()

    mock_file: Path = tmp_path / "bulk_downloads" / "test.zip"
    mock_file.write_text("contents")

    bulk_download_service.delete("test.zip")

    assert (tmp_path / "bulk_downloads").exists()
    assert len(os.listdir(tmp_path / "bulk_downloads")) == 0


def test_stop(tmp_path: Path):
    """Test that the stop method removes the bulk download file and not any directories."""

    bulk_download_service = BulkDownloadService()

    mock_file: Path = tmp_path / "bulk_downloads" / "test.zip"
    mock_file.write_text("contents")

    mock_dir: Path = tmp_path / "bulk_downloads" / "test"
    mock_dir.mkdir(parents=True, exist_ok=True)

    bulk_download_service.stop()

    assert not (tmp_path / "bulk_downloads").exists()
