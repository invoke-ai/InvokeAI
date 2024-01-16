from pathlib import Path
from typing import Any

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.bulk_download.bulk_download_default import BulkDownloadService
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invoker import Invoker
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database

client = TestClient(app)


@pytest.fixture
def mock_services(tmp_path: Path) -> InvocationServices:
    configuration = InvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    logger = InvokeAILogger.get_logger()
    db = create_mock_sqlite_database(configuration, logger)

    return InvocationServices(
        board_image_records=None,  # type: ignore
        board_images=None,  # type: ignore
        board_records=SqliteBoardRecordStorage(db=db),
        boards=None,  # type: ignore
        bulk_download=BulkDownloadService(tmp_path),
        configuration=None,  # type: ignore
        events=None,  # type: ignore
        graph_execution_manager=None,  # type: ignore
        image_files=None,  # type: ignore
        image_records=None,  # type: ignore
        images=ImageService(),
        invocation_cache=None,  # type: ignore
        latents=None,  # type: ignore
        logger=logger,
        model_manager=None,  # type: ignore
        model_records=None,  # type: ignore
        download_queue=None,  # type: ignore
        model_install=None,  # type: ignore
        names=None,  # type: ignore
        performance_statistics=None,  # type: ignore
        processor=None,  # type: ignore
        queue=None,  # type: ignore
        session_processor=None,  # type: ignore
        session_queue=None,  # type: ignore
        urls=None,  # type: ignore
        workflow_records=None,  # type: ignore
    )


@pytest.fixture()
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    return Invoker(services=mock_services)


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker) -> None:
        self.invoker = invoker


def test_download_images_from_list(monkeypatch: Any, mock_invoker: Invoker) -> None:
    prepare_download_images_test(monkeypatch, mock_invoker)

    response = client.post("/api/v1/images/download", json={"image_names": ["test.png"]})

    assert response.status_code == 202


def test_download_images_from_board_id_empty_image_name_list(monkeypatch: Any, mock_invoker: Invoker) -> None:
    prepare_download_images_test(monkeypatch, mock_invoker)

    response = client.post("/api/v1/images/download", json={"image_names": [], "board_id": "test"})

    assert response.status_code == 202


def prepare_download_images_test(monkeypatch: Any, mock_invoker: Invoker) -> None:
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", MockApiDependencies(mock_invoker))

    def mock_add_task(*args, **kwargs):
        return None

    monkeypatch.setattr(BackgroundTasks, "add_task", mock_add_task)


def test_download_images_with_empty_image_list_and_no_board_id(monkeypatch: Any, mock_invoker: Invoker) -> None:
    prepare_download_images_test(monkeypatch, mock_invoker)

    response = client.post("/api/v1/images/download", json={"image_names": []})

    assert response.status_code == 400


def test_get_bulk_download_image(tmp_path: Path, monkeypatch: Any, mock_invoker: Invoker) -> None:
    mock_file: Path = tmp_path / "test.zip"
    mock_file.write_text("contents")

    monkeypatch.setattr(mock_invoker.services.bulk_download, "get_path", lambda x: str(mock_file))
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", MockApiDependencies(mock_invoker))

    def mock_add_task(*args, **kwargs):
        return None

    monkeypatch.setattr(BackgroundTasks, "add_task", mock_add_task)

    response = client.get("/api/v1/images/download/test.zip")

    assert response.status_code == 200
    assert response.content == b"contents"


def test_get_bulk_download_image_not_found(monkeypatch: Any, mock_invoker: Invoker) -> None:
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", MockApiDependencies(mock_invoker))

    def mock_add_task(*args, **kwargs):
        return None

    monkeypatch.setattr(BackgroundTasks, "add_task", mock_add_task)

    response = client.get("/api/v1/images/download/test.zip")

    assert response.status_code == 404


def test_get_bulk_download_image_image_deleted_after_response(
    monkeypatch: Any, mock_invoker: Invoker, tmp_path: Path
) -> None:
    mock_file: Path = tmp_path / "test.zip"
    mock_file.write_text("contents")

    monkeypatch.setattr(mock_invoker.services.bulk_download, "get_path", lambda x: str(mock_file))
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", MockApiDependencies(mock_invoker))

    client.get("/api/v1/images/download/test.zip")

    assert not (tmp_path / "test.zip").exists()
