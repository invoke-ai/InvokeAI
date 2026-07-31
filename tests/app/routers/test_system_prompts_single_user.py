"""Single-user tests for the /api/v1/system_prompts router.

The multi-user tests cover the ownership checks. Single-user installs skip those checks
entirely, which is exactly where the delete contract used to diverge: DELETE reported 200 for
ids that GET 404s on, and deleting the same row twice succeeded twice.
"""

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invoker import Invoker
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_services() -> InvocationServices:
    from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
    from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
    from invokeai.app.services.boards.boards_default import BoardService
    from invokeai.app.services.bulk_download.bulk_download_default import BulkDownloadService
    from invokeai.app.services.client_state_persistence.client_state_persistence_sqlite import (
        ClientStatePersistenceSqlite,
    )
    from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
    from invokeai.app.services.images.images_default import ImageService
    from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
    from invokeai.app.services.invocation_stats.invocation_stats_default import InvocationStatsService
    from invokeai.app.services.system_prompt_records.system_prompt_records_sqlite import (
        SqliteSystemPromptRecordsStorage,
    )
    from invokeai.app.services.users.users_default import UserService
    from tests.test_nodes import TestEventService

    configuration = InvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    logger = InvokeAILogger.get_logger()
    db = create_mock_sqlite_database(configuration, logger)

    return InvocationServices(
        board_image_records=SqliteBoardImageRecordStorage(db=db),
        board_images=None,  # type: ignore
        board_records=SqliteBoardRecordStorage(db=db),
        boards=BoardService(),
        bulk_download=BulkDownloadService(),
        configuration=configuration,
        events=TestEventService(),
        image_files=None,  # type: ignore
        image_records=SqliteImageRecordStorage(db=db),
        images=ImageService(),
        invocation_cache=MemoryInvocationCache(max_cache_size=0),
        logger=logging,  # type: ignore
        model_images=None,  # type: ignore
        model_manager=None,  # type: ignore
        download_queue=None,  # type: ignore
        names=None,  # type: ignore
        performance_statistics=InvocationStatsService(),
        session_processor=None,  # type: ignore
        session_queue=None,  # type: ignore
        urls=None,  # type: ignore
        workflow_records=None,  # type: ignore
        tensors=None,  # type: ignore
        conditioning=None,  # type: ignore
        style_preset_records=None,  # type: ignore
        style_preset_image_files=None,  # type: ignore
        system_prompt_records=SqliteSystemPromptRecordsStorage(db=db),
        workflow_thumbnails=None,  # type: ignore
        model_relationship_records=None,  # type: ignore
        model_relationships=None,  # type: ignore
        client_state_persistence=ClientStatePersistenceSqlite(db=db),
        project_records=None,  # type: ignore
        users=UserService(db),
        wildcard_records=None,  # type: ignore
        external_generation=None,  # type: ignore
        videos=None,  # type: ignore
        video_files=None,  # type: ignore
        video_records=None,  # type: ignore
        board_video_records=None,  # type: ignore
        gallery=None,  # type: ignore
    )


@pytest.fixture
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    return Invoker(services=mock_services)


@pytest.fixture
def single_user(monkeypatch: Any, mock_invoker: Invoker):
    mock_invoker.services.configuration.multiuser = False
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.system_prompts.ApiDependencies", mock_deps)
    yield


def _create_prompt(client: TestClient, name: str = "p", content: str = "c") -> dict:
    r = client.post("/api/v1/system_prompts/", json={"name": name, "content": content})
    assert r.status_code == 200, r.text
    return r.json()


def test_delete_of_unknown_id_returns_404(single_user: Any, client: TestClient) -> None:
    # GET and DELETE must agree: an id that does not exist is a 404 for both.
    assert client.get("/api/v1/system_prompts/i/does-not-exist").status_code == 404
    assert client.delete("/api/v1/system_prompts/i/does-not-exist").status_code == 404


def test_delete_is_not_idempotent_success(single_user: Any, client: TestClient) -> None:
    created = _create_prompt(client, name="temp")
    assert client.delete(f"/api/v1/system_prompts/i/{created['id']}").status_code == 200
    assert client.get(f"/api/v1/system_prompts/i/{created['id']}").status_code == 404
    # The second delete must not report success for a row that is already gone.
    assert client.delete(f"/api/v1/system_prompts/i/{created['id']}").status_code == 404


def test_create_is_public_in_single_user_mode(single_user: Any, client: TestClient) -> None:
    created = _create_prompt(client, name="shared by default")
    assert created["is_public"] is True
