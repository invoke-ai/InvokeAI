from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.auth_dependencies import get_current_user_or_default
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.auth.token_service import TokenData
from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.boards.boards_common import BoardDTO
from invokeai.app.services.invoker import Invoker


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker) -> None:
        self.invoker = invoker


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("post", "/api/v1/board_images/", {"board_id": "board-id", "image_name": "image.png"}),
        ("delete", "/api/v1/board_images/", {"image_name": "image.png"}),
        ("post", "/api/v1/board_images/batch", {"board_id": "board-id", "image_names": ["image.png"]}),
        ("post", "/api/v1/board_images/batch/delete", {"image_names": ["image.png"]}),
    ],
)
def test_board_image_mutations_are_blocked_during_image_move_maintenance(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    client: TestClient,
    method: str,
    path: str,
    json_body: dict,
) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = True
    monkeypatch.setattr(mock_invoker.services.image_records, "get_user_id", MagicMock(return_value="system"))
    monkeypatch.setattr(mock_invoker.services.images, "get_dto", MagicMock(return_value=MagicMock(board_id=None)))
    monkeypatch.setattr(
        mock_invoker.services.boards,
        "get_dto",
        MagicMock(
            return_value=BoardDTO(
                board_id="board-id",
                board_name="Board",
                user_id="system",
                created_at="2024-01-01 00:00:00.000",
                updated_at="2024-01-01 00:00:00.000",
                archived=False,
                board_visibility=BoardVisibility.Private,
                cover_image_name=None,
                image_count=0,
                asset_count=0,
            )
        ),
    )
    monkeypatch.setattr("invokeai.app.api.routers.board_images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)

    response = client.request(method, path, json=json_body)

    assert response.status_code == 409
    assert response.json()["detail"] == "Image storage maintenance is active"


def test_board_image_mutation_checks_access_before_image_move_maintenance(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    client: TestClient,
) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = True
    monkeypatch.setattr(mock_invoker.services.image_records, "get_user_id", MagicMock(return_value="other-user"))
    monkeypatch.setattr(
        mock_invoker.services.boards,
        "get_dto",
        MagicMock(
            return_value=BoardDTO(
                board_id="board-id",
                board_name="Board",
                user_id="system",
                created_at="2024-01-01 00:00:00.000",
                updated_at="2024-01-01 00:00:00.000",
                archived=False,
                board_visibility=BoardVisibility.Private,
                cover_image_name=None,
                image_count=0,
                asset_count=0,
            )
        ),
    )
    monkeypatch.setattr("invokeai.app.api.routers.board_images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)

    async def current_user_override() -> TokenData:
        return TokenData(user_id="request-user", email="request-user@example.com", is_admin=False)

    app.dependency_overrides[get_current_user_or_default] = current_user_override
    try:
        response = client.post("/api/v1/board_images/", json={"board_id": "board-id", "image_name": "image.png"})

        assert response.status_code == 403
        mock_invoker.services.image_moves.is_maintenance_active.assert_not_called()
    finally:
        app.dependency_overrides.pop(get_current_user_or_default, None)


def test_delete_board_with_images_is_blocked_during_image_move_maintenance(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    client: TestClient,
) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = True
    mock_invoker.services.images.delete_images_on_board = MagicMock()
    mock_invoker.services.boards.get_dto = MagicMock(
        return_value=BoardDTO(
            board_id="board-id",
            board_name="Board",
            user_id="system",
            created_at="2024-01-01 00:00:00.000",
            updated_at="2024-01-01 00:00:00.000",
            archived=False,
            board_visibility=BoardVisibility.Private,
            cover_image_name=None,
            image_count=0,
            asset_count=0,
        )
    )
    monkeypatch.setattr("invokeai.app.api.routers.boards.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)

    response = client.delete("/api/v1/boards/board-id?include_images=true")

    assert response.status_code == 409
    assert response.json()["detail"] == "Image storage maintenance is active"
    mock_invoker.services.images.delete_images_on_board.assert_not_called()
