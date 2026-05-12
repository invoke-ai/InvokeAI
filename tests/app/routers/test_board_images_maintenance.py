from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
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
    monkeypatch.setattr("invokeai.app.api.routers.board_images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)

    response = client.request(method, path, json=json_body)

    assert response.status_code == 409
    assert response.json()["detail"] == "Image storage maintenance is active"
