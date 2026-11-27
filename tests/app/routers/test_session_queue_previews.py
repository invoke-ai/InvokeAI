"""`GET /queue/{queue_id}/previews` returns the caller's own latest preview frames only."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.events.events_common import InvocationProgressEvent
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


@pytest.fixture
def setup_jwt_secret():
    from invokeai.app.services.auth.token_service import set_jwt_secret

    set_jwt_secret("test-secret-key-for-unit-tests-only-do-not-use-in-production")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def enable_multiuser(monkeypatch: Any, mock_invoker: Invoker):
    mock_invoker.services.configuration.multiuser = True
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.session_queue.ApiDependencies", mock_deps)
    yield


def _create_user(mock_invoker: Invoker, email: str, display_name: str, is_admin: bool = False) -> str:
    user = mock_invoker.services.users.create(
        UserCreateRequest(email=email, display_name=display_name, password="TestPass123", is_admin=is_admin)
    )
    return user.user_id


def _login(client: TestClient, email: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": "TestPass123", "remember_me": False})
    assert response.status_code == 200, response.text
    return response.json()["token"]


def _event(item_id: int, user_id: str) -> InvocationProgressEvent:
    return InvocationProgressEvent(
        queue_id="default",
        item_id=item_id,
        batch_id="batch-1",
        user_id=user_id,
        session_id=f"session-{item_id}",
        invocation={"type": "add", "id": "node-1", "a": 1, "b": 2},
        invocation_source_id="node-1",
        message="Denoising",
        percentage=0.5,
        image={"width": 64, "height": 64, "dataURL": "data:image/jpeg;base64,frame"},
        revision=2,
    )


def test_previews_are_scoped_to_the_caller_even_for_admins(
    setup_jwt_secret: None, enable_multiuser: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    admin_id = _create_user(mock_invoker, "admin@test.com", "Admin", is_admin=True)
    user_id = _create_user(mock_invoker, "user@test.com", "User")
    previews = mock_invoker.services.progress_previews
    previews.record(_event(1, admin_id))
    previews.record(_event(2, user_id))

    admin_token = _login(client, "admin@test.com")
    user_token = _login(client, "user@test.com")

    admin_response = client.get("/api/v1/queue/default/previews", headers={"Authorization": f"Bearer {admin_token}"})
    user_response = client.get("/api/v1/queue/default/previews", headers={"Authorization": f"Bearer {user_token}"})

    assert admin_response.status_code == 200, admin_response.text
    assert [entry["item_id"] for entry in admin_response.json()] == [1]
    assert admin_response.json()[0]["revision"] == 2
    assert admin_response.json()[0]["image"]["dataURL"] == "data:image/jpeg;base64,frame"
    assert user_response.status_code == 200
    assert [entry["item_id"] for entry in user_response.json()] == [2]


def test_previews_require_authentication_in_multiuser_mode(
    setup_jwt_secret: None, enable_multiuser: Any, client: TestClient
) -> None:
    assert client.get("/api/v1/queue/default/previews").status_code == 401
