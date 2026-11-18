from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.routers import app_info
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.server_output.server_output_store import MAX_CURSOR_LENGTH, ServerOutputStore
from invokeai.app.services.users.users_common import UserUpdateRequest
from tests.app.routers.conftest import (
    _PATCHED_API_DEPENDENCIES_MODULES,
    MockApiDependencies,
    _auth,
    _create_user,
    _login,
)

READ = "/api/v1/app/server_output"
CONFIG = "/api/v1/app/frontend_config"


@pytest.fixture
def store() -> ServerOutputStore:
    store = ServerOutputStore()
    store.open_channel().write("secret transcript line\n")
    return store


@pytest.fixture
def single_user(monkeypatch: pytest.MonkeyPatch, mock_invoker: Invoker) -> None:
    dependencies = MockApiDependencies(mock_invoker)
    for module_path in _PATCHED_API_DEPENDENCIES_MODULES:
        monkeypatch.setattr(f"{module_path}.ApiDependencies", dependencies)


def provide(monkeypatch: pytest.MonkeyPatch, store: ServerOutputStore | None) -> None:
    monkeypatch.setattr(app_info.ApiDependencies, "server_output", store, raising=False)


def can_view(client: TestClient, headers: dict[str, str] | None = None) -> bool:
    response = client.get(CONFIG, headers=headers)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    return response.json()["can_view_server_output"]


def test_single_user_reads_snapshot_then_an_unchanged_cursor(
    single_user: None, monkeypatch: pytest.MonkeyPatch, client: TestClient, store: ServerOutputStore
) -> None:
    provide(monkeypatch, store)
    assert can_view(client)

    first = client.get(READ)
    assert first.status_code == 200
    assert first.headers["cache-control"] == "no-store"
    body = first.json()
    assert body["reset"] == "initial"
    assert [row["text"] for row in body["rows"]] == ["secret transcript line"]

    idle = client.get(READ, params={"cursor": body["cursor"]})
    assert idle.status_code == 200 and idle.headers["cache-control"] == "no-store"
    assert idle.json()["rows"] == [] and idle.json()["cursor"] == body["cursor"]


@pytest.mark.parametrize("capture", ["absent", "stopped"])
def test_without_live_capture_reads_are_unavailable(
    single_user: None, monkeypatch: pytest.MonkeyPatch, client: TestClient, store: ServerOutputStore, capture: str
) -> None:
    if capture == "stopped":
        store.stop()  # what a failed mirror does: the transcript no longer follows the console
    provide(monkeypatch, store if capture == "stopped" else None)
    assert not can_view(client)

    response = client.get(READ)
    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"detail": app_info.SERVER_OUTPUT_UNAVAILABLE}


@pytest.mark.parametrize("cursor", ["not-a-cursor", "f" * 32 + ".1.1.1x", "x" * (MAX_CURSOR_LENGTH + 1)])
def test_malformed_cursors_get_a_fixed_error(
    single_user: None, monkeypatch: pytest.MonkeyPatch, client: TestClient, store: ServerOutputStore, cursor: str
) -> None:
    provide(monkeypatch, store)

    response = client.get(READ, params={"cursor": cursor})
    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"detail": "Invalid server output cursor"}
    assert cursor not in response.text


def test_multiuser_reads_need_a_current_administrator(
    enable_multiuser: Any,
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    mock_invoker: Invoker,
    store: ServerOutputStore,
    admin_token: str,
    user1_token: str,
) -> None:
    provide(monkeypatch, store)

    for headers, status in [(None, 401), (_auth(user1_token), 403), (_auth(admin_token), 200)]:
        response = client.get(READ, params={"cursor": "malformed"} if status != 200 else None, headers=headers)
        assert response.status_code == status
        assert response.headers["cache-control"] == "no-store"
        assert ("secret transcript line" in response.text) is (status == 200)
    assert client.get(CONFIG).status_code == 401
    assert not can_view(client, _auth(user1_token))
    assert can_view(client, _auth(admin_token))

    # Demoting an administrator through the user service refuses their very next read, with the same token.
    _create_user(mock_invoker, "admin2@test.com", "Admin Two", is_admin=True)
    second_admin = _login(client, "admin2@test.com")
    assert client.get(READ, headers=_auth(second_admin)).status_code == 200
    user = mock_invoker.services.users.get_by_email("admin2@test.com")
    assert user is not None
    mock_invoker.services.users.update(user.user_id, UserUpdateRequest(is_admin=False))

    assert client.get(READ, headers=_auth(second_admin)).status_code == 403
    assert not can_view(client, _auth(second_admin))
