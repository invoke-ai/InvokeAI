"""Shared fixtures and helpers for router-level multiuser/auth tests.

Note: This conftest intentionally does NOT redefine `mock_services` / `mock_invoker`
to avoid shadowing the project-level fixtures in `tests/conftest.py`. Instead, the
`enable_multiuser` fixture below injects MagicMock services for the routers that
have no real backing service in the default mock_services (download_queue,
style_preset_image_files, model_relationships, model_manager).

WARNING for future authors: `enable_multiuser` *unconditionally* replaces several
services on `mock_invoker.services` with MagicMocks. If you add a router test that
expects the real service (e.g. a SQLite-backed implementation), you must restore
that service in your own fixture before the request — otherwise your test will
pass against a MagicMock that accepts every call. `style_preset_records` is wired
up to a real SQLite storage here because cross-user filtering needs real SQL.

WARNING for new routers: every router that calls `ApiDependencies.invoker.services`
must be added to `_PATCHED_API_DEPENDENCIES_MODULES` below, or its routes will hit
the un-patched singleton and fail with `AttributeError: type object
'ApiDependencies' has no attribute 'invoker'`.

Existing test files that define their own `enable_multiuser` / `admin_token` / etc.
fixtures locally are NOT affected — pytest's local-shadows-conftest rule applies.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest


class MockApiDependencies(ApiDependencies):
    """Minimal stand-in that lets tests inject their own Invoker."""

    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


# Every module that reads `ApiDependencies.invoker.services` must be patched here
# so the in-test invoker is reachable. Add new routers / shared helpers to this
# list when they start touching ApiDependencies.
_PATCHED_API_DEPENDENCIES_MODULES = (
    "invokeai.app.api.auth_dependencies",
    "invokeai.app.api.routers.auth",
    "invokeai.app.api.routers.download_queue",
    "invokeai.app.api.routers.style_presets",
    "invokeai.app.api.routers.model_relationships",
    "invokeai.app.api.routers.utilities",
    "invokeai.app.api.routers.virtual_boards",
    "invokeai.app.api.routers.images",
    "invokeai.app.api.routers.workflows",
    "invokeai.app.api.routers._access",
    "invokeai.app.api.routers.model_manager",
    "invokeai.app.api.routers.app_info",
)


@pytest.fixture
def setup_jwt_secret():
    from invokeai.app.services.auth.token_service import set_jwt_secret

    set_jwt_secret("test-secret-key-for-unit-tests-only-do-not-use-in-production")


@pytest.fixture
def client():
    return TestClient(app)


def _create_user(mock_invoker: Invoker, email: str, display_name: str, is_admin: bool = False) -> str:
    user = mock_invoker.services.users.create(
        UserCreateRequest(email=email, display_name=display_name, password="TestPass123", is_admin=is_admin)
    )
    return user.user_id


def _login(client: TestClient, email: str) -> str:
    r = client.post("/api/v1/auth/login", json={"email": email, "password": "TestPass123", "remember_me": False})
    assert r.status_code == 200, f"Login failed for {email}: {r.text}"
    return r.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def enable_multiuser(monkeypatch: Any, mock_invoker: Invoker):
    """Enable multiuser mode and patch ApiDependencies across the routers covered by router-level tests.

    Replaces None-valued services with MagicMocks so that routes can run end-to-end.
    """
    from invokeai.app.services.style_preset_records.style_preset_records_sqlite import (
        SqliteStylePresetRecordsStorage,
    )

    mock_invoker.services.configuration.multiuser = True

    # Replace services that are None in the default mock_services with MagicMocks.
    mock_invoker.services.download_queue = MagicMock()
    mock_invoker.services.style_preset_image_files = MagicMock()
    mock_invoker.services.model_relationships = MagicMock()
    mock_invoker.services.model_manager = MagicMock()
    mock_invoker.services.workflow_thumbnails = MagicMock()

    # Style preset records uses a real SQLite-backed storage on the same in-memory
    # database that image_records was wired up against. This lets cross-user tests
    # exercise the actual filter SQL instead of asserting on MagicMock calls.
    mock_invoker.services.style_preset_records = SqliteStylePresetRecordsStorage(
        db=mock_invoker.services.image_records._db
    )

    # Required by board_image_records-touching helpers in some tests.
    if mock_invoker.services.board_images is None:
        mock_board_images = MagicMock()
        mock_board_images.get_all_board_image_names_for_board.return_value = []
        mock_invoker.services.board_images = mock_board_images

    mock_deps = MockApiDependencies(mock_invoker)
    for module_path in _PATCHED_API_DEPENDENCIES_MODULES:
        monkeypatch.setattr(f"{module_path}.ApiDependencies", mock_deps)
    yield


@pytest.fixture
def admin_token(setup_jwt_secret: None, enable_multiuser: Any, mock_invoker: Invoker, client: TestClient) -> str:
    _create_user(mock_invoker, "admin@test.com", "Admin", is_admin=True)
    return _login(client, "admin@test.com")


@pytest.fixture
def user1_token(enable_multiuser: Any, mock_invoker: Invoker, client: TestClient, admin_token: str) -> str:
    _create_user(mock_invoker, "user1@test.com", "User One")
    return _login(client, "user1@test.com")


@pytest.fixture
def user2_token(enable_multiuser: Any, mock_invoker: Invoker, client: TestClient, admin_token: str) -> str:
    _create_user(mock_invoker, "user2@test.com", "User Two")
    return _login(client, "user2@test.com")
