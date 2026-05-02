"""Tests for the utilities router."""

import os
from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
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
def client(invokeai_root_dir: Path):
    os.environ["INVOKEAI_ROOT"] = invokeai_root_dir.as_posix()
    return TestClient(app)


def setup_test_user(
    mock_invoker: Invoker,
    email: str,
    display_name: str,
    password: str = "TestPass123",
    is_admin: bool = False,
) -> str:
    user_service = mock_invoker.services.users
    user_data = UserCreateRequest(
        email=email,
        display_name=display_name,
        password=password,
        is_admin=is_admin,
    )
    user = user_service.create(user_data)
    return user.user_id


def get_user_token(client: TestClient, email: str, password: str = "TestPass123") -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password, "remember_me": False},
    )
    assert response.status_code == 200
    return response.json()["token"]


@pytest.fixture
def patch_utilities_dependencies(monkeypatch: Any, mock_invoker: Invoker, invokeai_root_dir: Path):
    mock_invoker.services.configuration._root = invokeai_root_dir
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.utilities.ApiDependencies", mock_deps)
    yield mock_invoker


@pytest.fixture
def enable_multiuser(
    patch_utilities_dependencies: Invoker,
):
    patch_utilities_dependencies.services.configuration.multiuser = True
    yield patch_utilities_dependencies


@pytest.fixture
def admin_token(setup_jwt_secret: None, enable_multiuser: Invoker, client: TestClient):
    setup_test_user(enable_multiuser, "admin@test.com", "Test Admin", is_admin=True)
    return get_user_token(client, "admin@test.com")


def test_list_user_fonts_requires_auth(enable_multiuser: Invoker, client: TestClient, invokeai_root_dir: Path) -> None:
    fonts_dir = invokeai_root_dir / "Fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    response = client.get("/api/v1/utilities/fonts")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_user_font_file_requires_auth(
    enable_multiuser: Invoker, client: TestClient, invokeai_root_dir: Path
) -> None:
    fonts_dir = invokeai_root_dir / "Fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    response = client.get("/api/v1/utilities/fonts/MyFont.ttf")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_list_user_fonts_allows_authenticated_access(
    admin_token: str, client: TestClient, invokeai_root_dir: Path
) -> None:
    fonts_dir = invokeai_root_dir / "Fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    (fonts_dir / "MyFont.ttf").write_bytes(b"not-a-real-font")

    response = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["fonts"]) == 1
    assert data["fonts"][0]["id"] == "user:MyFont.ttf"


def test_get_user_font_file_rejects_symlink(
    admin_token: str, client: TestClient, invokeai_root_dir: Path, tmp_path: Path
) -> None:
    fonts_dir = invokeai_root_dir / "Fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    outside_file = tmp_path / "outside.ttf"
    outside_file.write_bytes(b"outside-font")
    symlink_path = fonts_dir / "linked.ttf"

    try:
        symlink_path.symlink_to(outside_file)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are not available in this test environment")

    response = client.get("/api/v1/utilities/fonts/linked.ttf", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_list_user_fonts_skips_symlinked_files(
    admin_token: str, client: TestClient, invokeai_root_dir: Path, tmp_path: Path
) -> None:
    fonts_dir = invokeai_root_dir / "Fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    outside_file = tmp_path / "outside.ttf"
    outside_file.write_bytes(b"outside-font")
    symlink_path = fonts_dir / "linked.ttf"

    try:
        symlink_path.symlink_to(outside_file)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are not available in this test environment")

    response = client.get("/api/v1/utilities/fonts", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["fonts"] == []
